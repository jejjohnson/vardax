"""ConvLSTM gradient modulators for FourDVarNet.

The gradient modulator takes the current gradient of the variational cost
(with respect to the state) and the current LSTM hidden state, and outputs
a modulated gradient update plus the new LSTM state.

Implemented in Equinox; convolutions are channels-first
(`eqx.nn.Conv1d` / `Conv2d`), with `jax.vmap` applied over the leading
batch axis.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._types import LSTMState1D, LSTMState2D


class ConvLSTMGradMod1D(eqx.Module):
    r"""1-D ConvLSTM-based gradient modulator.

    Accepts the concatenation of the current state and its gradient as input
    and produces a modulated gradient update (and updated LSTM state).  With
    $g = \nabla J(u)$ and previous memory $(h, c)$ the gates are

    $$
    [i; f; \tilde c; o] = W_x [g; u] + W_h h, \quad
    c' = \sigma(f) \odot c + \sigma(i) \odot \tanh(\tilde c), \quad
    h' = \sigma(o) \odot \tanh(c'),
    $$

    and the update is $W_o h'$.

    Algorithm:
        1. Stack the gradient over the state along the channel axis and apply
           the input convolution; add the convolution of the hidden state.
        2. Split into the input, forget, candidate and output gates.
        3. Update the cell state $c'$ and the hidden state $h'$.
        4. Map $h'$ back to the state channels with the output convolution.

    Args:
        state_channels: Number of channels in the state / gradient (``T``).
        hidden_dim: Number of hidden channels in the LSTM.
        kernel_size: 1-D convolution kernel size.
        key: PRNG key for the convolution weights.

    Examples:
        The update has the shape of the gradient, the memory the shape of the
        hidden state:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> mod = vdx.ConvLSTMGradMod1D(
        ...     state_channels=3, hidden_dim=5, key=jax.random.PRNGKey(0)
        ... )
        >>> grad = jnp.ones((2, 3, 4))
        >>> update, lstm = mod(grad, grad, vdx.LSTMState1D.zeros(2, 5, 4))
        >>> update.shape, lstm.h.shape
        ((2, 3, 4), (2, 5, 4))

        The memory is carried across solver steps, so a second call with the
        returned state gives a different update:

        >>> update2, _ = mod(grad, grad, lstm)
        >>> bool(jnp.allclose(update, update2))
        False
    """

    state_channels: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    _gates_input_conv: eqx.nn.Conv1d
    _gates_hidden_conv: eqx.nn.Conv1d
    _output_conv: eqx.nn.Conv1d

    def __init__(
        self,
        state_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.state_channels = state_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        key_input, key_hidden, key_output = jax.random.split(key, 3)
        pad = kernel_size // 2
        self._gates_input_conv = eqx.nn.Conv1d(
            in_channels=2 * state_channels,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=key_input,
        )
        self._gates_hidden_conv = eqx.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=key_hidden,
        )
        self._output_conv = eqx.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=state_channels,
            kernel_size=kernel_size,
            padding=pad,
            key=key_output,
        )

    def __call__(
        self,
        grad: Float[Array, "B T N"],
        state: Float[Array, "B T N"],
        lstm_state: LSTMState1D,
    ) -> tuple[Float[Array, "B T N"], LSTMState1D]:
        """Forward pass.

        Args:
            grad: Gradient of the variational cost with respect to the state.
            state: Current state estimate.
            lstm_state: Current LSTM hidden/cell state with ``hidden_dim`` channels.

        Returns:
            The modulated gradient update and the new LSTM state.
        """
        # eqx.nn.Conv1d operates on (channels, spatial). Our arrays are already
        # (B, channels, spatial) with T=channels for grad/state and H=channels
        # for the LSTM state — vmap over the leading batch dim.

        def _forward(
            grad_sample: Float[Array, "T N"],
            state_sample: Float[Array, "T N"],
            hidden_sample: Float[Array, "H N"],
            cell_sample: Float[Array, "H N"],
        ) -> tuple[Float[Array, "T N"], Float[Array, "H N"], Float[Array, "H N"]]:
            # the input channels are the gradient stacked over the state
            stacked_input = einx.id("a n, b n -> (a + b) n", grad_sample, state_sample)
            # gates = W_x [∇J; u] + W_h h
            gates = self._gates_input_conv(stacked_input) + self._gates_hidden_conv(
                hidden_sample
            )
            # [i; f; g; o] = split(gates)
            input_gate, forget_gate, cell_candidate, output_gate = einx.id(
                "(gate h) n -> gate h n", gates, gate=4
            )
            input_gate = jax.nn.sigmoid(input_gate)
            forget_gate = jax.nn.sigmoid(forget_gate)
            cell_candidate = jnp.tanh(cell_candidate)
            output_gate = jax.nn.sigmoid(output_gate)
            # c' = f ⊙ c + i ⊙ g
            new_cell = forget_gate * cell_sample + input_gate * cell_candidate
            # h' = o ⊙ tanh(c')
            new_hidden = output_gate * jnp.tanh(new_cell)
            update = self._output_conv(new_hidden)
            return update, new_hidden, new_cell

        update, new_hidden, new_cell = jax.vmap(_forward)(
            grad, state, lstm_state.h, lstm_state.c
        )
        return update, LSTMState1D(h=new_hidden, c=new_cell)


class ConvLSTMGradMod2D(eqx.Module):
    r"""2-D ConvLSTM-based gradient modulator.

    Same recurrence as [`ConvLSTMGradMod1D`][vardax.ConvLSTMGradMod1D], with 2-D
    convolutions: with $g = \nabla J(u)$ and memory $(h, c)$,

    $$
    [i; f; \tilde c; o] = W_x [g; u] + W_h h, \quad
    c' = \sigma(f) \odot c + \sigma(i) \odot \tanh(\tilde c), \quad
    h' = \sigma(o) \odot \tanh(c'),
    $$

    and the update is $W_o h'$.

    Algorithm:
        1. Stack the gradient over the state along the channel axis and apply
           the input convolution; add the convolution of the hidden state.
        2. Split into the input, forget, candidate and output gates.
        3. Update the cell state $c'$ and the hidden state $h'$.
        4. Map $h'$ back to the state channels with the output convolution.

    Args:
        state_channels: Number of time channels in the state / gradient.
        hidden_dim: Number of hidden channels in the LSTM.
        kernel_size: 2-D convolution kernel size.
        key: PRNG key for the convolution weights.

    Examples:
        The update has the shape of the gradient:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> mod = vdx.ConvLSTMGradMod2D(
        ...     state_channels=3, hidden_dim=2, key=jax.random.PRNGKey(0)
        ... )
        >>> grad = jnp.ones((1, 3, 4, 5))
        >>> update, lstm = mod(grad, grad, vdx.LSTMState2D.zeros(1, 2, 4, 5))
        >>> update.shape, lstm.c.shape
        ((1, 3, 4, 5), (1, 2, 4, 5))

        A zero gradient and zero state still yield a finite update:

        >>> zeros = jnp.zeros((1, 3, 4, 5))
        >>> out, _ = mod(zeros, zeros, vdx.LSTMState2D.zeros(1, 2, 4, 5))
        >>> bool(jnp.all(jnp.isfinite(out)))
        True
    """

    state_channels: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    _gates_input_conv: eqx.nn.Conv2d
    _gates_hidden_conv: eqx.nn.Conv2d
    _output_conv: eqx.nn.Conv2d

    def __init__(
        self,
        state_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.state_channels = state_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        key_input, key_hidden, key_output = jax.random.split(key, 3)
        pad = kernel_size // 2
        self._gates_input_conv = eqx.nn.Conv2d(
            in_channels=2 * state_channels,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=key_input,
        )
        self._gates_hidden_conv = eqx.nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=key_hidden,
        )
        self._output_conv = eqx.nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=state_channels,
            kernel_size=kernel_size,
            padding=pad,
            key=key_output,
        )

    def __call__(
        self,
        grad: Float[Array, "B T H W"],
        state: Float[Array, "B T H W"],
        lstm_state: LSTMState2D,
    ) -> tuple[Float[Array, "B T H W"], LSTMState2D]:
        """Forward pass.

        Args:
            grad: Gradient of the variational cost with respect to the state.
            state: Current state estimate.
            lstm_state: Current LSTM hidden/cell state.

        Returns:
            The modulated gradient update and the new LSTM state.
        """

        def _forward(
            grad_sample: Float[Array, "T H W"],
            state_sample: Float[Array, "T H W"],
            hidden_sample: Float[Array, "H_dim H W"],
            cell_sample: Float[Array, "H_dim H W"],
        ) -> tuple[
            Float[Array, "T H W"],
            Float[Array, "H_dim H W"],
            Float[Array, "H_dim H W"],
        ]:
            # the input channels are the gradient stacked over the state
            stacked_input = einx.id(
                "a h w, b h w -> (a + b) h w", grad_sample, state_sample
            )
            # gates = W_x [∇J; u] + W_h h
            gates = self._gates_input_conv(stacked_input) + self._gates_hidden_conv(
                hidden_sample
            )
            # [i; f; g; o] = split(gates)
            input_gate, forget_gate, cell_candidate, output_gate = einx.id(
                "(gate c) h w -> gate c h w", gates, gate=4
            )
            input_gate = jax.nn.sigmoid(input_gate)
            forget_gate = jax.nn.sigmoid(forget_gate)
            cell_candidate = jnp.tanh(cell_candidate)
            output_gate = jax.nn.sigmoid(output_gate)
            # c' = f ⊙ c + i ⊙ g
            new_cell = forget_gate * cell_sample + input_gate * cell_candidate
            # h' = o ⊙ tanh(c')
            new_hidden = output_gate * jnp.tanh(new_cell)
            update = self._output_conv(new_hidden)
            return update, new_hidden, new_cell

        update, new_hidden, new_cell = jax.vmap(_forward)(
            grad, state, lstm_state.h, lstm_state.c
        )
        return update, LSTMState2D(h=new_hidden, c=new_cell)
