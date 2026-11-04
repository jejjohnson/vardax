"""ConvLSTM gradient modulators for FourDVarNet: g_mod: (∇ₓ𝒥, x, c) ↦ (Δx, c′).

Implemented in Equinox with channels-first convolutions (`eqx.nn.Conv1d`, `Conv2d`),
with `jax.vmap` applied over the leading batch axis.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._types import LSTMState1D, LSTMState2D


class ConvLSTMGradMod1D(eqx.Module):
    """1-D ConvLSTM gradient modulator.

        g_mod: (∇ₓ𝒥, x, c) ↦ (Δx, c′),   c = (c_hid, c_cell)
        [a_in; a_forget; a_cand; a_out] = W_in [∇ₓ𝒥; x] + W_hid c_hid
        c_cell′ = σ(a_forget) ⊙ c_cell + σ(a_in) ⊙ tanh(a_cand)
        c_hid′ = σ(a_out) ⊙ tanh(c_cell′)
        Δx = W_out c_hid′

    The carry c holds the LSTM hidden and cell states, σ is the logistic sigmoid, and
    W_in, W_hid and W_out are convolutions.

    Algorithm:
        1. Stack the gradient over the state along the channel axis and apply W_in;
           add W_hid applied to the hidden state.
        2. Split into the input, forget, candidate and output activations.
        3. Update the cell state c_cell′ and the hidden state c_hid′.
        4. Map c_hid′ back to the state channels with W_out.

    Args:
        state_channels: Number of channels in the state and gradient (``T``).
        hidden_dim: Number of hidden channels in the LSTM.
        kernel_size: 1-D convolution kernel size.
        key: PRNG key for the convolution weights.

    Examples:
        The update has the shape of the gradient, the carry the shape of the hidden
        state:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> mod = vdx.ConvLSTMGradMod1D(
        ...     state_channels=3, hidden_dim=5, key=jax.random.PRNGKey(0)
        ... )
        >>> grad = jnp.ones((2, 3, 4))
        >>> update, lstm = mod(grad, grad, vdx.LSTMState1D.zeros(2, 5, 4))
        >>> update.shape, lstm.hidden.shape
        ((2, 3, 4), (2, 5, 4))

        The carry persists across solver steps, so a second call with the returned
        carry gives a different update:

        >>> update2, _ = mod(grad, grad, lstm)
        >>> bool(jnp.allclose(update, update2))
        False

        It conforms to the [`GradModulator`][vardax.GradModulator] protocol:

        >>> isinstance(mod, vdx.GradModulator)
        True
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
        # an odd kernel keeps the spatial size
        padding = kernel_size // 2
        self._gates_input_conv = eqx.nn.Conv1d(
            in_channels=2 * state_channels,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            key=key_input,
        )
        self._gates_hidden_conv = eqx.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            key=key_hidden,
        )
        self._output_conv = eqx.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=state_channels,
            kernel_size=kernel_size,
            padding=padding,
            key=key_output,
        )

    def __call__(
        self,
        grad: Float[Array, "B T N"],
        state: Float[Array, "B T N"],
        lstm_state: LSTMState1D,
    ) -> tuple[Float[Array, "B T N"], LSTMState1D]:
        """Turn a cost gradient into a state update: (Δx, c′) = g_mod(∇ₓ𝒥, x, c).

        Args:
            grad: Variational-cost gradient ∇ₓ𝒥.
            state: Current state x.
            lstm_state: Carry c = (c_hid, c_cell) with ``hidden_dim`` channels.

        Returns:
            The state update Δx and the new carry c′.
        """

        # eqx.nn.Conv1d takes (channels, space), and the arrays already have the
        # channels second (T for the gradient and state, H for the carry), so only
        # the batch axis needs a vmap
        def modulate_sample(
            grad_sample: Float[Array, "T N"],
            state_sample: Float[Array, "T N"],
            hidden_sample: Float[Array, "H N"],
            cell_sample: Float[Array, "H N"],
        ) -> tuple[Float[Array, "T N"], Float[Array, "H N"], Float[Array, "H N"]]:
            # [∇ₓ𝒥; x]
            stacked_input = einx.id("a n, b n -> (a + b) n", grad_sample, state_sample)
            # W_in [∇ₓ𝒥; x]
            input_term = self._gates_input_conv(stacked_input)
            # W_hid c_hid
            hidden_term = self._gates_hidden_conv(hidden_sample)
            # [a_in; a_forget; a_cand; a_out] = W_in [∇ₓ𝒥; x] + W_hid c_hid
            activations = input_term + hidden_term
            input_logit, forget_logit, candidate_logit, output_logit = einx.id(
                "(gate h) n -> gate h n", activations, gate=4
            )
            # σ(a_in)
            input_gate = jax.nn.sigmoid(input_logit)
            # σ(a_forget)
            forget_gate = jax.nn.sigmoid(forget_logit)
            # tanh(a_cand)
            cell_candidate = jnp.tanh(candidate_logit)
            # σ(a_out)
            output_gate = jax.nn.sigmoid(output_logit)
            # c_cell′ = σ(a_forget) ⊙ c_cell + σ(a_in) ⊙ tanh(a_cand)
            new_cell = forget_gate * cell_sample + input_gate * cell_candidate
            # c_hid′ = σ(a_out) ⊙ tanh(c_cell′)
            new_hidden = output_gate * jnp.tanh(new_cell)
            # Δx = W_out c_hid′
            update = self._output_conv(new_hidden)
            return update, new_hidden, new_cell

        update, new_hidden, new_cell = jax.vmap(modulate_sample)(
            grad, state, lstm_state.hidden, lstm_state.cell
        )
        return update, LSTMState1D(hidden=new_hidden, cell=new_cell)


class ConvLSTMGradMod2D(eqx.Module):
    """2-D ConvLSTM gradient modulator.

        g_mod: (∇ₓ𝒥, x, c) ↦ (Δx, c′),   c = (c_hid, c_cell)
        [a_in; a_forget; a_cand; a_out] = W_in [∇ₓ𝒥; x] + W_hid c_hid
        c_cell′ = σ(a_forget) ⊙ c_cell + σ(a_in) ⊙ tanh(a_cand)
        c_hid′ = σ(a_out) ⊙ tanh(c_cell′)
        Δx = W_out c_hid′

    The same recurrence as [`ConvLSTMGradMod1D`][vardax.ConvLSTMGradMod1D], with 2-D
    convolutions: the carry c holds the LSTM hidden and cell states, σ is the logistic
    sigmoid, and W_in, W_hid and W_out are convolutions.

    Algorithm:
        1. Stack the gradient over the state along the channel axis and apply W_in;
           add W_hid applied to the hidden state.
        2. Split into the input, forget, candidate and output activations.
        3. Update the cell state c_cell′ and the hidden state c_hid′.
        4. Map c_hid′ back to the state channels with W_out.

    Args:
        state_channels: Number of time channels in the state and gradient.
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
        >>> update.shape, lstm.cell.shape
        ((1, 3, 4, 5), (1, 2, 4, 5))

        A zero gradient and zero state still yield a finite update:

        >>> zeros = jnp.zeros((1, 3, 4, 5))
        >>> out, _ = mod(zeros, zeros, vdx.LSTMState2D.zeros(1, 2, 4, 5))
        >>> bool(jnp.all(jnp.isfinite(out)))
        True

        It conforms to the [`GradModulator`][vardax.GradModulator] protocol:

        >>> isinstance(mod, vdx.GradModulator)
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
        # an odd kernel keeps the spatial size
        padding = kernel_size // 2
        self._gates_input_conv = eqx.nn.Conv2d(
            in_channels=2 * state_channels,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            key=key_input,
        )
        self._gates_hidden_conv = eqx.nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            key=key_hidden,
        )
        self._output_conv = eqx.nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=state_channels,
            kernel_size=kernel_size,
            padding=padding,
            key=key_output,
        )

    def __call__(
        self,
        grad: Float[Array, "B T H W"],
        state: Float[Array, "B T H W"],
        lstm_state: LSTMState2D,
    ) -> tuple[Float[Array, "B T H W"], LSTMState2D]:
        """Turn a cost gradient into a state update: (Δx, c′) = g_mod(∇ₓ𝒥, x, c).

        Args:
            grad: Variational-cost gradient ∇ₓ𝒥.
            state: Current state x.
            lstm_state: Carry c = (c_hid, c_cell) with ``hidden_dim`` channels.

        Returns:
            The state update Δx and the new carry c′.
        """

        def modulate_sample(
            grad_sample: Float[Array, "T H W"],
            state_sample: Float[Array, "T H W"],
            hidden_sample: Float[Array, "H_dim H W"],
            cell_sample: Float[Array, "H_dim H W"],
        ) -> tuple[
            Float[Array, "T H W"],
            Float[Array, "H_dim H W"],
            Float[Array, "H_dim H W"],
        ]:
            # [∇ₓ𝒥; x]
            stacked_input = einx.id(
                "a h w, b h w -> (a + b) h w", grad_sample, state_sample
            )
            # W_in [∇ₓ𝒥; x]
            input_term = self._gates_input_conv(stacked_input)
            # W_hid c_hid
            hidden_term = self._gates_hidden_conv(hidden_sample)
            # [a_in; a_forget; a_cand; a_out] = W_in [∇ₓ𝒥; x] + W_hid c_hid
            activations = input_term + hidden_term
            input_logit, forget_logit, candidate_logit, output_logit = einx.id(
                "(gate c) h w -> gate c h w", activations, gate=4
            )
            # σ(a_in)
            input_gate = jax.nn.sigmoid(input_logit)
            # σ(a_forget)
            forget_gate = jax.nn.sigmoid(forget_logit)
            # tanh(a_cand)
            cell_candidate = jnp.tanh(candidate_logit)
            # σ(a_out)
            output_gate = jax.nn.sigmoid(output_logit)
            # c_cell′ = σ(a_forget) ⊙ c_cell + σ(a_in) ⊙ tanh(a_cand)
            new_cell = forget_gate * cell_sample + input_gate * cell_candidate
            # c_hid′ = σ(a_out) ⊙ tanh(c_cell′)
            new_hidden = output_gate * jnp.tanh(new_cell)
            # Δx = W_out c_hid′
            update = self._output_conv(new_hidden)
            return update, new_hidden, new_cell

        update, new_hidden, new_cell = jax.vmap(modulate_sample)(
            grad, state, lstm_state.hidden, lstm_state.cell
        )
        return update, LSTMState2D(hidden=new_hidden, cell=new_cell)
