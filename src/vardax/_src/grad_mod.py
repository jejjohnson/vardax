"""ConvLSTM gradient modulators for FourDVarNet.

The gradient modulator takes the current gradient of the variational cost
(with respect to the state) and the current LSTM hidden state, and outputs
a modulated gradient update plus the new LSTM state.

Implemented in Equinox; convolutions are channels-first
(`eqx.nn.Conv1d` / `Conv2d`), with `jax.vmap` applied over the leading
batch axis.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._types import LSTMState1D, LSTMState2D


class ConvLSTMGradMod1D(eqx.Module):
    """1-D ConvLSTM-based gradient modulator.

    Accepts the concatenation of the current state and its gradient as input
    and produces a modulated gradient update (and updated LSTM state).

    Attributes:
        state_channels: Number of channels in the state / gradient (``T``).
        hidden_dim: Number of hidden channels in the LSTM.
        kernel_size: 1-D convolution kernel size.
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
        k1, k2, k3 = jax.random.split(key, 3)
        pad = kernel_size // 2
        self._gates_input_conv = eqx.nn.Conv1d(
            in_channels=2 * state_channels,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=k1,
        )
        self._gates_hidden_conv = eqx.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=k2,
        )
        self._output_conv = eqx.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=state_channels,
            kernel_size=kernel_size,
            padding=pad,
            key=k3,
        )

    def __call__(
        self,
        grad: Float[Array, "B T N"],
        state: Float[Array, "B T N"],
        lstm_state: LSTMState1D,
    ) -> tuple[Float[Array, "B T N"], LSTMState1D]:
        """Forward pass.

        Args:
            grad: Gradient of variational cost w.r.t. state, shape ``(B, T, N)``.
            state: Current state estimate, shape ``(B, T, N)``.
            lstm_state: Current LSTM hidden/cell state with hidden_dim channels.

        Returns:
            Tuple of (modulated gradient update, new LSTM state).
        """
        # eqx.nn.Conv1d operates on (channels, spatial). Our arrays are already
        # (B, channels, spatial) with T=channels for grad/state and H=channels
        # for the LSTM state — vmap over the leading batch dim.

        def _forward(
            grad_i: Float[Array, "T N"],
            state_i: Float[Array, "T N"],
            h_i: Float[Array, "H N"],
            c_i: Float[Array, "H N"],
        ) -> tuple[Float[Array, "T N"], Float[Array, "H N"], Float[Array, "H N"]]:
            x = jnp.concatenate([grad_i, state_i], axis=0)  # (2T, N)
            gates = self._gates_input_conv(x) + self._gates_hidden_conv(h_i)
            i, f, g, o = jnp.split(gates, 4, axis=0)
            i = jax.nn.sigmoid(i)
            f = jax.nn.sigmoid(f)
            g = jnp.tanh(g)
            o = jax.nn.sigmoid(o)
            c_new = f * c_i + i * g
            h_new = o * jnp.tanh(c_new)
            out = self._output_conv(h_new)
            return out, h_new, c_new

        out, h_new, c_new = jax.vmap(_forward)(grad, state, lstm_state.h, lstm_state.c)
        return out, LSTMState1D(h=h_new, c=c_new)


class ConvLSTMGradMod2D(eqx.Module):
    """2-D ConvLSTM-based gradient modulator.

    Attributes:
        state_channels: Number of time channels in the state / gradient.
        hidden_dim: Number of hidden channels in the LSTM.
        kernel_size: 2-D convolution kernel size.
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
        k1, k2, k3 = jax.random.split(key, 3)
        pad = kernel_size // 2
        self._gates_input_conv = eqx.nn.Conv2d(
            in_channels=2 * state_channels,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=k1,
        )
        self._gates_hidden_conv = eqx.nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            key=k2,
        )
        self._output_conv = eqx.nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=state_channels,
            kernel_size=kernel_size,
            padding=pad,
            key=k3,
        )

    def __call__(
        self,
        grad: Float[Array, "B T H W"],
        state: Float[Array, "B T H W"],
        lstm_state: LSTMState2D,
    ) -> tuple[Float[Array, "B T H W"], LSTMState2D]:
        """Forward pass.

        Args:
            grad: Gradient of variational cost w.r.t. state, shape ``(B, T, H, W)``.
            state: Current state estimate, shape ``(B, T, H, W)``.
            lstm_state: Current LSTM hidden/cell state.

        Returns:
            Tuple of (modulated gradient update, new LSTM state).
        """

        def _forward(
            grad_i: Float[Array, "T H W"],
            state_i: Float[Array, "T H W"],
            h_i: Float[Array, "H_dim H W"],
            c_i: Float[Array, "H_dim H W"],
        ) -> tuple[
            Float[Array, "T H W"],
            Float[Array, "H_dim H W"],
            Float[Array, "H_dim H W"],
        ]:
            x = jnp.concatenate([grad_i, state_i], axis=0)  # (2T, H, W)
            gates = self._gates_input_conv(x) + self._gates_hidden_conv(h_i)
            i, f, g, o = jnp.split(gates, 4, axis=0)
            i = jax.nn.sigmoid(i)
            f = jax.nn.sigmoid(f)
            g = jnp.tanh(g)
            o = jax.nn.sigmoid(o)
            c_new = f * c_i + i * g
            h_new = o * jnp.tanh(c_new)
            out = self._output_conv(h_new)
            return out, h_new, c_new

        out, h_new, c_new = jax.vmap(_forward)(grad, state, lstm_state.h, lstm_state.c)
        return out, LSTMState2D(h=h_new, c=c_new)
