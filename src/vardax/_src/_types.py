"""Batch and LSTM state type definitions for vardax.

All containers are ``eqx.Module``s (not ``NamedTuple``) so they are proper
JAX pytrees with method support and compose cleanly with
``eqx.filter_jit`` / ``eqx.filter_vmap`` / ``eqx.filter_value_and_grad``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Batch1D(eqx.Module):
    """Batch of 1-D spatiotemporal data.

    Attributes:
        input: Observed (masked) input field of shape ``(B, T, N)``.
        mask: Binary observation mask of shape ``(B, T, N)``.
        target: Ground-truth field of shape ``(B, T, N)``. Optional; absent
            in operational use where only observations are available.
    """

    input: Float[Array, "B T N"]
    mask: Float[Array, "B T N"]
    target: Float[Array, "B T N"] | None = None


class Batch2D(eqx.Module):
    """Batch of 2-D spatiotemporal data.

    Attributes:
        input: Observed (masked) input field of shape ``(B, T, H, W)``.
        mask: Binary observation mask of shape ``(B, T, H, W)``.
        target: Ground-truth field of shape ``(B, T, H, W)``. Optional.
    """

    input: Float[Array, "B T H W"]
    mask: Float[Array, "B T H W"]
    target: Float[Array, "B T H W"] | None = None


class Batch2DMultivar(eqx.Module):
    """Batch of 2-D multivariate spatiotemporal data.

    Attributes:
        input: Observed (masked) input field of shape ``(B, T, C, H, W)``.
        mask: Binary observation mask of shape ``(B, T, C, H, W)``.
        target: Ground-truth field of shape ``(B, T, C, H, W)``. Optional.
    """

    input: Float[Array, "B T C H W"]
    mask: Float[Array, "B T C H W"]
    target: Float[Array, "B T C H W"] | None = None


class LSTMState1D(eqx.Module):
    """Hidden state for a 1-D ConvLSTM gradient modulator.

    Attributes:
        h: Hidden state tensor of shape ``(B, H_dim, N)``.
        c: Cell state tensor of shape ``(B, H_dim, N)``.
    """

    h: Float[Array, "B H_dim N"]
    c: Float[Array, "B H_dim N"]

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        hidden_dim: int,
        seq_len: int,
    ) -> LSTMState1D:
        """Create a zero-initialised LSTM state."""
        return cls(
            h=jnp.zeros((batch_size, hidden_dim, seq_len)),
            c=jnp.zeros((batch_size, hidden_dim, seq_len)),
        )


class LSTMState2D(eqx.Module):
    """Hidden state for a 2-D ConvLSTM gradient modulator.

    Attributes:
        h: Hidden state tensor of shape ``(B, H_dim, H, W)``.
        c: Cell state tensor of shape ``(B, H_dim, H, W)``.
    """

    h: Float[Array, "B H_dim H W"]
    c: Float[Array, "B H_dim H W"]

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        hidden_dim: int,
        height: int,
        width: int,
    ) -> LSTMState2D:
        """Create a zero-initialised LSTM state."""
        return cls(
            h=jnp.zeros((batch_size, hidden_dim, height, width)),
            c=jnp.zeros((batch_size, hidden_dim, height, width)),
        )
