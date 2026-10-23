"""Batch and LSTM state type definitions for vardax.

All containers are ``eqx.Module``s (not ``NamedTuple``) so they are proper
JAX pytrees with method support and compose cleanly with
``eqx.filter_jit`` / ``eqx.filter_vmap`` / ``eqx.filter_value_and_grad``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


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


class PointObsBatch(eqx.Module):
    r"""Scattered point observations, padded to a fixed count.

    Observation $i$ has coordinates $s_i$, value $y_i$, error standard
    deviation $\sigma_i$ and a validity flag $m_i$; padding rows have
    ``valid=False`` and contribute exactly nothing to the cost or its
    gradient, whatever their other fields hold (``NaN`` included). The
    observation cost is
    $J_o = \tfrac12 \sum_i m_i (y_i - \mathcal H(x)(s_i, t_i))^2 / \sigma_i^2$
    ([`point_obs_cost`][vardax.point_obs_cost]).

    Every array field may carry a leading batch axis ``B`` (``coords``
    ``(B, N_obs, D)`` and so on); the models then return one analysis per
    sample, and without it a single analysis.

    Attributes:
        coords: Coordinates ``(N_obs, D)``, e.g. ``(x,)``, ``(y, x)`` or
            ``(lon, lat, t)``.
        values: Observed values ``(N_obs,)``.
        sigma: Per-observation error standard deviations ``(N_obs,)``.
        valid: ``False`` on padding rows ``(N_obs,)``.
        time_index: Window step of each observation ``(N_obs,)``, for
            gridded-state models (the trajectory state it is compared
            with). ``None``: every observation is at step 0.
        n_times: Number of states in the assimilation window (rollout
            steps + 1) for the 4D-Var models; required with
            ``time_index``. Static. ``None`` means 1.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.5], [2.0], [0.0]]),
        ...     values=jnp.array([1.0, 2.0, 0.0]),
        ...     sigma=jnp.ones(3),
        ...     valid=jnp.array([True, True, False]),
        ...     time_index=jnp.array([0, 3, 0]),
        ...     n_times=4,
        ... )
        >>> batch.values.shape
        (3,)
    """

    coords: Float[Array, "N_obs D"]
    values: Float[Array, " N_obs"]
    sigma: Float[Array, " N_obs"]
    valid: Bool[Array, " N_obs"]
    time_index: Int[Array, " N_obs"] | None = None
    n_times: int | None = eqx.field(static=True, default=None)


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
