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
    r"""Batch of 1-D spatiotemporal data.

    Holds the observed field $y$, the observation mask $m$ and, optionally,
    the ground truth $x^\mathrm{gt}$, for samples of a 1-D state over time.

    Attributes:
        input: Observed (masked) input field $y$.
        mask: Binary observation mask $m$.
        target: Ground-truth field $x^\mathrm{gt}$. Optional; absent in
            operational use where only observations are available.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.Batch1D(input=jnp.zeros((2, 5, 8)), mask=jnp.ones((2, 5, 8)))
        >>> batch.input.shape, batch.target is None
        ((2, 5, 8), True)

        The batch is a pytree, so it passes through JAX transforms.

        >>> import jax
        >>> len(jax.tree_util.tree_leaves(batch))
        2

        It feeds the cost functions.

        >>> float(vardax.variational_cost(batch.input, batch, vardax.IdentityPrior()))
        0.0
    """

    # 4DVarNet batches are (B, T, N); OI and 3D-Var put (B, M) observations
    # through the same type, so only the batch axis is fixed
    input: Float[Array, " B *sample"]
    mask: Float[Array, " B *sample"]
    target: Float[Array, " B *sample"] | None = None


class Batch2D(eqx.Module):
    r"""Batch of 2-D spatiotemporal data.

    Holds the observed field $y$, the observation mask $m$ and, optionally,
    the ground truth $x^\mathrm{gt}$, for samples of a 2-D state over time.

    Attributes:
        input: Observed (masked) input field $y$.
        mask: Binary observation mask $m$.
        target: Ground-truth field $x^\mathrm{gt}$. Optional.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.Batch2D(
        ...     input=jnp.zeros((1, 3, 4, 4)),
        ...     mask=jnp.ones((1, 3, 4, 4)),
        ...     target=jnp.zeros((1, 3, 4, 4)),
        ... )
        >>> batch.input.shape
        (1, 3, 4, 4)

        It feeds the 2-D observation cost.

        >>> float(vardax.obs_cost_2d(batch.input + 1.0, batch.input, batch.mask))
        1.0
    """

    input: Float[Array, "B T H W"]
    mask: Float[Array, "B T H W"]
    target: Float[Array, "B T H W"] | None = None


class Batch2DMultivar(eqx.Module):
    r"""Batch of 2-D multivariate spatiotemporal data.

    Like [`Batch2D`][vardax.Batch2D] with an extra channel axis for the
    variables.

    Attributes:
        input: Observed (masked) input field $y$.
        mask: Binary observation mask $m$.
        target: Ground-truth field $x^\mathrm{gt}$. Optional.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.Batch2DMultivar(
        ...     input=jnp.zeros((1, 3, 2, 4, 4)), mask=jnp.ones((1, 3, 2, 4, 4))
        ... )
        >>> batch.mask.shape
        (1, 3, 2, 4, 4)

        The ground truth is optional.

        >>> batch.target is None
        True
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

    Every array field may carry a leading batch axis; the models then
    return one analysis per sample, and without it a single analysis.

    Attributes:
        coords: Coordinates $s_i$, e.g. ``(x,)``, ``(y, x)`` or
            ``(lon, lat, t)``.
        values: Observed values $y_i$.
        sigma: Per-observation error standard deviations $\sigma_i$.
        valid: Validity flags $m_i$, ``False`` on padding rows.
        time_index: Window step of each observation, for gridded-state
            models (the trajectory state it is compared with). ``None``:
            every observation is at step 0.
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

        Padding rows contribute nothing, even when they hold ``NaN``.

        >>> padded = vardax.PointObsBatch(
        ...     coords=jnp.zeros((2, 1)),
        ...     values=jnp.array([1.0, jnp.nan]),
        ...     sigma=jnp.ones(2),
        ...     valid=jnp.array([True, False]),
        ... )
        >>> int(padded.valid.sum())
        1
    """

    coords: Float[Array, "*batch N_obs D"]
    values: Float[Array, "*batch N_obs"]
    sigma: Float[Array, "*batch N_obs"]
    valid: Bool[Array, "*batch N_obs"]
    time_index: Int[Array, "*batch N_obs"] | None = None
    n_times: int | None = eqx.field(static=True, default=None)


class LSTMState1D(eqx.Module):
    """Hidden state for a 1-D ConvLSTM gradient modulator.

    Attributes:
        h: Hidden state tensor.
        c: Cell state tensor.

    Examples:
        >>> import vardax
        >>> state = vardax.LSTMState1D.zeros(batch_size=2, hidden_dim=3, seq_len=5)
        >>> state.h.shape, state.c.shape
        ((2, 3, 5), (2, 3, 5))

        Both tensors start at zero.

        >>> float(abs(state.h).sum() + abs(state.c).sum())
        0.0
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
        """Create a zero-initialised LSTM state.

        Args:
            batch_size: Number of samples $B$.
            hidden_dim: Hidden channel count.
            seq_len: Length $N$ of the 1-D axis.

        Returns:
            A state with zero hidden and cell tensors.
        """
        return cls(
            h=jnp.zeros((batch_size, hidden_dim, seq_len)),
            c=jnp.zeros((batch_size, hidden_dim, seq_len)),
        )


class LSTMState2D(eqx.Module):
    """Hidden state for a 2-D ConvLSTM gradient modulator.

    Attributes:
        h: Hidden state tensor.
        c: Cell state tensor.

    Examples:
        >>> import vardax
        >>> state = vardax.LSTMState2D.zeros(1, 4, height=3, width=2)
        >>> state.h.shape
        (1, 4, 3, 2)

        It is a pytree with two leaves.

        >>> import jax
        >>> len(jax.tree_util.tree_leaves(state))
        2
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
        """Create a zero-initialised LSTM state.

        Args:
            batch_size: Number of samples $B$.
            hidden_dim: Hidden channel count.
            height: Grid height $H$.
            width: Grid width $W$.

        Returns:
            A state with zero hidden and cell tensors.
        """
        return cls(
            h=jnp.zeros((batch_size, hidden_dim, height, width)),
            c=jnp.zeros((batch_size, hidden_dim, height, width)),
        )
