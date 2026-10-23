r"""Observation operators and cost for scattered point observations.

A [`PointObsBatch`][vardax.PointObsBatch] carries $n_\text{obs}$
observations as they are — coordinates $s_i$, values $y_i$, error
standard deviations $\sigma_i$ and a ``valid`` mask $m_i$ that is
``False`` on padding rows. The observation cost is

$$
J_o = \tfrac12 \sum_i m_i \,
      \frac{\big(y_i - \mathcal H(x)(s_i, t_i)\big)^2}{\sigma_i^2},
$$

computed by [`point_obs_cost`][vardax.point_obs_cost]. Padding rows
contribute exactly zero value **and** zero gradient: the whitened residual
is masked with ``jnp.where`` before squaring, and the padding's
coordinates, values and $\sigma$ are replaced by safe placeholders first,
so a ``NaN`` in a padding row never reaches a derivative.

Two operators evaluate $\mathcal H(x)$ at the points:

- [`GridPointObs`][vardax.GridPointObs] for a gridded state or
  trajectory: multilinear interpolation in space (1-D or 2-D, clamped at
  the grid edges) of the state at each observation's ``time_index``.
- [`FieldPointObs`][vardax.FieldPointObs] for a state that is a callable
  field: the field evaluated at the coordinates.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from vardax._src._types import PointObsBatch


def _safe(batch: PointObsBatch) -> PointObsBatch:
    """Replace padding rows by finite placeholders (zero value, unit sigma).

    ``jnp.where`` alone stops the padding's *value* but not a ``NaN``
    gradient: ``0 * NaN`` is ``NaN``. Sanitising the inputs first keeps
    every intermediate on padding rows finite.
    """
    valid = batch.valid
    coords = jnp.where(valid[:, None], batch.coords, jnp.zeros_like(batch.coords))
    values = jnp.where(valid, batch.values, jnp.zeros_like(batch.values))
    sigma = jnp.where(valid, batch.sigma, jnp.ones_like(batch.sigma))
    time_index = (
        None
        if batch.time_index is None
        else jnp.where(valid, batch.time_index, jnp.zeros_like(batch.time_index))
    )
    return PointObsBatch(
        coords=coords,
        values=values,
        sigma=sigma,
        valid=valid,
        time_index=time_index,
        n_times=batch.n_times,
    )


def point_obs_cost(
    predictions: Float[Array, " N_obs"], batch: PointObsBatch
) -> Float[Array, ""]:
    r"""Masked, $\sigma$-weighted observation cost of a point batch.

    $J_o = \tfrac12 \sum_i m_i (y_i - \hat y_i)^2 / \sigma_i^2$, with the
    whitened residual set to zero by ``jnp.where`` on padding rows before
    squaring, so padding contributes zero value and zero gradient.

    Args:
        predictions: $\hat y_i = \mathcal H(x)(s_i, t_i)$, shape
            ``(N_obs,)``.
        batch: The observations (one sample, no batch axis).

    Returns:
        The scalar cost.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.zeros((3, 1)),
        ...     values=jnp.array([1.0, 2.0, jnp.nan]),
        ...     sigma=jnp.array([1.0, 2.0, 0.0]),
        ...     valid=jnp.array([True, True, False]),
        ... )
        >>> float(vardax.point_obs_cost(jnp.zeros(3), batch))  # ½ (1 + 1)
        1.0
    """
    return 0.5 * jnp.sum(jnp.square(whitened_residual(predictions, batch)))


def whitened_residual(
    predictions: Float[Array, " N_obs"], batch: PointObsBatch
) -> Float[Array, " N_obs"]:
    """$m_i (y_i - \\hat y_i) / \\sigma_i$, exactly zero (value and gradient) on padding."""
    safe = _safe(batch)
    return jnp.where(
        batch.valid,
        (safe.values - predictions) / safe.sigma,
        jnp.zeros_like(predictions),
    )


def _bracket(
    axis: Float[Array, " K"], points: Float[Array, " N_obs"]
) -> tuple[Int[Array, " N_obs"], Float[Array, " N_obs"]]:
    """Left neighbour index and fractional offset, clamped to the axis."""
    idx = jnp.clip(jnp.searchsorted(axis, points) - 1, 0, axis.shape[0] - 2)
    span = axis[idx + 1] - axis[idx]
    frac = jnp.clip((points - axis[idx]) / span, 0.0, 1.0)
    return idx, frac


class GridPointObs(eqx.Module):
    r"""$\mathcal H_t(x_t)$ at scattered points for a gridded state.

    The state carries a leading time axis, ``(n_times, *grid)`` (a 4D-Var
    trajectory, or a ``ThreeDVar`` state ``(T, N)``), with ``grid``
    ``(N,)`` (1-D) or ``(Ny, Nx)`` (2-D) on the axes ``grid_axes``. For
    observation $i$ the operator takes the state at ``time_index[i]``
    ($0$ when the batch has no ``time_index``) and interpolates it
    linearly in each spatial dimension at the first ``len(grid_axes)``
    columns of ``coords[i]``, clamping points outside the grid to its
    edge — the ``order=1`` scheme of `interp_obs_from_coords`, with the
    stencil computed at call time so the coordinates can be traced. Any
    further coordinate columns (a time) are ignored: the time enters
    through ``time_index``.

    Attributes:
        grid_axes: One strictly increasing 1-D axis per spatial
            dimension: ``(x_axis,)`` or ``(y_axis, x_axis)``.

    Examples:
        A 1-D grid at $0, 1, 2, 3$: linear interpolation between nodes.

        >>> import jax.numpy as jnp, vardax
        >>> H = vardax.GridPointObs((jnp.arange(4.0),))
        >>> state = jnp.array([[0.0, 10.0, 20.0, 30.0]])  # (n_times=1, N=4)
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.5], [2.25]]),
        ...     values=jnp.zeros(2),
        ...     sigma=jnp.ones(2),
        ...     valid=jnp.ones(2, bool),
        ... )
        >>> H(state, batch)
        Array([ 5. , 22.5], dtype=float32)
    """

    grid_axes: tuple[Float[Array, " K"], ...]

    def __init__(self, grid_axes: tuple[Float[Array, " K"], ...]) -> None:
        axes = tuple(jnp.asarray(a) for a in grid_axes)
        if len(axes) not in (1, 2):
            raise ValueError(f"grid_axes must hold 1 or 2 axes; got {len(axes)}.")
        for k, axis in enumerate(axes):
            if axis.ndim != 1 or axis.shape[0] < 2:
                raise ValueError(
                    f"grid_axes[{k}] must be 1-D with at least 2 points; "
                    f"got shape {axis.shape}."
                )
            if not bool(jnp.all(jnp.diff(axis) > 0)):
                raise ValueError(f"grid_axes[{k}] must be strictly increasing.")
        self.grid_axes = axes

    def __call__(
        self, state: Float[Array, "n_times ..."], batch: PointObsBatch
    ) -> Float[Array, " N_obs"]:
        """Interpolate ``state[time_index]`` at the observation coordinates."""
        safe = _safe(batch)
        n_obs = safe.values.shape[0]
        grid_shape = tuple(a.shape[0] for a in self.grid_axes)
        frames = state.reshape((state.shape[0], *grid_shape))
        t_idx = (
            jnp.zeros(n_obs, dtype=jnp.int32)
            if safe.time_index is None
            else jnp.clip(safe.time_index, 0, state.shape[0] - 1)
        )
        if len(self.grid_axes) == 1:
            i0, w1 = _bracket(self.grid_axes[0], safe.coords[:, 0])
            left = frames[t_idx, i0]
            right = frames[t_idx, i0 + 1]
            return (1.0 - w1) * left + w1 * right
        iy, wy = _bracket(self.grid_axes[0], safe.coords[:, 0])
        ix, wx = _bracket(self.grid_axes[1], safe.coords[:, 1])
        f00 = frames[t_idx, iy, ix]
        f01 = frames[t_idx, iy, ix + 1]
        f10 = frames[t_idx, iy + 1, ix]
        f11 = frames[t_idx, iy + 1, ix + 1]
        return (1.0 - wy) * ((1.0 - wx) * f00 + wx * f01) + wy * (
            (1.0 - wx) * f10 + wx * f11
        )


class FieldPointObs(eqx.Module):
    r"""$\mathcal H(f)$ for a state that is a callable field.

    Evaluates the single-example field $f(s) \mapsto$ ``(1,)`` (or a
    scalar) at every observation coordinate, $\hat y_i = f(s_i)$; all
    ``D`` coordinate columns (including a time column) are passed to the
    field. Padding rows are evaluated at the origin, a finite placeholder.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> H = vardax.FieldPointObs()
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        ...     values=jnp.zeros(2),
        ...     sigma=jnp.ones(2),
        ...     valid=jnp.ones(2, bool),
        ... )
        >>> H(lambda s: jnp.sum(s, keepdims=True), batch)
        Array([3., 7.], dtype=float32)
    """

    def __call__(
        self, field: Callable[[Array], Array], batch: PointObsBatch
    ) -> Float[Array, " N_obs"]:
        """Evaluate ``field`` at each coordinate row."""
        safe = _safe(batch)
        values = jax.vmap(field)(safe.coords)
        return values.reshape(values.shape[0])


# --- helpers for the models' PointObsBatch path ------------------------------


def map_point_batch(fn: Callable, batch: PointObsBatch):
    """Apply ``fn`` to one sample, or ``vmap`` it over a leading batch axis."""
    if batch.values.ndim == 1:
        return fn(batch)
    if batch.values.ndim != 2:
        raise ValueError(
            "PointObsBatch.values must be (N_obs,) or (B, N_obs); "
            f"got shape {batch.values.shape}."
        )
    return jax.vmap(fn)(batch)


def window_steps(batch: PointObsBatch) -> int:
    """Rollout steps ``n_times - 1`` of a point batch (static)."""
    if batch.n_times is None:
        if batch.time_index is not None:
            raise ValueError(
                "PointObsBatch.time_index needs n_times (the number of "
                "states in the window) for a 4D-Var model."
            )
        return 0
    if batch.n_times < 1:
        raise ValueError(f"PointObsBatch.n_times must be >= 1, got {batch.n_times}.")
    return batch.n_times - 1


def grid_point_obs_cost(
    obs_op: object, states: Float[Array, "n_times ..."], batch: PointObsBatch
) -> Float[Array, ""]:
    """Point cost of a gridded state or trajectory under a `GridPointObs`."""
    return point_obs_cost(require_grid_point_obs(obs_op)(states, batch), batch)


def require_grid_point_obs(obs_op: object) -> GridPointObs:
    """Return ``obs_op`` if it is a `GridPointObs`, else raise ``TypeError``."""
    if not isinstance(obs_op, GridPointObs):
        raise TypeError(
            "A PointObsBatch needs a point observation operator: build the "
            f"model with obs_op=GridPointObs(...), not {type(obs_op).__name__}."
        )
    return obs_op


__all__ = ["FieldPointObs", "GridPointObs", "point_obs_cost"]
