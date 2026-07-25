"""Sparse bilinear interpolation observation operator.

Maps a regular 2-D grid to scattered obs locations via cached bilinear
weights: ``(Hx)_k = Σ_{i ∈ stencil(p_k)} w_{k,i} x_i``. Both forward
and ``H^T`` are ``O(nnz)``; tangent-linear is the operator itself
(``H`` is linear in the state).

Reference: MASSH `Obsop_interp_l3_jax._sparse_op` /
`explicit_proj_operation` (`mapping/src/obsop.py:401`).

Satisfies ``pipekit_cycle.ObservationOperator``: implements
``__call__(state, mask=None) -> obs`` and ``linearize(x) ->
lineax.AbstractLinearOperator``. The optional ``mask`` kwarg matches
``MaskedIdentity`` so vardax's ``_accepts_mask`` dispatch (in
``ThreeDVar`` / ``StrongFourDVar`` / ``IncrementalFourDVar``) picks it
up automatically.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import lineax as lx
import numpy as _np


class InterpObs(eqx.Module):
    """Grid → scattered-obs sparse bilinear interpolation.

    Build a precomputed sparse ``(n_obs, 4)`` weight stencil from a 2-D
    grid and an ``(n_obs, 2)`` array of obs coordinates. The forward
    indexing is pure JAX (so ``jit`` / ``grad`` / ``vmap`` all work);
    ``linearize`` wraps the same forward as a
    ``lineax.FunctionLinearOperator`` (its transpose comes from
    ``jax.vjp`` automatically).

    Attributes:
        indices: ``(n_obs, 4)`` int array of flat grid indices for each
            obs's 2x2 stencil.
        weights: ``(n_obs, 4)`` float array of bilinear weights summing
            to 1 along each row.
        grid_shape: static ``(ny, nx)`` of the input grid.
    """

    indices: Int[Array, "n_obs 4"]
    weights: Float[Array, "n_obs 4"]
    grid_shape: tuple[int, int] = eqx.field(static=True)

    def __call__(
        self,
        state: Float[Array, "Ny Nx"],
        mask: Float[Array, " n_obs"] | None = None,
    ) -> Float[Array, " n_obs"]:
        """Apply ``H`` (optionally masked) to a grid state."""
        flat = state.reshape(-1)
        out = (flat[self.indices] * self.weights).sum(axis=-1)
        if mask is not None:
            out = out * mask
        return out

    def linearize(
        self,
        x: Float[Array, ...],
    ) -> lx.AbstractLinearOperator:
        """Tangent-linear at ``x``: ``H`` itself (linear operator).

        Wraps ``__call__`` as a ``lineax.FunctionLinearOperator`` so it
        plugs straight into ``IncrementalFourDVar``'s GN Hessian
        assembly and ``LaplaceCovariance``. The transpose ``H^T`` is
        derived via ``jax.vjp`` (cached numpy adjoint is no longer
        exposed; use ``op.transpose().mv(y)`` instead).

        The input structure is taken from ``x.shape`` so the operator
        composes with both 2-D ``(Ny, Nx)`` and flat ``(Ny*Nx,)`` state
        layouts — variational models that carry flat state vectors then
        get a flat-shaped ``H`` instead of being forced through the
        grid shape.
        """
        in_struct = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return lx.FunctionLinearOperator(self.__call__, in_struct)


def interp_obs_from_coords(
    grid_coords: tuple[Float[Array, " Ny"], Float[Array, " Nx"]],
    obs_coords: Float[Array, "n_obs 2"],
    *,
    order: int = 1,
) -> InterpObs:
    """Build an `InterpObs` from grid axis arrays + obs locations.

    Args:
        grid_coords: ``(y_axis, x_axis)`` tuple of 1-D axis arrays.
        obs_coords: ``(n_obs, 2)`` array of ``(y, x)`` obs locations.
        order: interpolation order. Only ``order=1`` (bilinear) is
            currently supported.

    Returns:
        Configured ``InterpObs`` instance with precomputed cached
        ``(indices, weights)``.
    """
    if order != 1:
        raise NotImplementedError(
            f"interp_obs_from_coords supports order=1 (bilinear), got {order}."
        )
    if len(grid_coords) != 2:
        raise ValueError(
            "interp_obs_from_coords expects grid_coords=(y_axis, x_axis); "
            f"got len={len(grid_coords)}."
        )

    y_axis = _np.asarray(grid_coords[0], dtype=float)
    x_axis = _np.asarray(grid_coords[1], dtype=float)
    for name, axis in (("y_axis", y_axis), ("x_axis", x_axis)):
        if axis.size < 2 or _np.any(_np.diff(axis) <= 0):
            raise ValueError(
                f"{name} must be strictly increasing with at least 2 points; "
                "reverse a descending axis before calling interp_obs_from_coords."
            )
    pts = _np.asarray(obs_coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"obs_coords must be (n_obs, 2); got {pts.shape}.")

    ny, nx = y_axis.size, x_axis.size
    iy0, wy1 = _bracket(y_axis, pts[:, 0])
    ix0, wx1 = _bracket(x_axis, pts[:, 1])
    iy1, ix1 = iy0 + 1, ix0 + 1
    wy0, wx0 = 1.0 - wy1, 1.0 - wx1

    indices = _np.stack(
        [iy0 * nx + ix0, iy0 * nx + ix1, iy1 * nx + ix0, iy1 * nx + ix1],
        axis=1,
    )
    weights = _np.stack(
        [wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1],
        axis=1,
    )

    return InterpObs(
        indices=jnp.asarray(indices, dtype=jnp.int32),
        weights=jnp.asarray(weights, dtype=jnp.float32),
        grid_shape=(int(ny), int(nx)),
    )


def _bracket(axis: _np.ndarray, pts: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    """For each point ``p`` in ``pts``, return ``(idx, frac)`` such that
    ``axis[idx] + frac * (axis[idx+1] - axis[idx])`` matches ``p``.

    Points outside ``axis`` are clamped to the edges.
    """
    n = axis.size
    idx = _np.clip(_np.searchsorted(axis, pts) - 1, 0, n - 2)
    span = axis[idx + 1] - axis[idx]
    frac = _np.where(span > 0, (pts - axis[idx]) / span, 0.0)
    return idx, _np.clip(frac, 0.0, 1.0)
