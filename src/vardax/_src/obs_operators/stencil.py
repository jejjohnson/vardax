"""The linear interpolation stencil shared by ``InterpObs`` and ``GridPointObs``."""

from __future__ import annotations

from collections.abc import Sequence

import einx
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Linear, Policies
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def linear_stencil(
    points: Float[Array, "target axis"], axes: Sequence[Float[Array, " _"]]
) -> tuple[Int[Array, "target corner"], Float[Array, "target corner"]]:
    """Flat indices and weights of linear interpolation on a rectilinear grid.

        ŷᵢ = ∑ₖ wᵢₖ x[nᵢₖ],   k = 1, …, 2ᴰ

    A point outside the grid takes the edge value. A point with a coordinate that is
    not finite gets NaN weights, so its prediction is NaN instead of a silent zero.

    Args:
        points: One point sᵢ per row, in the order of ``axes``.
        axes: The D axes of the grid, each strictly increasing.

    Returns:
        The indices nᵢₖ into the row-major flattened grid and the weights wᵢₖ. The
        corners are row-major: (lo, lo), (lo, hi), (hi, lo), (hi, hi) for D = 2.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax._src.obs_operators.stencil import linear_stencil
        >>> axis = jnp.array([0.0, 1.0, 3.0])
        >>> indices, weights = linear_stencil(jnp.array([[1.5], [9.0]]), [axis])
        >>> indices.tolist(), weights.tolist()
        ([[1, 2], [1, 2]], [[0.75, 0.25], [0.0, 1.0]])
    """
    grid = RectilinearGrid(tuple(Axis(axis) for axis in axes))
    plan = Linear().plan(grid, Points(points), policies=Policies(outside="clamp"))
    indices, weights = plan.to_gaussx()
    if points.shape[0] == 0:
        return indices, weights
    # wᵢₖ ← NaN where sᵢ is not finite: clamping leaves no other invalid row
    nan = jnp.full((), jnp.nan, weights.dtype)
    weights = einx.where(
        "target, target corner, -> target corner", plan.valid, weights, nan
    )
    return indices, weights
