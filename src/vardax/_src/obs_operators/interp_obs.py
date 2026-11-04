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
``MaskedIdentity`` so vardax's ``accepts_mask`` dispatch (in
``ThreeDVar`` / ``StrongFourDVar`` / ``IncrementalFourDVar``) picks it
up automatically.
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int
import lineax as lx
import numpy as _np

from vardax._src.errors import GridError, ShapeError
from vardax._src.utils.shapes import flatten


class InterpObs(eqx.Module):
    r"""Grid → scattered-obs sparse bilinear interpolation.

    $$(Hx)_k = \sum_{i \in \mathrm{stencil}(p_k)} w_{k,i} \, x_i$$

    Build a precomputed sparse ``(n_obs, 4)`` weight stencil from a 2-D
    grid and an ``(n_obs, 2)`` array of obs coordinates. The forward
    indexing is pure JAX (so ``jit`` / ``grad`` / ``vmap`` all work);
    ``linearize`` wraps the same forward as a
    ``lineax.FunctionLinearOperator`` (its transpose comes from
    ``jax.vjp`` automatically).

    Algorithm:
        1. Flatten the grid state.
        2. Gather the four stencil corners of each observation with
           ``indices``.
        3. Take the weighted sum of the corners with ``weights``.
        4. Multiply by ``mask`` if one is given.

    Attributes:
        indices: Integer flat grid indices of each observation's 2x2
            stencil.
        weights: Bilinear weights of the four corners, summing to 1 for
            each observation.
        grid_shape: Static ``(ny, nx)`` extent of the input grid.

    Examples:
        Interpolate a linear field at the centre of a unit cell:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> axis = jnp.array([0.0, 1.0])
        >>> op = vdx.interp_obs_from_coords((axis, axis), jnp.array([[0.5, 0.5]]))
        >>> field = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        >>> bool(jnp.allclose(op(field), 1.5))
        True

        A mask zeroes out dropped observations:

        >>> pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> op2 = vdx.interp_obs_from_coords((axis, axis), pts)
        >>> op2(field, mask=jnp.array([1.0, 0.0]))
        Array([0., 0.], dtype=float32)

        The operator is linear, so its linearisation applies the same map:

        >>> bool(jnp.allclose(op2.linearize(field).mv(field), op2(field)))
        True
    """

    indices: Int[Array, "n_obs 4"]
    weights: Float[Array, "n_obs 4"]
    grid_shape: tuple[int, int] = eqx.field(static=True)

    def __call__(
        self,
        state: Float[Array, "*grid"],
        mask: Float[Array, " n_obs"] | None = None,
    ) -> Float[Array, " n_obs"]:
        """Apply ``H`` (optionally masked) to a grid state.

        Args:
            state: Grid state, either ``(ny, nx)`` or flattened.
            mask: Optional observation mask multiplying the predictions.

        Returns:
            The interpolated observations, one per location.
        """
        flat = flatten(state)
        corners = flat[self.indices]
        # yₒ = Σₖ wₒₖ x[iₒₖ]: bilinear weights over the four corners;
        # jnp, not einx: a time step may have no observations
        predicted_obs = jnp.sum(corners * self.weights, axis=-1)
        if mask is not None:
            predicted_obs = predicted_obs * mask
        return predicted_obs

    def linearize(
        self,
        x: Float[Array, ...],
    ) -> lx.AbstractLinearOperator:
        """Tangent-linear at ``x``: ``H`` itself (linear operator).

        ``gaussx.interpolation_operator`` holds the stencil, so it plugs
        straight into ``IncrementalFourDVar``'s GN Hessian assembly and
        ``LaplaceCovariance``, and ``op.transpose().mv(y)`` is its
        scatter-add.

        The input structure is taken from ``x.shape`` so the operator
        composes with both 2-D ``(Ny, Nx)`` and flat ``(Ny*Nx,)`` state
        layouts — variational models that carry flat state vectors then
        get a flat-shaped ``H`` instead of being forced through the
        grid shape.

        Args:
            x: State whose shape and dtype define the operator input.

        Returns:
            The operator $H$, which does not depend on ``x`` otherwise.
        """
        in_struct = jax.ShapeDtypeStruct(x.shape, x.dtype)
        flatten_state = lx.FunctionLinearOperator(jnp.ravel, in_struct)
        # W, the gather of the four corners, with its scatter-add transpose
        interpolation = gaussx.interpolation_operator(
            self.indices,
            self.weights.astype(x.dtype),
            self.grid_shape[0] * self.grid_shape[1],
        )
        return interpolation @ flatten_state


def interp_obs_from_coords(
    grid_coords: tuple[Float[ArrayLike, " Ny"], Float[ArrayLike, " Nx"]],
    obs_coords: Float[ArrayLike, "n_obs 2"],
    *,
    order: int = 1,
) -> InterpObs:
    """Build an `InterpObs` from grid axis arrays + obs locations.

    Locates each observation in its grid cell and caches the four corner
    indices and bilinear weights. Points outside the grid are clamped to
    the edge.

    Args:
        grid_coords: ``(y_axis, x_axis)`` tuple of 1-D axis arrays.
        obs_coords: ``(y, x)`` observation locations, one row each.
        order: interpolation order. Only ``order=1`` (bilinear) is
            currently supported.

    Returns:
        Configured ``InterpObs`` instance with precomputed cached
        ``(indices, weights)``.

    Raises:
        GridError: If ``grid_coords`` does not hold two axes, or an axis
            is not strictly increasing with at least two points.
        ShapeError: If ``obs_coords`` is not an ``(n_obs, 2)`` array.
        NotImplementedError: If ``order`` is not 1.

    Examples:
        Cache the stencil for two observations on a 3 x 3 grid:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> axis = jnp.array([0.0, 1.0, 2.0])
        >>> obs = jnp.array([[0.5, 1.5], [2.0, 0.0]])
        >>> op = vdx.interp_obs_from_coords((axis, axis), obs)
        >>> op.indices.shape, op.weights.shape
        ((2, 4), (2, 4))

        Predict the observations from a state whose value is 3 * y + x:

        >>> op(jnp.arange(9.0).reshape(3, 3))
        Array([3., 6.], dtype=float32)

        A descending axis is rejected:

        >>> vdx.interp_obs_from_coords(
        ...     (axis[::-1], axis), obs
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        GridError: y_axis must be strictly increasing ...
    """
    if order != 1:
        raise NotImplementedError(
            f"interp_obs_from_coords supports order=1 (bilinear), got {order}."
        )
    if len(grid_coords) != 2:
        raise GridError(
            "interp_obs_from_coords expects grid_coords=(y_axis, x_axis); "
            f"got len={len(grid_coords)}."
        )

    y_axis = _np.asarray(grid_coords[0], dtype=float)
    x_axis = _np.asarray(grid_coords[1], dtype=float)
    for name, axis in (("y_axis", y_axis), ("x_axis", x_axis)):
        if axis.size < 2 or _np.any(_np.diff(axis) <= 0):
            raise GridError(
                f"{name} must be strictly increasing with at least 2 points; "
                "reverse a descending axis before calling interp_obs_from_coords."
            )
    points = _np.asarray(obs_coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ShapeError(f"obs_coords must be (n_obs, 2); got {points.shape}.")

    if points.shape[0] == 0:
        # gaussx's weights reject an empty batch on a 2-D grid
        indices = _np.zeros((0, 4), dtype=_np.int32)
        weights = _np.zeros((0, 4))
    else:
        # Scale the coordinates here, in float64: as float32, projected
        # coordinates of order 1e6 would lose the offset within a grid cell.
        # The stencil is row-major: (lo, lo), (lo, hi), (hi, lo), (hi, hi).
        y_origin, y_span = y_axis[0], y_axis[-1] - y_axis[0]
        x_origin, x_span = x_axis[0], x_axis[-1] - x_axis[0]
        indices, weights = gaussx.linear_interpolation_weights(
            jnp.asarray(
                _np.stack(
                    [
                        (points[:, 0] - y_origin) / y_span,
                        (points[:, 1] - x_origin) / x_span,
                    ],
                    axis=1,
                )
            ),
            [
                jnp.asarray((y_axis - y_origin) / y_span),
                jnp.asarray((x_axis - x_origin) / x_span),
            ],
        )

    return InterpObs(
        indices=jnp.asarray(indices, dtype=jnp.int32),
        # JAX's default float dtype, so the weights are float64 under x64
        weights=jnp.asarray(weights),
        grid_shape=(y_axis.size, x_axis.size),
    )
