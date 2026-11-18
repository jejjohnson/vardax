"""Grid-to-points bilinear observation operator with a cached sparse stencil."""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int
import lineax as lx
import numpy as np

from vardax._src.errors import GridError, ShapeError
from vardax._src.utils.shapes import flatten


class InterpObs(eqx.Module):
    """Sparse bilinear interpolation from a 2-D grid to scattered observations.

        (H x)ᵢ = ∑ₖ wᵢₖ x[nᵢₖ],   k = 1, …, 4

    Observation i reads the four corners k of its grid cell, at flat grid index nᵢₖ
    with bilinear weight wᵢₖ, so both H and Hᵀ cost O(nnz). The forward indexing is
    pure JAX, so ``jit``, ``grad`` and ``vmap`` all work.

    It satisfies ``pipekit_cycle.ObservationOperator``. The optional ``mask`` keyword
    matches ``MaskedIdentity``, so the ``accepts_mask`` dispatch of ``ThreeDVar``,
    ``StrongFourDVar`` and ``IncrementalFourDVar`` picks it up.

    Algorithm:
        1. Flatten the grid state.
        2. Gather the four stencil corners of each observation with ``indices``.
        3. Take the weighted sum of the corners with ``weights``.
        4. Multiply by ``mask`` if one is given.

    Attributes:
        indices: Flat grid indices nᵢₖ of each observation's 2 × 2 stencil.
        weights: Bilinear weights wᵢₖ of the four corners, summing to 1 for each
            observation.
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
        >>> op2(field, mask=jnp.array([1.0, 0.0])).tolist()
        [0.0, 0.0]

        The operator is linear, so its linearisation applies the same map:

        >>> bool(jnp.allclose(op2.linearize(field).mv(field), op2(field)))
        True

    References:
        MASSH, ``Obsop_interp_l3_jax._sparse_op`` and ``explicit_proj_operation`` in
        ``mapping/src/obsop.py``.
    """

    indices: Int[Array, "n_obs 4"]
    weights: Float[Array, "n_obs 4"]
    grid_shape: tuple[int, int] = eqx.field(static=True)

    def __call__(
        self,
        state: Float[Array, "*grid"],
        mask: Float[Array, " n_obs"] | None = None,
    ) -> Float[Array, " n_obs"]:
        """Apply H, optionally masked, to a grid state: (H x)ᵢ = ∑ₖ wᵢₖ x[nᵢₖ].

        Args:
            state: Grid state x, either ``(ny, nx)`` or flattened.
            mask: Optional observation mask multiplying the predictions.

        Returns:
            The interpolated observations, one per location.
        """
        flat_state = flatten(state)
        # x[nᵢₖ]
        corners = flat_state[self.indices]
        # wᵢₖ x[nᵢₖ]
        weighted_corners = corners * self.weights
        # (H x)ᵢ = ∑ₖ wᵢₖ x[nᵢₖ]; jnp, not einx: a time step may have no observations
        predicted_obs = jnp.sum(weighted_corners, axis=-1)
        if mask is not None:
            # mᵢ (H x)ᵢ
            predicted_obs = predicted_obs * mask
        return predicted_obs

    def linearize(
        self,
        x: Float[Array, ...],
    ) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``x``: J_h = H, since h is linear.

        ``gaussx.interpolation_operator`` holds the stencil, so it plugs straight into
        the Gauss-Newton Hessian of ``IncrementalFourDVar`` and into
        ``LaplaceCovariance``, and ``op.transpose().mv(y)`` is its scatter-add.

        The input structure comes from ``x.shape``, so the operator composes with both
        the 2-D ``(Ny, Nx)`` and the flat ``(Ny*Nx,)`` state layouts: a variational
        model that carries a flat state gets a flat-shaped H.

        Args:
            x: State whose shape and dtype define the operator input.

        Returns:
            The operator H, which does not depend on ``x`` otherwise.
        """
        state_structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
        # vec; jnp.ravel, not einx: einx raises on a zero-size state at mv time
        flatten_state = lx.FunctionLinearOperator(jnp.ravel, state_structure)
        if self.indices.shape[0] == 0:
            # The empty slice is linear and its transpose scatters zeros into x.
            # gaussx's einx gather cannot resolve a zero-length observation axis.
            flat_structure = jax.ShapeDtypeStruct((x.size,), x.dtype)
            empty = lx.FunctionLinearOperator(lambda state: state[:0], flat_structure)
            return empty @ flatten_state
        # W, the gather of the four corners, with its scatter-add transpose
        interpolation = gaussx.interpolation_operator(
            self.indices,
            self.weights.astype(x.dtype),
            self.grid_shape[0] * self.grid_shape[1],
        )
        # H = W vec
        return interpolation @ flatten_state


def interp_obs_from_coords(
    grid_coords: tuple[Float[ArrayLike, " Ny"], Float[ArrayLike, " Nx"]],
    obs_coords: Float[ArrayLike, "n_obs 2"],
    *,
    order: int = 1,
) -> InterpObs:
    """Build an `InterpObs` from grid axis arrays and observation locations.

    Locates each observation in its grid cell and caches the four corner indices nᵢₖ
    and bilinear weights wᵢₖ. Points outside the grid are clamped to the edge.

    Args:
        grid_coords: ``(y_axis, x_axis)`` tuple of 1-D axis arrays.
        obs_coords: ``(y, x)`` observation locations, one row each.
        order: Interpolation order. Only ``order=1`` (bilinear) is supported.

    Returns:
        The configured ``InterpObs``, with its cached ``indices`` and ``weights``.

    Raises:
        GridError: If ``grid_coords`` does not hold two axes, or an axis is not
            strictly increasing with at least two points.
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

        >>> state = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        >>> bool(jnp.allclose(op(state), jnp.array([3.0, 6.0])))
        True

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

    y_axis = np.asarray(grid_coords[0], dtype=float)
    x_axis = np.asarray(grid_coords[1], dtype=float)
    for name, axis in (("y_axis", y_axis), ("x_axis", x_axis)):
        if axis.size < 2 or np.any(np.diff(axis) <= 0):
            raise GridError(
                f"{name} must be strictly increasing with at least 2 points; "
                "reverse a descending axis before calling interp_obs_from_coords."
            )
    points = np.asarray(obs_coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ShapeError(f"obs_coords must be (n_obs, 2); got {points.shape}.")

    if points.shape[0] == 0:
        # gaussx's weights reject an empty batch on a 2-D grid
        indices = np.zeros((0, 4), dtype=np.int32)
        weights = np.zeros((0, 4))
    else:
        # In float64 whatever JAX's default, as these coordinates are concrete:
        # in float32, a projected coordinate of order 1e6 or a narrow cell on a
        # wide axis would lose its offset. Only the finished weights are cast.
        # The stencil is row-major: (lo, lo), (lo, hi), (hi, lo), (hi, hi).
        with jax.ensure_compile_time_eval(), jax.enable_x64(True):
            indices, weights = gaussx.linear_interpolation_weights(
                jnp.asarray(points), [jnp.asarray(y_axis), jnp.asarray(x_axis)]
            )
            indices, weights = np.asarray(indices), np.asarray(weights)

    return InterpObs(
        indices=jnp.asarray(indices, dtype=jnp.int32),
        # JAX's default float dtype, so the weights are float64 under x64
        weights=jnp.asarray(weights),
        grid_shape=(y_axis.size, x_axis.size),
    )
