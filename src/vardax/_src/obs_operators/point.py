"""Operators and masked cost for scattered point observations.

``point_obs_cost`` is 𝒥_o = ½ ∑ᵢ mᵢ (yᵢ − ŷᵢ)² / σᵢ², ``GridPointObs`` gives ŷᵢ from a
gridded state or trajectory and ``FieldPointObs`` from a callable field.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx
import numpy as np

from vardax._src._types import PointObsBatch
from vardax._src.errors import (
    GridError,
    InputTypeError,
    MissingInputError,
    ObservationError,
    ShapeError,
)
from vardax._src.utils.shapes import as_shape


def _safe(batch: PointObsBatch) -> PointObsBatch:
    """Replace padding rows by finite placeholders."""
    valid = batch.valid
    # Padding takes the coordinates of the first valid row (the origin if there is
    # none), so an operator never evaluates the state or field where no real
    # observation is: a field singular at the origin stays finite. jnp.where alone
    # stops the padding's value but not a NaN gradient (0 · NaN = NaN), so the
    # inputs are sanitised first and every intermediate on a padding row stays finite.
    first_valid = batch.coords[jnp.argmax(valid)]
    origin = jnp.zeros_like(first_valid)
    placeholder = jnp.where(jnp.any(valid), first_valid, origin)
    coords = einx.where(
        "obs, obs coord, coord -> obs coord", valid, batch.coords, placeholder
    )
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
    """Masked, σ-weighted observation cost of a point batch.

        𝒥_o = ½ ∑ᵢ mᵢ (yᵢ − ŷᵢ)² / σᵢ²

    A [`PointObsBatch`][vardax.PointObsBatch] holds the observations as they are:
    coordinates sᵢ, values yᵢ, error standard deviations σᵢ and a ``valid`` mask mᵢ
    that is ``False`` on padding rows. Padding contributes zero value and zero
    gradient, even when it holds ``NaN``.

    Algorithm:
        1. Replace padding rows by finite placeholders.
        2. Form the whitened residual rᵢ = (yᵢ − ŷᵢ) / σᵢ and set it to zero where mᵢ
           is false.
        3. Return half the sum of squares.

    Args:
        predictions: Predicted observations ŷᵢ = h(x)(sᵢ, tᵢ).
        batch: The observations (one sample, no batch axis).

    Returns:
        The scalar cost 𝒥_o.

    Examples:
        A batch whose third row is padding (``NaN`` value, zero sigma), so
        𝒥_o = ½ (1 + 1):

        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.zeros((3, 1)),
        ...     values=jnp.array([1.0, 2.0, jnp.nan]),
        ...     sigma=jnp.array([1.0, 2.0, 0.0]),
        ...     valid=jnp.array([True, True, False]),
        ... )
        >>> float(vardax.point_obs_cost(jnp.zeros(3), batch))
        1.0

        Padding contributes zero gradient, not ``NaN``:

        >>> import jax
        >>> grad = jax.grad(lambda p: vardax.point_obs_cost(p, batch))(jnp.zeros(3))
        >>> bool(jnp.all(jnp.isfinite(grad)))
        True

        Composed with a point operator on a gridded state:

        >>> point_op = vardax.GridPointObs((jnp.arange(3.0),))
        >>> single = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.5]]),
        ...     values=jnp.array([1.0]),
        ...     sigma=jnp.array([1.0]),
        ...     valid=jnp.array([True]),
        ... )
        >>> float(vardax.point_obs_cost(point_op(jnp.zeros((1, 3)), single), single))
        0.5
    """
    # rᵢ = mᵢ (yᵢ − ŷᵢ) / σᵢ
    whitened = whitened_residual(predictions, batch)
    # rᵢ²
    squared_residual = jnp.square(whitened)
    # ∑ᵢ rᵢ²
    total = jnp.sum(squared_residual)
    # 𝒥_o = ½ ∑ᵢ rᵢ²
    return 0.5 * total


def whitened_residual(
    predictions: Float[Array, " N_obs"], batch: PointObsBatch
) -> Float[Array, " N_obs"]:
    """Whitened residual of a point batch, exactly zero (value and gradient) on padding.

        rᵢ = mᵢ (yᵢ − ŷᵢ) / σᵢ

    Args:
        predictions: Predicted observations ŷᵢ.
        batch: The observations (one sample, no batch axis).

    Returns:
        The whitened residual rᵢ of every row, padding included.

    Examples:
        The second row is whitened by σ₂ = 2, and the third is padding:

        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.zeros((3, 1)),
        ...     values=jnp.array([1.0, 2.0, jnp.nan]),
        ...     sigma=jnp.array([1.0, 2.0, 0.0]),
        ...     valid=jnp.array([True, True, False]),
        ... )
        >>> whitened_residual(jnp.zeros(3), batch).tolist()
        [1.0, 1.0, 0.0]

        A perfect prediction has a zero residual:

        >>> whitened_residual(jnp.array([1.0, 2.0, 0.0]), batch).tolist()
        [0.0, 0.0, 0.0]
    """
    safe = _safe(batch)
    # yᵢ − ŷᵢ
    residual = safe.values - predictions
    # (yᵢ − ŷᵢ) / σᵢ
    whitened = residual / safe.sigma
    # rᵢ = mᵢ (yᵢ − ŷᵢ) / σᵢ
    return jnp.where(batch.valid, whitened, jnp.zeros_like(predictions))


class GridPointObs(eqx.Module):
    """Gridded state or trajectory at scattered, time-stamped points.

        ŷᵢ = ∑ₖ wᵢₖ x[τᵢ, nᵢₖ]

    The state carries a leading time axis, ``(n_times, *grid)`` (a 4D-Var trajectory,
    or a ``ThreeDVar`` state ``(T, N)``), with ``grid`` ``(N,)`` (1-D) or ``(Ny, Nx)``
    (2-D) on the axes ``grid_axes``. Observation i takes the state at its window step
    τᵢ = ``time_index[i]`` (0 when the batch has no ``time_index``) and interpolates it
    linearly in each spatial dimension at the first ``len(grid_axes)`` columns of
    ``coords[i]``, over corners k with flat grid index nᵢₖ and weight wᵢₖ. Points
    outside the grid clamp to its edge: the ``order=1`` scheme of
    `interp_obs_from_coords`, with the stencil computed at call time so the coordinates
    can be traced.

    A ``time_index`` outside ``[0, n_times - 1]`` on a valid row is an error when the
    indices are concrete (the models check it); traced indices are clamped to that
    range. Any further coordinate columns (a time) are ignored: the time enters through
    ``time_index``.

    Algorithm:
        1. Replace padding rows by finite placeholders.
        2. Pick each observation's frame x[τᵢ].
        3. Bracket the coordinates in the grid axes, clamping at the edges.
        4. Combine the stencil corners with linear (1-D) or bilinear (2-D) weights.

    Attributes:
        grid_axes: One strictly increasing 1-D axis per spatial dimension:
            ``(x_axis,)`` or ``(y_axis, x_axis)``.

    Raises:
        GridError: At construction, if ``grid_axes`` holds neither one nor two axes, or
            an axis is not 1-D, strictly increasing and at least two points long.

    Examples:
        A 1-D grid at 0, 1, 2, 3: linear interpolation between nodes.

        >>> import jax.numpy as jnp, vardax
        >>> point_op = vardax.GridPointObs((jnp.arange(4.0),))
        >>> state = jnp.array([[0.0, 10.0, 20.0, 30.0]])
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.5], [2.25]]),
        ...     values=jnp.zeros(2),
        ...     sigma=jnp.ones(2),
        ...     valid=jnp.ones(2, bool),
        ... )
        >>> point_op(state, batch).tolist()
        [5.0, 22.5]

        Points outside the grid are clamped to the edge value:

        >>> far = vardax.PointObsBatch(
        ...     coords=jnp.array([[-5.0], [9.0]]),
        ...     values=jnp.zeros(2),
        ...     sigma=jnp.ones(2),
        ...     valid=jnp.ones(2, bool),
        ... )
        >>> point_op(state, far).tolist()
        [0.0, 30.0]

        A non-increasing axis is rejected:

        >>> vardax.GridPointObs(
        ...     (jnp.array([1.0, 0.0]),)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        GridError: grid_axes[0] must be strictly increasing.
    """

    grid_axes: tuple[Float[Array, " _"], ...]

    def __init__(self, grid_axes: tuple[Float[Array, " _"], ...]) -> None:
        """Validate and store the grid axes."""
        axes = tuple(jnp.asarray(axis_values) for axis_values in grid_axes)
        if len(axes) not in (1, 2):
            raise GridError(f"grid_axes must hold 1 or 2 axes; got {len(axes)}.")
        for axis_number, axis in enumerate(axes):
            if axis.ndim != 1 or axis.shape[0] < 2:
                raise GridError(
                    f"grid_axes[{axis_number}] must be 1-D with at least 2 points; "
                    f"got shape {axis.shape}."
                )
            if not bool(jnp.all(jnp.diff(axis) > 0)):
                raise GridError(
                    f"grid_axes[{axis_number}] must be strictly increasing."
                )
        self.grid_axes = axes

    def __call__(
        self, state: Float[Array, "n_times ..."], batch: PointObsBatch
    ) -> Float[Array, " N_obs"]:
        """Interpolate each observation's frame at its coordinates.

        Args:
            state: Gridded state x with a leading time axis.
            batch: The point observations.

        Returns:
            The interpolated prediction ŷᵢ for every row, padding included.
        """
        safe = _safe(batch)
        num_obs = safe.values.shape[0]
        grid_shape = tuple(axis.shape[0] for axis in self.grid_axes)
        flat_frames = einx.id("time cell... -> time (cell...)", state)
        # raises when the state does not hold this grid
        einx.id("time (cell...) -> time cell...", flat_frames, cell=grid_shape)
        # τᵢ
        time_indices = (
            jnp.zeros(num_obs, dtype=jnp.int32)
            if safe.time_index is None
            else jnp.clip(safe.time_index, 0, state.shape[0] - 1)
        )
        num_axes = len(self.grid_axes)
        spatial_coords = safe.coords[:, :num_axes]
        # flat grid indices nᵢₖ and weights wᵢₖ of each point's 2^D corners
        indices, weights = gaussx.linear_interpolation_weights(
            spatial_coords, list(self.grid_axes)
        )
        # τᵢ, one row per observation, broadcasting over its corners; einx is safe on
        # an empty batch because _safe has already rejected it
        time_rows = einx.id("obs -> obs 1", time_indices)
        # x[τᵢ, nᵢₖ]
        corners = flat_frames[time_rows, indices]
        # wᵢₖ x[τᵢ, nᵢₖ]
        weighted_corners = weights * corners
        # ŷᵢ = ∑ₖ wᵢₖ x[τᵢ, nᵢₖ], bilinear in 2-D
        return einx.sum("obs [corner]", weighted_corners)


class FieldPointObs(eqx.Module):
    """Callable field evaluated at scattered points.

        ŷᵢ = g(sᵢ)

    Evaluates the single-example field g(s), which returns ``(1,)`` or a scalar, at
    every observation coordinate; all coordinate columns (including a time column) are
    passed to the field. Padding rows are evaluated at the first valid row's
    coordinates, so the field is never queried where no observation is.

    Algorithm:
        1. Replace padding rows by finite placeholders.
        2. ``vmap`` the field over the coordinate rows.
        3. Reduce each result to one value per observation.

    Examples:
        A field of two coordinates:

        >>> import jax.numpy as jnp, vardax
        >>> point_op = vardax.FieldPointObs()
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        ...     values=jnp.zeros(2),
        ...     sigma=jnp.ones(2),
        ...     valid=jnp.ones(2, bool),
        ... )
        >>> point_op(lambda s: jnp.sum(s, keepdims=True), batch).tolist()
        [3.0, 7.0]

        A scalar-returning field works too, and a padding row reuses the first valid
        coordinates:

        >>> padded = vardax.PointObsBatch(
        ...     coords=jnp.array([[1.0, 2.0], [9.0, 9.0]]),
        ...     values=jnp.zeros(2),
        ...     sigma=jnp.ones(2),
        ...     valid=jnp.array([True, False]),
        ... )
        >>> point_op(lambda s: s[0] * s[1], padded).tolist()
        [2.0, 2.0]
    """

    def __call__(
        self, field: Callable[[Array], Array], batch: PointObsBatch
    ) -> Float[Array, " N_obs"]:
        """Evaluate ``field`` at each coordinate row.

        Args:
            field: Field g, mapping one coordinate row to a value.
            batch: The point observations.

        Returns:
            The field value ŷᵢ = g(sᵢ) for every row.
        """
        safe = _safe(batch)
        # g(sᵢ)
        values = jax.vmap(field)(safe.coords)
        # a scalar field may return (1,) per point; one value per observation
        return as_shape(values, (values.shape[0],))


# --- helpers for the models' PointObsBatch path ------------------------------


def map_point_batch(fn: Callable, batch: PointObsBatch):
    """Apply ``fn`` to one sample, or ``vmap`` it over a leading batch axis.

    Args:
        fn: Function of a single-sample batch.
        batch: The point observations, with ``values`` ``(N_obs,)`` or
            ``(B, N_obs)``.

    Returns:
        ``fn(batch)``, with a leading batch axis if ``batch`` has one.

    Raises:
        ShapeError: If ``values`` has neither one nor two dimensions.

    Examples:
        One sample, then the same function over a batch of two:

        >>> import jax.numpy as jnp, vardax
        >>> def one_sample(n):
        ...     return vardax.PointObsBatch(
        ...         coords=jnp.zeros((*n, 3, 1)),
        ...         values=jnp.ones((*n, 3)),
        ...         sigma=jnp.ones((*n, 3)),
        ...         valid=jnp.ones((*n, 3), bool),
        ...     )
        >>> count = lambda window: jnp.sum(window.values)
        >>> float(map_point_batch(count, one_sample(())))
        3.0
        >>> map_point_batch(count, one_sample((2,))).shape
        (2,)

        Any other rank is rejected:

        >>> map_point_batch(
        ...     count, one_sample((2, 2))
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapeError: PointObsBatch.values must be (N_obs,) or (B, N_obs) ...
    """
    if batch.values.ndim == 1:
        return fn(batch)
    if batch.values.ndim != 2:
        raise ShapeError(
            "PointObsBatch.values must be (N_obs,) or (B, N_obs); "
            f"got shape {batch.values.shape}."
        )
    return jax.vmap(fn)(batch)


def window_steps(batch: PointObsBatch) -> int:
    """Rollout steps ``n_times - 1`` of a point batch (static).

    Args:
        batch: The point observations.

    Returns:
        The number of steps T, which is 0 when ``n_times`` is ``None``.

    Raises:
        MissingInputError: If ``time_index`` is given without ``n_times``.
        ObservationError: If ``n_times`` is below 1, or a concrete ``time_index`` on a
            valid row lies outside ``[0, n_times - 1]``.

    Examples:
        A window of three states has two steps, and no ``n_times`` means none:

        >>> import jax.numpy as jnp, vardax
        >>> def points(**window):
        ...     return vardax.PointObsBatch(
        ...         coords=jnp.zeros((2, 1)),
        ...         values=jnp.zeros(2),
        ...         sigma=jnp.ones(2),
        ...         valid=jnp.ones(2, bool),
        ...         **window,
        ...     )
        >>> window_steps(points(time_index=jnp.array([0, 2]), n_times=3))
        2
        >>> window_steps(points())
        0

        A step outside the window is rejected:

        >>> window_steps(
        ...     points(time_index=jnp.array([0, 5]), n_times=3)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ObservationError: PointObsBatch.time_index has valid entries outside ...
    """
    if batch.n_times is None:
        if batch.time_index is not None:
            raise MissingInputError(
                "PointObsBatch.time_index needs n_times (the number of "
                "states in the window) for a 4D-Var model."
            )
        return 0
    if batch.n_times < 1:
        raise ObservationError(
            f"PointObsBatch.n_times must be >= 1, got {batch.n_times}."
        )
    if batch.time_index is not None:
        # Validate eagerly when the indices are concrete; traced indices are
        # clamped to [0, n_times - 1] by GridPointObs.
        try:
            time_indices = np.asarray(batch.time_index)
            valid = np.asarray(batch.valid, dtype=bool)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            return batch.n_times - 1
        out_of_range = valid & ((time_indices < 0) | (time_indices >= batch.n_times))
        if out_of_range.any():
            raise ObservationError(
                f"PointObsBatch.time_index has valid entries outside "
                f"[0, n_times - 1] = [0, {batch.n_times - 1}]: "
                f"{np.unique(time_indices[out_of_range]).tolist()}."
            )
    return batch.n_times - 1


def check_point_obs_cov(obs_cov_op: object) -> None:
    """Refuse a non-trivial Σ_ε with a point batch, which carries it as its σᵢ.

    Args:
        obs_cov_op: The model's observation-error covariance Σ_ε, or ``None``.

    Raises:
        ObservationError: If ``obs_cov_op`` is neither ``None`` nor an identity.

    Examples:
        ``None`` passes, and anything else non-trivial is refused:

        >>> import jax.numpy as jnp, lineax as lx
        >>> check_point_obs_cov(None)
        >>> check_point_obs_cov(
        ...     lx.DiagonalLinearOperator(jnp.ones(2))
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ObservationError: A PointObsBatch carries its observation errors in sigma ...
    """
    if obs_cov_op is None or isinstance(obs_cov_op, lx.IdentityLinearOperator):
        return
    raise ObservationError(
        "A PointObsBatch carries its observation errors in sigma, so the "
        "model's obs_cov_op would be ignored: build the model with "
        f"obs_cov_op=None (or an identity), not {type(obs_cov_op).__name__}."
    )


def require_obs_cov(
    obs_cov_op: lx.AbstractLinearOperator | None,
) -> lx.AbstractLinearOperator:
    """Return Σ_ε for a gridded batch, refusing the ``None`` only point batches allow.

    Args:
        obs_cov_op: The model's observation-error covariance Σ_ε, or ``None``.

    Returns:
        The covariance ``obs_cov_op``.

    Raises:
        MissingInputError: If ``obs_cov_op`` is ``None``.

    Examples:
        A covariance passes through, and ``None`` is refused:

        >>> import jax.numpy as jnp, lineax as lx
        >>> obs_covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> require_obs_cov(obs_covariance) is obs_covariance
        True
        >>> require_obs_cov(None)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        MissingInputError: obs_cov_op=None is only valid for a PointObsBatch ...
    """
    if obs_cov_op is None:
        raise MissingInputError(
            "obs_cov_op=None is only valid for a PointObsBatch, which carries "
            "its observation errors in sigma; a gridded Batch1D needs R."
        )
    return obs_cov_op


def grid_point_obs_cost(
    obs_op: object, states: Float[Array, "n_times ..."], batch: PointObsBatch
) -> Float[Array, ""]:
    """Point cost of a gridded state or trajectory under a `GridPointObs`.

        𝒥_o = ½ ∑ᵢ mᵢ (yᵢ − ŷᵢ)² / σᵢ²

    Args:
        obs_op: The observation operator, which must be a `GridPointObs`.
        states: Gridded state or trajectory x with a leading time axis.
        batch: The point observations.

    Returns:
        The scalar cost 𝒥_o.

    Raises:
        InputTypeError: If ``obs_op`` is not a `GridPointObs`.

    Examples:
        A zero state against one observation of value 1, so 𝒥_o = ½:

        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.5]]),
        ...     values=jnp.array([1.0]),
        ...     sigma=jnp.array([1.0]),
        ...     valid=jnp.array([True]),
        ... )
        >>> point_op = vardax.GridPointObs((jnp.arange(3.0),))
        >>> float(grid_point_obs_cost(point_op, jnp.zeros((1, 3)), batch))
        0.5

        Another operator is refused:

        >>> grid_point_obs_cost(
        ...     object(), jnp.zeros((1, 3)), batch
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        InputTypeError: A PointObsBatch needs a point observation operator ...
    """
    point_op = require_grid_point_obs(obs_op)
    # ŷᵢ = h(x)(sᵢ, τᵢ)
    predictions = point_op(states, batch)
    # 𝒥_o = ½ ∑ᵢ mᵢ (yᵢ − ŷᵢ)² / σᵢ²
    return point_obs_cost(predictions, batch)


def require_grid_point_obs(obs_op: object) -> GridPointObs:
    """Return ``obs_op`` if it is a `GridPointObs`, else raise ``InputTypeError``."""
    if not isinstance(obs_op, GridPointObs):
        raise InputTypeError(
            "A PointObsBatch needs a point observation operator: build the "
            f"model with obs_op=GridPointObs(...), not {type(obs_op).__name__}."
        )
    return obs_op


__all__ = ["FieldPointObs", "GridPointObs", "point_obs_cost"]
