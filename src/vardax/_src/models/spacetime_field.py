"""Neural-field 4D-Var: the whole window is one continuous field x(t, s; θ).

The field is fitted to scattered observations, with the physics as a soft constraint and
no rollout (Oh 2025), so there is no background term and the network's spectral bias
plays the role of Σ_x.
"""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, Real
import numpy as np

from vardax._src._types import PointObsBatch
from vardax._src.costs import ResidualPenalty
from vardax._src.errors import (
    ConfigurationError,
    GridError,
    InputTypeError,
    MissingInputError,
    ObservationError,
    ShapeError,
)
from vardax._src.minimisers import OptaxMinimiser, OptimistixMinimiser
from vardax._src.obs_operators.point import FieldPointObs, point_obs_cost
from vardax._src.parameterization import NeuralFieldParameterization
from vardax._src.utils.shapes import as_shape


def _evaluate(field: Any, coords: Float[Array, "N P"]) -> Array:
    """``field`` at every row of ``coords``; a single channel is squeezed."""
    parameterization = NeuralFieldParameterization(field, coords, 0.0)
    params = parameterization.init_control()
    return parameterization.decode(params)


class SpaceTimeFieldResult(eqx.Module):
    """A fitted space-time field and its minimisation diagnostics.

    An [`ensemble`][vardax.SpaceTimeFieldVar.ensemble] returns one result whose array
    leaves carry a leading member axis; evaluate it with
    ``eqx.filter_vmap(lambda r: r.initial_condition(grid))(result)``.

    Attributes:
        field: The analysis x(·; θ⋆), a single-example callable of one space-time point
            ``(t, *s)``.
        t0: Start of the window, t₀, used by `initial_condition`.
        cost: 𝒥(θ⋆).
        grad_norm: ‖∇_θ 𝒥(θ⋆)‖₂; a first-order optimiser stopped by its step budget
            leaves it non-zero.
        num_steps: Optimiser steps taken.

    Examples:
        Fit a line to three points, then query the fitted field:

        >>> import equinox as eqx, jax, jax.numpy as jnp, vardax
        >>> class Line(eqx.Module):
        ...     w: jax.Array
        ...
        ...     def __call__(self, s):
        ...         return jnp.stack([self.w[0] + self.w[1] * s[1]])
        >>> obs = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.0, -1.0], [0.5, 0.0], [1.0, 1.0]]),
        ...     values=jnp.array([-1.0, 0.0, 1.0]),
        ...     sigma=jnp.ones(3),
        ...     valid=jnp.ones(3, bool),
        ... )
        >>> ana = vardax.SpaceTimeFieldVar(
        ...     field=Line(jnp.zeros(2)),
        ...     weight_decay=1e-6,
        ...     minimiser=vardax.OptimistixMinimiser(),
        ... )
        >>> result = ana(obs)
        >>> query = jnp.array([[0.0, 2.0], [0.0, 3.0]])
        >>> bool(jnp.allclose(result.evaluate(query), jnp.array([2.0, 3.0]), atol=1e-2))
        True
        >>> bool(result.grad_norm < 1e-3)
        True

        The analysis at the window start on a spatial grid is the initial condition for
        a forecast:

        >>> result.initial_condition(jnp.array([[0.5], [2.0]])).shape
        (2,)
    """

    field: Any
    t0: float = eqx.field(static=True)
    cost: Float[Array, ""]
    grad_norm: Float[Array, ""]
    num_steps: Int[Array, ""]

    def evaluate(self, coords: Float[Array, "N P"]) -> Float[Array, "N ..."]:
        """Evaluate the field at space-time points: x(tₖ, sₖ; θ⋆).

        Args:
            coords: Space-time points ``(t, *s)``, one per row.

        Returns:
            One value per point for a single-channel field, else one row of channels
            per point.
        """
        return _evaluate(self.field, jnp.asarray(coords))

    def initial_condition(self, grid: Real[Array, "N D"]) -> Float[Array, "N ..."]:
        """The analysis at the window start on a grid: x(t₀, sₙ; θ⋆).

        This is the initial state for a forecast with a ``ForwardModel``.

        Args:
            grid: Spatial points sₙ, one per row.

        Returns:
            One value per grid point for a single-channel field, else one row of
            channels per point.

        Raises:
            GridError: If ``grid`` is not 2-D.
        """
        grid = jnp.asarray(grid)
        if grid.ndim != 2:
            raise GridError(f"grid must be 2-D (N, D); got shape {grid.shape}.")
        if not jnp.issubdtype(grid.dtype, jnp.inexact):
            # an integer grid would truncate t₀ in the time column
            grid = grid.astype(jnp.result_type(float))
        time = jnp.full((grid.shape[0],), self.t0, dtype=grid.dtype)
        # (t₀, sₙ), time first
        points = einx.id("n, n d -> n (1 + d)", time, grid)
        return _evaluate(self.field, points)


class SpaceTimeFieldVar(eqx.Module):
    """Neural-field 4D-Var: fit one continuous field x(t, s; θ) to a whole window.

        𝒥(θ) = 𝒥_o(θ) + 𝒥_res(θ) + ½ β ‖θ‖²
        𝒥_o(θ) = ½ ∑ᵢ mᵢ (yᵢ − x(tᵢ, sᵢ; θ))² / σᵢ²
        𝒥_res(θ) = λ / (2 N_c) ∑ⱼ ‖𝒩[x(·; θ)](cⱼ)‖²

    The observations are those of a [`PointObsBatch`][vardax.PointObsBatch] (mᵢ = 0 on
    padding rows, which contribute nothing, ``NaN`` included), the physics term is an
    optional [`ResidualPenalty`][vardax.ResidualPenalty] with PDE operator 𝒩, weight λ
    and collocation points cⱼ, and β is the weight decay. There is no background
    covariance and no rollout: the whole window is the field, which can be queried and
    differentiated anywhere.

    Coordinates put **time first**: the field maps a point ``(t, *s)`` to its channels
    (or a scalar), and the observation coordinates and collocation points use the same
    order, as the residuals of `vardax.examples` expect. The observed quantity is the
    field's single output (`FieldPointObs`); a multi-channel field needs its own
    wrapper.

    Scale of λ: the observation term is a *sum* over the N_obs valid observations, each
    weighted by 1 / σᵢ², while the residual term is a *mean* over the collocation
    points. A unit change of the field everywhere moves the first by about
    N_obs / (2 σ²) and the second by about λ / 2 times the squared residual it causes,
    so λ trades off against N_obs / (2 σ²) and must grow with it to keep the same
    balance (the advection test uses λ = 10³ for 100 observations with σ = 0.05, where
    N_obs / (2 σ²) = 2 × 10⁴).

    The fit runs in one ``eqx.filter_jit``-compiled function of the model, the
    observations and the key, so repeated analyses with new observations (or a new
    ``weight_decay`` or penalty weight, both array leaves) of the same shapes do not
    recompile.

    The penalty's collocation points are fixed during a fit: a ``ResidualPenalty`` with
    a ``sampler`` draws them once, from ``key``, at the start of the analysis, so the
    cost the minimiser sees is deterministic (its patience test, best-iterate tracking
    and implicit adjoint all assume that). Draw more points, or refit with new keys, for
    a better estimate of the expected residual.

    Algorithm:
        1. Split the field's leaves into the control θ (by ``filter_spec``) and the
           frozen rest.
        2. Draw the penalty's collocation points once from ``key`` when it has a
           ``sampler``.
        3. Evaluate the field at the observation coordinates and form 𝒥_o.
        4. Add the weight decay ½ β ‖θ‖² and, if present, 𝒥_res.
        5. Minimise over θ with ``minimiser`` in one compiled function and return the
           fitted field with the cost, gradient norm and step count.

    Attributes:
        field: The initial network, a single-example ``eqx.Module`` of one point
            ``(t, *s)``, such as `geonnax.SIREN` or `geonnax.NeuralFieldMLP`. Its
            parameters are the initial control.
        residual: Optional [`ResidualPenalty`][vardax.ResidualPenalty], the physics
            term; ``None`` fits the observations alone.
        weight_decay: β ≥ 0, stored as a 0-d array (a leaf: it can be changed without
            recompiling and differentiated).
        minimiser: An [`OptaxMinimiser`][vardax.OptaxMinimiser] (the default, Adam with
            learning rate 10⁻³) or an
            [`OptimistixMinimiser`][vardax.OptimistixMinimiser].
        filter_spec: Which leaves of ``field`` are the control, as in
            [`NeuralFieldParameterization`][vardax.NeuralFieldParameterization]: exclude
            frozen arrays (random Fourier frequencies) so the weight decay does not
            shrink them. Default ``eqx.is_inexact_array``.
        t0: Start of the window, t₀ (default ``0.0``).

    Raises:
        ShapeError: If ``weight_decay`` is not a scalar.
        ConfigurationError: If ``weight_decay`` is negative.
        InputTypeError: If ``residual`` is not a ``ResidualPenalty`` or ``None``, or
            ``minimiser`` is neither an ``OptaxMinimiser`` nor an
            ``OptimistixMinimiser``.

    Examples:
        A field linear in fixed features, x(t, s; w) = w₀ + w₁ s, with no physics is
        Bayesian linear regression: the analysis is the ridge solution, here
        w⋆ = (0, 1) to within the small weight decay.

        >>> import equinox as eqx, jax, jax.numpy as jnp, vardax
        >>> class Line(eqx.Module):
        ...     w: jax.Array
        ...
        ...     def __call__(self, s):
        ...         return jnp.stack([self.w[0] + self.w[1] * s[1]])
        >>> obs = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.0, -1.0], [0.5, 0.0], [1.0, 1.0], [0.0, 9.0]]),
        ...     values=jnp.array([-1.0, 0.0, 1.0, jnp.nan]),  # last row: padding
        ...     sigma=jnp.ones(4),
        ...     valid=jnp.array([True, True, True, False]),
        ... )
        >>> ana = vardax.SpaceTimeFieldVar(
        ...     field=Line(jnp.zeros(2)),
        ...     weight_decay=1e-6,
        ...     minimiser=vardax.OptimistixMinimiser(),
        ... )
        >>> result = ana(obs)
        >>> initial = result.initial_condition(jnp.array([[0.5], [2.0]]))
        >>> bool(jnp.allclose(initial, jnp.array([0.5, 2.0]), atol=1e-2))
        True

        A deep ensemble refits from random initialisations; the result's array leaves
        carry a member axis:

        >>> def init(key):
        ...     return Line(jax.random.normal(key, (2,)))
        >>> members = ana.ensemble(obs, init, 3, key=jax.random.PRNGKey(0))
        >>> members.cost.shape
        (3,)

        The cycle adapter returns the analysis on a grid directly:

        >>> step = ana.as_analysis_step(grid=jnp.array([[0.5], [2.0]]))
        >>> step(None, obs).shape
        (2,)

        A negative weight decay is rejected:

        >>> vardax.SpaceTimeFieldVar(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     field=Line(jnp.zeros(2)), weight_decay=-1.0
        ... )
        Traceback (most recent call last):
        ...
        ConfigurationError: weight_decay must be >= 0, got -1.0.

    References:
        Oh, Y. (2025). On the effect of neural field reparameterization for
        4DVAR. arXiv:2509.21751.
    """

    field: Any
    residual: ResidualPenalty | None
    weight_decay: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    minimiser: OptaxMinimiser | OptimistixMinimiser = eqx.field(static=True)
    filter_spec: Any = eqx.field(static=True)
    t0: float = eqx.field(static=True)

    def __init__(
        self,
        field: Any,
        residual: ResidualPenalty | None = None,
        weight_decay: float | Float[Array, ""] = 0.0,
        minimiser: OptaxMinimiser | OptimistixMinimiser | None = None,
        *,
        filter_spec: Any = eqx.is_inexact_array,
        t0: float = 0.0,
    ) -> None:
        if jnp.ndim(weight_decay) != 0:
            raise ShapeError(
                f"weight_decay must be a scalar; got shape {jnp.shape(weight_decay)}."
            )
        try:
            negative = float(weight_decay) < 0
        except jax.errors.ConcretizationTypeError:
            # built under a transform: the sign is not known
            negative = False
        if negative:
            raise ConfigurationError(f"weight_decay must be >= 0, got {weight_decay}.")
        if residual is not None and not isinstance(residual, ResidualPenalty):
            raise InputTypeError(
                "residual must be a vardax.ResidualPenalty or None; got "
                f"{type(residual).__name__}."
            )
        if minimiser is None:
            minimiser = OptaxMinimiser()
        if not isinstance(minimiser, OptaxMinimiser | OptimistixMinimiser):
            raise InputTypeError(
                "minimiser must be a vardax.OptaxMinimiser or "
                f"vardax.OptimistixMinimiser; got {type(minimiser).__name__}."
            )
        self.field = field
        self.residual = residual
        self.weight_decay = weight_decay
        self.minimiser = minimiser
        self.filter_spec = filter_spec
        self.t0 = t0

    @property
    def _parameterization(self) -> NeuralFieldParameterization:
        """The control split of ``field``, with an empty grid and unit weight."""
        # Only the control split, ``as_field`` and the penalty are used: the field is
        # evaluated at the observation and collocation points, never on a grid, so the
        # grid is empty. The weight is 1, for ℛ(θ) = ½ ‖θ‖², because β multiplies it in
        # the cost as an array leaf.
        return NeuralFieldParameterization(
            self.field,
            jnp.zeros((0, 1)),
            1.0,
            filter_spec=self.filter_spec,
        )

    def _fixed_residual(self, key: PRNGKeyArray | None) -> ResidualPenalty | None:
        """The residual penalty with its collocation points drawn and fixed."""
        residual = self.residual
        if residual is None or residual.sampler is None:
            return residual
        if key is None:
            raise MissingInputError(
                "The residual penalty has a sampler: pass key= to draw its "
                "collocation points."
            )
        # cⱼ, drawn once so that the cost is deterministic during a fit
        collocation = residual.sampler(key)
        return ResidualPenalty(
            residual_fn=residual.residual_fn,
            collocation=collocation,
            weight=residual.weight,
        )

    def __call__(
        self, obs: PointObsBatch, *, key: PRNGKeyArray | None = None
    ) -> SpaceTimeFieldResult:
        """Fit the field to one sample of point observations.

        Args:
            obs: The observations, one sample, with ``coords`` ordered time
                first (``time_index`` and ``n_times`` are not used). For a
                batch of samples, ``eqx.filter_vmap`` over this call.
            key: Draws the collocation points when the residual penalty
                has a ``sampler``; unused otherwise.

        Returns:
            A [`SpaceTimeFieldResult`][vardax.SpaceTimeFieldResult].

        Raises:
            InputTypeError: If ``obs`` is not a `PointObsBatch`.
            ShapeError: If ``obs`` carries a batch axis.
            MissingInputError: If the penalty needs a key and none is
                given.
        """
        if not isinstance(obs, PointObsBatch):
            raise InputTypeError(
                f"obs must be a vardax.PointObsBatch; got {type(obs).__name__}."
            )
        if obs.values.ndim != 1:
            raise ShapeError(
                "SpaceTimeFieldVar fits one sample: obs.values must be "
                f"(N_obs,), got shape {obs.values.shape}; use eqx.filter_vmap "
                "for a batch."
            )
        return _fit(self, obs, key)

    def ensemble(
        self,
        obs: PointObsBatch,
        field_init: Any,
        n_members: int,
        *,
        key: PRNGKeyArray,
    ) -> SpaceTimeFieldResult:
        """A deep ensemble: refit from ``n_members`` random initialisations.

        Each member re-initialises the field with ``field_init`` and draws its own
        collocation points (when the penalty has a sampler), then fits as `__call__`
        does; the members run under ``eqx.filter_vmap``. The spread of the members is a
        heuristic uncertainty: small where the observations pin the field down, large
        away from them. A θ-space Laplace push-forward J Σ_θ Jᵀ, with
        J = ∂x/∂θ, is not implemented.

        Args:
            obs: The observations, one sample.
            field_init: ``key -> field``, building a network with the same structure as
                ``self.field`` (e.g.
                ``lambda k: geonnax.SIREN.init(2, 32, 1, depth=3, key=k)``).
            n_members: Number of members, at least 1.
            key: Split into one initialisation key and one collocation key per member.

        Returns:
            A [`SpaceTimeFieldResult`][vardax.SpaceTimeFieldResult] whose array leaves
            carry a leading axis of size ``n_members``.

        Raises:
            ConfigurationError: If ``n_members < 1``.
        """
        if n_members < 1:
            raise ConfigurationError(f"n_members must be >= 1, got {n_members}.")

        def member(member_key: PRNGKeyArray) -> SpaceTimeFieldResult:
            init_key, collocation_key = jax.random.split(member_key)
            model = eqx.tree_at(lambda tree: tree.field, self, field_init(init_key))
            return model(obs, key=collocation_key)

        return eqx.filter_vmap(member)(jax.random.split(key, n_members))

    def as_analysis_step(
        self,
        grid: Float[Array, "N D"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> _SpaceTimeFieldAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        The step fits the field to the cycle's observations. With a ``grid`` it returns
        the analysis at the window start on it, x(t₀, sₙ; θ⋆)
        (`SpaceTimeFieldResult.initial_condition`), the state a ``ForwardModel`` steps
        on; without one it returns the
        [`SpaceTimeFieldResult`][vardax.SpaceTimeFieldResult] itself.

        The collocation points are drawn **once**, here, from ``key``, and reused by
        every cycle. ``pipekit_cycle.DACycle`` passes no time: the times are the first
        column of each cycle's ``PointObsBatch``, so they must be **window-relative**,
        in the same window [t₀, t₁] every cycle as the collocation points. The step
        checks that (when the times are concrete): t₀ and the valid observation times
        must lie in the collocation points' time range, widened by 10 % on each side for
        the gaps a random draw leaves at the ends.

        Args:
            grid: Spatial points sₙ of the returned state, one per row, or ``None``.
            key: Collocation key; required when the penalty has a ``sampler``.

        Returns:
            The adapter.

        Raises:
            GridError: If ``grid`` is not 2-D.
            MissingInputError: If the penalty has a sampler and ``key`` is ``None``.
            ConfigurationError: If t₀ is outside the collocation window.
        """
        if grid is not None:
            grid = jnp.asarray(grid)
            if grid.ndim != 2:
                raise GridError(f"grid must be 2-D (N, D); got shape {grid.shape}.")
        if self.residual is not None and self.residual.sampler is not None:
            if key is None:
                raise MissingInputError(
                    "The residual penalty has a sampler: pass key= to "
                    "as_analysis_step to draw the collocation points, which "
                    "every cycle then reuses."
                )
        model = eqx.tree_at(
            lambda tree: tree.residual,
            self,
            self._fixed_residual(key),
            is_leaf=lambda leaf: leaf is None,
        )
        window = _collocation_window(model.residual)
        if window is not None and not window[0] <= self.t0 <= window[1]:
            raise ConfigurationError(
                f"t0 = {self.t0} is outside the collocation time window "
                f"[{window[0]:.4g}, {window[1]:.4g}]: the initial condition "
                "would be unconstrained by the physics."
            )
        return _SpaceTimeFieldAnalysisStep(model, grid)


def _collocation_window(residual: ResidualPenalty | None) -> tuple[float, float] | None:
    """Time range of fixed collocation points, widened by 10 % each side."""
    if residual is None or residual.collocation is None:
        return None
    try:
        times = np.asarray(residual.collocation[:, 0])
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None
    time_min, time_max = float(times.min()), float(times.max())
    # 10 % of the range, for the gaps a random draw leaves at the ends
    padding = 0.1 * (time_max - time_min)
    return time_min - padding, time_max + padding


def _cost(
    params: PyTree[Array],
    args: tuple[
        NeuralFieldParameterization,
        PointObsBatch,
        ResidualPenalty | None,
        Float[Array, ""],
    ],
) -> Float[Array, ""]:
    """𝒥(θ) = 𝒥_o(θ) + ½ β ‖θ‖² + 𝒥_res(θ); every input is a pytree argument."""
    parameterization, obs, residual, weight_decay = args
    # x(·; θ)
    field = parameterization.as_field(params)
    # ŷᵢ = x(tᵢ, sᵢ; θ)
    predictions = FieldPointObs()(field, obs)
    # 𝒥_o = ½ ∑ᵢ mᵢ (yᵢ − ŷᵢ)² / σᵢ²
    obs_cost = point_obs_cost(predictions, obs)
    # ½ β ‖θ‖²
    decay_cost = weight_decay * parameterization.penalty(params)
    cost = obs_cost + decay_cost
    if residual is not None:
        # 𝒥_res = λ / (2 N_c) ∑ⱼ ‖𝒩[x(·; θ)](cⱼ)‖²
        cost = cost + residual(field)
    return cost


@eqx.filter_jit
def _fit(
    model: SpaceTimeFieldVar, obs: PointObsBatch, key: PRNGKeyArray | None
) -> SpaceTimeFieldResult:
    """Fit ``model`` to one sample: compiled once per model structure and shapes."""
    parameterization = model._parameterization
    residual = model._fixed_residual(key)
    # __init__ skips the sign check when weight_decay is traced; catch it here
    weight_decay = eqx.error_if(
        model.weight_decay, model.weight_decay < 0, "weight_decay must be >= 0"
    )
    result = model.minimiser.run(
        _cost,
        parameterization.init_control(),
        (parameterization, obs, residual, weight_decay),
    )
    return SpaceTimeFieldResult(
        field=parameterization.as_field(result.value),
        t0=model.t0,
        cost=result.cost,
        grad_norm=result.grad_norm,
        num_steps=result.num_steps,
    )


@eqx.filter_jit
def _fit_on_grid(
    model: SpaceTimeFieldVar, obs: PointObsBatch, grid: Float[Array, "N D"]
) -> Array:
    """x(t₀, ·; θ⋆) on ``grid``, from a fit with fixed collocation points."""
    return _fit(model, obs, None).initial_condition(grid)


class _SpaceTimeFieldAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``SpaceTimeFieldVar``.

    ``obs`` is one sample of a [`PointObsBatch`][vardax.PointObsBatch] with
    window-relative times, whose σᵢ are the observation errors, so ``obs_err_cov`` must
    be ``None``. There is no background term, so ``forecast`` is not used in the cost.
    The model's collocation points are already drawn and fixed.

    Attributes:
        model: The model to fit with.
        grid: Spatial points sₙ of the returned state, or ``None`` to return the fitted
            result.
    """

    model: SpaceTimeFieldVar
    grid: Float[Array, "N D"] | None

    def __call__(
        self,
        forecast: Array | None,
        obs: PointObsBatch,
        *,
        obs_op: Any = None,
        obs_err_cov: Any = None,
    ) -> Array | SpaceTimeFieldResult:
        """Fit the field to the cycle's observations.

        Args:
            forecast: With a grid, the shape and dtype of the returned analysis, or
                ``None``.
            obs: The observations, one sample.
            obs_op: Not used: the field is evaluated at the observation coordinates
                directly, and ``DACycle``'s gridded h does not apply to a field.
            obs_err_cov: Must be ``None``.

        Returns:
            The analysis x(t₀, sₙ; θ⋆) on the grid, or the fitted result without one.

        Raises:
            ObservationError: If ``obs_err_cov`` is given, or a valid observation time
                is outside the collocation window.
            ShapeError: If ``forecast`` has a different size from the analysis.
        """
        # the field is observed at the points (FieldPointObs)
        del obs_op
        if obs_err_cov is not None:
            raise ObservationError(
                "A PointObsBatch carries its observation errors in sigma: "
                "pass obs_err_cov=None."
            )
        self._check_window(obs)
        if self.grid is None:
            return self.model(obs)
        analysis = _fit_on_grid(self.model, obs, self.grid)
        if forecast is None:
            return analysis
        forecast = jnp.asarray(forecast)
        if forecast.size != analysis.size:
            raise ShapeError(
                f"forecast has shape {forecast.shape}, but the analysis on the "
                f"grid has {analysis.size} values."
            )
        return as_shape(analysis, forecast.shape).astype(forecast.dtype)

    def _check_window(self, obs: PointObsBatch) -> None:
        """Valid observation times must lie in the collocation window."""
        window = _collocation_window(self.model.residual)
        if window is None or not isinstance(obs, PointObsBatch):
            return
        try:
            times = np.asarray(obs.coords[..., 0])
            valid = np.asarray(obs.valid, dtype=bool)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            return
        times = times[valid]
        if times.size and (times.min() < window[0] or times.max() > window[1]):
            raise ObservationError(
                f"Observation times span [{times.min():.4g}, {times.max():.4g}], "
                f"outside the collocation window [{window[0]:.4g}, "
                f"{window[1]:.4g}]: give each cycle window-relative times."
            )


__all__ = ["SpaceTimeFieldResult", "SpaceTimeFieldVar"]
