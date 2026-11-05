"""Shared plumbing for the ``AnalysisStep`` adapters, which analyse around the forecast.

``analyse_forecast`` serves the single-time models and ``analyse_window`` the 4D-Var
models, each taking the observation operator h and Σ_ε from the cycle when given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D
from vardax._src.errors import InputTypeError, ShapeError
from vardax._src.obs_operators.point import require_obs_cov
from vardax._src.utils.covariance import as_covariance

#: ``analyse_one(μ_x, y, m, h, Σ_ε)`` for one sample.
AnalyseOne = Callable[
    [Float[Array, "*state"], Array, Array, Any, lx.AbstractLinearOperator],
    Float[Array, "*state"],
]


def analyse_forecast(
    analyse_one: AnalyseOne,
    *,
    state_shape: tuple[int, ...],
    default_obs_op: Any,
    default_obs_cov_op: lx.AbstractLinearOperator | None,
    forecast: Float[Array, ...],
    obs: Array,
    obs_op: Any,
    obs_err_cov: Any,
) -> Float[Array, ...]:
    """Analyse around ``forecast``, with the cycle's h and Σ_ε when given.

        x⋆ = analyse_one(μ_x, y, m, h, Σ_ε),   μ_x = ``forecast``

    ``forecast`` is either one state or a batch of states, and ``obs`` is batched the
    same way. Missing observations are ``NaN`` and are masked out of the analysis.

    Args:
        analyse_one: The model's per-sample analysis around a prior mean.
        state_shape: Shape of one model state.
        default_obs_op: The model's h, used when ``obs_op`` is ``None``.
        default_obs_cov_op: The model's Σ_ε, used when ``obs_err_cov`` is ``None``.
        forecast: The forecast, one state or a batch, which is the prior mean μ_x.
        obs: Observations y.
        obs_op: The cycle's observation operator h, or ``None``.
        obs_err_cov: The cycle's Σ_ε: ``None``, a ``lineax.AbstractLinearOperator`` on
            one observation sample, or a scalar or array of error variances
            broadcastable to one observation sample (a diagonal Σ_ε).

    Returns:
        The analysis x⋆, with the shape of ``forecast``.

    Raises:
        ShapeError: If ``forecast`` is neither one state nor a batch of states, the
            batch sizes of ``forecast`` and ``obs`` differ, or ``obs_err_cov`` does not
            act on one observation sample.
        InputTypeError: If ``obs_err_cov`` is of none of the accepted kinds.
        MissingInputError: If neither the cycle nor the model has a Σ_ε.

    Examples:
        A stand-in analysis that replaces each observed entry by its observation shows
        how a ``NaN`` becomes a masked entry:

        >>> import jax.numpy as jnp, lineax as lx
        >>> def replace(prior_mean, obs, mask, obs_op, obs_covariance):
        ...     return mask * obs + (1 - mask) * prior_mean
        >>> options = dict(
        ...     state_shape=(2,),
        ...     default_obs_op=None,
        ...     default_obs_cov_op=lx.DiagonalLinearOperator(jnp.ones(2)),
        ...     obs_op=None,
        ...     obs_err_cov=None,
        ... )
        >>> obs = jnp.array([3.0, jnp.nan])
        >>> analyse_forecast(replace, forecast=jnp.ones(2), obs=obs, **options).tolist()
        [3.0, 1.0]

        A batch of forecasts is analysed sample by sample:

        >>> batch = jnp.ones((4, 2))
        >>> analyse_forecast(replace, forecast=batch, obs=batch, **options).shape
        (4, 2)

        A forecast of any other shape is rejected:

        >>> analyse_forecast(
        ...     replace, forecast=jnp.ones(3), obs=obs, **options
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapeError: forecast must be one state of shape (2,) or a batch of them, ...
    """
    forecast = jnp.asarray(forecast)
    obs = jnp.asarray(obs)
    state_shape = tuple(state_shape)
    if forecast.shape == state_shape:
        batched = False
    elif forecast.ndim == len(state_shape) + 1 and forecast.shape[1:] == state_shape:
        batched = True
        if obs.ndim == 0 or obs.shape[0] != forecast.shape[0]:
            raise ShapeError(
                f"A batch of {forecast.shape[0]} forecasts needs observations "
                f"with the same leading axis; got shape {obs.shape}."
            )
    else:
        raise ShapeError(
            f"forecast must be one state of shape {state_shape} or a batch of "
            f"them, (B, *{state_shape}); got shape {forecast.shape}."
        )

    obs_shape = obs.shape[1:] if batched else obs.shape
    resolved_obs_op = default_obs_op if obs_op is None else obs_op
    resolved_obs_cov_op = _resolve_obs_err_cov(
        obs_err_cov, default_obs_cov_op, obs_shape, obs.dtype
    )

    finite = jnp.isfinite(obs)
    # m = 1 where y is finite
    mask = finite.astype(obs.dtype)
    # y with the gaps set to zero
    filled_obs = jnp.where(finite, obs, jnp.zeros((), obs.dtype))

    def analyse_sample(
        prior_mean: Float[Array, "*state"], sample_obs: Array, sample_mask: Array
    ) -> Float[Array, "*state"]:
        return analyse_one(
            prior_mean,
            sample_obs,
            sample_mask,
            resolved_obs_op,
            resolved_obs_cov_op,
        )

    if batched:
        return jax.vmap(analyse_sample)(forecast, filled_obs, mask)
    return analyse_sample(forecast, filled_obs, mask)


def _resolve_obs_err_cov(
    obs_err_cov: Any,
    default: lx.AbstractLinearOperator | None,
    obs_shape: tuple[int, ...],
    dtype: Any,
) -> lx.AbstractLinearOperator:
    """Map the cycle's ``obs_err_cov`` onto a ``lineax`` operator Σ_ε."""
    if obs_err_cov is None:
        return require_obs_cov(default)
    if isinstance(obs_err_cov, lx.AbstractLinearOperator):
        in_shape = obs_err_cov.in_structure().shape
        if in_shape != tuple(obs_shape):
            raise ShapeError(
                f"obs_err_cov acts on shape {in_shape}, but one observation "
                f"sample has shape {tuple(obs_shape)}."
            )
        return obs_err_cov
    try:
        variances = jnp.asarray(obs_err_cov, dtype=dtype)
    except TypeError as exc:
        raise InputTypeError(
            "obs_err_cov must be None, a lineax.AbstractLinearOperator or an "
            f"array of error variances; got {type(obs_err_cov).__name__}."
        ) from exc
    try:
        variances = jnp.broadcast_to(variances, obs_shape)
    except ValueError as exc:
        raise ShapeError(
            f"obs_err_cov variances of shape {variances.shape} do not "
            f"broadcast to one observation sample, {tuple(obs_shape)}; pass "
            "a lineax operator for a non-diagonal R."
        ) from exc
    # Σ_ε = diag(σ²); variances are positive, so the tag lets lineax.CG solve it
    return lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(variances), lx.positive_semidefinite_tag
    )


def analyse_window(
    model: Any,
    end_of_window: Callable[[Any, Any, int], Float[Array, " N"]],
    *,
    forecast: Any,
    obs: Any,
    obs_op: Any,
    obs_err_cov: Any,
) -> Float[Array, " N"]:
    """Cycled 4D-Var: analyse a window around the forecast, return its end state.

        x₀⋆ = model(y₀, …, y_T),   μ_x = forecast at t = 0

    ``SmootherCycle`` passes lists of length T + 1, the forecast and the observations
    at each step of the window; ``DACycle`` passes one state and one observation, a
    one-time window.

    Algorithm:
        1. Take the forecast at the start of the window as the prior mean μ_x, and
           stack the observations over time; ``NaN`` entries are missing.
        2. Swap μ_x, and the cycle's h and Σ_ε when given, into the model (one h and
           Σ_ε at every time of the window).
        3. Analyse, then propagate the analysis to the end of the window, the state
           the cycle steps forward from.

    Args:
        model: A 4D-Var model with ``prior_mean``, ``obs_op`` and ``obs_cov_op``.
        end_of_window: ``(model, analysis, n_steps) ↦`` the analysed state ``n_steps``
            steps after the start of the window.
        forecast: The forecasts of the window (a list), or one state.
        obs: The observations of the window (a list), or one sample.
        obs_op: The cycle's observation operator h, or ``None`` for the model's.
        obs_err_cov: The cycle's Σ_ε, as in ``analyse_forecast``, or ``None`` for the
            model's.

    Returns:
        The analysed state at the end of the window.

    Raises:
        ShapeError: If ``obs_err_cov`` does not act on one observation sample.
        InputTypeError: If ``obs_err_cov`` is of none of the accepted kinds.

    Examples:
        One state and one observation are a one-time window, so with Σ_x = Σ_ε = I and
        μ_x = 0 the analysis is y / 2:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Still:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> model = vardax.StrongFourDVar(
        ...     forward=Still(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.ones(2),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> end = analyse_window(
        ...     model,
        ...     rollout_end,
        ...     forecast=jnp.zeros(2),
        ...     obs=jnp.ones(2),
        ...     obs_op=None,
        ...     obs_err_cov=None,
        ... )
        >>> bool(jnp.allclose(end, 0.5, atol=1e-3))
        True

        Lists are a window, and a ``NaN`` is missing, so that entry stays at μ_x:

        >>> forecasts = [jnp.zeros(2), jnp.zeros(2)]
        >>> window_obs = [jnp.array([1.0, jnp.nan]), jnp.array([1.0, jnp.nan])]
        >>> end = analyse_window(
        ...     model,
        ...     rollout_end,
        ...     forecast=forecasts,
        ...     obs=window_obs,
        ...     obs_op=None,
        ...     obs_err_cov=None,
        ... )
        >>> bool(jnp.allclose(end, jnp.array([2 / 3, 0.0]), atol=1e-3))
        True
    """
    if isinstance(obs, (list, tuple)):
        obs_stack = jnp.stack([jnp.asarray(sample) for sample in obs])
    else:
        obs_stack = einx.id("m... -> 1 m...", jnp.asarray(obs))
    num_times = obs_stack.shape[0]
    # μ_x, the forecast at the start of the window
    prior_mean = jnp.asarray(
        forecast[0] if isinstance(forecast, (list, tuple)) else forecast
    )
    finite = jnp.isfinite(obs_stack)
    # mₜ = 1 where yₜ is finite
    mask = finite.astype(obs_stack.dtype)
    # yₜ with the gaps set to zero
    filled_obs = jnp.where(finite, obs_stack, jnp.zeros((), obs_stack.dtype))
    batch = Batch1D(
        input=einx.id("t m... -> 1 t m...", filled_obs),
        mask=einx.id("t m... -> 1 t m...", mask),
        target=None,
    )

    swapped = eqx.tree_at(lambda m: m.prior_mean, model, prior_mean)
    if obs_op is not None:
        swapped = eqx.tree_at(lambda m: m.obs_op, swapped, obs_op)
    if obs_err_cov is not None:
        resolved_covariance = _resolve_obs_err_cov(
            obs_err_cov, None, obs_stack.shape[1:], obs_stack.dtype
        )
        obs_covariance = as_covariance(resolved_covariance)
        swapped = eqx.tree_at(
            lambda m: m.obs_cov_op,
            swapped,
            obs_covariance,
            is_leaf=lambda x: x is None,
        )
    analysis = swapped(batch)
    # the window holds T + 1 times, so T steps to its end
    return end_of_window(swapped, analysis, num_times - 1)


def rollout_end(model: Any, analysis: Array, n_steps: int) -> Float[Array, " N"]:
    """The analysed initial state of one window, stepped T = ``n_steps`` forward.

        xₜ = f(xₜ₋₁),   t = 1, …, T,   from x₀⋆

    Args:
        model: A model with ``_rollout(initial_state, n_steps)``.
        analysis: The analysed initial states, with a leading window axis of size 1.
        n_steps: Number of steps T.

    Returns:
        The state x_T at the end of the window.

    Examples:
        With f(x) = x / 2, two steps give a quarter of x₀⋆:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Halve:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> model = vardax.StrongFourDVar(
        ...     forward=Halve(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(2),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> rollout_end(model, jnp.ones((1, 2)), 2).tolist()
        [0.25, 0.25]

        Zero steps return the analysed state itself:

        >>> rollout_end(model, jnp.ones((1, 2)), 0).tolist()
        [1.0, 1.0]
    """
    # x₀⋆ of the single window
    initial_state = analysis[0]
    # xₜ = f(xₜ₋₁)
    trajectory = model._rollout(initial_state, n_steps)
    return trajectory[-1]
