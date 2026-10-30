"""Shared plumbing for the ``AnalysisStep`` adapters.

``pipekit_cycle.AnalysisStep`` hands an adapter the forecast $x_b$, the
observations $y$, the observation operator $H$ and the observation-error
covariance $R$. The models carry their own background, $H$ and $R$ for
standalone use; their adapters analyse around the forecast instead, with
the $H$ and $R$ the cycle passes. ``analyse_forecast`` does this for the
single-time models (OI, 3D-Var) and ``analyse_window`` for the 4D-Var
models, which see a window of observations.
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

#: ``analyse_one(x_b, y, mask, obs_op, obs_cov_op)`` for one sample.
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
    r"""Analyse around ``forecast`` with the cycle's $H$ and $R$.

    ``forecast`` is either one state (``state_shape``) or a batch of
    states (``(B, *state_shape)``), and ``obs`` is batched the same way.
    Missing observations are ``NaN`` and are masked out of the analysis.

    Args:
        analyse_one: The model's per-sample analysis around a background.
        state_shape: Shape of one model state (the model's background).
        default_obs_op: The model's $H$, used when ``obs_op`` is ``None``.
        default_obs_cov_op: The model's $R$, used when ``obs_err_cov`` is
            ``None``.
        forecast: The forecast (background) $x_b$, one state or a batch.
        obs: Observations $y$.
        obs_op: $H$ from the cycle, or ``None``.
        obs_err_cov: $R$ from the cycle: ``None``, a
            ``lineax.AbstractLinearOperator`` on one observation sample,
            or a scalar / array of error variances broadcastable to one
            observation sample (a diagonal $R$).

    Returns:
        The analysis $x_a$, with ``forecast``'s shape.

    Raises:
        ValueError: If ``forecast`` is neither one state nor a batch of
            states, the batch sizes of ``forecast`` and ``obs`` differ, or
            ``obs_err_cov`` does not act on one observation sample.
        TypeError: If ``obs_err_cov`` is of none of the accepted kinds.
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
    mask = finite.astype(obs.dtype)
    obs_clean = jnp.where(finite, obs, jnp.zeros((), obs.dtype))

    def _one(
        background: Float[Array, "*state"], sample_obs: Array, sample_mask: Array
    ) -> Float[Array, "*state"]:
        return analyse_one(
            background,
            sample_obs,
            sample_mask,
            resolved_obs_op,
            resolved_obs_cov_op,
        )

    if batched:
        return jax.vmap(_one)(forecast, obs_clean, mask)
    return _one(forecast, obs_clean, mask)


def _resolve_obs_err_cov(
    obs_err_cov: Any,
    default: lx.AbstractLinearOperator | None,
    obs_shape: tuple[int, ...],
    dtype: Any,
) -> lx.AbstractLinearOperator:
    """Map the cycle's ``obs_err_cov`` onto a ``lineax`` operator $R$."""
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
    # Variances are positive, so R is PSD; the tag lets lineax.CG solve it.
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
    r"""Cycled 4D-Var: analyse a window around the forecast, return its end state.

    ``SmootherCycle`` passes length-$W$ lists, the forecast and the
    observations at each step of the window; ``DACycle`` passes one state
    and one observation, a one-time window.

    Algorithm:
        1. Take the forecast at the start of the window as the background
           $x_b$, and stack the observations to ``(1, W, M)``; ``NaN``
           entries are missing.
        2. Swap $x_b$, and the cycle's $H$ and $R$ when given, into the
           model (one $H$ and $R$ at every time of the window).
        3. Analyse, then propagate the analysis to the end of the window,
           the state the cycle steps forward from.

    Args:
        model: A 4D-Var model with ``prior_mean``, ``obs_op`` and
            ``obs_cov_op``.
        end_of_window: ``(model, analysis, n_steps) ↦`` the analysed state
            ``n_steps`` steps after the window start.
        forecast: The forecasts of the window (a list), or one state.
        obs: The observations of the window (a list), or one sample.
        obs_op: $H$ from the cycle, or ``None`` for the model's.
        obs_err_cov: $R$ from the cycle, as in ``analyse_forecast``, or
            ``None`` for the model's.

    Returns:
        The analysed state at the end of the window.
    """
    if isinstance(obs, (list, tuple)):
        obs_stack = jnp.stack([jnp.asarray(sample) for sample in obs])
    else:
        obs_stack = einx.id("m... -> 1 m...", jnp.asarray(obs))
    n_times = obs_stack.shape[0]
    background = jnp.asarray(
        forecast[0] if isinstance(forecast, (list, tuple)) else forecast
    )
    finite = jnp.isfinite(obs_stack)
    mask = finite.astype(obs_stack.dtype)
    observed = jnp.where(finite, obs_stack, jnp.zeros((), obs_stack.dtype))
    batch = Batch1D(
        input=einx.id("t m... -> 1 t m...", observed),
        mask=einx.id("t m... -> 1 t m...", mask),
        target=None,
    )

    swapped = eqx.tree_at(lambda m: m.prior_mean, model, background)
    if obs_op is not None:
        swapped = eqx.tree_at(lambda m: m.obs_op, swapped, obs_op)
    if obs_err_cov is not None:
        obs_cov = as_covariance(
            _resolve_obs_err_cov(
                obs_err_cov, None, obs_stack.shape[1:], obs_stack.dtype
            )
        )
        swapped = eqx.tree_at(
            lambda m: m.obs_cov_op, swapped, obs_cov, is_leaf=lambda x: x is None
        )
    analysis = swapped(batch)
    return end_of_window(swapped, analysis, n_times - 1)


def rollout_end(model: Any, analysis: Array, n_steps: int) -> Float[Array, " N"]:
    """The analysed initial state of one window, stepped ``n_steps`` forward."""
    return model._rollout(analysis[0], n_steps)[-1]
