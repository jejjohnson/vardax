"""Shared plumbing for the single-time ``AnalysisStep`` adapters (OI, 3D-Var).

``pipekit_cycle.AnalysisStep`` hands an adapter the forecast $x_b$, the
observations $y$, the observation operator $H$ and the observation-error
covariance $R$. ``OptimalInterpolation`` and ``ThreeDVar`` carry their own
background, $H$ and $R$ for standalone use; their adapters analyse around
the forecast instead, with the $H$ and $R$ the cycle passes. The helper
here maps the cycle's arguments onto the models' representation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src.errors import InputTypeError, ShapeError
from vardax._src.obs_operators.point import require_obs_cov

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
    H = default_obs_op if obs_op is None else obs_op
    R = _resolve_obs_err_cov(obs_err_cov, default_obs_cov_op, obs_shape, obs.dtype)

    finite = jnp.isfinite(obs)
    mask = finite.astype(obs.dtype)
    obs_clean = jnp.where(finite, obs, jnp.zeros((), obs.dtype))

    def _one(x_b: Float[Array, "*state"], y: Array, m: Array) -> Float[Array, "*state"]:
        return analyse_one(x_b, y, m, H, R)

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
