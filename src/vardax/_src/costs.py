"""Cost functions for 4DVarNet variational data assimilation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._types import Batch1D


def obs_cost_1d(
    state: Float[Array, "B T N"],
    obs: Float[Array, "B T N"],
    mask: Float[Array, "B T N"],
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    r"""Observation cost for 1-D data.

    Computes the masked mean-squared error between the state and observations:

    $$
    J_{obs} = \frac{1}{|\Omega|} \sum_{i \in \Omega} (x_i - y_i)^2
    $$

    where $\Omega$ is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, N)``.
        obs: Observations of shape ``(B, T, N)``.
        mask: Binary observation mask of shape ``(B, T, N)``.
            A value of ``1`` indicates an observed location.
        nan_to_num: When ``True``, replace ``NaN`` entries in ``obs`` with
            zero (via :func:`jax.numpy.nan_to_num`) *before* masking. Use this
            for real-world geophysical products (e.g. satellite altimetry)
            where gaps are stored as ``NaN`` rather than an explicit binary
            mask. Without it, a ``NaN`` observation poisons both the cost and
            its gradient even at masked-out locations, since ``0 * NaN`` is
            ``NaN`` in JAX. Ported from ``mfourdvar``'s ``ObsOperator.loss``.

    Returns:
        Scalar observation cost.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import obs_cost_1d
        >>> state = jnp.ones((1, 1, 4))
        >>> obs = jnp.zeros((1, 1, 4))
        >>> mask = jnp.ones((1, 1, 4))
        >>> float(obs_cost_1d(state, obs, mask))
        1.0
    """
    if nan_to_num:
        obs = jnp.nan_to_num(obs)
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def obs_cost_2d(
    state: Float[Array, "B T H W"],
    obs: Float[Array, "B T H W"],
    mask: Float[Array, "B T H W"],
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    r"""Observation cost for 2-D data.

    Computes the masked mean-squared error between the state and observations:

    $$
    J_{obs} = \frac{1}{|\Omega|} \sum_{i \in \Omega} (x_i - y_i)^2
    $$

    where $\Omega$ is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, H, W)``.
        obs: Observations of shape ``(B, T, H, W)``.
        mask: Binary observation mask of shape ``(B, T, H, W)``.
        nan_to_num: When ``True``, replace ``NaN`` entries in ``obs`` with
            zero before masking (see :func:`obs_cost_1d` for the rationale).

    Returns:
        Scalar observation cost.
    """
    if nan_to_num:
        obs = jnp.nan_to_num(obs)
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def prior_cost(
    state: Float[Array, ...],
    prior_reconstruction: Float[Array, ...],
) -> Float[Array, ""]:
    r"""Prior cost based on learned autoencoder reconstruction.

    Computes the mean-squared error between the state and its reconstruction
    through the learned prior (autoencoder):

    $$
    J_{prior} = \|x - \varphi(x)\|^2
    $$

    Args:
        state: Current state estimate of arbitrary shape.
        prior_reconstruction: Autoencoder reconstruction of the state,
            same shape as ``state``.

    Returns:
        Scalar prior cost.
    """
    return jnp.mean((state - prior_reconstruction) ** 2)


# ---------------------------------------------------------------------------
# Structured variational cost utilities
# ---------------------------------------------------------------------------


def variational_cost(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> Float[Array, ""]:
    r"""Compute the variational cost $U(x)$.

    $$
    U(x) = \alpha_{obs} \|m \odot (x - y)\|^2
          + \alpha_{prior} \|x - \varphi(x)\|^2
    $$

    Args:
        x: Current state estimate.
        batch: Observed data batch with ``input`` (``y``) and ``mask``
            (``m``).
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term (default ``0.5``).
        alpha_prior: Weight for the prior term (default ``0.5``).

    Returns:
        Scalar cost value.

    Examples:
        With the trivial [`IdentityPrior`][vardax.IdentityPrior] the
        prior term vanishes, leaving the weighted observation MSE.

        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.Batch1D(input=jnp.zeros((1, 2, 4)), mask=jnp.ones((1, 2, 4)))
        >>> x = jnp.ones((1, 2, 4))
        >>> float(vardax.variational_cost(x, batch, vardax.IdentityPrior()))
        0.5
    """
    obs_diff = batch.mask * (x - batch.input)
    j_obs = jnp.mean(obs_diff**2)
    j_prior = jnp.mean((x - prior_fn(x)) ** 2)
    return alpha_obs * j_obs + alpha_prior * j_prior


def variational_cost_grad(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> Float[Array, ...]:
    """Gradient of [`variational_cost`][vardax.variational_cost] w.r.t. ``x``.

    Args:
        x: Current state estimate.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term.
        alpha_prior: Weight for the prior term.

    Returns:
        Gradient array with the same shape as ``x``.
    """
    return jax.grad(variational_cost)(x, batch, prior_fn, alpha_obs, alpha_prior)


def decomposed_loss(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> dict[str, Float[Array, ""]]:
    """Compute the decomposed variational loss.

    Returns individual observation and prior components alongside the
    total, matching the ``ModelLoss`` pattern from the legacy codebase.

    Args:
        x: Current state estimate.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term.
        alpha_prior: Weight for the prior term.

    Returns:
        Dictionary with keys ``"obs"``, ``"prior"``, and ``"total"``.
    """
    obs_diff = batch.mask * (x - batch.input)
    obs = alpha_obs * jnp.mean(obs_diff**2)
    prior = alpha_prior * jnp.mean((x - prior_fn(x)) ** 2)
    return {"obs": obs, "prior": prior, "total": obs + prior}


# ---------------------------------------------------------------------------
# Strong-constraint variational cost (ported from mfourdvar)
# ---------------------------------------------------------------------------


def background_cost(
    x0: Float[Array, ...],
    xb: Float[Array, ...],
) -> Float[Array, ""]:
    r"""Background cost $\|x_0 - x_b\|^2$.

    Penalises departure of the initial state $x_0$ (the control variable
    in strong-constraint 4DVar) from the background / first guess $x_b$.
    Uses the same mean-squared convention as the other functional costs in
    this module.

    Args:
        x0: Initial state estimate of arbitrary shape.
        xb: Background state, same shape as ``x0``.

    Returns:
        Scalar background cost.
    """
    return jnp.mean((x0 - xb) ** 2)


def strong_variational_cost(
    x0: Float[Array, ...],
    ts: Float[Array, T],  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
    batch: Batch1D,
    forward_fn: Callable[..., Any],
    *,
    xb: Float[Array, ...] | None = None,
    alpha_obs: float = 0.5,
    alpha_bg: float = 0.5,
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    r"""Strong-constraint variational cost $U(x_0)$.

    $$
    U(x_0) = \alpha_{obs}\,\|m \odot (\varphi(x_0) - y)\|^2
           + \alpha_{bg}\,\|x_0 - x_b\|^2 .
    $$

    The dynamical model is enforced as a **hard** constraint: the initial
    state ``x0`` is propagated through the dynamics ``forward_fn(x0, ts)``
    and only the resulting trajectory is scored against the observations.
    This differs from the weak / soft-constraint
    [`variational_cost`][vardax.variational_cost], where the model appears
    as an *additive* prior penalty and the whole state field is free.

    Ported from ``mfourdvar``'s ``StrongVarCost``, adapted to a functional
    form that mirrors [`variational_cost`][vardax.variational_cost].

    Args:
        x0: Initial state (control variable). Whatever shape ``forward_fn``
            expects — e.g. ``(B, N)`` for a batched 1-D system.
        ts: Time coordinates of shape ``(T,)`` passed to ``forward_fn``.
        batch: Observed data batch supplying ``input`` (``y``) and ``mask``
            (``m``), each of shape matching ``forward_fn``'s trajectory output
            (e.g. ``(B, T, N)``).
        forward_fn: Callable ``(x0, ts) -> trajectory`` producing a state
            trajectory the same shape as ``batch.input``. A
            [`DynTrajectory`][vardax.DynTrajectory] (optionally ``vmap``-ed
            over the batch axis) is the canonical choice.
        xb: Background state for the background term. Defaults to ``x0``
            (i.e. no background penalty).
        alpha_obs: Weight for the observation term (default ``0.5``).
        alpha_bg: Weight for the background term (default ``0.5``).
        nan_to_num: When ``True``, replace ``NaN`` entries in the
            observations with zero before masking (see
            [`obs_cost_1d`][vardax.obs_cost_1d]).

    Returns:
        Scalar strong-constraint variational cost.
    """
    if xb is None:
        xb = x0
    traj = forward_fn(x0, ts)
    obs = batch.input
    if nan_to_num:
        obs = jnp.nan_to_num(obs)
    j_obs = jnp.mean((batch.mask * (traj - obs)) ** 2)
    j_bg = background_cost(x0, xb)
    return alpha_obs * j_obs + alpha_bg * j_bg
