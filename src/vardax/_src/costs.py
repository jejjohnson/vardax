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
) -> Float[Array, ""]:
    """Observation cost for 1-D data.

    Computes the masked mean-squared error between the state and observations:

    .. math::

        J_{obs} = \\frac{1}{|\\Omega|} \\sum_{i \\in \\Omega} (x_i - y_i)^2

    where :math:`\\Omega` is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, N)``.
        obs: Observations of shape ``(B, T, N)``.
        mask: Binary observation mask of shape ``(B, T, N)``.
            A value of ``1`` indicates an observed location.

    Returns:
        Scalar observation cost.
    """
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def obs_cost_2d(
    state: Float[Array, "B T H W"],
    obs: Float[Array, "B T H W"],
    mask: Float[Array, "B T H W"],
) -> Float[Array, ""]:
    """Observation cost for 2-D data.

    Computes the masked mean-squared error between the state and observations:

    .. math::

        J_{obs} = \\frac{1}{|\\Omega|} \\sum_{i \\in \\Omega} (x_i - y_i)^2

    where :math:`\\Omega` is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, H, W)``.
        obs: Observations of shape ``(B, T, H, W)``.
        mask: Binary observation mask of shape ``(B, T, H, W)``.

    Returns:
        Scalar observation cost.
    """
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def prior_cost(
    state: Float[Array, ...],
    prior_reconstruction: Float[Array, ...],
) -> Float[Array, ""]:
    """Prior cost based on learned autoencoder reconstruction.

    Computes the mean-squared error between the state and its reconstruction
    through the learned prior (autoencoder):

    .. math::

        J_{prior} = \\|x - \\varphi(x)\\|^2

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
    """Compute the variational cost :math:`U(x)`.

    .. math::

        U(x) = \\alpha_{obs} \\|m \\odot (x - y)\\|^2
              + \\alpha_{prior} \\|x - \\varphi(x)\\|^2

    Args:
        x: Current state estimate.
        batch: Observed data batch with ``input`` (``y``) and ``mask``
            (``m``).
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term (default ``0.5``).
        alpha_prior: Weight for the prior term (default ``0.5``).

    Returns:
        Scalar cost value.
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
    """Return the gradient of :func:`variational_cost` with respect to ``x``.

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
