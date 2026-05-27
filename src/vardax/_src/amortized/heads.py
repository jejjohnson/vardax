"""Density heads for ``AmortizedPosterior``.

Three head variants per the design (chapter 10):

- ``RegressionHead`` — Gaussian ``q_φ(x|y) = N(μ_φ(y), Σ_φ(y))`` with
  diagonal covariance. Fully implemented.
- ``ConditionalFlowHead`` — normalising flow. Requires ``gauss_flows``;
  ships as a stub until that dependency lands.
- ``ScoreDiffusionHead`` — score-based diffusion. Stub pending the
  diffrax-based reverse-SDE pipeline.

Heads expose three methods:

- ``map_estimate(ctx)`` — mode of ``q_φ(x|ctx)``.
- ``sample(ctx, key, n)`` — ``n`` posterior samples shaped
  ``(n, *state_shape)``.
- ``log_prob(x, ctx)`` — scalar log-density of ``x`` under ``q_φ(·|ctx)``.
  ``ScoreDiffusionHead.log_prob`` raises ``NotImplementedError``.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

_LOG_2PI = math.log(2.0 * math.pi)


class RegressionHead(eqx.Module):
    """Gaussian regression head: ``q_φ(x|y) = N(μ_φ(y), diag(σ²_φ(y)))``.

    Two MLPs share the context: one for the mean, one for the log
    variance. Outputs are reshaped to the user-specified
    ``state_shape``.

    Attributes:
        mlp_mu: MLP from context to flat mean (size ``prod(state_shape)``).
        mlp_log_var: MLP from context to flat log variance.
        state_shape: Shape of a single posterior sample (e.g.
            ``(T, N)`` for ``Batch1D`` targets).
    """

    mlp_mu: eqx.nn.MLP
    mlp_log_var: eqx.nn.MLP
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        context_dim: int,
        state_shape: tuple[int, ...],
        *,
        hidden_dim: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
    ) -> None:
        self.state_shape = tuple(state_shape)
        flat_dim = 1
        for d in self.state_shape:
            flat_dim *= d
        k_mu, k_lv = jax.random.split(key)
        self.mlp_mu = eqx.nn.MLP(
            in_size=context_dim,
            out_size=flat_dim,
            width_size=hidden_dim,
            depth=depth,
            key=k_mu,
        )
        self.mlp_log_var = eqx.nn.MLP(
            in_size=context_dim,
            out_size=flat_dim,
            width_size=hidden_dim,
            depth=depth,
            key=k_lv,
        )

    def map_estimate(self, ctx: Float[Array, " D"]) -> Float[Array, ...]:
        return self.mlp_mu(ctx).reshape(self.state_shape)

    def sample(
        self,
        ctx: Float[Array, " D"],
        key: PRNGKeyArray,
        n: int,
    ) -> Float[Array, ...]:
        mu = self.mlp_mu(ctx).reshape(self.state_shape)
        log_var = self.mlp_log_var(ctx).reshape(self.state_shape)
        sigma = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(key, (n, *self.state_shape))
        return mu + sigma * eps

    def log_prob(
        self,
        x: Float[Array, ...],
        ctx: Float[Array, " D"],
    ) -> Float[Array, ""]:
        mu = self.mlp_mu(ctx).reshape(self.state_shape)
        log_var = self.mlp_log_var(ctx).reshape(self.state_shape)
        # log N(x | mu, diag(exp(log_var))) — sum over state dims.
        return -0.5 * jnp.sum(((x - mu) ** 2) * jnp.exp(-log_var) + log_var + _LOG_2PI)


class ConditionalFlowHead(eqx.Module):
    """Conditional normalising flow head (stub).

    Implements ``x = f_φ(z; c_ψ(y))`` with ``z ~ N(0, I)``. Exact
    density via change-of-variables. Requires ``gauss_flows`` for the
    flow primitives; ships as a stub until that dependency is added to
    ``pyproject.toml``.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "ConditionalFlowHead requires `gauss_flows`, which is not yet "
            "a vardax dependency. Use `RegressionHead` for now; flow support "
            "lands in a follow-up PR alongside the gauss_flows dependency."
        )


class ScoreDiffusionHead(eqx.Module):
    """Score-based diffusion head (stub).

    Learns ``s_φ(x, t | y) ≈ ∇_x log p_t(x | y)``; samples via reverse
    SDE solved with diffrax. Ships as a stub pending the reverse-SDE
    pipeline.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "ScoreDiffusionHead is not yet implemented. Pending the diffrax-based "
            "reverse-SDE pipeline. Use `RegressionHead` for now."
        )
