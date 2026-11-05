"""Density heads q(x | c; ν) for ``AmortizedPosterior``.

``RegressionHead`` is a diagonal Gaussian and is fully implemented.
``ConditionalFlowHead`` (needs ``gauss_flows``) and ``ScoreDiffusionHead``
(pending a diffrax reverse-SDE pipeline) are stubs that raise on construction.
Heads expose ``map_estimate(ctx)``, ``sample(ctx, key, n)`` and
``log_prob(x, ctx)``.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

_LOG_2PI = math.log(2.0 * math.pi)


class RegressionHead(eqx.Module):
    """Gaussian regression head with diagonal covariance.

        q(x | c; ν) = 𝒩(x; μ(c; ν), diag(σ²(c; ν)))
        log q(x | c; ν) = −½ ∑ᵢ [(xᵢ − μᵢ)² / σᵢ² + log σᵢ² + log 2π]

    Two MLPs share the context c: one for the mean μ and one for the log
    variance log σ². Their outputs are laid out on ``state_shape``.

    Algorithm:
        1. Map c to the flat μ and log σ² with the two MLPs and lay them out on
           ``state_shape``.
        2. ``map_estimate`` returns μ.
        3. ``sample`` returns x = μ + σ ⊙ z with σ = exp(½ log σ²) and
           z ∼ 𝒩(0, I).
        4. ``log_prob`` returns the log-density above, summed over all state
           entries.

    Args:
        context_dim: Length of the context vector c from the encoder.
        state_shape: Shape of a single posterior sample x (e.g. ``(T, N)`` for
            ``Batch1D`` targets).
        hidden_dim: Width of the MLP hidden layers.
        depth: Number of hidden layers of each MLP.
        key: PRNG key for the MLP weights.

    Attributes:
        mlp_mu: MLP from c to the flat mean μ (size ``prod(state_shape)``).
        mlp_log_var: MLP from c to the flat log variance log σ².

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> head = vdx.RegressionHead(
        ...     context_dim=6, state_shape=(3, 4), key=jax.random.PRNGKey(0)
        ... )
        >>> ctx = jnp.ones(6)
        >>> head.map_estimate(ctx).shape
        (3, 4)
        >>> head.sample(ctx, jax.random.PRNGKey(1), 5).shape
        (5, 3, 4)

        The density is highest at the mean:

        >>> mode = head.map_estimate(ctx)
        >>> bool(head.log_prob(mode, ctx) > head.log_prob(mode + 3.0, ctx))
        True

        The head scalar log-density has no batch axis:

        >>> head.log_prob(mode, ctx).shape
        ()
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
        # Product of the state axes: size of the flat MLP output.
        flat_dim = 1
        for dim_size in self.state_shape:
            flat_dim *= dim_size
        # One key per MLP, so the mean and log-variance weights are independent.
        key_mean, key_log_var = jax.random.split(key)
        self.mlp_mu = eqx.nn.MLP(
            in_size=context_dim,
            out_size=flat_dim,
            width_size=hidden_dim,
            depth=depth,
            key=key_mean,
        )
        self.mlp_log_var = eqx.nn.MLP(
            in_size=context_dim,
            out_size=flat_dim,
            width_size=hidden_dim,
            depth=depth,
            key=key_log_var,
        )

    def map_estimate(self, ctx: Float[Array, " D"]) -> Float[Array, ...]:
        """Return the mode μ(c; ν), which equals the mean of a Gaussian."""
        # μ = lay_out(MLP_μ(c))
        return einx.id("(s...) -> s...", self.mlp_mu(ctx), s=self.state_shape)

    def sample(
        self,
        ctx: Float[Array, " D"],
        key: PRNGKeyArray,
        n: int,
    ) -> Float[Array, ...]:
        """Draw ``n`` samples x = μ + σ ⊙ z, shaped ``(n, *state_shape)``."""
        # μ = lay_out(MLP_μ(c))
        mean = einx.id("(s...) -> s...", self.mlp_mu(ctx), s=self.state_shape)
        # log σ² = lay_out(MLP_logσ²(c))
        log_var = einx.id("(s...) -> s...", self.mlp_log_var(ctx), s=self.state_shape)
        # σ = exp(½ log σ²)
        std = jnp.exp(0.5 * log_var)
        # z ∼ 𝒩(0, I), one draw per sample
        noise = jax.random.normal(key, (n, *self.state_shape))
        # x = μ + σ ⊙ z
        return mean + std * noise

    def log_prob(
        self,
        x: Float[Array, ...],
        ctx: Float[Array, " D"],
    ) -> Float[Array, ""]:
        """Return log q(x | c; ν), summed over every state entry."""
        # μ = lay_out(MLP_μ(c))
        mean = einx.id("(s...) -> s...", self.mlp_mu(ctx), s=self.state_shape)
        # log σ² = lay_out(MLP_logσ²(c))
        log_var = einx.id("(s...) -> s...", self.mlp_log_var(ctx), s=self.state_shape)
        # (x − μ)²
        squared_error = (x - mean) ** 2
        # σ⁻² = exp(−log σ²)
        inverse_variance = jnp.exp(-log_var)
        # (x − μ)² / σ² + log σ² + log 2π
        terms = squared_error * inverse_variance + log_var + _LOG_2PI
        # log q = −½ ∑ᵢ [(xᵢ − μᵢ)² / σᵢ² + log σᵢ² + log 2π]
        return -0.5 * jnp.sum(terms)


class ConditionalFlowHead(eqx.Module):
    """Conditional normalising flow head (stub).

    The planned head is x = g_flow(z; c, ν) with z ∼ 𝒩(0, I), whose density
    follows from the change of variables. It needs ``gauss_flows`` for the flow
    primitives. This class is a stub that raises ``NotImplementedError`` on
    construction until that dependency is added to ``pyproject.toml``.

    Examples:
        Constructing the stub raises until the dependency lands:

        >>> import vardax as vdx
        >>> vdx.ConditionalFlowHead()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        NotImplementedError: ConditionalFlowHead requires `gauss_flows` ...
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "ConditionalFlowHead requires `gauss_flows`, which is not yet "
            "a vardax dependency. Use `RegressionHead` for now; flow support "
            "lands in a follow-up PR alongside the gauss_flows dependency."
        )


class ScoreDiffusionHead(eqx.Module):
    """Score-based diffusion head (stub).

    The planned head learns s(x, t | y; ν) ≈ ∇ₓ log p_t(x | y) and samples by a
    reverse SDE solved with diffrax. This class is a stub that raises
    ``NotImplementedError`` on construction, pending the reverse-SDE pipeline.

    Examples:
        Constructing the stub raises until the pipeline lands:

        >>> import vardax as vdx
        >>> vdx.ScoreDiffusionHead()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        NotImplementedError: ScoreDiffusionHead is not yet implemented. ...
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "ScoreDiffusionHead is not yet implemented. Pending the diffrax-based "
            "reverse-SDE pipeline. Use `RegressionHead` for now."
        )
