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

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

_LOG_2PI = math.log(2.0 * math.pi)


class RegressionHead(eqx.Module):
    r"""Gaussian regression head: ``q_φ(x|y) = N(μ_φ(y), diag(σ²_φ(y)))``.

    $$
    q_\phi(x \mid y) = \mathcal{N}(\mu_\phi(y), \mathrm{diag}\,\sigma^2_\phi(y))
    $$

    Two MLPs share the context: one for the mean, one for the log
    variance. Outputs are reshaped to the user-specified
    ``state_shape``.

    Algorithm:
        1. Map the context to the flat mean $\mu$ and log variance
           $\log \sigma^2$ with the two MLPs, and reshape them to ``state_shape``.
        2. ``map_estimate`` returns $\mu$.
        3. ``sample`` returns $\mu + \sigma \odot \varepsilon$ with
           $\varepsilon \sim \mathcal{N}(0, I)$.
        4. ``log_prob`` returns the diagonal-Gaussian log-density summed over
           all state entries.

    Args:
        context_dim: Dimension of the context vector from the encoder.
        state_shape: Shape of a single posterior sample (e.g.
            ``(T, N)`` for ``Batch1D`` targets).
        hidden_dim: Width of the MLP hidden layers.
        depth: Number of hidden layers of each MLP.
        key: PRNG key for the MLP weights.

    Attributes:
        mlp_mu: MLP from context to flat mean (size ``prod(state_shape)``).
        mlp_log_var: MLP from context to flat log variance.

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
        flat_dim = 1
        for dim_size in self.state_shape:
            flat_dim *= dim_size
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
        return einx.id("(s...) -> s...", self.mlp_mu(ctx), s=self.state_shape)

    def sample(
        self,
        ctx: Float[Array, " D"],
        key: PRNGKeyArray,
        n: int,
    ) -> Float[Array, ...]:
        mean = einx.id("(s...) -> s...", self.mlp_mu(ctx), s=self.state_shape)
        log_var = einx.id("(s...) -> s...", self.mlp_log_var(ctx), s=self.state_shape)
        # σ = exp(½ log σ²)
        std = jnp.exp(0.5 * log_var)
        noise = jax.random.normal(key, (n, *self.state_shape))
        # x = μ + σ ⊙ ε
        return mean + std * noise

    def log_prob(
        self,
        x: Float[Array, ...],
        ctx: Float[Array, " D"],
    ) -> Float[Array, ""]:
        mean = einx.id("(s...) -> s...", self.mlp_mu(ctx), s=self.state_shape)
        log_var = einx.id("(s...) -> s...", self.mlp_log_var(ctx), s=self.state_shape)
        # log N(x | mu, diag(exp(log_var))) — sum over state dims.
        return -0.5 * jnp.sum(
            ((x - mean) ** 2) * jnp.exp(-log_var) + log_var + _LOG_2PI
        )


class ConditionalFlowHead(eqx.Module):
    r"""Conditional normalising flow head (stub).

    Implements $x = f_\phi(z; c_\psi(y))$ with $z \sim \mathcal{N}(0, I)$. Exact
    density via change-of-variables. Requires ``gauss_flows`` for the
    flow primitives; ships as a stub until that dependency is added to
    ``pyproject.toml``.

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
    r"""Score-based diffusion head (stub).

    Learns $s_\phi(x, t \mid y) \approx \nabla_x \log p_t(x \mid y)$; samples via
    reverse SDE solved with diffrax. Ships as a stub pending the reverse-SDE
    pipeline.

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
