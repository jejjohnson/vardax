"""``Posterior`` container — the standard output of every analysis.

Carries the posterior mean, covariance (as an ``AbstractLinearOperator``
when a Gaussian / Laplace approximation is used), optional ensemble
samples, and provenance metadata for downstream auditing.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import jax
from jaxtyping import Array, Float, PRNGKeyArray
import lineax as lx

from vardax._src.errors import ShapeError, UnsupportedInputError


class Posterior(eqx.Module):
    r"""Posterior container — Decision D10.

    A plain container for $p(x \mid y) \approx \mathcal{N}(\text{mean}, \text{cov})$,
    plus optional samples and provenance.

    Attributes:
        mean: Posterior mean / MAP estimate.
        cov: Posterior covariance as ``lineax.AbstractLinearOperator``,
            or ``None`` for amortized / sampling-only heads. Stored
            lazily — never materialised into a dense matrix unless the
            user asks.
        samples: Optional posterior samples (from a flow head, ensemble,
            or score-based diffusion). ``None`` for analytical adapters.
        basis: $\Phi$ of a linear control $x = x^* + \Phi \theta$, as an
            operator from coefficients to the state, or ``None``. Set by
            the Gauss–Newton adapters together with ``coefficient_cov``,
            so that ``cov`` $= \Phi P_\theta \Phi^\top$.
        coefficient_cov: $P_\theta$, the coefficient-space covariance of
            a linear control, or ``None``.
        provenance: Free-form dict carrying audit info — at minimum
            ``{forward_model_id, n_iter, J_star, converged,
            vardax_version}``. Consumed by
            [`GaussianMarkLikelihood`][vardax.GaussianMarkLikelihood]
            when serialising for downstream population models.

    Examples:
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> post = vdx.Posterior(mean=jnp.zeros(4))
        >>> post.mean.shape
        (4,)
        >>> post.cov is None
        True
        >>> post.provenance
        {}

        A Gaussian posterior carries a lazy covariance operator and audit
        metadata:

        >>> import jax, lineax as lx
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> gauss = vdx.Posterior(
        ...     mean=jnp.zeros(2), cov=eye, provenance={"adapter": "demo"}
        ... )
        >>> gauss.cov.mv(jnp.ones(2)).shape, gauss.provenance["adapter"]
        ((2,), 'demo')

        Draws from it, reproducible for a fixed key:

        >>> draws = gauss.sample(jax.random.PRNGKey(0), 5)
        >>> draws.shape
        (5, 2)
        >>> bool(jnp.all(draws == gauss.sample(jax.random.PRNGKey(0), 5)))
        True
    """

    mean: Float[Array, ...]
    cov: lx.AbstractLinearOperator | None = None
    samples: Float[Array, ...] | None = None
    provenance: dict[str, Any] = eqx.field(default_factory=dict)
    basis: lx.AbstractLinearOperator | None = None
    coefficient_cov: lx.AbstractLinearOperator | None = None

    def sample(
        self, key: PRNGKeyArray, n_samples: int
    ) -> Float[Array, "n_samples ..."]:
        r"""Draw ``n_samples`` states from the posterior.

        Algorithm:
            1. With ``basis`` and ``coefficient_cov`` (a linear control),
               draw in coefficient space:
               $x^{(e)} = x^* + \Phi P_\theta^{1/2} \xi^{(e)}$.
            2. Else with ``cov``:
               $x^{(e)} = x^* + P^{1/2} \xi^{(e)}$, $\xi^{(e)} \sim
               \mathcal{N}(0, I)$, with the square root from
               ``gaussx.sqrt`` (closed form for structured operators, a
               dense eigendecomposition otherwise).
            3. Else with ``samples``: a random subset of them, without
               replacement.

        Any square root gives the same distribution; a different root
        only changes which draws a key produces.

        Args:
            key: PRNG key.
            n_samples: Number of draws $N_E$; static under ``jit``.

        Returns:
            The draws, shape ``(n_samples, *mean.shape)``.

        Raises:
            UnsupportedInputError: If the posterior has neither a
                covariance nor samples.
            ShapeError: If ``n_samples`` exceeds the stored samples.
        """
        basis = self.basis
        if basis is not None and self.coefficient_cov is not None:
            # P_θ^{1/2}, then Φ maps it to the state
            root = gaussx.sqrt(self.coefficient_cov)
            noise = jax.random.normal(
                key, (n_samples, *root.in_structure().shape), self.mean.dtype
            )
            draws = jax.vmap(lambda xi: basis.mv(root.mv(xi)))(noise)
            # x⁽ᵉ⁾ = x* + Φ P_θ^{1/2} ξ⁽ᵉ⁾
            return self.mean + draws.reshape(n_samples, *self.mean.shape)
        if self.cov is not None:
            # L Lᵀ = P; gaussx may act on the flattened state
            root = gaussx.sqrt(self.cov)
            noise = jax.random.normal(
                key, (n_samples, *root.in_structure().shape), self.mean.dtype
            )
            draws = jax.vmap(root.mv)(noise)
            # x⁽ᵉ⁾ = x* + P^{1/2} ξ⁽ᵉ⁾
            return self.mean + draws.reshape(n_samples, *self.mean.shape)
        if self.samples is not None:
            n_stored = self.samples.shape[0]
            if n_samples > n_stored:
                raise ShapeError(
                    f"Asked for {n_samples} draws but the posterior holds "
                    f"{n_stored} samples."
                )
            chosen = jax.random.choice(key, n_stored, (n_samples,), replace=False)
            return self.samples[chosen]
        raise UnsupportedInputError(
            "This posterior has neither a covariance nor samples to draw from."
        )
