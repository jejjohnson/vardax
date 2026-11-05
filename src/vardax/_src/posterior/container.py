"""``Posterior``, the standard output of every analysis.

It carries the mean μ_x|y, the covariance Σ_x|y as a lazy operator when the
approximation is Gaussian, optional samples and the provenance for auditing.
"""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import gaussx
import jax
from jaxtyping import Array, Float, PRNGKeyArray
import lineax as lx

from vardax._src.errors import ShapeError, UnsupportedInputError
from vardax._src.utils.shapes import flatten


class Posterior(eqx.Module):
    """Posterior of an analysis, with optional samples and provenance.

        p(x ∣ y) ≈ 𝒩(x; μ_x|y, Σ_x|y)

    A linear control x = x⋆ + Φ z also keeps its factors, Σ_x|y = Φ Σ_z|y Φᵀ.

    Attributes:
        mean: Posterior mean μ_x|y, the analysis x⋆ for a variational model.
        cov: Posterior covariance Σ_x|y as a ``lineax.AbstractLinearOperator``, or
            ``None`` for amortized or sampling-only heads. It stays lazy and is
            never materialised into a dense matrix unless the user asks.
        samples: Optional posterior samples x⁽ᵉ⁾ from a flow head, an ensemble or
            score-based diffusion; ``None`` for the analytical adapters.
        provenance: Free-form dict of audit information, at minimum
            ``{forward_model_id, n_iter, J_star, converged, vardax_version}``,
            consumed by [`GaussianMarkLikelihood`][vardax.GaussianMarkLikelihood]
            when serialising for downstream population models.
        basis: Φ of a linear control x = x⋆ + Φ z, as an operator from coefficients
            to the state, or ``None``. The Gauss–Newton adapters set it together
            with ``coefficient_cov``.
        coefficient_cov: Σ_z|y, the coefficient-space posterior covariance of a
            linear control, or ``None``.
        precision: Λ_x|y = Σ_x|y⁻¹ when an adapter built ``cov`` by inverting it,
            or ``None``. It gives the marginal variances through
            ``gaussx.diag_inv`` without forming the inverse.

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
        >>> mean = jnp.zeros(2)
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), mean.dtype))
        >>> gauss = vdx.Posterior(mean=mean, cov=eye, provenance={"adapter": "demo"})
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
    precision: lx.AbstractLinearOperator | None = None

    def sample(
        self, key: PRNGKeyArray, n_samples: int
    ) -> Float[Array, "n_samples ..."]:
        """Draw ``n_samples`` states from the posterior.

            x⁽ᵉ⁾ = μ_x|y + L ξ⁽ᵉ⁾,   L Lᵀ = Σ_x|y,   ξ⁽ᵉ⁾ ~ 𝒩(0, I)

        Algorithm:
            1. With ``basis`` and ``coefficient_cov`` (a linear control), draw in
               coefficient space: x⁽ᵉ⁾ = μ_x|y + Φ Σ_z|y^½ ξ⁽ᵉ⁾.
            2. Else with ``cov``, use ``gaussx.sample_mvn``: an FFT circulant
               embedding for ``Toeplitz``, a block-banded Cholesky for
               ``BlockTriDiag``, closed forms for other structured operators and
               the symmetric square root otherwise, so a singular Σ_x|y still has
               finite gradients.
            3. Else with ``samples``, take a random subset without replacement.

        Any square root gives the same distribution; a different root only changes
        which draws a key produces.

        Args:
            key: PRNG key.
            n_samples: Number of draws N_ℰ; static under ``jit``.

        Returns:
            The draws, with the sample axis first and then ``mean``'s shape.

        Raises:
            UnsupportedInputError: If the posterior has neither a covariance nor
                samples.
            ShapeError: If ``n_samples`` exceeds the stored samples.
        """
        basis = self.basis
        if basis is not None and self.coefficient_cov is not None:
            # Σ_z|y^½
            coefficient_root = gaussx.sqrt(self.coefficient_cov)
            # ξ⁽ᵉ⁾ ~ 𝒩(0, I)
            noise = jax.random.normal(
                key,
                (n_samples, *coefficient_root.in_structure().shape),
                self.mean.dtype,
            )

            def draw_increment(noise_draw: Array) -> Array:
                # Σ_z|y^½ ξ⁽ᵉ⁾
                coefficient_draw = coefficient_root.mv(noise_draw)
                # Φ Σ_z|y^½ ξ⁽ᵉ⁾
                return basis.mv(coefficient_draw)

            increments = jax.vmap(draw_increment)(noise)
            # .reshape, not einx: einx cannot size an empty stack (n_samples = 0)
            shaped_increments = increments.reshape(n_samples, *self.mean.shape)
            # x⁽ᵉ⁾ = μ_x|y + Φ Σ_z|y^½ ξ⁽ᵉ⁾
            return self.mean + shaped_increments
        if self.cov is not None:
            flat_mean = flatten(self.mean)
            # x⁽ᵉ⁾ = μ_x|y + L ξ⁽ᵉ⁾, on the flattened state; sample_mvn picks L by
            # Σ_x|y's structure
            flat_draws = gaussx.sample_mvn(
                flat_mean, self.cov, key=key, num_samples=n_samples
            )
            return einx.id(
                "member (state...) -> member state...",
                flat_draws,
                state=self.mean.shape,
            )
        if self.samples is not None:
            num_stored = self.samples.shape[0]
            if n_samples > num_stored:
                raise ShapeError(
                    f"Asked for {n_samples} draws but the posterior holds "
                    f"{num_stored} samples."
                )
            chosen = jax.random.choice(key, num_stored, (n_samples,), replace=False)
            return self.samples[chosen]
        raise UnsupportedInputError(
            "This posterior has neither a covariance nor samples to draw from."
        )
