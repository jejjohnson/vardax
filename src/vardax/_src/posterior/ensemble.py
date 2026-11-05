"""Ensemble posterior: the sample covariance of an ensemble of analyses x⁽ᵉ⁾.

Localisation and other ensemble fixes live in ``filterax``, not here.
"""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import gaussx
from jaxtyping import Array, Float

from .container import Posterior


class EnsembleCovariance(eqx.Module):
    """Sample-covariance posterior from an ensemble ℰ of analyses.

        x̄ = (1 / N_ℰ) ∑_e x⁽ᵉ⁾,   x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
        Σ_x|y ≈ ζ (1 / (N_ℰ − 1)) ∑_e x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ

    Algorithm:
        1. Average the analyses over the member axis to get x̄.
        2. Flatten each member and form the Bessel-corrected sample covariance as
           a low-rank operator, never a dense matrix.
        3. Scale it by the inflation factor ζ.
        4. Package the mean, the covariance, the members as samples and the
           provenance into a [`Posterior`][vardax.Posterior].

    Attributes:
        n_members: Expected ensemble size N_ℰ, recorded in the configuration; the
            stack's own size is what enters the covariance.
        inflation: Multiplicative inflation factor ζ applied to the sample
            covariance. Default 1.0, no inflation.

    Examples:
        Three scalar-pair analyses: the mean is the member average.

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> analyses = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 8.0]])
        >>> post = vdx.EnsembleCovariance(n_members=3)(analyses, None, None)
        >>> bool(jnp.allclose(post.mean, jnp.array([2.0, 4.0])))
        True
        >>> post.samples.shape
        (3, 2)

        Inflation scales the covariance (here by 4):

        >>> big = vdx.EnsembleCovariance(n_members=3, inflation=4.0)(
        ...     analyses, None, None
        ... )
        >>> v = jnp.array([1.0, 1.0])
        >>> bool(jnp.allclose(big.cov.mv(v), 4.0 * post.cov.mv(v)))
        True

        Members of any shape are flattened, so a ``(M, 2, 2)`` stack gives a
        4 x 4 covariance:

        >>> grid = jnp.arange(12.0).reshape(3, 2, 2) ** 2
        >>> vdx.EnsembleCovariance(n_members=3)(grid, None, None).cov.mv(
        ...     jnp.ones(4)
        ... ).shape
        (4,)
    """

    n_members: int = eqx.field(static=True)
    inflation: float = eqx.field(static=True, default=1.0)

    def __call__(
        self,
        analyses: Float[Array, "M ..."],
        model: Any,
        batch: Any,
    ) -> Posterior:
        """Build the ensemble posterior from per-member analyses.

        Args:
            analyses: Stack of the members x⁽ᵉ⁾, member axis first.
            model: The analysis model; unused, kept for the adapter interface.
            batch: The batch; unused, kept for the adapter interface.

        Returns:
            The [`Posterior`][vardax.Posterior] with mean x̄, the members as
            samples, and ``cov`` the low-rank operator Σ_x|y.
        """
        # N_ℰ, from the stack: the configured n_members is advisory
        num_members = analyses.shape[0]

        # x̄ = (1 / N_ℰ) ∑_e x⁽ᵉ⁾
        ensemble_mean = einx.mean("[member] ...", analyses)
        flat_analyses = einx.id("member state... -> member (state...)", analyses)
        # (1 / (N_ℰ − 1)) ∑_e x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ, of rank N_ℰ − 1
        sample_covariance = gaussx.ensemble_covariance(flat_analyses, bessel=True)
        # Σ_x|y ≈ ζ (1 / (N_ℰ − 1)) ∑_e x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ
        covariance = self.inflation * sample_covariance

        return Posterior(
            mean=ensemble_mean,
            cov=covariance,
            samples=analyses,
            provenance={
                "adapter": "EnsembleCovariance",
                "n_members": int(num_members),
                "inflation": float(self.inflation),
            },
        )

    def get_config(self) -> dict[str, Any]:
        """Return the fields that rebuild this adapter."""
        return {
            "n_members": self.n_members,
            "inflation": self.inflation,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EnsembleCovariance:
        """Rebuild an adapter from ``get_config``'s output."""
        return cls(**config)
