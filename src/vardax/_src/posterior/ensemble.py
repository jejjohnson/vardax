r"""Ensemble-covariance posterior adapter.

Builds the posterior covariance from an ensemble of analyses
(per perturbed initial condition or observation realisation):

$$
P^* \approx \frac{1}{M-1} \sum_m (x^{(m)} - \bar x)(x^{(m)} - \bar x)^\top.
$$

When ``filterax`` is installed and used for the propagation, this
adapter just packages the sample covariance into a ``Posterior``.
Otherwise the user supplies the ensemble explicitly via the
``analyses`` argument (shape ``(M, ...)``).

Localisation and inflation are standard ensemble fixes — they live
in ``filterax`` and aren't reimplemented here.
"""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import gaussx
from jaxtyping import Array, Float

from .container import Posterior


class EnsembleCovariance(eqx.Module):
    r"""Sample-covariance posterior from an ensemble of analyses.

    Attributes:
        n_members: Expected ensemble size (used for diagnostics).
        inflation: Multiplicative inflation factor $\lambda$ applied
            to the sample covariance. Default 1.0 (no inflation).
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
            analyses: Stack of ``M`` analyses with shape ``(M, ...)``.
            model: AnalysisStep instance (not used directly; kept for
                interface compatibility).
            batch: The batch (not used directly; kept for interface
                compatibility).

        Returns:
            [`Posterior`][vardax.Posterior] with mean = ensemble mean,
            ``cov`` = sample covariance as a dense
            ``MatrixLinearOperator`` (only assembled if you ask for
            it; safe for moderate M).
        """
        m = analyses.shape[0]
        if m != self.n_members:
            # Allow runtime mismatch; the configured ``n_members`` is
            # advisory rather than enforced.
            pass

        # x̄ = (1/m) Σᵢ xᵢ
        mean = einx.mean("[m] ...", analyses)
        # Flatten everything but the leading ensemble dim; the
        # Bessel-corrected sample covariance comes from gaussx as a
        # rank-(m-1) LowRankUpdate operator (never dense), scaled by
        # the multiplicative inflation factor.
        flat_analyses = einx.id("m s... -> m (s...)", analyses)
        # Σ = λ (1/(m−1)) Σᵢ (xᵢ − x̄)(xᵢ − x̄)ᵀ
        cov_op = self.inflation * gaussx.ensemble_covariance(flat_analyses, bessel=True)

        return Posterior(
            mean=mean,
            cov=cov_op,
            samples=analyses,
            provenance={
                "adapter": "EnsembleCovariance",
                "n_members": int(m),
                "inflation": float(self.inflation),
            },
        )
