"""Ensemble-covariance posterior adapter.

Builds the posterior covariance from an ensemble of analyses
(per perturbed initial condition or observation realisation):

.. math::

    P^* \\approx \\frac{1}{M-1} \\sum_m (x^{(m)} - \\bar x)(x^{(m)} - \\bar x)^\\top.

When ``filterax`` is installed and used for the propagation, this
adapter just packages the sample covariance into a ``Posterior``.
Otherwise the user supplies the ensemble explicitly via the
``analyses`` argument (shape ``(M, ...)``).

Localisation and inflation are standard ensemble fixes — they live
in ``filterax`` and aren't reimplemented here.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .container import Posterior


class EnsembleCovariance(eqx.Module):
    """Sample-covariance posterior from an ensemble of analyses.

    Attributes:
        n_members: Expected ensemble size (used for diagnostics).
        inflation: Multiplicative inflation factor :math:`\\lambda` applied
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
            ``Posterior`` with mean = ensemble mean, ``cov`` = sample
            covariance as a dense ``MatrixLinearOperator`` (only
            assembled if you ask for it; safe for moderate M).
        """
        m = analyses.shape[0]
        if m != self.n_members:
            # Allow runtime mismatch; the configured ``n_members`` is
            # advisory rather than enforced.
            pass

        mean = jnp.mean(analyses, axis=0)
        # Flatten everything but the leading ensemble dim; the
        # Bessel-corrected sample covariance comes from gaussx as a
        # rank-(m-1) LowRankUpdate operator (never dense), scaled by
        # the multiplicative inflation factor.
        cov_op = self.inflation * gaussx.ensemble_covariance(
            analyses.reshape(m, -1), bessel=True
        )

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
