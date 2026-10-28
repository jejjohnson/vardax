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

    $$\bar x = \frac{1}{M} \sum_m x^{(m)}, \quad
    P^* = \lambda \, \frac{1}{M-1} \sum_m (x^{(m)} - \bar x)(x^{(m)} - \bar x)^\top$$

    Algorithm:
        1. Average the analyses over the ensemble axis to get $\bar x$.
        2. Flatten each member and form the Bessel-corrected sample
           covariance as a low-rank operator (never dense).
        3. Scale it by the inflation factor $\lambda$.
        4. Package mean, covariance, the members as samples and the
           provenance into a [`Posterior`][vardax.Posterior].

    Attributes:
        n_members: Expected ensemble size (used for diagnostics).
        inflation: Multiplicative inflation factor $\lambda$ applied
            to the sample covariance. Default 1.0 (no inflation).

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
            analyses: Stack of the ``M`` member analyses, member axis
                first.
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
        n_analyses = analyses.shape[0]
        if n_analyses != self.n_members:
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
                "n_members": int(n_analyses),
                "inflation": float(self.inflation),
            },
        )
