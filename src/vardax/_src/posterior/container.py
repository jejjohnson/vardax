"""``Posterior`` container — the standard output of every analysis.

Carries the posterior mean, covariance (as an ``AbstractLinearOperator``
when a Gaussian / Laplace approximation is used), optional ensemble
samples, and provenance metadata for downstream auditing.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx


class Posterior(eqx.Module):
    """Posterior container — Decision D10.

    Attributes:
        mean: Posterior mean / MAP estimate.
        cov: Posterior covariance as ``lineax.AbstractLinearOperator``,
            or ``None`` for amortized / sampling-only heads. Stored
            lazily — never materialised into a dense matrix unless the
            user asks.
        samples: Optional ``(B, M, ...)`` posterior samples (from a
            flow head, ensemble, or score-based diffusion). ``None``
            for analytical adapters.
        provenance: Free-form dict carrying audit info — at minimum
            ``{forward_model_id, n_iter, J_star, converged,
            vardax_version}``. Consumed by ``GaussianMarkLikelihood``
            when serialising for downstream population models.
    """

    mean: Float[Array, ...]
    cov: lx.AbstractLinearOperator | None = None
    samples: Float[Array, ...] | None = None
    provenance: dict[str, Any] = eqx.field(default_factory=dict)
