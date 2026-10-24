r"""Posterior adapters (Decision D10).

Every vardax analysis emits a [`Posterior`][vardax.Posterior]
container — not just a point estimate. Three adapter families compute
the posterior covariance via different approximations:

- [`LaplaceCovariance`][vardax.LaplaceCovariance] — inverse
  Gauss–Newton Hessian at MAP: $(B^{-1} + H'^\top R^{-1} H')^{-1}$ for a
  single time, $(B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} H'_t M'_t)^{-1}$
  through the tangent-linear model for 4DVar. Gaussian-likelihood-only.
- [`GaussNewtonHessian`][vardax.GaussNewtonHessian] — the same operator
  with an explicit Krylov budget.
- [`EnsembleCovariance`][vardax.EnsembleCovariance] — sample covariance
  from an ensemble of analyses (delegates to ``filterax`` when
  supplied).

Plus [`GaussianMarkLikelihood`][vardax.GaussianMarkLikelihood] —
serialises a [`Posterior`][vardax.Posterior] to a JSON-friendly dict
for downstream population models (Tier V).
"""

from __future__ import annotations

from .adapter import GaussianMarkLikelihood
from .container import Posterior
from .ensemble import EnsembleCovariance
from .gauss_newton import GaussNewtonHessian
from .laplace import LaplaceCovariance

__all__ = [
    "EnsembleCovariance",
    "GaussNewtonHessian",
    "GaussianMarkLikelihood",
    "LaplaceCovariance",
    "Posterior",
]
