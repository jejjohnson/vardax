"""Posterior adapters (Decision D10).

Every vardax analysis emits a ``Posterior`` container — not just a
point estimate. Three adapter families compute the posterior covariance
via different approximations:

- :class:`LaplaceCovariance`   — :math:`P^* = (B^{-1} + H^\\top R^{-1} H)^{-1}`
  at MAP. Cheap, Gaussian-likelihood-only.
- :class:`GaussNewtonHessian`  — Krylov / Lanczos inversion of the
  Gauss-Newton Hessian. Exact at MAP, structured.
- :class:`EnsembleCovariance`  — sample covariance from an ensemble of
  analyses (delegates to ``filterax`` when supplied).

Plus :class:`GaussianMarkLikelihood` — serialises a ``Posterior``
to a JSON-friendly dict for downstream population models (Tier V).
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
