"""Posterior adapters: each turns an analysis into a [`Posterior`][vardax.Posterior].

[`LaplaceCovariance`][vardax.LaplaceCovariance] and
[`GaussNewtonHessian`][vardax.GaussNewtonHessian] invert the Gauss–Newton precision
Λ_x|y at the analysis, [`EnsembleCovariance`][vardax.EnsembleCovariance] takes the
sample covariance of an ensemble, and
[`GaussianMarkLikelihood`][vardax.GaussianMarkLikelihood] serialises a posterior.
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
