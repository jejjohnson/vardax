"""Laplace approximation at MAP.

For Gaussian-likelihood / Gaussian-prior problems at the MAP
:math:`x^*`:

.. math::

    P^* \\approx \\big((H')^\\top R^{-1} H' + B^{-1}\\big)^{-1}.

The result is returned as an ``AbstractLinearOperator`` so mat-vec
(samples, marginals) compose lazily via ``lineax.CG`` without
materialising :math:`P^*`.

Use when:

- Gaussian likelihood + Gaussian prior + single posterior mode
  (verified by SBC).
- Default for ``ThreeDVar``, ``StrongFourDVar``, ``WeakFourDVar``,
  ``FourDVarNet``.

For multimodal posteriors / non-Gaussian likelihoods, use
``EnsembleCovariance``. For ``IncrementalFourDVar`` the
``GaussNewtonHessian`` adapter reuses the Hessian assembled during
the last outer iteration.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx

from .container import Posterior


class LaplaceCovariance(eqx.Module):
    """Laplace approximation at MAP.

    Attributes:
        prior_cov_op: Background-error covariance :math:`B`.
        obs_cov_op: Observation-error covariance :math:`R`.

    Both required so the adapter can build :math:`P^*` lazily. They
    should match the operators used by the analysis method that
    produced ``analysis``.
    """

    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator

    def __call__(
        self,
        analysis: Float[Array, ...],
        model: Any,  # AnalysisStep-compliant
        batch: Any,
    ) -> Posterior:
        """Build the Laplace posterior at ``analysis``.

        Args:
            analysis: MAP / posterior mean.
            model: The analysis-step instance (used to recover the
                observation operator). Expected to have a ``.model``
                or be the underlying ``eqx.Module`` directly.
            batch: The batch the analysis was computed against (used
                to recover the mask, instrument bookkeeping, etc.).

        Returns:
            ``Posterior`` with ``mean = analysis`` and ``cov`` as a
            lazy ``AbstractLinearOperator`` representing :math:`P^*`.
        """
        # Pull the obs operator off either an .as_analysis_step()
        # wrapper or the raw model. Both expose ``obs_op``.
        underlying = getattr(model, "model", model)
        obs_op = getattr(underlying, "obs_op", None)
        if obs_op is None:
            raise AttributeError(
                "LaplaceCovariance requires the model to expose `obs_op`; "
                "got a model without one."
            )

        H = obs_op.linearize(analysis)
        # P*^{-1} = H^T R^{-1} H + B^{-1}
        # Build lazily — represent as the inverse of a composed
        # AbstractLinearOperator. Concrete mat-vec evaluation is via
        # lineax.CG on the precision operator.
        precision = H.transpose() @ _inverse_op(self.obs_cov_op) @ H + _inverse_op(
            self.prior_cov_op
        )
        precision_tagged = lx.TaggedLinearOperator(
            precision, lx.positive_semidefinite_tag
        )

        return Posterior(
            mean=analysis,
            cov=_InverseLinearOperator(precision_tagged),
            samples=None,
            provenance={"adapter": "LaplaceCovariance"},
        )


def _inverse_op(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Lazy inverse: a wrapper that applies ``op^{-1}`` via lineax.CG."""
    return _InverseLinearOperator(
        lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)
    )


class _InverseLinearOperator(lx.FunctionLinearOperator):
    """Inverse of a positive-semidefinite operator, applied via CG.

    Wraps any ``AbstractLinearOperator`` ``A`` such that mat-vec
    returns ``A^{-1} v`` (solved by ``lineax.CG``). The inverse is
    never materialised.
    """

    def __init__(self, op: lx.AbstractLinearOperator) -> None:
        def _solve(v: Float[Array, ...]) -> Float[Array, ...]:
            return lx.linear_solve(
                op, v, solver=lx.CG(atol=1e-6, rtol=1e-6, max_steps=200)
            ).value

        # FunctionLinearOperator needs an input_structure; reuse the
        # underlying operator's.
        super().__init__(_solve, op.in_structure())
