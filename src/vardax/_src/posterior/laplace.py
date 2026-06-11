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
import gaussx
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
        # Build lazily via operator composition (gaussx.sandwich would
        # materialise the Jacobian, breaking the matrix-free design);
        # the inverses are gaussx.inv, which dispatches to closed forms
        # for structured operators and falls back to lineax.CG mat-vecs.
        precision = H.transpose() @ _inverse_op(self.obs_cov_op) @ H + _inverse_op(
            self.prior_cov_op
        )
        precision_tagged = lx.TaggedLinearOperator(
            precision, lx.positive_semidefinite_tag
        )

        return Posterior(
            mean=analysis,
            cov=gaussx.inv(precision_tagged, solver=_CG_SOLVER),
            samples=None,
            provenance={"adapter": "LaplaceCovariance"},
        )


_CG_SOLVER = lx.CG(atol=1e-6, rtol=1e-6, max_steps=200)


def _inverse_op(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Lazy PSD inverse — delegates to :func:`gaussx.inv`.

    Structured operators (diagonal, Kronecker, block-diagonal,
    low-rank) invert in closed form; everything else solves via
    ``lineax.CG`` mat-vecs, never materialising the inverse. The
    gaussx operator is re-wrapped as a ``FunctionLinearOperator`` so
    lineax can linearise it when it appears inside a composed
    precision that is itself solved (gaussx's ``InverseOperator``
    does not register ``lx.linearise``).
    """
    inv_op = gaussx.inv(
        lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag),
        solver=_CG_SOLVER,
    )
    return lx.FunctionLinearOperator(inv_op.mv, op.in_structure())
