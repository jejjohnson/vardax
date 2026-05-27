"""Optimal Interpolation / BLUE (Decision D16).

Closed-form linear-Gaussian analysis: the Best Linear Unbiased
Estimator. When :math:`H` is linear and :math:`B`, :math:`R` are
Gaussian, the posterior is exact in one matrix expression — no
iteration, no convergence criterion.

.. math::

    x^* = x_b + K (y - H x_b), \\quad K = B H^\\top (H B H^\\top + R)^{-1}.

Implementation notes:

- The inversion is solved by ``lineax.CG`` against an
  ``AbstractLinearOperator``; we never materialise :math:`HBH^\\top + R`.
- ``B_op`` and ``R_op`` are supplied as ``lineax.AbstractLinearOperator``
  (typically ``gaussx`` structured operators for Matérn priors and
  diagonal :math:`R`).
- The class refuses non-linear observation operators on construction.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D


class OptimalInterpolation(eqx.Module):
    """BLUE / OI — closed-form linear-Gaussian analysis.

    Attributes:
        obs_op: Observation operator. Must be linear (its
            ``linearize(x)`` must return an operator independent of
            ``x``). Use ``ThreeDVar`` for non-linear ``H``.
        prior_mean: Background :math:`x_b` of shape ``(T, N)``.
        prior_cov_op: Background-error covariance :math:`B` as a
            ``lineax.AbstractLinearOperator``.
        obs_cov_op: Observation-error covariance :math:`R` as a
            ``lineax.AbstractLinearOperator``.
        cg_atol: CG absolute tolerance for the inner solve.
        cg_rtol: CG relative tolerance for the inner solve.
        cg_max_steps: CG iteration cap.
    """

    obs_op: Any  # ObservationOperator (linear)
    prior_mean: Float[Array, "T N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    cg_atol: float = eqx.field(static=True, default=1e-6)
    cg_rtol: float = eqx.field(static=True, default=1e-6)
    cg_max_steps: int = eqx.field(static=True, default=200)

    def __call__(self, batch: Batch1D) -> Float[Array, "B T N"]:
        """BLUE analysis for each sample in the batch."""

        def _one(
            input_i: Float[Array, "T N"], mask_i: Float[Array, "T N"]
        ) -> Float[Array, "T N"]:
            # Observation vector: masked input projected through obs_op.
            y_pred = (
                self.obs_op(self.prior_mean, mask=mask_i)
                if _accepts_mask(self.obs_op)
                else self.obs_op(self.prior_mean)
            )
            innovation = (input_i * mask_i) - y_pred

            # Build (H B H^T + R) as a composed lineax operator. We
            # tag it positive-semidefinite so lineax.CG accepts it
            # (the composition itself doesn't carry the tag).
            H = self.obs_op.linearize(self.prior_mean)
            inner_raw = H @ self.prior_cov_op @ H.transpose() + self.obs_cov_op
            inner_op = lx.TaggedLinearOperator(inner_raw, lx.positive_semidefinite_tag)

            solver = lx.CG(
                atol=self.cg_atol,
                rtol=self.cg_rtol,
                max_steps=self.cg_max_steps,
            )
            v = lx.linear_solve(inner_op, innovation, solver=solver).value

            # x_star = x_b + B H^T v
            return self.prior_mean + self.prior_cov_op.mv(H.transpose().mv(v))

        return jax.vmap(_one)(batch.input, batch.mask)

    def as_analysis_step(self) -> _OIAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
        return _OIAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    """Whether ``obs_op.__call__`` takes a ``mask`` kwarg.

    ``MaskedIdentity`` does; ``LinearObs`` doesn't. The dispatch keeps
    the same OI implementation usable across operator types.
    """
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _OIAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``OptimalInterpolation``."""

    model: OptimalInterpolation

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        mask = jnp.where(jnp.isfinite(obs), 1.0, 0.0)
        obs_clean = jnp.nan_to_num(obs)
        batch = Batch1D(input=obs_clean, mask=mask, target=None)
        return self.model(batch)
