"""3D Variational analysis.

Minimises

.. math::

    J(x) = \\tfrac{1}{2} \\|x - x_b\\|^2_{B^{-1}}
         + \\tfrac{1}{2} \\|y - H(x)\\|^2_{R^{-1}}

over a single timestep. When :math:`H` is linear and :math:`B`, :math:`R`
Gaussian, ``ThreeDVar`` reduces to ``OptimalInterpolation`` — the
canonical correctness gate enforced by the linear-Gaussian agreement
test.

The inner minimisation is delegated to an
``optimistix.AbstractMinimiser`` (``GaussNewton``, ``BFGS``,
``NonlinearCG``, ``LevenbergMarquardt``, …). The choice of minimiser
is the user's; vardax provides a sensible default.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx
import optimistix as optx

from vardax._src._types import Batch1D


class ThreeDVar(eqx.Module):
    """3D variational analysis.

    Attributes:
        obs_op: Observation operator (linear or nonlinear).
        prior_mean: Background :math:`x_b` of shape ``(T, N)``.
        prior_cov_op: :math:`B`. ``lineax.AbstractLinearOperator``.
        obs_cov_op: :math:`R`. ``lineax.AbstractLinearOperator``.
        minimiser: ``optimistix.AbstractMinimiser`` for the inner
            iteration. Default ``optimistix.BFGS(rtol=1e-6, atol=1e-6)``.
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for
            differentiating *through* the minimum (used only if the
            ``ThreeDVar`` analysis is itself nested inside a larger
            training loop). Default ``ImplicitAdjoint`` — exact at the
            optimum.
        max_steps: Iteration cap on the inner solver.
    """

    obs_op: Any
    prior_mean: Float[Array, "T N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    minimiser: optx.AbstractMinimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)

    def __init__(
        self,
        obs_op: Any,
        prior_mean: Float[Array, "T N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        minimiser: optx.AbstractMinimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
    ) -> None:
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps

    def __call__(self, batch: Batch1D) -> Float[Array, "B T N"]:
        """3DVar analysis for each sample in the batch."""

        def _one(
            input_i: Float[Array, "T N"], mask_i: Float[Array, "T N"]
        ) -> Float[Array, "T N"]:
            def cost(x: Float[Array, "T N"], _args: Any) -> Float[Array, ""]:
                # Background term — apply B^{-1} via lineax solve.
                dx = x - self.prior_mean
                B_inv_dx = lx.linear_solve(
                    self.prior_cov_op,
                    dx,
                    solver=lx.CG(atol=1e-6, rtol=1e-6),
                ).value
                j_bg = 0.5 * jnp.sum(dx * B_inv_dx)
                # Observation term — apply R^{-1} via lineax solve.
                # Mask predictions AND observations symmetrically: for
                # obs_ops that don't take a `mask` kwarg (e.g.
                # LinearObs), unmasked predictions at missing entries
                # would otherwise penalise the analysis there.
                y_pred = (
                    self.obs_op(x, mask=mask_i)
                    if _accepts_mask(self.obs_op)
                    else self.obs_op(x)
                )
                residual = mask_i * (input_i - y_pred)
                R_inv_r = lx.linear_solve(
                    self.obs_cov_op,
                    residual,
                    solver=lx.CG(atol=1e-6, rtol=1e-6),
                ).value
                j_obs = 0.5 * jnp.sum(residual * R_inv_r)
                return j_bg + j_obs

            result = optx.minimise(
                fn=cost,
                solver=self.minimiser,
                y0=self.prior_mean,
                args=None,
                max_steps=self.max_steps,
                adjoint=self.minimiser_adjoint,
                throw=False,
            )
            return result.value

        return jax.vmap(_one)(batch.input, batch.mask)

    def as_analysis_step(self) -> _ThreeDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
        return _ThreeDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _ThreeDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``ThreeDVar``."""

    model: ThreeDVar

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
