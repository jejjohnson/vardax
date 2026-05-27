"""Weak-constraint 4DVar.

Multi-time analysis with control variable
:math:`(x_0, \\eta_1, \\ldots, \\eta_T)` — initial state plus per-step
model-error increments. Cost:

.. math::

    J(x_0, \\boldsymbol{\\eta}) =
        \\tfrac{1}{2} \\|x_0 - x_b\\|^2_{B^{-1}}
      + \\tfrac{1}{2} \\sum_t \\|y_t - H_t(x_t)\\|^2_{R_t^{-1}}
      + \\tfrac{1}{2} \\sum_t \\|\\eta_t\\|^2_{Q_t^{-1}}

where :math:`x_t = M_t(x_{t-1}) + \\eta_t`.

The control vector has dimension :math:`(T + 1) \\cdot N` (initial
state + T model-error increments). For long windows this is
computationally heavier than ``StrongFourDVar`` — use only when the
free model :math:`M_t^\\text{free}` is known to have biases.
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


class WeakFourDVar(eqx.Module):
    """Weak-constraint 4DVar with augmented control vector.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel`` (the free model :math:`M_t^\\text{free}`).
        obs_op: Observation operator.
        prior_mean: Background :math:`x_b` — initial state ``(N,)``.
        prior_cov_op: :math:`B`.
        obs_cov_op: :math:`R`.
        model_err_cov_op: :math:`Q` — covariance of the per-step model
            error :math:`\\eta_t`. Defaults to identity scaled by a
            small variance if not supplied (effectively
            strong-constraint with a tiny relaxation).
        minimiser, minimiser_adjoint, max_steps: as in
            ``StrongFourDVar``.
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, N]  # ty:ignore[unresolved-reference]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    model_err_cov_op: lx.AbstractLinearOperator
    minimiser: optx.AbstractMinimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, N],  # ty:ignore[unresolved-reference]
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        model_err_cov_op: lx.AbstractLinearOperator,
        minimiser: optx.AbstractMinimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
    ) -> None:
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.model_err_cov_op = model_err_cov_op
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps

    def _rollout_with_error(
        self,
        x_0: Float[Array, N],  # ty:ignore[unresolved-reference]
        etas: Float[Array, "T N"],
    ) -> Float[Array, "T_plus_1 N"]:
        """Roll out with per-step model error: :math:`x_t = M_t^\\text{free}(x_{t-1}) + \\eta_t`."""
        dt = self.forward.dt

        def step_fn(
            x: Float[Array, N],  # ty:ignore[unresolved-reference]
            eta_t: Float[Array, N],  # ty:ignore[unresolved-reference]
        ) -> tuple[Float[Array, N], Float[Array, N]]:  # ty:ignore[unresolved-reference]
            x_new = self.forward.step(x, dt) + eta_t
            return x_new, x_new

        _, trajectory = jax.lax.scan(step_fn, x_0, etas)
        return jnp.concatenate([x_0[None, :], trajectory], axis=0)

    def __call__(
        self,
        batch: Batch1D,
    ) -> tuple[Float[Array, "B N"], Float[Array, "B T N"]]:
        """Weak-4DVar analysis.

        Returns ``(x_0_star, eta_star)`` — the analysis initial state
        and the per-step model-error trajectory (one row per
        timestep, total ``T``).
        """
        # T_plus_1 timesteps in the batch ⇒ T model-error steps.
        T_plus_1 = batch.input.shape[1]
        T = T_plus_1 - 1
        N = batch.input.shape[2]

        def _one(
            input_i: Float[Array, "T_plus_1 N"], mask_i: Float[Array, "T_plus_1 N"]
        ):
            def cost(
                control: Float[Array, "T_plus_1 N"], _args: Any
            ) -> Float[Array, ""]:
                x_0 = control[0]
                etas = control[1:]

                # Background term
                dx = x_0 - self.prior_mean
                B_inv_dx = lx.linear_solve(
                    self.prior_cov_op,
                    dx,
                    solver=lx.CG(atol=1e-6, rtol=1e-6),
                ).value
                j_bg = 0.5 * jnp.sum(dx * B_inv_dx)

                # Model-error term
                def _eta_cost(eta_t):
                    Q_inv_eta = lx.linear_solve(
                        self.model_err_cov_op,
                        eta_t,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    return 0.5 * jnp.sum(eta_t * Q_inv_eta)

                j_eta = jnp.sum(jax.vmap(_eta_cost)(etas))

                # Observation term
                trajectory = self._rollout_with_error(x_0, etas)

                def _per_step(x_t, y_t, m_t):
                    y_pred = (
                        self.obs_op(x_t, mask=m_t)
                        if _accepts_mask(self.obs_op)
                        else self.obs_op(x_t)
                    )
                    residual = m_t * (y_t - y_pred)
                    R_inv_r = lx.linear_solve(
                        self.obs_cov_op,
                        residual,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    return 0.5 * jnp.sum(residual * R_inv_r)

                per_step_costs = jax.vmap(_per_step)(trajectory, input_i, mask_i)
                j_obs = jnp.sum(per_step_costs)

                return j_bg + j_obs + j_eta

            # Initial guess: x_0 at the background, etas at zero.
            init_etas = jnp.zeros((T, N))
            y0 = jnp.concatenate([self.prior_mean[None, :], init_etas], axis=0)

            result = optx.minimise(
                fn=cost,
                solver=self.minimiser,
                y0=y0,
                args=None,
                max_steps=self.max_steps,
                adjoint=self.minimiser_adjoint,
                throw=False,
            )
            control_star = result.value
            return control_star[0], control_star[1:]

        x0_star, eta_star = jax.vmap(_one)(batch.input, batch.mask)
        return x0_star, eta_star

    def as_analysis_step(self) -> _WeakFourDVarAnalysisStep:
        return _WeakFourDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _WeakFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``WeakFourDVar``.

    Returns the analysis initial state only (the model-error
    trajectory is also available via direct ``model(batch)`` calls).
    """

    model: WeakFourDVar

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
        x0_star, _ = self.model(batch)
        return x0_star
