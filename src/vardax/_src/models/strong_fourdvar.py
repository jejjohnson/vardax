"""Strong-constraint 4DVar.

Multi-time analysis with the control variable being the initial state
:math:`x_0`. The forward model :math:`M_t` is treated as exact (no
model-error term). Cost:

.. math::

    J(x_0) = \\tfrac{1}{2} \\|x_0 - x_b\\|^2_{B^{-1}}
           + \\tfrac{1}{2} \\sum_{t=0}^{T} \\|y_t - H_t(M_t(x_0))\\|^2_{R_t^{-1}}.

Gradients of :math:`J` w.r.t. :math:`x_0` flow through the rollout
``trajectory = [forward.step(x, dt) for _ in range(T)]`` and back —
the adjoint composition is whatever JAX gives by default. For long
windows where the forward is a diffrax ODE solve, the
``forward_adjoint`` slot threads through to ``diffrax.diffeqsolve(...,
adjoint=...)`` (per-step rollouts use plain reverse-mode autodiff).
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


class StrongFourDVar(eqx.Module):
    """Strong-constraint 4DVar.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel`` supplying
            ``step(state, dt) -> state``.
        obs_op: Observation operator.
        prior_mean: Background :math:`x_b` — initial state of shape
            ``(N,)`` (state vector at :math:`t=0`).
        prior_cov_op: :math:`B`.
        obs_cov_op: :math:`R`.
        minimiser: ``optimistix.AbstractMinimiser`` for the outer
            optimisation over :math:`x_0`.
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for
            differentiating through the minimum.
        forward_adjoint: ``diffrax.AbstractAdjoint``-like — currently
            stored as a tag, threaded through if the forward delegates
            to ``diffrax``. Default ``None`` (use forward's own).
        max_steps: Iteration cap on the outer solver.
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, N]  # ty:ignore[unresolved-reference]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    minimiser: optx.AbstractMinimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    forward_adjoint: Any = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, N],  # ty:ignore[unresolved-reference]
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        minimiser: optx.AbstractMinimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        forward_adjoint: Any = None,
        max_steps: int = 200,
    ) -> None:
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.forward_adjoint = forward_adjoint
        self.max_steps = max_steps

    def _rollout(self, x_0: Float[Array, N], n_steps: int) -> Float[Array, "T N"]:  # ty:ignore[unresolved-reference]
        """Step the forward model ``n_steps`` times, returning the trajectory."""
        dt = self.forward.dt

        def step_fn(
            x: Float[Array, N], _: None  # ty:ignore[unresolved-reference]
        ) -> tuple[Float[Array, N], Float[Array, N]]:  # ty:ignore[unresolved-reference]
            x_new = self.forward.step(x, dt)
            return x_new, x_new

        _, trajectory = jax.lax.scan(step_fn, x_0, None, length=n_steps)
        # Prepend x_0 so the trajectory has length T+1.
        return jnp.concatenate([x_0[None, :], trajectory], axis=0)

    def __call__(self, batch: Batch1D) -> Float[Array, "B N"]:
        """Strong-4DVar analysis: minimise over :math:`x_0`."""

        T = batch.input.shape[1] - 1  # rollout steps (input has T+1 timesteps)

        def _one(
            input_i: Float[Array, "T_plus_1 N"], mask_i: Float[Array, "T_plus_1 N"]
        ) -> Float[Array, N]:  # ty:ignore[unresolved-reference]
            def cost(x_0: Float[Array, N], _args: Any) -> Float[Array, ""]:  # ty:ignore[unresolved-reference]
                # Background term
                dx = x_0 - self.prior_mean
                B_inv_dx = lx.linear_solve(
                    self.prior_cov_op,
                    dx,
                    solver=lx.CG(atol=1e-6, rtol=1e-6),
                ).value
                j_bg = 0.5 * jnp.sum(dx * B_inv_dx)

                # Roll out the trajectory.
                trajectory = self._rollout(x_0, n_steps=T)

                # Observation term, summed across time.
                def _per_step(x_t, y_t, m_t):
                    y_pred = (
                        self.obs_op(x_t, mask=m_t)
                        if _accepts_mask(self.obs_op)
                        else self.obs_op(x_t)
                    )
                    residual = (y_t * m_t) - y_pred
                    R_inv_r = lx.linear_solve(
                        self.obs_cov_op,
                        residual,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    return 0.5 * jnp.sum(residual * R_inv_r)

                per_step_costs = jax.vmap(_per_step)(trajectory, input_i, mask_i)
                j_obs = jnp.sum(per_step_costs)
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

    def as_analysis_step(self) -> _StrongFourDVarAnalysisStep:
        return _StrongFourDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _StrongFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``StrongFourDVar``."""

    model: StrongFourDVar

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
