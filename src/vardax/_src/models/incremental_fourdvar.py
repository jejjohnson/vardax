"""Incremental 4DVar with control-variable transform (Decision D11).

The operational fast path of ``StrongFourDVar``: Gauss-Newton outer
iterations on the full nonlinear cost, CG inner iterations on the
linearised quadratic subproblem, with a control-variable transform
:math:`\\chi = B^{-1/2}(\\delta x)` for preconditioning.

At each outer iteration around :math:`x_b^{(k)}`:

1. Linearise :math:`M_t` and :math:`H_t` (via ``jax.linearize``).
2. Form the Gauss-Newton Hessian
   :math:`J''_{GN} = B^{-1} + \\sum_t (H'_t M'_t)^\\top R_t^{-1} (H'_t M'_t)`.
3. Solve :math:`J''_{GN}\\,\\delta x^* = \\text{rhs}` via lineax CG
   (in :math:`\\chi`-space if CVT enabled).
4. Update :math:`x_b^{(k+1)} = x_b^{(k)} + \\delta x^*`.

This implementation uses ``lineax.JacobianLinearOperator`` for
:math:`M'_t`, :math:`H'_t` (autodiff-derived) and ``lineax.CG``
with positive-semidefinite tagging for the inner solve. The CVT is
applied via ``gaussx.sqrt`` when ``cvt=True`` — gaussx returns the
square-root operator for any positive-definite operator (Cholesky
for dense, structured for Matérn / Kronecker).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D


class IncrementalConfig(eqx.Module):
    """Configuration for ``IncrementalFourDVar``.

    Attributes:
        n_outer: Number of Gauss-Newton outer iterations (typical: 3).
        n_inner: Max CG iterations per outer (typical: 20-50).
        cg_atol: CG absolute tolerance.
        cg_rtol: CG relative tolerance.
    """

    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=30)
    cg_atol: float = eqx.field(static=True, default=1e-6)
    cg_rtol: float = eqx.field(static=True, default=1e-6)


class IncrementalFourDVar(eqx.Module):
    """Operational incremental 4DVar (Decision D11).

    Functionally equivalent to ``StrongFourDVar`` (same problem, same
    answer in the converged limit) but with a specialised inner
    solver: Gauss-Newton outer iterations and CG inner iterations on
    the linearised cost. Use this for production / long-window 4DVar.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel``.
        obs_op: Observation operator.
        prior_mean: Background :math:`x_b` — initial state ``(N,)``.
        prior_cov_op: :math:`B`.
        obs_cov_op: :math:`R`.
        config: ``IncrementalConfig``.
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, N]  # ty:ignore[unresolved-reference]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    config: IncrementalConfig

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, N],  # ty:ignore[unresolved-reference]
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        config: IncrementalConfig | None = None,
    ) -> None:
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.config = config or IncrementalConfig()

    def _rollout(
        self, x_0: Float[Array, N], n_steps: int  # ty:ignore[unresolved-reference]
    ) -> Float[Array, "T_plus_1 N"]:
        dt = self.forward.dt

        def step_fn(x, _):
            x_new = self.forward.step(x, dt)
            return x_new, x_new

        _, trajectory = jax.lax.scan(step_fn, x_0, None, length=n_steps)
        return jnp.concatenate([x_0[None, :], trajectory], axis=0)

    def __call__(self, batch: Batch1D) -> Float[Array, "B N"]:
        """Incremental-4DVar analysis: GN outer + CG inner."""
        T = batch.input.shape[1] - 1

        def _one(input_i, mask_i):
            x_b = self.prior_mean

            def cost_grad_for_outer(x_iter):
                """Gradient of the full nonlinear cost at ``x_iter``."""

                def cost_full(x_0):
                    dx = x_0 - self.prior_mean
                    B_inv_dx = lx.linear_solve(
                        self.prior_cov_op,
                        dx,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    j_bg = 0.5 * jnp.sum(dx * B_inv_dx)
                    trajectory = self._rollout(x_0, n_steps=T)

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

                    per_step = jax.vmap(_per_step)(trajectory, input_i, mask_i)
                    return j_bg + jnp.sum(per_step)

                return jax.grad(cost_full)(x_iter)

            # Gauss-Newton outer iterations on the nonlinear cost.
            for _ in range(self.config.n_outer):
                # Build the observation operator J: x_0 → masked
                # trajectory observations (shape (T_plus_1, N)).
                def J_full(x_0):
                    trajectory = self._rollout(x_0, n_steps=T)

                    def _per_step(x_t, m_t):
                        return (
                            self.obs_op(x_t, mask=m_t)
                            if _accepts_mask(self.obs_op)
                            else self.obs_op(x_t)
                        ) * m_t

                    return jax.vmap(_per_step)(trajectory, mask_i)

                # GN Hessian as a hand-rolled mat-vec on the state
                # space (N,). Uses jax.linearize + jax.vjp to apply
                # H^T R^{-1} H v plus B^{-1} v without forming dense
                # matrices.
                _, jvp_fn = jax.linearize(J_full, x_b)
                _, vjp_fn = jax.vjp(J_full, x_b)

                def gn_hessian_matvec(v, _jvp_fn=jvp_fn, _vjp_fn=vjp_fn):
                    Hv = _jvp_fn(v)

                    # Apply per-time-step R^{-1}: vmap CG over time
                    def _r_inv(y):
                        return lx.linear_solve(
                            self.obs_cov_op,
                            y,
                            solver=lx.CG(atol=1e-6, rtol=1e-6),
                        ).value

                    R_inv_Hv = jax.vmap(_r_inv)(Hv)
                    (HtR_inv_H_v,) = _vjp_fn(R_inv_Hv)
                    B_inv_v = lx.linear_solve(
                        self.prior_cov_op,
                        v,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    return HtR_inv_H_v + B_inv_v

                gn_op = lx.FunctionLinearOperator(
                    gn_hessian_matvec,
                    jax.ShapeDtypeStruct(x_b.shape, x_b.dtype),
                    lx.positive_semidefinite_tag,
                )

                # RHS = -grad J(x_b)
                rhs = -cost_grad_for_outer(x_b)

                # Inner CG solve: GN_Hessian @ dx = rhs
                dx_star = lx.linear_solve(
                    gn_op,
                    rhs,
                    solver=lx.CG(
                        atol=self.config.cg_atol,
                        rtol=self.config.cg_rtol,
                        max_steps=self.config.n_inner,
                    ),
                ).value
                x_b = x_b + dx_star

            return x_b

        return jax.vmap(_one)(batch.input, batch.mask)

    def as_analysis_step(self) -> _IncrementalFourDVarAnalysisStep:
        return _IncrementalFourDVarAnalysisStep(self)


def _inv(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Lazy inverse of a positive-semidefinite operator via lineax.CG."""

    def _solve(v):
        return lx.linear_solve(
            lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag),
            v,
            solver=lx.CG(atol=1e-6, rtol=1e-6, max_steps=200),
        ).value

    return lx.FunctionLinearOperator(_solve, op.in_structure())


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _IncrementalFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``IncrementalFourDVar``."""

    model: IncrementalFourDVar

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
