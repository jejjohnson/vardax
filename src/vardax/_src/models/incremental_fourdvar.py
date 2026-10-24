r"""Incremental 4DVar: Gauss-Newton outer, CG inner (Decision D11).

The operational fast path of [`StrongFourDVar`][vardax.StrongFourDVar]:
Gauss-Newton outer iterations on the full nonlinear cost, CG inner
iterations on the linearised quadratic subproblem.

Starting from $x^{(0)} = x_b$, each outer iteration around the current
iterate $x^{(k)}$ (the background $x_b$ stays fixed):

1. Linearise $M_t$ and $H_t$ at $x^{(k)}$ (via ``jax.linearize``).
2. Form the Gauss-Newton Hessian
   $J''_{GN} = B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t)$.
3. Solve $J''_{GN}\,\delta x^* = -\nabla J(x^{(k)})$ via lineax CG, where
   $-\nabla J(x^{(k)}) = -B^{-1}(x^{(k)} - x_b)
   + \sum_t (H'_t M'_t)^\top R_t^{-1} (y_t - H_t(M_t(x^{(k)})))$
   is the negative gradient of the full nonlinear cost.
4. Update $x^{(k+1)} = x^{(k)} + \delta x^*$.

This implementation applies $M'_t$, $H'_t$ through ``jax.linearize`` /
``jax.vjp`` (autodiff-derived, matrix-free) and solves the inner system
with ``lineax.CG`` on a positive-semidefinite-tagged operator, in state
space. There is no control-variable-transform option; for a
reduced-basis or other reparameterised control use
[`ReducedBasisFourDVar`][vardax.ReducedBasisFourDVar] with ``basis=`` or
``parameterization=``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.models._rollout import (
    checkpoint_every,
    observed_trajectory,
    resolve_forward,
    rollout,
)
from vardax._src.obs_operators.point import (
    check_point_obs_cov,
    map_point_batch,
    require_grid_point_obs,
    whitened_residual,
    window_steps,
)


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
    r"""Operational incremental 4DVar (Decision D11).

    Functionally equivalent to [`StrongFourDVar`][vardax.StrongFourDVar]
    (same problem, same answer in the converged limit) but with a
    specialised inner solver: Gauss-Newton outer iterations and CG
    inner iterations on the linearised cost. Use this for production /
    long-window 4DVar.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel``.
        obs_op: Observation operator.
        prior_mean: Background $x_b$ — initial state ``(N,)``.
        prior_cov_op: $B$.
        obs_cov_op: $R$.
        config: [`IncrementalConfig`][vardax.IncrementalConfig].
        forward_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]:
            a ``diffrax.AbstractAdjoint`` for a forward implementing
            [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint].
        checkpoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]:
            ``False``, ``True`` or a block length ``k`` for
            ``jax.checkpoint`` over the rollout (the
            outer nonlinear run and its tangent-linear / adjoint).

    Examples:
        Trivial dynamics ($M_t(x) = x$) and a single timestep reduce
        the problem to 3DVar, so with $B = R = I$ the converged
        analysis is $x_0^* = y / 2$.

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Identity:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
        >>> inc = vardax.IncrementalFourDVar(
        ...     forward=Identity(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 1, 3)), mask=jnp.ones((1, 1, 3)))
        >>> xa = inc(batch)
        >>> xa.shape
        (1, 3)
        >>> bool(jnp.allclose(xa, 0.5, atol=1e-2))
        True
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, N]  # ty:ignore[unresolved-reference]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    config: IncrementalConfig
    forward_adjoint: Any = eqx.field(static=True, default=None)
    checkpoint: bool | int = eqx.field(static=True, default=False)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, N],  # ty:ignore[unresolved-reference]
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        config: IncrementalConfig | None = None,
        forward_adjoint: Any = None,
        checkpoint: bool | int = False,
    ) -> None:
        # fail at construction, not on the first traced call
        checkpoint_every(checkpoint)
        resolve_forward(forward, forward_adjoint)
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.config = config or IncrementalConfig()
        self.forward_adjoint = forward_adjoint
        self.checkpoint = checkpoint

    def _rollout(
        self,
        x_0: Float[Array, N],  # ty:ignore[unresolved-reference]
        n_steps: int,
    ) -> Float[Array, "T_plus_1 N"]:
        forward = resolve_forward(self.forward, self.forward_adjoint)
        return rollout(forward, x_0, n_steps, checkpoint=self.checkpoint)

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, "B N"]:
        """Incremental-4DVar analysis: GN outer + CG inner.

        A [`PointObsBatch`][vardax.PointObsBatch] needs
        ``obs_op=GridPointObs(...)``: observation $i$ is compared with the
        trajectory state at ``time_index[i]`` over a window of ``n_times``
        states, with error variance $\\sigma_i^2$ (``obs_cov_op`` must
        be ``None`` or an identity, else ``ValueError``).
        """
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            n_steps = window_steps(batch)
            return map_point_batch(lambda b: self._analyse_points(b, n_steps), batch)

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
                        residual = m_t * (y_t - y_pred)
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
                # G(x₀) = [mₜ ⊙ Hₜ(Mₜ(x₀))]ₜ, shared with the posterior
                # adapters so both use the same tangent-linear operator
                J_full = observed_trajectory(self, mask_i)

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

    def _analyse_points(self, batch: PointObsBatch, T: int) -> Float[Array, " N"]:
        r"""GN outer + CG inner on a point batch.

        With the whitened, padding-masked residual
        $r_i(x_0) = m_i (y_i - \mathcal H(M^{\tau_i}(x_0))(s_i)) / \sigma_i$,
        the cost is $\tfrac12 \lVert x_0 - x_b \rVert^2_{B^{-1}} +
        \tfrac12 \lVert r \rVert^2$ and the Gauss-Newton Hessian is
        $B^{-1} + r'^\top r'$.
        """
        obs_op = require_grid_point_obs(self.obs_op)

        def residual(x_0):
            preds = obs_op(self._rollout(x_0, n_steps=T), batch)
            return whitened_residual(preds, batch)

        def b_inv(v):
            return lx.linear_solve(
                self.prior_cov_op, v, solver=lx.CG(atol=1e-6, rtol=1e-6)
            ).value

        def cost_full(x_0):
            dx = x_0 - self.prior_mean
            return 0.5 * jnp.sum(dx * b_inv(dx)) + 0.5 * jnp.sum(residual(x_0) ** 2)

        x_b = self.prior_mean
        for _ in range(self.config.n_outer):
            _, jvp_fn = jax.linearize(residual, x_b)
            _, vjp_fn = jax.vjp(residual, x_b)

            def gn_hessian_matvec(v, _jvp_fn=jvp_fn, _vjp_fn=vjp_fn):
                (jt_j_v,) = _vjp_fn(_jvp_fn(v))
                return jt_j_v + b_inv(v)

            gn_op = lx.FunctionLinearOperator(
                gn_hessian_matvec,
                jax.ShapeDtypeStruct(x_b.shape, x_b.dtype),
                lx.positive_semidefinite_tag,
            )
            dx_star = lx.linear_solve(
                gn_op,
                -jax.grad(cost_full)(x_b),
                solver=lx.CG(
                    atol=self.config.cg_atol,
                    rtol=self.config.cg_rtol,
                    max_steps=self.config.n_inner,
                ),
            ).value
            x_b = x_b + dx_star
        return x_b

    def as_analysis_step(self) -> _IncrementalFourDVarAnalysisStep:
        return _IncrementalFourDVarAnalysisStep(self)


def _inv(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Lazy PSD inverse — gaussx.inv (structured closed forms, CG fallback).

    Re-wrapped as a ``FunctionLinearOperator`` so lineax can linearise
    it inside composed operators that are themselves solved.
    """
    inv_op = gaussx.inv(
        lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag),
        solver=lx.CG(atol=1e-6, rtol=1e-6, max_steps=200),
    )
    return lx.FunctionLinearOperator(inv_op.mv, op.in_structure())


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
