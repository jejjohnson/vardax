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
with ``lineax.CG`` on a positive-semidefinite-tagged operator. By default
it does so over the control variable $\chi$, $x_0 = x_b + B^{1/2}\chi$
(the control-variable transform, ``IncrementalConfig.cvt``), where the
inner Hessian is $I + B^{1/2\top} G'^\top R^{-1} G' B^{1/2}$ and no
$B^{-1}$ is needed. For a reduced-basis or other reparameterised control
use
[`ReducedBasisFourDVar`][vardax.ReducedBasisFourDVar] with ``basis=`` or
``parameterization=``.
"""

from __future__ import annotations

from collections.abc import Callable
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
        n_inner: Max CG iterations per outer (typical: 20-50). The inner
            solve stops there whether or not it has met the tolerance
            (truncated CG, as in operational incremental 4DVar).
        cg_atol: Inner CG absolute tolerance.
        cg_rtol: Inner CG relative tolerance.
        cvt: Solve the inner system over the control variable χ with
            $x_0 = x_b + B^{1/2} \\chi$ (default). The background term is
            $\tfrac12 \\lVert \\chi \rVert^2$, the inner Hessian
            $I + B^{1/2\top} G'^\top R^{-1} G' B^{1/2}$ has eigenvalues
            ≥ 1, and $B$ is never solved against. ``False`` solves in
            state space with $B^{-1}$ applied by CG at every Hessian
            product.
        covariance_tol: Absolute and relative tolerance of the CG solves
            that apply $R^{-1}$ (and $B^{-1}$ when ``cvt=False``). Outer
            accuracy cannot beat it.
    """

    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=30)
    cg_atol: float = eqx.field(static=True, default=1e-6)
    cg_rtol: float = eqx.field(static=True, default=1e-6)
    cvt: bool = eqx.field(static=True, default=True)
    covariance_tol: float = eqx.field(static=True, default=1e-6)


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
        prior_sqrt_op: Optional square root $S$ of $B$ ($S S^\top = B$),
            used by the control-variable transform: a Cholesky factor, a
            structured operator's own square root, or a Lanczos
            ``gaussx.sqrt(B, lanczos_order=k)`` for a large unstructured
            $B$. Default ``None``: ``gaussx.sqrt(prior_cov_op)``, closed
            form for diagonal / Kronecker / block-diagonal $B$ and a dense
            eigendecomposition otherwise.
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
    prior_sqrt_op: lx.AbstractLinearOperator | None = None
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
        prior_sqrt_op: lx.AbstractLinearOperator | None = None,
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
        self.prior_sqrt_op = prior_sqrt_op
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

        tol = self.config.covariance_tol
        obs_cov = _as_covariance(self.obs_cov_op)

        def apply_obs_precision(
            residuals: Float[Array, "T_plus_1 N"],
        ) -> Float[Array, "T_plus_1 N"]:
            # R⁻¹ rₜ at each time
            def solve_one(residual_t: Float[Array, " N"]) -> Float[Array, " N"]:
                solver = lx.CG(atol=tol, rtol=tol)
                return lx.linear_solve(obs_cov, residual_t, solver=solver).value

            return jax.vmap(solve_one)(residuals)

        def analyse_one(
            input_i: Float[Array, "T_plus_1 N"], mask_i: Float[Array, "T_plus_1 N"]
        ) -> Float[Array, " N"]:
            # G(x₀) = [mₜ ⊙ Hₜ(Mₜ(x₀))]ₜ, shared with the posterior adapters
            observed = observed_trajectory(self, mask_i)
            masked_observations = mask_i * input_i

            def residual(
                initial_state: Float[Array, " N"],
            ) -> Float[Array, "T_plus_1 N"]:
                # rₜ = mₜ ⊙ (yₜ - Hₜ(Mₜ(x₀)))
                return masked_observations - observed(initial_state)

            return self._gauss_newton(residual, apply_obs_precision)

        return jax.vmap(analyse_one)(batch.input, batch.mask)

    def _analyse_points(self, batch: PointObsBatch, T: int) -> Float[Array, " N"]:
        r"""GN outer + CG inner on a point batch.

        With the whitened, padding-masked residual
        $r_i(x_0) = m_i (y_i - \mathcal H(M^{\tau_i}(x_0))(s_i)) / \sigma_i$,
        the cost is $\tfrac12 \lVert x_0 - x_b \rVert^2_{B^{-1}} +
        \tfrac12 \lVert r \rVert^2$ and the Gauss-Newton Hessian is
        $B^{-1} + r'^\top r'$ ($I + S^\top r'^\top r' S$ with the
        control-variable transform).
        """
        obs_op = require_grid_point_obs(self.obs_op)

        def residual(initial_state: Float[Array, " N"]) -> Float[Array, " n_obs"]:
            predictions = obs_op(self._rollout(initial_state, n_steps=T), batch)
            return whitened_residual(predictions, batch)

        def identity(residuals: Float[Array, " n_obs"]) -> Float[Array, " n_obs"]:
            # the residual is already whitened: R = I
            return residuals

        return self._gauss_newton(residual, identity)

    def _gauss_newton(
        self,
        residual: Callable[[Array], Array],
        apply_obs_precision: Callable[[Array], Array],
    ) -> Float[Array, " N"]:
        r"""Gauss–Newton outer loop with a CG inner solve, for one window.

        J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ r(x₀)ᵀ W r(x₀),  W = R⁻¹.

        Algorithm:
            1. Pick the control c and its map to the state: c = x₀
               (state space), or c = χ with x₀ = x_b + S χ, S Sᵀ = B
               (control-variable transform), where the background term
               becomes ½ ‖χ‖².
            2. At the current control c⁽ᵏ⁾, linearise the residual,
               r(c⁽ᵏ⁾ + δc) ≈ r(c⁽ᵏ⁾) + r' δc.
            3. Solve (P + r'ᵀ W r') δc = −∇J(c⁽ᵏ⁾) by CG, with P = B⁻¹ in
               state space and P = I under the transform.
            4. Update c⁽ᵏ⁺¹⁾ = c⁽ᵏ⁾ + δc and repeat ``n_outer`` times.

        Args:
            residual: x₀ ↦ r(x₀), masked or whitened observation misfits.
            apply_obs_precision: r ↦ W r.

        Returns:
            The analysis x₀*.
        """
        config = self.config
        background = self.prior_mean
        tol = config.covariance_tol

        if config.cvt:
            prior_sqrt = self._prior_sqrt()

            def to_state(control: Float[Array, " N"]) -> Float[Array, " N"]:
                # x₀ = x_b + S χ
                return background + prior_sqrt.mv(control)

            def background_cost(control: Float[Array, " N"]) -> Float[Array, ""]:
                # ½ ‖χ‖²
                return 0.5 * jnp.sum(control**2)

            def apply_prior_precision(direction: Float[Array, " N"]):
                # the prior Hessian in χ is the identity
                return direction

            control = jnp.zeros_like(background)
        else:

            def to_state(control: Float[Array, " N"]) -> Float[Array, " N"]:
                return control

            prior_cov = _as_covariance(self.prior_cov_op)

            def apply_prior_precision(direction: Float[Array, " N"]):
                # B⁻¹ v
                solver = lx.CG(atol=tol, rtol=tol)
                return lx.linear_solve(prior_cov, direction, solver=solver).value

            def background_cost(control: Float[Array, " N"]) -> Float[Array, ""]:
                # ½ (x₀ - x_b)ᵀ B⁻¹ (x₀ - x_b)
                departure = control - background
                return 0.5 * jnp.sum(departure * apply_prior_precision(departure))

            control = background

        def control_residual(control: Float[Array, " N"]) -> Array:
            return residual(to_state(control))

        def cost(control: Float[Array, " N"]) -> Float[Array, ""]:
            # J = background + ½ rᵀ W r
            misfit = control_residual(control)
            weighted = apply_obs_precision(misfit)
            return background_cost(control) + 0.5 * jnp.sum(misfit * weighted)

        structure = jax.ShapeDtypeStruct(control.shape, control.dtype)
        inner_solver = lx.CG(
            atol=config.cg_atol, rtol=config.cg_rtol, max_steps=config.n_inner
        )
        for _ in range(config.n_outer):
            # --- linearise at c⁽ᵏ⁾ ---
            _, residual_tangent = jax.linearize(control_residual, control)
            _, residual_adjoint = jax.vjp(control_residual, control)

            def hessian_mv(
                direction: Float[Array, " N"],
                tangent: Callable = residual_tangent,
                adjoint: Callable = residual_adjoint,
            ) -> Float[Array, " N"]:
                # (P + r'ᵀ W r') v
                weighted = apply_obs_precision(tangent(direction))
                (data_term,) = adjoint(weighted)
                return apply_prior_precision(direction) + data_term

            hessian = lx.FunctionLinearOperator(
                hessian_mv, structure, lx.positive_semidefinite_tag
            )

            # --- inner solve: (P + r'ᵀ W r') δc = -∇J(c⁽ᵏ⁾) ---
            negative_gradient = -jax.grad(cost)(control)
            # truncated CG: stop at n_inner whether or not it has converged
            solution = lx.linear_solve(
                hessian, negative_gradient, solver=inner_solver, throw=False
            )
            # c⁽ᵏ⁺¹⁾ = c⁽ᵏ⁾ + δc
            control = control + solution.value

        return to_state(control)

    def _prior_sqrt(self) -> lx.AbstractLinearOperator:
        """S with S Sᵀ = B: the given ``prior_sqrt_op``, else ``gaussx.sqrt(B)``."""
        if self.prior_sqrt_op is not None:
            return self.prior_sqrt_op
        return gaussx.sqrt(_as_covariance(self.prior_cov_op))

    def as_analysis_step(self) -> _IncrementalFourDVarAnalysisStep:
        return _IncrementalFourDVarAnalysisStep(self)


def _as_covariance(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Tag a covariance operator PSD, so CG accepts e.g. a plain diagonal."""
    if lx.is_positive_semidefinite(operator):
        return operator
    return lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)


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
