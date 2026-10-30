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
   $-\nabla J(x^{(k)}) = -B^{-1}(x^{(k)} - x_b) + \sum_t (H'_t M'_t)^\top R_t^{-1} (y_t - H_t(M_t(x^{(k)})))$
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
from vardax._src.models._observations import fill_gaps
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
    require_obs_cov,
    whitened_residual,
    window_steps,
)
from vardax._src.utils.covariance import as_covariance


class IncrementalConfig(eqx.Module):
    r"""Configuration for ``IncrementalFourDVar``.

    A plain container of the outer/inner iteration budget and the
    control-variable-transform switch.

    Attributes:
        n_outer: Number of Gauss-Newton outer iterations (typical: 3).
        n_inner: Max CG iterations per outer (typical: 20-50). The inner
            solve stops there whether or not it has met the tolerance
            (truncated CG, as in operational incremental 4DVar).
        cg_atol: Inner CG absolute tolerance.
        cg_rtol: Inner CG relative tolerance.
        cvt: Solve the inner system over the control variable $\chi$ with
            $x_0 = x_b + B^{1/2} \chi$ (default). The background term is
            $\tfrac12 \lVert \chi \rVert^2$, the inner Hessian
            $I + B^{1/2\top} G'^\top R^{-1} G' B^{1/2}$ has eigenvalues
            ≥ 1, and $B$ is never solved against. ``False`` solves in
            state space with $B^{-1}$ applied by CG at every Hessian
            product.
        covariance_tol: Absolute and relative tolerance of the CG solves
            that apply $R^{-1}$ (and $B^{-1}$ when ``cvt=False``). Outer
            accuracy cannot beat it.

    Examples:
        The defaults are three outer and thirty inner iterations with the
        control-variable transform on:

        >>> import vardax as vdx
        >>> cfg = vdx.IncrementalConfig()
        >>> cfg.n_outer, cfg.n_inner, cfg.cvt
        (3, 30, True)

        State-space solving (``cvt=False``) with a bigger budget:

        >>> vdx.IncrementalConfig(n_outer=5, cvt=False).n_outer
        5
    """

    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=30)
    cg_atol: float = eqx.field(static=True, default=1e-6)
    cg_rtol: float = eqx.field(static=True, default=1e-6)
    cvt: bool = eqx.field(static=True, default=True)
    covariance_tol: float = eqx.field(static=True, default=1e-6)

    def get_config(self) -> dict[str, Any]:
        return {
            "n_outer": self.n_outer,
            "n_inner": self.n_inner,
            "cg_atol": self.cg_atol,
            "cg_rtol": self.cg_rtol,
            "cvt": self.cvt,
            "covariance_tol": self.covariance_tol,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IncrementalConfig:
        return cls(**config)


class IncrementalFourDVar(eqx.Module):
    r"""Operational incremental 4DVar (Decision D11).

    $$J(x_0) = \tfrac12 \|x_0 - x_b\|^2_{B^{-1}} + \tfrac12 \|r(x_0)\|^2_{R^{-1}}$$

    $$r_t(x_0) = m_t \odot (y_t - H_t(M_t(x_0)))$$

    Algorithm:
        1. Start from $x^{(0)} = x_b$ (the control $\chi = 0$ under the
           control-variable transform).
        2. For each of ``n_outer`` iterations, linearise the residual
           $r$ at the current iterate with ``jax.linearize`` and
           ``jax.vjp``.
        3. Solve the Gauss-Newton system
           $(P + r'^\top R^{-1} r') \, \delta = -\nabla J$ with CG for at
           most ``n_inner`` steps, where $P = B^{-1}$ in state space and
           $P = I$ under the transform.
        4. Update the iterate with $\delta$ and map back to the state.

    Functionally equivalent to [`StrongFourDVar`][vardax.StrongFourDVar]
    (same problem, same answer in the converged limit) but with a
    specialised inner solver: Gauss-Newton outer iterations and CG
    inner iterations on the linearised cost. Use this for production /
    long-window 4DVar.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel``.
        obs_op: Observation operator.
        prior_mean: Background $x_b$ — the initial state.
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

        On a linear decaying model $M(x) = x / 2$ with observations at
        $t = 0, 1$ the Gauss-Newton step is exact, so one outer iteration
        reproduces the strong-constraint answer $x_0^* = 2/3$:

        >>> class Decay:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        >>> window = vardax.Batch1D(input=jnp.ones((1, 2, 3)), mask=jnp.ones((1, 2, 3)))
        >>> one = vardax.IncrementalFourDVar(
        ...     forward=Decay(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     config=vardax.IncrementalConfig(n_outer=1, cvt=False),
        ... )
        >>> bool(jnp.allclose(one(window), 2 / 3, atol=1e-3))
        True

        Composed with the Laplace adapter (via ``posterior``), the window's
        covariance is $P^* = 1 / 2.25$:

        >>> post = one.posterior(window)
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(3)), 1 / 2.25, atol=1e-3))
        True
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator | None
    config: IncrementalConfig
    prior_sqrt_op: lx.AbstractLinearOperator | None = None
    forward_adjoint: Any = eqx.field(static=True, default=None)
    checkpoint: bool | int = eqx.field(static=True, default=False)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, " N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator | None,
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
        self.prior_cov_op = as_covariance(prior_cov_op)
        self.obs_cov_op = None if obs_cov_op is None else as_covariance(obs_cov_op)
        self.config = config or IncrementalConfig()
        self.prior_sqrt_op = prior_sqrt_op
        self.forward_adjoint = forward_adjoint
        self.checkpoint = checkpoint

    def _rollout(
        self,
        initial_state: Float[Array, " N"],
        n_steps: int,
    ) -> Float[Array, "T_plus_1 N"]:
        forward = resolve_forward(self.forward, self.forward_adjoint)
        return rollout(forward, initial_state, n_steps, checkpoint=self.checkpoint)

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, "*batch N"]:
        r"""Incremental-4DVar analysis: GN outer + CG inner.

        A [`PointObsBatch`][vardax.PointObsBatch] needs
        ``obs_op=GridPointObs(...)``: observation $i$ is compared with the
        trajectory state at ``time_index[i]`` over a window of ``n_times``
        states, with error variance $\sigma_i^2$ (``obs_cov_op`` must
        be ``None`` or an identity).

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] whose time axis
                holds the window, or a
                [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The analysed initial state $x_0^*$ of every window.

        Raises:
            ObservationError: For a point batch with a non-trivial
                ``obs_cov_op`` or an out-of-range ``time_index``.
            MissingInputError: For a gridded batch when ``obs_cov_op`` is
                ``None``, or a point batch with ``time_index`` but no
                ``n_times``.
            InputTypeError: For a point batch whose ``obs_op`` is not a
                ``GridPointObs``.
        """
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            n_steps = window_steps(batch)
            return map_point_batch(
                lambda window: self._analyse_points(window, n_steps), batch
            )

        tol = self.config.covariance_tol
        obs_cov = require_obs_cov(self.obs_cov_op)

        def apply_obs_precision(
            residuals: Float[Array, "T_plus_1 M"],
        ) -> Float[Array, "T_plus_1 M"]:
            # R⁻¹ rₜ at each time
            def solve_one(step_residual: Float[Array, " M"]) -> Float[Array, " M"]:
                solver = lx.CG(atol=tol, rtol=tol)
                return lx.linear_solve(obs_cov, step_residual, solver=solver).value

            return jax.vmap(solve_one)(residuals)

        def analyse_one(
            window_obs: Float[Array, "T_plus_1 M"],
            window_mask: Float[Array, "T_plus_1 M"],
        ) -> Float[Array, " N"]:
            # G(x₀) = [mₜ ⊙ Hₜ(Mₜ(x₀))]ₜ, shared with the posterior adapters
            observed = observed_trajectory(self, window_mask)
            masked_observations = window_mask * window_obs

            def residual(
                initial_state: Float[Array, " N"],
            ) -> Float[Array, "T_plus_1 M"]:
                # rₜ = mₜ ⊙ (yₜ - Hₜ(Mₜ(x₀)))
                return masked_observations - observed(initial_state)

            return self._gauss_newton(residual, apply_obs_precision)

        return jax.vmap(analyse_one)(fill_gaps(batch.input, batch.mask), batch.mask)

    def _analyse_points(self, batch: PointObsBatch, n_steps: int) -> Float[Array, " N"]:
        r"""GN outer + CG inner on a point batch.

        With the whitened, padding-masked residual
        $r_i(x_0) = m_i (y_i - \mathcal H(M^{\tau_i}(x_0))(s_i)) / \sigma_i$,
        the cost is
        $\tfrac12 \lVert x_0 - x_b \rVert^2_{B^{-1}} + \tfrac12 \lVert r \rVert^2$
        and the Gauss-Newton Hessian is $B^{-1} + r'^\top r'$
        ($I + S^\top r'^\top r' S$ with the control-variable transform).
        """
        obs_op = require_grid_point_obs(self.obs_op)

        def residual(initial_state: Float[Array, " N"]) -> Float[Array, " n_obs"]:
            predictions = obs_op(self._rollout(initial_state, n_steps=n_steps), batch)
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

            prior_cov = self.prior_cov_op

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
        return gaussx.sqrt(self.prior_cov_op)

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        r"""Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the
        default adapter is [`LaplaceCovariance`][vardax.LaplaceCovariance]
        with this model's $B$ and $R$. With ``config.cvt`` the covariance
        is built over the whitened control, as the inner loop is,
        $P^* = S (I + (G' S)^\top R^{-1} G' S)^{-1} S^\top$ with
        $S S^\top = B$, so $B^{-1}$ is never applied; its CG solves use
        the inner tolerances ``cg_atol`` and ``cg_rtol``.

        Args:
            batch: A ``Batch1D`` holding one window.
            adapter: Optional [`PosteriorAdapter`][vardax.PosteriorAdapter],
                e.g. [`GaussNewtonHessian`][vardax.GaussNewtonHessian].

        Returns:
            The window's posterior: mean and lazy covariance.

        Raises:
            ShapeError: If ``batch`` holds more than one window.
            UnsupportedInputError: For a point batch.
        """
        from vardax._src.posterior._precision import model_posterior
        from vardax._src.posterior.laplace import DEFAULT_MAX_STEPS, LaplaceCovariance

        if adapter is None and self.config.cvt:
            # the control-space precision is the inner Hessian: solve it to
            # the inner tolerances, with at least the default step budget
            solver = lx.CG(
                atol=self.config.cg_atol,
                rtol=self.config.cg_rtol,
                max_steps=max(self.config.n_inner, DEFAULT_MAX_STEPS),
            )
            adapter = LaplaceCovariance(solver=solver)
        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _IncrementalFourDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        Returns:
            An analysis step that treats non-finite observations as
            missing and returns the analysed initial state.
        """
        return _IncrementalFourDVarAnalysisStep(self)


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
