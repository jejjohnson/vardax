"""Incremental 4D-Var: Gauss–Newton outer iterations with CG on the inner quadratic.

The operational fast path of [`StrongFourDVar`][vardax.StrongFourDVar]: f and h are
applied through ``jax.linearize`` and ``jax.vjp``, matrix-free, and by default the inner
system is solved over the whitened control z with x₀ = μ_x + L_x z, L_x L_xᵀ = Σ_x.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.errors import ConfigurationError
from vardax._src.models._analysis_step import analyse_window, rollout_end
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
from vardax._src.utils.covariance import (
    as_covariance,
    guarded_solve,
)


class IncrementalConfig(eqx.Module):
    """Outer and inner iteration budget and solver switches of ``IncrementalFourDVar``.

    Attributes:
        n_outer: Number of Gauss–Newton outer iterations (typically 3).
        n_inner: Maximum CG iterations per outer iteration (typically 20 to 50). The
            inner solve stops there whether or not it has met the tolerance (truncated
            CG, as in operational incremental 4D-Var).
        cg_atol: Inner CG absolute tolerance.
        cg_rtol: Inner CG relative tolerance.
        cvt: Solve the inner system over the whitened control z with x₀ = μ_x + L_x z
            (default). The background term is ½ ‖z‖², the inner Hessian
            I + L_xᵀ J_rᵀ Λ̃_ε J_r L_x has eigenvalues ≥ 1, and Σ_x is never solved
            against. ``False`` solves in state space with Σ_x⁻¹ applied by CG at every
            Hessian product.
        covariance_tol: Absolute and relative tolerance of the CG solves that apply
            Σ_ε⁻¹ (and Σ_x⁻¹ when ``cvt=False``). Outer accuracy cannot beat it.
        jac: How each outer iteration linearises the residual of the window. ``"fwd"``
            (default) applies the tangent-linear model by ``jax.linearize``,
            matrix-free. ``"bwd"`` forms the residual Jacobian with ``jax.jacrev``, one
            VJP per observation, and runs the inner CG on that matrix; pass it when the
            forward or observation map defines only a reverse-mode rule (a
            ``jax.custom_vjp``, e.g. a diffrax solve under
            ``RecursiveCheckpointAdjoint``). It suits windows of up to a few thousand
            observations. The posterior uses the same mode.

    Raises:
        ConfigurationError: If ``jac`` is neither ``"fwd"`` nor ``"bwd"``.

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

        An unknown ``jac`` is rejected at construction, and a configuration
        round-trips:

        >>> vdx.IncrementalConfig(jac="central")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: jac must be 'fwd' or 'bwd'; got 'central'.
        >>> vdx.IncrementalConfig.from_config(cfg.get_config()) == cfg
        True
    """

    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=30)
    cg_atol: float = eqx.field(static=True, default=1e-6)
    cg_rtol: float = eqx.field(static=True, default=1e-6)
    cvt: bool = eqx.field(static=True, default=True)
    covariance_tol: float = eqx.field(static=True, default=1e-6)
    # "fwd" or "bwd", checked in __check_init__
    jac: str = eqx.field(static=True, default="fwd")

    def __check_init__(self) -> None:
        if self.jac not in ("fwd", "bwd"):
            raise ConfigurationError(f"jac must be 'fwd' or 'bwd'; got {self.jac!r}.")

    def get_config(self) -> dict[str, Any]:
        """The configuration as a plain dict, which ``from_config`` reads back."""
        return {
            "n_outer": self.n_outer,
            "n_inner": self.n_inner,
            "cg_atol": self.cg_atol,
            "cg_rtol": self.cg_rtol,
            "cvt": self.cvt,
            "covariance_tol": self.covariance_tol,
            "jac": self.jac,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IncrementalConfig:
        """Rebuild the configuration from the dict of ``get_config``."""
        return cls(**config)


class IncrementalFourDVar(eqx.Module):
    """Operational incremental 4D-Var of an assimilation window.

        𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖rₜ‖²_Σ̃_ε,t⁻¹
        rₜ = mₜ ⊙ (yₜ − h(xₜ)),   xₜ = f(xₜ₋₁),   t = 0, …, T
        Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ),   Mₜ = diag(mₜ)
        (Λ_c + J_rᵀ Λ̃_ε J_r) δc = −∇_c 𝒥,   c⁽ⁱ⁺¹⁾ = c⁽ⁱ⁾ + δc

    It solves the same problem as [`StrongFourDVar`][vardax.StrongFourDVar], with the
    same answer in the converged limit, but with Gauss–Newton outer iterations and CG on
    the linearised cost. Here c is the control and J_r = ∂r/∂c the Jacobian of the
    stacked residual r at c⁽ⁱ⁾. Use it for production and long windows. For a
    reduced-basis control on the full nonlinear cost, use
    [`ReducedBasisFourDVar`][vardax.ReducedBasisFourDVar].

    Algorithm:
        1. Start from x₀ = μ_x, which is the control z = 0 under the control-variable
           transform x₀ = μ_x + L_x z.
        2. For each of ``n_outer`` iterations, linearise the residual r at the current
           iterate with ``jax.linearize`` and ``jax.vjp``, or form its Jacobian with
           ``jax.jacrev`` when ``config.jac="bwd"``.
        3. Solve the Gauss–Newton system with CG for at most ``n_inner`` steps, where
           Λ_c = Σ_x⁻¹ in state space and Λ_c = I under the transform.
        4. Update the iterate with δc and map back to the state.

    Attributes:
        forward: Forward model f, a ``pipekit_cycle.ForwardModel``.
        obs_op: Observation operator h.
        prior_mean: Prior mean μ_x of the state at t = 0.
        prior_cov_op: Prior covariance Σ_x.
        obs_cov_op: Observation-error covariance Σ_ε.
        config: [`IncrementalConfig`][vardax.IncrementalConfig].
        prior_sqrt_op: Optional control-variable root L_x with L_x L_xᵀ = Σ_x: a
            Cholesky factor, a structured operator's own square root, or a Lanczos
            ``gaussx.sqrt(Σ_x, lanczos_order=k)`` for a large unstructured Σ_x. Default
            ``None``: ``gaussx.sqrt(prior_cov_op)``, closed form for diagonal, Kronecker
            and block-diagonal Σ_x and a dense eigendecomposition otherwise. It may be
            rectangular, D_x × D_z with Σ_x ≈ L_x L_xᵀ (e.g.
            ``lx.MatrixLinearOperator(basis.design_matrix)``): each inner system is then
            D_z × D_z, and the analysis stays in μ_x + span(L_x) (reduced incremental
            4D-Var).
        forward_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]: a
            ``diffrax.AbstractAdjoint`` for a forward implementing
            [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint].
        checkpoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]: ``False``, ``True``
            or a block length k for ``jax.checkpoint`` over the rollout (the outer
            nonlinear run and its tangent-linear and adjoint).

    Raises:
        ConfigurationError: If ``checkpoint`` is neither a bool nor an int ≥ 1.
        UnsupportedInputError: If ``forward_adjoint`` is set and ``forward`` has no
            ``with_adjoint`` method.

    Examples:
        Trivial dynamics f(x) = x and a single time reduce the problem to 3D-Var, so
        with Σ_x = Σ_ε = I the converged analysis is x₀⋆ = y / 2:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Identity:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> structure = jax.ShapeDtypeStruct((3,), jnp.result_type(float))
        >>> eye = lx.IdentityLinearOperator(structure)
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

        On a linear decaying model f(x) = x / 2 with observations at t = 0 and t = 1 the
        Gauss–Newton step is exact, so one outer iteration reproduces the
        strong-constraint answer x₀⋆ = ⅔:

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

        Composed with the Laplace adapter (via ``posterior``), the window's covariance
        is Σ_x|y = 1 / 2.25:

        >>> post = one.posterior(window)
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(3)), 1 / 2.25, atol=1e-3))
        True

        A rectangular root restricts the increment to its span (reduced incremental
        4D-Var). With one constant feature L_x = (1, 1, 1)ᵀ, x₀ = L_x z and the inner
        system (1 + L_xᵀ L_x) z = L_xᵀ y is 1 × 1, so y = (1, 2, 3) gives z⋆ = 6 / 4 at
        every cell:

        >>> constant = lx.MatrixLinearOperator(jnp.ones((3, 1)))
        >>> reduced = vardax.IncrementalFourDVar(
        ...     forward=Identity(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     prior_sqrt_op=constant,
        ... )
        >>> ramp = vardax.Batch1D(
        ...     input=jnp.array([[[1.0, 2.0, 3.0]]]), mask=jnp.ones((1, 1, 3))
        ... )
        >>> bool(jnp.allclose(reduced(ramp), 1.5, atol=1e-4))
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
        """Step f for T = ``n_steps`` steps from x₀, returning x₀, …, x_T.

        Args:
            initial_state: Initial state x₀.
            n_steps: Number of steps T.

        Returns:
            The trajectory, including x₀.
        """
        forward = resolve_forward(self.forward, self.forward_adjoint)
        return rollout(forward, initial_state, n_steps, checkpoint=self.checkpoint)

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, "*batch N"]:
        """Analyse each window with Gauss–Newton outer and CG inner iterations.

        A [`PointObsBatch`][vardax.PointObsBatch] needs ``obs_op=GridPointObs(...)``:
        observation i is compared with the trajectory state x_τᵢ,
        τᵢ = ``time_index[i]``, over a window of ``n_times`` states, with error
        variance σᵢ² (``obs_cov_op`` must be ``None`` or an identity).

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] whose time axis holds the
                window, or a [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The analysed initial state x₀⋆ of every window. Gauss–Newton has no line
            search, so a window whose iteration blows up comes back as NaN instead of
            raising.

        Raises:
            ObservationError: For a point batch with a non-trivial ``obs_cov_op`` or an
                out-of-range ``time_index``.
            MissingInputError: For a gridded batch when ``obs_cov_op`` is ``None``, or a
                point batch with ``time_index`` but no ``n_times``.
            InputTypeError: For a point batch whose ``obs_op`` is not a
                ``GridPointObs``.
        """
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            num_steps = window_steps(batch)
            return map_point_batch(
                lambda window: self._analyse_points(window, num_steps), batch
            )

        tol = self.config.covariance_tol
        obs_covariance = require_obs_cov(self.obs_cov_op)

        def analyse_window_obs(
            window_obs: Float[Array, "T_plus_1 M"],
            window_mask: Float[Array, "T_plus_1 M"],
        ) -> Float[Array, " N"]:
            def apply_obs_precision(
                residuals: Float[Array, "T_plus_1 M"],
            ) -> Float[Array, "T_plus_1 M"]:
                def solve_step(
                    step_residual: Float[Array, " M"], step_mask: Float[Array, " M"]
                ) -> Float[Array, " M"]:
                    solver = lx.CG(atol=tol, rtol=tol)
                    # Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ), so masked entries drop out of
                    # the likelihood
                    masked_covariance = gaussx.masked_obs_covariance(
                        obs_covariance, step_mask
                    )
                    # Σ̃_ε,t⁻¹ rₜ
                    solved, _ = guarded_solve(masked_covariance, step_residual, solver)
                    return solved

                return jax.vmap(solve_step)(residuals, window_mask)

            # ŷₜ(x₀) = mₜ ⊙ h(xₜ), shared with the posterior adapters
            observed = observed_trajectory(self, window_mask)
            # mₜ ⊙ yₜ
            masked_observations = window_mask * window_obs

            def residual(
                initial_state: Float[Array, " N"],
            ) -> Float[Array, "T_plus_1 M"]:
                # rₜ = mₜ ⊙ yₜ − ŷₜ(x₀)
                return masked_observations - observed(initial_state)

            return self._gauss_newton(residual, apply_obs_precision)

        return jax.vmap(analyse_window_obs)(
            fill_gaps(batch.input, batch.mask), batch.mask
        )

    def _analyse_points(
        self, batch: PointObsBatch, num_steps: int
    ) -> Float[Array, " N"]:
        """Gauss–Newton outer and CG inner iterations on a point batch.

            𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ‖r‖²
            rᵢ(x₀) = mᵢ (yᵢ − h(x_τᵢ)(sᵢ)) / σᵢ

        Args:
            batch: One sample of point observations.
            num_steps: Number of steps T.

        Returns:
            The analysed initial state x₀⋆.

        Raises:
            InputTypeError: If ``obs_op`` is not a ``GridPointObs``.
        """
        obs_op = require_grid_point_obs(self.obs_op)

        def residual(initial_state: Float[Array, " N"]) -> Float[Array, " n_obs"]:
            # xₜ = f(xₜ₋₁)
            trajectory = self._rollout(initial_state, n_steps=num_steps)
            # ŷᵢ = h(x_τᵢ)(sᵢ)
            predictions = obs_op(trajectory, batch)
            # rᵢ = mᵢ (yᵢ − ŷᵢ) / σᵢ
            return whitened_residual(predictions, batch)

        def identity(residuals: Float[Array, " n_obs"]) -> Float[Array, " n_obs"]:
            # the residual is already whitened: Λ̃_ε = I
            return residuals

        return self._gauss_newton(residual, identity)

    def _gauss_newton(
        self,
        residual: Callable[[Array], Array],
        apply_obs_precision: Callable[[Array], Array],
    ) -> Float[Array, " N"]:
        """Gauss–Newton outer loop with a CG inner solve, for one window.

            𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ r(x₀)ᵀ Λ̃_ε r(x₀)
            (Λ_c + J_rᵀ Λ̃_ε J_r) δc = −∇_c 𝒥

        Algorithm:
            1. Pick the control c and its map to the state: c = x₀ (state space), or
               c = z with x₀ = μ_x + L_x z, L_x L_xᵀ = Σ_x (control-variable transform),
               where the background term becomes ½ ‖z‖².
            2. At the current control c⁽ⁱ⁾, linearise the residual,
               r(c⁽ⁱ⁾ + δc) ≈ r(c⁽ⁱ⁾) + J_r δc.
            3. Solve for δc by CG, with Λ_c = Σ_x⁻¹ in state space and Λ_c = I under the
               transform.
            4. Update c⁽ⁱ⁺¹⁾ = c⁽ⁱ⁾ + δc and repeat ``n_outer`` times.

        Args:
            residual: x₀ ↦ r(x₀), masked or whitened observation misfits.
            apply_obs_precision: r ↦ Λ̃_ε r.

        Returns:
            The analysis x₀⋆.
        """
        config = self.config
        prior_mean = self.prior_mean
        tol = config.covariance_tol

        if config.cvt:
            prior_sqrt = self._prior_sqrt()

            def to_state(control: Float[Array, " C"]) -> Float[Array, " N"]:
                # x₀ = μ_x + L_x z
                return prior_mean + prior_sqrt.mv(control)

            def background_cost(control: Float[Array, " C"]) -> Float[Array, ""]:
                # 𝒥_b = ½ ‖z‖²
                return 0.5 * jnp.sum(control**2)

            def apply_prior_precision(direction: Float[Array, " C"]):
                # Λ_c = I: the prior Hessian in z is the identity
                return direction

            # z has the input size of L_x: D_x for a square root, D_z ≪ D_x for features
            control = jnp.zeros(prior_sqrt.in_structure().shape, prior_mean.dtype)
        else:

            def to_state(control: Float[Array, " C"]) -> Float[Array, " N"]:
                return control

            prior_covariance = self.prior_cov_op

            def apply_prior_precision(direction: Float[Array, " C"]):
                solver = lx.CG(atol=tol, rtol=tol)
                # Λ_c v = Σ_x⁻¹ v
                solved, _ = guarded_solve(prior_covariance, direction, solver)
                return solved

            def background_cost(control: Float[Array, " C"]) -> Float[Array, ""]:
                # δx₀ = x₀ − μ_x
                departure = control - prior_mean
                # 𝒥_b = ½ δx₀ᵀ Σ_x⁻¹ δx₀
                return 0.5 * jnp.sum(departure * apply_prior_precision(departure))

            control = prior_mean

        def control_residual(control: Float[Array, " C"]) -> Array:
            # r(c) = r(x₀(c))
            return residual(to_state(control))

        def cost(control: Float[Array, " C"]) -> Float[Array, ""]:
            # r(c)
            misfit = control_residual(control)
            # Λ̃_ε r
            weighted = apply_obs_precision(misfit)
            # 𝒥(c) = 𝒥_b + ½ rᵀ Λ̃_ε r
            return background_cost(control) + 0.5 * jnp.sum(misfit * weighted)

        # truncated CG: throw=False stops at n_inner whether or not it has converged
        inner_solver = gaussx.CGSolver(
            rtol=config.cg_rtol,
            atol=config.cg_atol,
            max_steps=config.n_inner,
            throw=False,
        )
        for _ in range(config.n_outer):
            # Λ_c + J_rᵀ Λ̃_ε J_r, linearised at c⁽ⁱ⁾
            hessian = gaussx.gauss_newton_operator(
                control_residual,
                control,
                noise_precision=apply_obs_precision,
                prior_precision=apply_prior_precision,
                # IncrementalConfig validates jac on construction
                jac=cast(Literal["fwd", "bwd"], config.jac),
            )

            # --- inner solve ---
            # −∇_c 𝒥(c⁽ⁱ⁾)
            negative_gradient = -jax.grad(cost)(control)
            # δc = (Λ_c + J_rᵀ Λ̃_ε J_r)⁻¹ (−∇_c 𝒥)
            increment = inner_solver.solve(hessian, negative_gradient)
            # c⁽ⁱ⁺¹⁾ = c⁽ⁱ⁾ + δc
            control = control + increment

        return to_state(control)

    def _prior_sqrt(self) -> lx.AbstractLinearOperator:
        """L_x with L_x L_xᵀ = Σ_x: ``prior_sqrt_op``, else ``gaussx.sqrt(Σ_x)``."""
        if self.prior_sqrt_op is not None:
            return self.prior_sqrt_op
        return gaussx.sqrt(self.prior_cov_op)

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the default adapter
        is [`LaplaceCovariance`][vardax.LaplaceCovariance] with this model's Σ_x and
        Σ_ε. With ``config.cvt`` the covariance is built over the whitened control, as
        the inner loop is, Σ_x|y = L_x (I + (J L_x)ᵀ Λ̃_ε J L_x)⁻¹ L_xᵀ, so Σ_x⁻¹ is
        never applied; its CG solves use the inner tolerances ``cg_atol`` and
        ``cg_rtol``.

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
        # lazy: the posterior adapters import the models
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
            An analysis step that treats non-finite observations as missing and returns
            the analysed state at the end of the window.
        """
        return _IncrementalFourDVarAnalysisStep(self)


class _IncrementalFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``IncrementalFourDVar``.

    Analyses the window around the cycle's forecast, with the cycle's h and Σ_ε when
    given, and returns the analysed state at the end of the window, the state the cycle
    steps forward from (``analyse_window``).

    Attributes:
        model: The model to analyse with.
    """

    model: IncrementalFourDVar

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        """Analyse a window around ``forecast``.

        Args:
            forecast: The cycle's forecast over the window.
            obs: Observations over the window; non-finite entries are missing.
            obs_op: The cycle's observation operator h, or ``None``.
            obs_err_cov: The cycle's observation-error covariance Σ_ε, or ``None``.

        Returns:
            The analysed state at the end of the window.
        """
        return analyse_window(
            self.model,
            rollout_end,
            forecast=forecast,
            obs=obs,
            obs_op=obs_op,
            obs_err_cov=obs_err_cov,
        )
