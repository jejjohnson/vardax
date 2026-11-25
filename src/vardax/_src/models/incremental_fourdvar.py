"""Incremental 4D-Var: Gauss–Newton outer iterations with CG on the inner quadratic.

The operational fast path of [`StrongFourDVar`][vardax.StrongFourDVar]: the cost is
given as whitened residuals and solved by ``optimistix.least_squares``, by default with
``argminax.solve.GaussNewtonCG``, which applies f and h through ``jax.linearize`` and
``jax.vjp``, matrix-free. By default the solve runs over the whitened control z with
x₀ = μ_x + L_x z, L_x L_xᵀ = Σ_x.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from argminax.solve import GaussNewtonCG
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx
import optimistix as optx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.errors import ConfigurationError, InputTypeError
from vardax._src.minimisers import least_squares
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
from vardax._src.utils.covariance import as_covariance, masked_cholesky, whitener
from vardax._src.utils.deprecation import warn_deprecated

# The default budget, and the defaults of the deprecated IncrementalConfig fields:
# three outer iterations, at most thirty CG steps each, CG tolerances 1e-6.
_N_OUTER = 3
_N_INNER = 30
_CG_TOL = 1e-6
_DEPRECATED_BUDGET = ("n_outer", "n_inner", "cg_atol", "cg_rtol")


def _budget_solver(config: IncrementalConfig) -> GaussNewtonCG:
    """The ``GaussNewtonCG`` of the outer and inner budget of ``config``.

    The outer tolerances are zero, so the solve never stops early and runs exactly
    ``n_outer`` linearisations, as the loop it replaces did; each inner CG stops at
    its own tolerance or after ``n_inner`` steps (truncated CG).
    """
    return GaussNewtonCG(
        rtol=0.0,
        atol=0.0,
        max_cg_steps=_N_INNER if config.n_inner is None else config.n_inner,
        cg_rtol=_CG_TOL if config.cg_rtol is None else config.cg_rtol,
        cg_atol=_CG_TOL if config.cg_atol is None else config.cg_atol,
    )


class IncrementalConfig(eqx.Module):
    """Solver switches of ``IncrementalFourDVar``, and its deprecated budget.

    The iteration budget moved out of this configuration: the outer budget is the
    model's ``n_outer``, and the inner CG is the model's ``solver``, e.g.
    ``argminax.solve.GaussNewtonCG(rtol, atol, max_cg_steps=50)``. Setting
    ``n_outer``, ``n_inner``, ``cg_atol`` or ``cg_rtol`` here still works for one
    release: the model builds the equivalent ``GaussNewtonCG`` (outer tolerances 0,
    so exactly ``n_outer`` linearisations) and emits a ``DeprecationWarning``.

    Attributes:
        n_outer: Deprecated: pass ``n_outer`` to the model. Number of Gauss–Newton
            outer iterations; ``None`` (default) leaves it to the model.
        n_inner: Deprecated: pass ``solver=GaussNewtonCG(max_cg_steps=...)``. Maximum
            CG iterations per outer iteration (truncated CG); ``None`` means 30.
        cg_atol: Deprecated: ``GaussNewtonCG(cg_atol=...)``. Inner CG absolute
            tolerance; ``None`` means 1e-6.
        cg_rtol: Deprecated: ``GaussNewtonCG(cg_rtol=...)``. Inner CG relative
            tolerance; ``None`` means 1e-6.
        cvt: Solve over the whitened control z with x₀ = μ_x + L_x z (default). The
            background residual is z itself, so the Gauss–Newton matrix
            I + L_xᵀ J_rᵀ Λ̃_ε J_r L_x has eigenvalues ≥ 1. ``False`` solves in state
            space, with the background residual L_x⁻¹ (x₀ − μ_x) whitened by the
            Cholesky factor of Σ_x.
        covariance_tol: Absolute and relative tolerance of the CG solves of the
            default posterior under ``cvt`` (when no deprecated ``cg_atol`` or
            ``cg_rtol`` is set). Σ_ε and Σ_x enter the analysis through their
            Cholesky factors, so not through this tolerance.
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
        The default leaves the budget to the model and turns the control-variable
        transform on:

        >>> import vardax as vdx
        >>> cfg = vdx.IncrementalConfig()
        >>> cfg.n_outer, cfg.n_inner, cfg.cvt
        (None, None, True)

        State-space solving (``cvt=False``):

        >>> vdx.IncrementalConfig(cvt=False).cvt
        False

        An unknown ``jac`` is rejected at construction, and a configuration
        round-trips:

        >>> vdx.IncrementalConfig(jac="central")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: jac must be 'fwd' or 'bwd'; got 'central'.
        >>> vdx.IncrementalConfig.from_config(cfg.get_config()) == cfg
        True
    """

    n_outer: int | None = eqx.field(static=True, default=None)
    n_inner: int | None = eqx.field(static=True, default=None)
    cg_atol: float | None = eqx.field(static=True, default=None)
    cg_rtol: float | None = eqx.field(static=True, default=None)
    cvt: bool = eqx.field(static=True, default=True)
    covariance_tol: float = eqx.field(static=True, default=1e-6)
    # "fwd" or "bwd", checked in __check_init__
    jac: str = eqx.field(static=True, default="fwd")

    def __check_init__(self) -> None:
        if self.jac not in ("fwd", "bwd"):
            raise ConfigurationError(f"jac must be 'fwd' or 'bwd'; got {self.jac!r}.")

    def deprecated_budget(self) -> tuple[str, ...]:
        """The deprecated budget fields that are set (not ``None``)."""
        return tuple(
            name for name in _DEPRECATED_BUDGET if getattr(self, name) is not None
        )

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
        𝒥(c) = ½ ‖r̃(c)‖²,   r̃(c) = (r_b(c), L̃_ε,t⁻¹ rₜ),   L̃_ε,t L̃_ε,tᵀ = Σ̃_ε,t
        (J̃ᵀ J̃) δc = −J̃ᵀ r̃(c⁽ⁱ⁾),   c⁽ⁱ⁺¹⁾ = c⁽ⁱ⁾ + δc

    It solves the same problem as [`StrongFourDVar`][vardax.StrongFourDVar], with the
    same answer in the converged limit, but with Gauss–Newton outer iterations and CG on
    the linearised cost. Here c is the control, r_b its background residual (z under
    the control-variable transform, L_x⁻¹ (x₀ − μ_x) in state space) and J̃ = ∂r̃/∂c
    the Jacobian of the stacked whitened residual at c⁽ⁱ⁾. The solve is
    ``optimistix.least_squares`` with ``solver``, by default
    ``argminax.solve.GaussNewtonCG``. Use it for production and long windows. For a
    reduced-basis control on the full nonlinear cost, use
    [`ReducedBasisFourDVar`][vardax.ReducedBasisFourDVar].

    Algorithm:
        1. Start from x₀ = μ_x, which is the control z = 0 under the control-variable
           transform x₀ = μ_x + L_x z.
        2. Whiten the window: factor each Σ̃_ε,t = L̃_ε,t L̃_ε,tᵀ once (and Σ_x in state
           space), so 𝒥 is a sum of squares.
        3. For at most ``n_outer`` iterations, linearise r̃ at the current iterate
           with ``jax.linearize``, or form its Jacobian with ``jax.jacrev`` when
           ``config.jac="bwd"``, and take the solver's step, for ``GaussNewtonCG`` a
           CG solve of the normal equations of at most ``max_cg_steps`` steps.
        4. Map the control back to the state.

    Attributes:
        forward: Forward model f, a ``pipekit_cycle.ForwardModel``.
        obs_op: Observation operator h.
        prior_mean: Prior mean μ_x of the state at t = 0.
        prior_cov_op: Prior covariance Σ_x.
        obs_cov_op: Observation-error covariance Σ_ε.
        config: [`IncrementalConfig`][vardax.IncrementalConfig]: the control-variable
            transform, the Jacobian mode and the posterior tolerance.
        solver: The ``optimistix.AbstractLeastSquaresSolver`` of the outer loop. The
            default is ``argminax.solve.GaussNewtonCG(rtol=0, atol=0,
            max_cg_steps=30, cg_rtol=1e-6, cg_atol=1e-6)``: outer tolerances 0, so
            exactly ``n_outer`` linearisations, each with a truncated inner CG, as
            operational incremental 4D-Var runs. Pass
            ``GaussNewtonCG(rtol, atol, max_cg_steps=...)`` to stop the outer loop
            once it has converged, or ``trust_region=True`` for Levenberg–Marquardt.
        n_outer: The most Gauss–Newton outer iterations (linearisations), 3 by
            default.
        solver_adjoint: ``optimistix.AbstractAdjoint`` for differentiating the
            analysis. The default ``optimistix.RecursiveCheckpointAdjoint`` goes
            through the outer iterations, which is exact for any budget;
            ``optimistix.ImplicitAdjoint`` is cheaper and exact at a converged
            minimum.
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
        ConfigurationError: If ``checkpoint`` is neither a bool nor an int ≥ 1, if
            ``n_outer`` is not an int ≥ 1, or if ``solver`` or ``n_outer`` is given
            alongside the deprecated budget of ``config``.
        InputTypeError: If ``solver`` is not an
            ``optimistix.AbstractLeastSquaresSolver``.
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
        ...     config=vardax.IncrementalConfig(cvt=False),
        ...     n_outer=1,
        ... )
        >>> bool(jnp.allclose(one(window), 2 / 3, atol=1e-3))
        True

        A solver with outer tolerances stops once the iterates settle, here after the
        one increment that solves the linear problem:

        >>> from argminax.solve import GaussNewtonCG
        >>> converged = vardax.IncrementalFourDVar(
        ...     forward=Decay(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     solver=GaussNewtonCG(rtol=1e-5, atol=1e-5, max_cg_steps=50),
        ...     n_outer=20,
        ... )
        >>> bool(jnp.allclose(converged(window), 2 / 3, atol=1e-3))
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
    # not static: GaussNewtonCG holds its learning rate as an array
    solver: optx.AbstractLeastSquaresSolver
    n_outer: int = eqx.field(static=True)
    solver_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
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
        solver: optx.AbstractLeastSquaresSolver | None = None,
        n_outer: int | None = None,
        solver_adjoint: optx.AbstractAdjoint | None = None,
    ) -> None:
        # fail at construction, not on the first traced call
        checkpoint_every(checkpoint)
        resolve_forward(forward, forward_adjoint)
        config = config or IncrementalConfig()
        deprecated = config.deprecated_budget()
        if deprecated:
            if solver is not None or n_outer is not None:
                raise ConfigurationError(
                    "Pass the budget once: solver= and n_outer= replace "
                    f"IncrementalConfig({', '.join(deprecated)}); got both."
                )
            warn_deprecated(
                f"IncrementalConfig({', '.join(f'{name}=' for name in deprecated)}) "
                "is deprecated and will be removed in the next release; pass "
                "n_outer= and solver=argminax.solve.GaussNewtonCG(rtol, atol, "
                "max_cg_steps=..., cg_rtol=..., cg_atol=...) to IncrementalFourDVar."
            )
            n_outer = config.n_outer
        if solver is None:
            solver = _budget_solver(config)
        if isinstance(solver, optx.AbstractMinimiser) or not isinstance(
            solver, optx.AbstractLeastSquaresSolver
        ):
            raise InputTypeError(
                "solver must be an optimistix.AbstractLeastSquaresSolver, such as "
                f"argminax.solve.GaussNewtonCG; got {type(solver).__name__}."
            )
        n_outer = _N_OUTER if n_outer is None else n_outer
        if isinstance(n_outer, bool) or not isinstance(n_outer, int) or n_outer < 1:
            raise ConfigurationError(f"n_outer must be an int >= 1; got {n_outer!r}.")
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = as_covariance(prior_cov_op)
        self.obs_cov_op = None if obs_cov_op is None else as_covariance(obs_cov_op)
        self.config = config
        self.solver = solver
        self.n_outer = n_outer
        self.solver_adjoint = solver_adjoint or optx.RecursiveCheckpointAdjoint()
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

        obs_covariance = require_obs_cov(self.obs_cov_op)

        def analyse_window_obs(
            window_obs: Float[Array, "T_plus_1 M"],
            window_mask: Float[Array, "T_plus_1 M"],
        ) -> Float[Array, " N"]:
            def step_factor(step_mask: Float[Array, " M"]) -> lx.AbstractLinearOperator:
                # L̃_ε,t L̃_ε,tᵀ = Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ), factored once per
                # window, so masked entries drop out of the likelihood
                return masked_cholesky(obs_covariance, step_mask)

            factors = eqx.filter_vmap(step_factor)(window_mask)

            def whiten_obs(
                residuals: Float[Array, "T_plus_1 M"],
            ) -> Float[Array, "T_plus_1 M"]:
                def whiten_step(
                    step_residual: Float[Array, " M"],
                    factor: lx.AbstractLinearOperator,
                ) -> Float[Array, " M"]:
                    # L̃_ε,t⁻¹ rₜ
                    return gaussx.solve(factor, step_residual)

                return eqx.filter_vmap(whiten_step)(residuals, factors)

            # ŷₜ(x₀) = mₜ ⊙ h(xₜ), shared with the posterior adapters
            observed = observed_trajectory(self, window_mask)
            # mₜ ⊙ yₜ
            masked_observations = window_mask * window_obs

            def residual(
                initial_state: Float[Array, " N"],
            ) -> Float[Array, "T_plus_1 M"]:
                # rₜ = mₜ ⊙ yₜ − ŷₜ(x₀)
                return masked_observations - observed(initial_state)

            return self._gauss_newton(residual, whiten_obs)

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
            # the residual is already whitened: L̃_ε = I
            return residuals

        return self._gauss_newton(residual, identity)

    def _gauss_newton(
        self,
        residual: Callable[[Array], Array],
        whiten_obs: Callable[[Array], Array],
    ) -> Float[Array, " N"]:
        """Gauss–Newton on the whitened residuals of one window, by ``solver``.

            𝒥(c) = ½ ‖r_b(c)‖² + ½ ‖L̃_ε⁻¹ r(x₀(c))‖²
            (J̃ᵀ J̃) δc = −J̃ᵀ r̃(c⁽ⁱ⁾),   c⁽ⁱ⁺¹⁾ = c⁽ⁱ⁾ + δc

        Algorithm:
            1. Pick the control c and its map to the state: c = z with
               x₀ = μ_x + L_x z, L_x L_xᵀ = Σ_x (control-variable transform), where
               r_b = z; or c = x₀ (state space), where r_b = L_x⁻¹ (x₀ − μ_x) with the
               Cholesky factor L_x of Σ_x.
            2. Stack r̃(c) = (r_b(c), L̃_ε⁻¹ r(x₀(c))).
            3. Run ``optimistix.least_squares`` with ``solver`` for at most
               ``n_outer`` increments.

        Args:
            residual: x₀ ↦ r(x₀), masked or whitened observation misfits.
            whiten_obs: r ↦ L̃_ε⁻¹ r, with L̃_ε L̃_εᵀ = Σ̃_ε.

        Returns:
            The analysis x₀⋆.
        """
        config = self.config
        prior_mean = self.prior_mean

        if config.cvt:
            prior_sqrt = self._prior_sqrt()

            def to_state(control: Float[Array, " C"]) -> Float[Array, " N"]:
                # x₀ = μ_x + L_x z
                return prior_mean + prior_sqrt.mv(control)

            def background_residual(control: Float[Array, " C"]) -> Array:
                # r_b = z, so 𝒥_b = ½ ‖z‖²
                return control

            # z has the input size of L_x: D_x for a square root, D_z ≪ D_x for features
            control = jnp.zeros(prior_sqrt.in_structure().shape, prior_mean.dtype)
        else:
            whiten_prior = whitener(self.prior_cov_op)

            def to_state(control: Float[Array, " C"]) -> Float[Array, " N"]:
                return control

            def background_residual(control: Float[Array, " C"]) -> Array:
                # r_b = L_x⁻¹ (x₀ − μ_x), so 𝒥_b = ½ (x₀ − μ_x)ᵀ Σ_x⁻¹ (x₀ − μ_x)
                return whiten_prior(control - prior_mean)

            control = prior_mean

        def whitened_residuals(
            control: Float[Array, " C"], _args: Any
        ) -> tuple[Array, Array]:
            # r̃(c) = (r_b(c), L̃_ε⁻¹ r(x₀(c)))
            return background_residual(control), whiten_obs(residual(to_state(control)))

        solution = least_squares(
            self.solver,
            whitened_residuals,
            control,
            adjoint=self.solver_adjoint,
            # each optimistix step linearises at the accepted iterate and proposes
            # the next, so n_outer increments take n_outer + 1 steps
            max_steps=self.n_outer + 1,
            jac=config.jac,
        )
        return to_state(solution.value)

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
        the solve is, Σ_x|y = L_x (I + (J L_x)ᵀ Λ̃_ε J L_x)⁻¹ L_xᵀ, so Σ_x⁻¹ is never
        applied; its CG solves use ``config.covariance_tol``, or the deprecated
        ``cg_atol``, ``cg_rtol`` and ``n_inner`` when the configuration sets them.

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

        config = self.config
        if adapter is None and config.cvt:
            # the control-space precision is the inner Hessian: solve it to the
            # deprecated inner tolerances when the budget is set, as before, else to
            # covariance_tol, with at least the default step budget
            if config.deprecated_budget():
                atol = _CG_TOL if config.cg_atol is None else config.cg_atol
                rtol = _CG_TOL if config.cg_rtol is None else config.cg_rtol
            else:
                atol = rtol = config.covariance_tol
            inner_steps = _N_INNER if config.n_inner is None else config.n_inner
            solver = lx.CG(
                atol=atol, rtol=rtol, max_steps=max(inner_steps, DEFAULT_MAX_STEPS)
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
