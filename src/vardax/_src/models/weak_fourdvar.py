"""Weak-constraint 4D-Var: the forward model errs, the control is w = (x₀, η₁, …, η_T).

μ_x and Σ_x are the prior mean and covariance, Σ_ε the observation-error covariance,
Σ_η the model-error covariance, f the forward model and h the observation operator.
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
import optimistix as optx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.basis.model_error import ModelErrorBasis
from vardax._src.errors import ConfigurationError
from vardax._src.minimisers import minimise
from vardax._src.models._analysis_step import analyse_window
from vardax._src.models._observations import fill_gaps
from vardax._src.models._rollout import (
    accepts_mask,
    checkpoint_every,
    resolve_forward,
    rollout,
)
from vardax._src.obs_operators.point import (
    check_point_obs_cov,
    grid_point_obs_cost,
    map_point_batch,
    require_obs_cov,
    window_steps,
)
from vardax._src.protocols import Minimiser, StateParameterization
from vardax._src.utils.covariance import (
    as_covariance,
    half_weighted_norm,
)
from vardax._src.utils.shapes import as_shape


class WeakFourDVar(eqx.Module):
    """Weak-constraint 4D-Var analysis of an assimilation window.

        𝒥(w) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖rₜ‖²_Σ̃_ε,t⁻¹ + ½ ∑ₜ ‖ηₜ‖²_Σ_η⁻¹
        w = (x₀, η₁, …, η_T),   xₜ = f(xₜ₋₁) + ηₜ
        rₜ = mₜ ⊙ (yₜ − h(xₜ))
        Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ),   Mₜ = diag(mₜ)

    The control has (T + 1) D_x entries, so a long window costs more than
    [`StrongFourDVar`][vardax.StrongFourDVar]; use it when the free model f is known to
    have biases.

    Algorithm:
        1. Start from x₀ = μ_x and zero model errors.
        2. Roll f out with the forcing ηₜ added after every step.
        3. Sum the observation term over the trajectory, the Σ_η⁻¹-weighted model-error
           term over the T steps, and the background term or the penalty ℛ(θ) of
           ``parameterization``.
        4. Minimise jointly over w with ``minimiser`` and return the pair (x₀⋆, η⋆).

    Attributes:
        forward: Forward model f, a ``pipekit_cycle.ForwardModel``.
        obs_op: Observation operator h.
        prior_mean: Prior mean μ_x of the state at t = 0.
        prior_cov_op: Prior covariance Σ_x.
        obs_cov_op: Observation-error covariance Σ_ε.
        model_err_cov_op: Model-error covariance Σ_η of each ηₜ, which is then estimated
            at every step and cell. Give exactly one of this and ``model_err_basis``.
        minimiser: As in [`StrongFourDVar`][vardax.StrongFourDVar].
        minimiser_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar].
        max_steps: As in [`StrongFourDVar`][vardax.StrongFourDVar].
        cov_solver: ``lineax`` solver for Σ_x⁻¹, Σ_ε⁻¹ and Σ_η⁻¹ in the cost. The
            default ``lx.CG(atol=1e-6, rtol=1e-6)`` suits large matrix-free operators
            but is inexact on an ill-conditioned covariance (for example a tiny Σ_ε);
            pass ``lx.AutoLinearSolver(well_posed=True)`` or ``lx.Cholesky()`` for an
            exact solve when the operator can be materialised.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization] of the initial
            state. The control is then (θ, η₁, …, η_T), with x₀ = g_dec(θ), and ℛ(θ)
            replaces the background term, so ``prior_mean`` and ``prior_cov_op`` are not
            used. Default ``None``: the control is w.
        forward_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]: a
            ``diffrax.AbstractAdjoint`` for a forward implementing
            [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint].
        checkpoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]: ``False``, ``True``
            or a block length k for ``jax.checkpoint`` over the rollout.
        model_err_basis: A [`ModelErrorBasis`][vardax.ModelErrorBasis] in place of
            ``model_err_cov_op`` (reduced weak-constraint 4D-Var), holding χ, Φ, Σ_time
            and Σ_space. The model error is η = χ W Φᵀ with
            W = Σ_time^½ ω Σ_space^½, the control is (x₀, ω) with C P model-error
            unknowns, the model-error term is ½ ‖ω‖², and the call returns (x₀⋆, W⋆). It
            composes with ``parameterization``.

    Raises:
        ConfigurationError: If not exactly one of ``model_err_cov_op`` and
            ``model_err_basis`` is given, or ``checkpoint`` is neither a bool nor an
            int ≥ 1.
        UnsupportedInputError: If ``forward_adjoint`` is set and ``forward`` has no
            ``with_adjoint`` method.

    Examples:
        Trivial dynamics and a single time: no model-error steps remain (T = 0), and
        with Σ_x = Σ_ε = I the analysis reduces to the 3D-Var answer x₀⋆ = y / 2:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Identity:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
        >>> weak = vardax.WeakFourDVar(
        ...     forward=Identity(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     model_err_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 1, 3)), mask=jnp.ones((1, 1, 3)))
        >>> x0_star, eta_star = weak(batch)
        >>> x0_star.shape, eta_star.shape
        ((1, 3), (1, 0, 3))

        A two-time window has one model-error step. With a tiny Σ_η the free model is
        almost exact, and the analysis approaches the strong-constraint x₀⋆ = ⅔ for
        f(x) = x / 2:

        >>> class Decay:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        >>> tiny = lx.TaggedLinearOperator(
        ...     lx.DiagonalLinearOperator(jnp.full(3, 1e-4)),
        ...     lx.positive_semidefinite_tag,
        ... )
        >>> weak = vardax.WeakFourDVar(
        ...     forward=Decay(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     model_err_cov_op=tiny,
        ... )
        >>> window = vardax.Batch1D(input=jnp.ones((1, 2, 3)), mask=jnp.ones((1, 2, 3)))
        >>> x0_star, eta_star = weak(window)
        >>> eta_star.shape
        (1, 1, 3)
        >>> bool(jnp.allclose(x0_star, 2 / 3, atol=5e-2))
        True

        Composed with the Laplace adapter, ``posterior`` returns the x₀ block of the
        inverse precision over w:

        >>> weak.posterior(window).cov.mv(jnp.ones(3)).shape
        (3,)
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator | None
    model_err_cov_op: lx.AbstractLinearOperator | None
    minimiser: optx.AbstractMinimiser | Minimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    cov_solver: lx.AbstractLinearSolver = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)
    parameterization: StateParameterization | None = None
    forward_adjoint: Any = eqx.field(static=True, default=None)
    checkpoint: bool | int = eqx.field(static=True, default=False)
    model_err_basis: ModelErrorBasis | None = None

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, " N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator | None,
        model_err_cov_op: lx.AbstractLinearOperator | None = None,
        minimiser: optx.AbstractMinimiser | Minimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
        parameterization: StateParameterization | None = None,
        forward_adjoint: Any = None,
        checkpoint: bool | int = False,
        cov_solver: lx.AbstractLinearSolver | None = None,
        model_err_basis: ModelErrorBasis | None = None,
    ) -> None:
        # fail at construction, not on the first traced call
        checkpoint_every(checkpoint)
        if (model_err_cov_op is None) == (model_err_basis is None):
            raise ConfigurationError(
                "WeakFourDVar takes exactly one of model_err_cov_op (a full Q "
                "over every step) and model_err_basis (a reduced model error)."
            )
        resolve_forward(forward, forward_adjoint)
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = as_covariance(prior_cov_op)
        self.obs_cov_op = None if obs_cov_op is None else as_covariance(obs_cov_op)
        self.model_err_cov_op = (
            None if model_err_cov_op is None else as_covariance(model_err_cov_op)
        )
        self.model_err_basis = model_err_basis
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps
        self.cov_solver = cov_solver or lx.CG(atol=1e-6, rtol=1e-6)
        self.parameterization = parameterization
        self.forward_adjoint = forward_adjoint
        self.checkpoint = checkpoint

    def _rollout_with_error(
        self,
        initial_state: Float[Array, " N"],
        model_errors: Float[Array, "T N"],
    ) -> Float[Array, "T_plus_1 N"]:
        """Roll out with per-step model error: xₜ = f(xₜ₋₁) + ηₜ.

        Args:
            initial_state: Initial state x₀.
            model_errors: Model errors η₁, …, η_T.

        Returns:
            The trajectory, including x₀.
        """
        forward = resolve_forward(self.forward, self.forward_adjoint)
        num_steps = model_errors.shape[0]
        return rollout(
            forward,
            initial_state,
            num_steps,
            forcing=model_errors,
            checkpoint=self.checkpoint,
        )

    def __call__(
        self,
        batch: Batch1D | PointObsBatch,
    ) -> tuple[Float[Array, "*batch N"], Float[Array, ...]]:
        """Analyse each window: w⋆ = argmin_w 𝒥(w).

        A [`PointObsBatch`][vardax.PointObsBatch] needs ``obs_op=GridPointObs(...)``:
        observation i is compared with the trajectory state x_τᵢ,
        τᵢ = ``time_index[i]``, over a window of ``n_times`` states (so
        T = ``n_times`` − 1), with error variance σᵢ² (``obs_cov_op`` must be ``None``
        or an identity).

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] whose time axis holds the
                window, or a [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The pair (x₀⋆, η⋆) of every window: the analysed initial state and the T
            model errors. With ``model_err_basis`` the second element is the
            coefficients W⋆.

        Raises:
            ObservationError: For a point batch with a non-trivial ``obs_cov_op`` or an
                out-of-range ``time_index``.
            MissingInputError: For a gridded batch when ``obs_cov_op`` is ``None``, or a
                point batch with ``time_index`` but no ``n_times``.
            InputTypeError: For a point batch whose ``obs_op`` is not a
                ``GridPointObs``.
            ShapeError: If ``model_err_basis`` has a different number of model-error
                steps from the window.
        """
        state_size = self.prior_mean.shape[0]
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            num_steps = window_steps(batch)

            def analyse_points(window: PointObsBatch):
                def obs_cost(
                    trajectory: Float[Array, "T_plus_1 N"],
                ) -> Float[Array, ""]:
                    # 𝒥_o = ½ ∑ᵢ ‖rᵢ / σᵢ‖²
                    return grid_point_obs_cost(self.obs_op, trajectory, window)

                return self._solve(obs_cost, num_steps, state_size)

            return map_point_batch(analyse_points, batch)

        obs_covariance = require_obs_cov(self.obs_cov_op)
        # the window holds T + 1 times, so T model-error steps
        num_times = batch.input.shape[1]
        num_steps = num_times - 1

        def analyse_window_obs(
            window_obs: Float[Array, "T_plus_1 M"],
            window_mask: Float[Array, "T_plus_1 M"],
        ):
            def obs_cost(
                trajectory: Float[Array, "T_plus_1 N"],
            ) -> Float[Array, ""]:
                def step_cost(state, observation, step_mask):
                    # h(xₜ)
                    predicted_obs = (
                        self.obs_op(state, mask=step_mask)
                        if accepts_mask(self.obs_op)
                        else self.obs_op(state)
                    )
                    # rₜ = mₜ ⊙ (yₜ − h(xₜ))
                    residual = step_mask * (observation - predicted_obs)
                    # Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ), so masked entries drop out
                    # of the likelihood
                    masked_covariance = gaussx.masked_obs_covariance(
                        obs_covariance, step_mask
                    )
                    # 𝒥_o,t = ½ rₜᵀ Σ̃_ε,t⁻¹ rₜ
                    return half_weighted_norm(
                        masked_covariance, residual, self.cov_solver
                    )

                step_costs = jax.vmap(step_cost)(trajectory, window_obs, window_mask)
                # 𝒥_o = ∑ₜ 𝒥_o,t
                return jnp.sum(step_costs)

            return self._solve(obs_cost, num_steps, state_size)

        analysis, model_error_analysis = jax.vmap(analyse_window_obs)(
            fill_gaps(batch.input, batch.mask), batch.mask
        )
        return analysis, model_error_analysis

    def _solve(
        self,
        obs_cost: Callable[[Array], Array],
        num_steps: int,
        state_size: int,
    ) -> tuple[Float[Array, " N"], Array]:
        """Minimise over w, or (θ, η₁, …, η_T), with ω in place of η for a basis.

        Args:
            obs_cost: The observation term of a trajectory, x₀, …, x_T ↦ 𝒥_o.
            num_steps: Number of model-error steps T.
            state_size: Length D_x of one state.

        Returns:
            The pair (x₀⋆, η⋆), or (x₀⋆, W⋆) with ``model_err_basis``.

        Raises:
            ConfigurationError: If the model has neither a Σ_η nor a basis.
            ShapeError: If ``model_err_basis`` has a different number of steps.
        """
        basis = self.model_err_basis
        model_noise_covariance = self.model_err_cov_op
        if model_noise_covariance is not None:
            init_model_errors = jnp.zeros((num_steps, state_size))

            def to_errors(error_control: Array) -> Float[Array, "T N"]:
                return error_control

            def model_error_cost(error_control: Array) -> Float[Array, ""]:
                def step_cost(step_error):
                    # 𝒥_η,t = ½ ηₜᵀ Σ_η⁻¹ ηₜ
                    return half_weighted_norm(
                        model_noise_covariance, step_error, self.cov_solver
                    )

                # a one-state window has no model-error steps, and vmap over an
                # empty axis would trip lineax's CG on a zero-size operand
                if error_control.shape[0] == 0:
                    return jnp.zeros((), dtype=error_control.dtype)
                step_costs = jax.vmap(step_cost)(error_control)
                # 𝒥_η = ∑ₜ 𝒥_η,t
                return jnp.sum(step_costs)

            def to_output(error_control: Array) -> Array:
                return error_control

        elif basis is not None:
            basis.check_steps(num_steps)
            # ω, the whitened coefficients
            init_model_errors = jnp.zeros(basis.coefficient_shape)
            to_errors = basis.errors

            def model_error_cost(error_control: Array) -> Float[Array, ""]:
                # 𝒥_η = ½ ‖ω‖²
                return 0.5 * jnp.sum(error_control**2)

            # W = Σ_time^½ ω Σ_space^½
            to_output = basis.coefficients
        else:
            raise ConfigurationError(
                "WeakFourDVar needs model_err_cov_op or model_err_basis."
            )

        def model_and_obs_cost(
            initial_state: Float[Array, " N"],
            error_control: Array,
        ) -> Float[Array, ""]:
            # η₁, …, η_T
            model_errors = to_errors(error_control)
            # xₜ = f(xₜ₋₁) + ηₜ
            trajectory = self._rollout_with_error(initial_state, model_errors)
            # 𝒥_o
            observation_cost = obs_cost(trajectory)
            # 𝒥_o + 𝒥_η
            return observation_cost + model_error_cost(error_control)

        parameterization = self.parameterization
        if parameterization is not None:

            def decode(params: Any) -> Float[Array, " N"]:
                # x₀ = g_dec(θ)
                decoded = parameterization.decode(params)
                return as_shape(decoded, self.prior_mean.shape)

            def params_cost(control: Any, _args: Any) -> Float[Array, ""]:
                params, error_control = control
                # ℛ(θ)
                penalty = parameterization.penalty(params)
                # x₀ = g_dec(θ)
                initial_state = decode(params)
                # 𝒥 = ℛ(θ) + 𝒥_o + 𝒥_η
                return penalty + model_and_obs_cost(initial_state, error_control)

            params_star, error_star = minimise(
                self.minimiser,
                params_cost,
                (parameterization.init_control(), init_model_errors),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            return decode(params_star), to_output(error_star)

        def background_term(initial_state: Float[Array, " N"]) -> Float[Array, ""]:
            # δx₀ = x₀ − μ_x
            increment = initial_state - self.prior_mean
            # 𝒥_b = ½ δx₀ᵀ Σ_x⁻¹ δx₀
            return half_weighted_norm(self.prior_cov_op, increment, self.cov_solver)

        if basis is not None:

            def reduced_cost(control: Any, _args: Any) -> Float[Array, ""]:
                # the control is (x₀, ω)
                initial_state, error_control = control
                # 𝒥_b
                background_cost = background_term(initial_state)
                # 𝒥 = 𝒥_b + 𝒥_o + 𝒥_η
                return background_cost + model_and_obs_cost(
                    initial_state, error_control
                )

            initial_state_star, error_star = minimise(
                self.minimiser,
                reduced_cost,
                (self.prior_mean, init_model_errors),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            return initial_state_star, to_output(error_star)

        def cost(control: Float[Array, "T_plus_1 N"], _args: Any) -> Float[Array, ""]:
            # w = (x₀, η₁, …, η_T), stacked as rows
            initial_state = control[0]
            model_errors = control[1:]
            # 𝒥_b
            background_cost = background_term(initial_state)
            # 𝒥(w) = 𝒥_b + 𝒥_o + 𝒥_η
            return background_cost + model_and_obs_cost(initial_state, model_errors)

        # w⁽⁰⁾ = [μ_x; 0, …, 0]; jnp, not einx: einx cannot size T = 0
        initial_state = jnp.expand_dims(self.prior_mean, 0)
        initial_control = jnp.concatenate([initial_state, init_model_errors])

        control_star = minimise(
            self.minimiser,
            cost,
            initial_control,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )
        return control_star[0], control_star[1:]

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the default adapter
        is [`LaplaceCovariance`][vardax.LaplaceCovariance] with this model's Σ_x and
        Σ_ε. The covariance is that of x₀: the x₀ block of the inverse Gauss–Newton
        precision over w.

        Args:
            batch: A ``Batch1D`` holding one window.
            adapter: Optional [`PosteriorAdapter`][vardax.PosteriorAdapter],
                e.g. [`GaussNewtonHessian`][vardax.GaussNewtonHessian].

        Returns:
            The window's posterior: mean and lazy covariance.

        Raises:
            ShapeError: If ``batch`` holds more than one window.
            UnsupportedInputError: For a point batch, or for a model with a
                ``parameterization`` or a ``model_err_basis``.
        """
        # lazy: the posterior adapters import the models
        from vardax._src.posterior._precision import model_posterior

        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _WeakFourDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        Returns:
            An analysis step that treats non-finite observations as missing and returns
            the state at the end of the window, rolled out with the analysed model
            errors.
        """
        return _WeakFourDVarAnalysisStep(self)


def _weak_rollout_end(
    model: WeakFourDVar, analysis: tuple[Array, Array], n_steps: int
) -> Float[Array, " N"]:
    """xₜ = f(xₜ₋₁) + ηₜ from the analysed (x₀⋆, η⋆) of one window, at its end."""
    initial_states, errors = analysis
    basis = model.model_err_basis
    # η, or χ W Φᵀ for a reduced model error
    model_errors = errors[0] if basis is None else basis.pattern(errors[0])
    trajectory = model._rollout_with_error(initial_states[0], model_errors)
    return trajectory[n_steps]


class _WeakFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``WeakFourDVar``.

    Analyses the window around the cycle's forecast (``analyse_window``) and returns the
    state at its end, rolled out with the analysed model errors; the model errors
    themselves come from a direct ``model(batch)``.

    Attributes:
        model: The model to analyse with.
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
            _weak_rollout_end,
            forecast=forecast,
            obs=obs,
            obs_op=obs_op,
            obs_err_cov=obs_err_cov,
        )
