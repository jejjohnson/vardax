"""Strong-constraint 4D-Var: the forward model is exact, the control is x₀.

μ_x and Σ_x are the prior mean and covariance, Σ_ε the observation-error
covariance, f the forward model and h the observation operator.
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
from vardax._src.minimisers import minimise
from vardax._src.models._analysis_step import analyse_window, rollout_end
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


class StrongFourDVar(eqx.Module):
    """Strong-constraint 4D-Var analysis of an assimilation window.

        𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖rₜ‖²_Σ̃_ε,t⁻¹
        rₜ = mₜ ⊙ (yₜ − h(xₜ)),   xₜ = f(xₜ₋₁),   t = 0, …, T
        Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ),   Mₜ = diag(mₜ)

    The forward model f is exact, so the only control is the initial state x₀.

    Algorithm:
        1. For each window, under ``jax.vmap``, start from the prior mean μ_x,
           or from the initial control of ``parameterization``.
        2. Roll f out from x₀ for T steps, with optional ``jax.checkpoint``
           rematerialisation.
        3. Sum the observation term over t = 0, …, T; a point batch uses its
           σᵢ-weighted cost at each ``time_index`` instead.
        4. Add the background term ½ ‖x₀ − μ_x‖²_Σ_x⁻¹, or the penalty ℛ(θ) of
           ``parameterization``.
        5. Minimise over x₀ with ``minimiser``, differentiating through the
           rollout by reverse-mode autodiff.

    Reverse mode stores every intermediate of every step by default, and two
    slots bound that memory. ``checkpoint`` recomputes the steps inside each
    block and works for any per-step forward. ``forward_adjoint`` hands a
    ``diffrax.AbstractAdjoint`` to a forward that integrates with diffrax
    ([`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint]).

    Attributes:
        forward: Forward model f, a ``pipekit_cycle.ForwardModel`` supplying
            ``step(state, dt) -> state``.
        obs_op: Observation operator h.
        prior_mean: Prior mean μ_x of the state at t = 0.
        prior_cov_op: Prior covariance Σ_x.
        obs_cov_op: Observation-error covariance Σ_ε.
        minimiser: ``optimistix.AbstractMinimiser`` over x₀, by default
            ``optx.BFGS(rtol=1e-6, atol=1e-6)``, or a
            [`Minimiser`][vardax.Minimiser] such as
            [`OptaxMinimiser`][vardax.OptaxMinimiser] for a large control.
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for differentiating
            through the minimum.
        forward_adjoint: ``diffrax.AbstractAdjoint`` passed to
            ``forward.with_adjoint(...)`` before the rollout. A forward without
            that method raises ``UnsupportedInputError``. Default ``None`` uses
            the forward's own.
        max_steps: Iteration cap on the outer solver when ``minimiser`` is an
            ``optimistix`` solver; a vardax `Minimiser` keeps its own.
        cov_solver: ``lineax`` solver for Σ_x⁻¹ and Σ_ε⁻¹ in the cost. The
            default ``lx.CG(atol=1e-6, rtol=1e-6)`` suits large matrix-free
            operators but is inexact on an ill-conditioned covariance; pass
            ``lx.AutoLinearSolver(well_posed=True)`` or ``lx.Cholesky()`` for
            an exact solve when the operator can be materialised.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization]. The control
            is then θ, with x₀ = g_dec(θ) reshaped to ``prior_mean.shape``, and
            ℛ(θ) replaces the background term, so ``prior_mean`` and
            ``prior_cov_op`` are not used. Default ``None``: the control is x₀.
        checkpoint: Rematerialise the rollout in reverse mode. ``False`` stores
            every step's intermediates, ``True`` wraps each step in
            ``jax.checkpoint``, and an int k wraps blocks of k steps. The
            gradient is the same, for less memory and about one extra forward
            run.

    Examples:
        A decaying forward f(x) = x / 2 with observations at t = 0 and t = 1
        both inform x₀. With μ_x = 0 and Σ_x = Σ_ε = I, the analysis is
        x₀⋆ = (1 + ½) / (1 + 1 + ¼) = ⅔:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Decay:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
        >>> window = vardax.Batch1D(input=jnp.ones((1, 2, 3)), mask=jnp.ones((1, 2, 3)))
        >>> decay = vardax.StrongFourDVar(
        ...     forward=Decay(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> decay(window).shape
        (1, 3)
        >>> bool(jnp.allclose(decay(window), 2 / 3, atol=1e-3))
        True

        ``checkpoint=True`` gives the same analysis, and the Laplace posterior
        of the window is Σ_x|y = 1 / 2.25:

        >>> remat = vardax.StrongFourDVar(
        ...     forward=Decay(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     checkpoint=True,
        ... )
        >>> bool(jnp.allclose(remat(window), decay(window), atol=1e-4))
        True
        >>> post = remat.posterior(window)
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(3)), 1 / 2.25, atol=1e-3))
        True

        A forward without ``with_adjoint`` cannot take a ``forward_adjoint``:

        >>> vardax.StrongFourDVar(
        ...     forward=Decay(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     forward_adjoint=object(),
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        UnsupportedInputError: forward_adjoint was set but Decay has no ...
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator | None
    minimiser: optx.AbstractMinimiser | Minimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    forward_adjoint: Any = eqx.field(static=True)
    cov_solver: lx.AbstractLinearSolver = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)
    parameterization: StateParameterization | None = None
    checkpoint: bool | int = eqx.field(static=True, default=False)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, " N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator | None,
        minimiser: optx.AbstractMinimiser | Minimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        forward_adjoint: Any = None,
        max_steps: int = 200,
        parameterization: StateParameterization | None = None,
        checkpoint: bool | int = False,
        cov_solver: lx.AbstractLinearSolver | None = None,
    ) -> None:
        # fail at construction, not on the first traced call
        checkpoint_every(checkpoint)
        resolve_forward(forward, forward_adjoint)
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = as_covariance(prior_cov_op)
        self.obs_cov_op = None if obs_cov_op is None else as_covariance(obs_cov_op)
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.forward_adjoint = forward_adjoint
        self.max_steps = max_steps
        self.cov_solver = cov_solver or lx.CG(atol=1e-6, rtol=1e-6)
        self.parameterization = parameterization
        self.checkpoint = checkpoint

    def _rollout(
        self, initial_state: Float[Array, " N"], n_steps: int
    ) -> Float[Array, "T N"]:
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
        """Analyse each window: x₀⋆ = argmin_x₀ 𝒥(x₀).

        A [`PointObsBatch`][vardax.PointObsBatch] needs
        ``obs_op=GridPointObs(...)``: observation i is compared with the
        trajectory state x_τᵢ, τᵢ = ``time_index[i]``, over a window of
        ``n_times`` states, with error variance σᵢ² (``obs_cov_op`` must be
        ``None`` or an identity).

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] whose time axis holds
                the window, or a [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The analysed initial state x₀⋆ of every window.

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
            num_steps = window_steps(batch)

            def analyse_points(window: PointObsBatch) -> Float[Array, " N"]:
                def obs_cost(initial_state: Float[Array, " N"]) -> Float[Array, ""]:
                    # xₜ = f(xₜ₋₁)
                    trajectory = self._rollout(initial_state, n_steps=num_steps)
                    # 𝒥_o = ½ ∑ᵢ ‖rᵢ / σᵢ‖²
                    return grid_point_obs_cost(self.obs_op, trajectory, window)

                return self._solve(obs_cost)

            return map_point_batch(analyse_points, batch)

        obs_covariance = require_obs_cov(self.obs_cov_op)
        # the window holds T + 1 times, so T steps
        num_steps = batch.input.shape[1] - 1

        def analyse_window_obs(
            window_obs: Float[Array, "T_plus_1 M"],
            window_mask: Float[Array, "T_plus_1 M"],
        ) -> Float[Array, " N"]:
            def obs_cost(initial_state: Float[Array, " N"]) -> Float[Array, ""]:
                # xₜ = f(xₜ₋₁)
                trajectory = self._rollout(initial_state, n_steps=num_steps)

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

            return self._solve(obs_cost)

        return jax.vmap(analyse_window_obs)(
            fill_gaps(batch.input, batch.mask), batch.mask
        )

    def _solve(
        self,
        obs_cost: Callable[[Array], Array],
    ) -> Float[Array, " N"]:
        """Minimise the background term or penalty, plus 𝒥_o(x₀).

        Args:
            obs_cost: The observation term x₀ ↦ 𝒥_o(x₀).

        Returns:
            The analysed initial state x₀⋆.
        """
        parameterization = self.parameterization
        if parameterization is not None:

            def decode(params: Any) -> Float[Array, " N"]:
                # x₀ = g_dec(θ)
                decoded = parameterization.decode(params)
                return as_shape(decoded, self.prior_mean.shape)

            def params_cost(params: Any, _args: Any) -> Float[Array, ""]:
                # ℛ(θ)
                penalty = parameterization.penalty(params)
                # x₀ = g_dec(θ)
                initial_state = decode(params)
                # 𝒥(θ) = ℛ(θ) + 𝒥_o(g_dec(θ))
                return penalty + obs_cost(initial_state)

            params_star = minimise(
                self.minimiser,
                params_cost,
                parameterization.init_control(),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            return decode(params_star)

        def cost(initial_state: Float[Array, " N"], _args: Any) -> Float[Array, ""]:
            # δx = x₀ − μ_x
            increment = initial_state - self.prior_mean
            # 𝒥_b = ½ δxᵀ Σ_x⁻¹ δx
            background_cost = half_weighted_norm(
                self.prior_cov_op, increment, self.cov_solver
            )
            # 𝒥(x₀) = 𝒥_b + 𝒥_o(x₀)
            return background_cost + obs_cost(initial_state)

        return minimise(
            self.minimiser,
            cost,
            self.prior_mean,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the
        default adapter is [`LaplaceCovariance`][vardax.LaplaceCovariance]
        with this model's Σ_x and Σ_ε.

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

        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _StrongFourDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        Returns:
            An analysis step that treats non-finite observations as
            missing and returns the analysed initial state.
        """
        return _StrongFourDVarAnalysisStep(self)


class _StrongFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``StrongFourDVar``.

    Analyses the window around the cycle's forecast, with the cycle's h and Σ_ε
    when given, and returns the analysed state at the end of the window, the
    state the cycle steps forward from (``analyse_window``).

    Attributes:
        model: The model to analyse with.
    """

    model: StrongFourDVar

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
            obs_err_cov: The cycle's observation-error covariance Σ_ε, or
                ``None``.

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
