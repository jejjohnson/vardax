"""Strong-constraint 4D-Var on the full nonlinear cost, optionally in a reduced basis.

There is no linearisation, outer loop or inner CG, which is what separates it from
[`IncrementalFourDVar`][vardax.IncrementalFourDVar]; the control is x₀ or the
coefficients z of a reduced basis with x₀ = μ_x + Φ z.
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
from vardax._src.errors import ConfigurationError
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


class ReducedBasisFourDVar(eqx.Module):
    """Strong-constraint 4D-Var with the control in a reduced basis.

        𝒥(z) = ½ zᵀ Σ_z⁻¹ z + 𝒥_o(μ_x + Φ z)
        𝒥_o(x₀) = ½ ∑ₜ ‖rₜ‖²_Σ̃_ε,t⁻¹
        rₜ = mₜ ⊙ (yₜ − h(xₜ)),   xₜ = f(xₜ₋₁),   t = 0, …, T
        Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ),   Mₜ = diag(mₜ)

    Without a basis the control is x₀ itself and the first term is
    ½ ‖x₀ − μ_x‖²_Σ_x⁻¹. The class was called ``Incremental4DVar`` until vardax 0.1.10,
    after MASSH's name for the increment δx₀ = Φ z. In vardax "incremental" means
    Gauss–Newton, so the old name is a deprecated alias.

    Algorithm:
        1. Pick the control and its map to the initial state: x₀ itself, the
           coefficients z with x₀ = μ_x + Φ z (``basis``), or θ with x₀ = g_dec(θ)
           (``parameterization``).
        2. Roll f out from x₀ for T steps.
        3. Add the prior term (Σ_x⁻¹, ``basis.prior_inv`` or the penalty ℛ(θ)) to the
           observation term summed over the trajectory; Σ̃_ε,t marginalises masked
           entries out, so a correlated Σ_ε is not biased by the gaps.
        4. Minimise the full nonlinear cost with ``minimiser`` and decode the minimum
           to x₀⋆.

    Attributes:
        forward: Forward model f, a ``pipekit_cycle.ForwardModel`` supplying
            ``step(state, dt) -> state``.
        obs_op: Observation operator h, linear or nonlinear. Supplies ``__call__`` and
            ``linearize``; ``__call__`` may accept a ``mask`` keyword (`MaskedIdentity`
            and `InterpObs` do).
        prior_mean: Prior mean μ_x of the state at t = 0.
        prior_cov_op: Prior covariance Σ_x, used only when ``basis`` is ``None``.
        obs_cov_op: Observation-error covariance Σ_ε.
        basis: Optional `ReducedBasis`. The control is then the coefficient vector z
            (``basis.nbasis`` entries) and Σ_z⁻¹, applied by ``basis.prior_inv``,
            replaces Σ_x⁻¹, the same as
            ``parameterization=LinearParameterization(basis, prior_mean)``.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization]. The control is
            then θ, with x₀ = g_dec(θ), and ℛ(θ) replaces the background term.
            Exclusive with ``basis``.
        minimiser: ``optimistix.AbstractMinimiser`` over the control, by default
            ``optx.BFGS(rtol=1e-6, atol=1e-6)``, or a
            [`Minimiser`][vardax.Minimiser] such as
            [`OptaxMinimiser`][vardax.OptaxMinimiser].
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for differentiating through
            the minimum, by default ``optx.ImplicitAdjoint``.
        max_steps: Iteration cap on the outer solver when ``minimiser`` is an
            ``optimistix`` solver; a vardax `Minimiser` keeps its own.
        cov_solver: ``lineax`` solver for Σ_x⁻¹ and Σ_ε⁻¹ in the cost. The default
            ``lx.CG(atol=1e-6, rtol=1e-6)`` suits large matrix-free operators but is
            inexact on an ill-conditioned covariance (for example a tiny Σ_ε); pass
            ``lx.AutoLinearSolver(well_posed=True)`` or ``lx.Cholesky()`` for an exact
            solve when the operator can be materialised.
        forward_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]: a
            ``diffrax.AbstractAdjoint`` for a forward implementing
            [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint].
        checkpoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]: ``False``, ``True``
            or a block length k for ``jax.checkpoint`` over the rollout.

    Raises:
        ConfigurationError: If both ``basis`` and ``parameterization`` are given, or
            ``checkpoint`` is neither a bool nor an int ≥ 1.
        UnsupportedInputError: If ``forward_adjoint`` is set and ``forward`` has no
            ``with_adjoint`` method.

    Examples:
        Trivial dynamics, one time, Σ_x = Σ_ε = I and no basis: the cost is that of
        3D-Var, so x₀⋆ = y / 2:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Identity:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float32))
        >>> common = dict(
        ...     forward=Identity(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(4),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 1, 4)), mask=jnp.ones((1, 1, 4)))
        >>> xa = vardax.ReducedBasisFourDVar(**common)(batch)
        >>> xa.shape
        (1, 4)
        >>> bool(jnp.allclose(xa, 0.5, atol=1e-3))
        True

        With a basis the increment is confined to span(Φ). A single unit-variance
        constant mode gives z⋆ = (∑ₙ yₙ) / (1 + D_x) and x₀⋆ = z⋆ 1:

        >>> basis = vardax.linear_basis(jnp.ones((4, 1)))
        >>> xa = vardax.ReducedBasisFourDVar(**common, basis=basis)(batch)
        >>> bool(jnp.allclose(xa, 4.0 / 5.0, atol=1e-3))
        True

        Composed with a Fourier basis from [`fourier_basis`][vardax.fourier_basis], the
        analysis is still a state-space array:

        >>> fb = vardax.fourier_basis([jnp.linspace(0.0, 1.0, 4)], 2)
        >>> xb = vardax.ReducedBasisFourDVar(**common, basis=fb)(batch)
        >>> xb.shape
        (1, 4)

        ``basis`` and ``parameterization`` are exclusive:

        >>> param = vardax.LinearParameterization(basis, jnp.zeros(4))
        >>> vardax.ReducedBasisFourDVar(
        ...     **common, basis=basis, parameterization=param
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: Pass basis= or parameterization=, not both; ...

    References:
        MASSH ``mapping/src/tools_4Dvar.py`` (``Variational``) and
        ``mapping/src/inv.py`` (``Inv_4Dvar``), reduced-basis 4D-Var with L-BFGS-B.
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator | None
    basis: Any
    parameterization: StateParameterization | None
    minimiser: optx.AbstractMinimiser | Minimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    cov_solver: lx.AbstractLinearSolver = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)
    forward_adjoint: Any = eqx.field(static=True, default=None)
    checkpoint: bool | int = eqx.field(static=True, default=False)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, " N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator | None,
        *,
        basis: Any = None,
        parameterization: StateParameterization | None = None,
        minimiser: optx.AbstractMinimiser | Minimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
        forward_adjoint: Any = None,
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
        if basis is not None and parameterization is not None:
            raise ConfigurationError(
                "Pass basis= or parameterization=, not both; "
                "LinearParameterization(basis, prior_mean) is the basis path."
            )
        self.basis = basis
        self.parameterization = parameterization
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps
        self.cov_solver = cov_solver or lx.CG(atol=1e-6, rtol=1e-6)
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
        """Analyse each window: minimise over x₀, z or θ, and return x₀⋆.

        A [`PointObsBatch`][vardax.PointObsBatch] needs ``obs_op=GridPointObs(...)``:
        observation i is compared with the trajectory state x_τᵢ,
        τᵢ = ``time_index[i]``, over a window of ``n_times`` states, with error
        variance σᵢ² (``obs_cov_op`` must be ``None`` or an identity).

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] whose time axis holds the
                window, or a [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The analysed initial state x₀⋆ of every window.

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

            def analyse_points(window: PointObsBatch) -> Float[Array, " N"]:
                def obs_cost(
                    trajectory: Float[Array, "T_plus_1 N"],
                ) -> Float[Array, ""]:
                    # 𝒥_o = ½ ∑ᵢ ‖rᵢ / σᵢ‖²
                    return grid_point_obs_cost(self.obs_op, trajectory, window)

                return self._solve(obs_cost, num_steps)

            return map_point_batch(analyse_points, batch)

        obs_covariance = require_obs_cov(self.obs_cov_op)
        # the window holds T + 1 times, so T steps
        num_steps = batch.input.shape[1] - 1

        def analyse_window_obs(
            window_obs: Float[Array, "T_plus_1 M"],
            window_mask: Float[Array, "T_plus_1 M"],
        ) -> Float[Array, " N"]:
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

            return self._solve(obs_cost, num_steps)

        return jax.vmap(analyse_window_obs)(
            fill_gaps(batch.input, batch.mask), batch.mask
        )

    def _solve(
        self, obs_cost: Callable[[Array], Array], num_steps: int
    ) -> Float[Array, " N"]:
        """Minimise the prior term or penalty, plus 𝒥_o of the trajectory.

        Args:
            obs_cost: The observation term of a trajectory, x₀, …, x_T ↦ 𝒥_o.
            num_steps: Number of steps T.

        Returns:
            The analysed initial state x₀⋆.
        """
        parameterization = self.parameterization
        if self.basis is not None:
            # the same cost as LinearParameterization(basis, prior_mean)
            def cost(coefficients: Float[Array, " M"], _args: Any) -> Float[Array, ""]:
                # Σ_z⁻¹ z
                precision_product = self.basis.prior_inv(coefficients)
                # 𝒥_b = ½ zᵀ Σ_z⁻¹ z
                background_cost = 0.5 * jnp.sum(coefficients * precision_product)
                # δx₀ = Φ z, with a basis evaluated at t = 0
                increment = self.basis.operg(0.0, coefficients)
                # x₀ = μ_x + δx₀
                initial_state = self.prior_mean + as_shape(
                    increment, self.prior_mean.shape
                )
                # xₜ = f(xₜ₋₁)
                trajectory = self._rollout(initial_state, n_steps=num_steps)
                # 𝒥(z) = 𝒥_b + 𝒥_o(x₀)
                return background_cost + obs_cost(trajectory)

            initial_control = jnp.zeros(
                (self.basis.nbasis,), dtype=self.prior_mean.dtype
            )
        elif parameterization is not None:

            def cost(params: Any, _args: Any) -> Float[Array, ""]:
                # x₀ = g_dec(θ)
                decoded = parameterization.decode(params)
                initial_state = as_shape(decoded, self.prior_mean.shape)
                # ℛ(θ)
                penalty = parameterization.penalty(params)
                # xₜ = f(xₜ₋₁)
                trajectory = self._rollout(initial_state, n_steps=num_steps)
                # 𝒥(θ) = ℛ(θ) + 𝒥_o(g_dec(θ))
                return penalty + obs_cost(trajectory)

            initial_control = parameterization.init_control()
        else:

            def cost(initial_state: Float[Array, " N"], _args: Any) -> Float[Array, ""]:
                # δx₀ = x₀ − μ_x
                increment = initial_state - self.prior_mean
                # 𝒥_b = ½ δx₀ᵀ Σ_x⁻¹ δx₀
                background_cost = half_weighted_norm(
                    self.prior_cov_op, increment, self.cov_solver
                )
                # xₜ = f(xₜ₋₁)
                trajectory = self._rollout(initial_state, n_steps=num_steps)
                # 𝒥(x₀) = 𝒥_b + 𝒥_o(x₀)
                return background_cost + obs_cost(trajectory)

            initial_control = self.prior_mean

        control_star = minimise(
            self.minimiser,
            cost,
            initial_control,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )
        if self.basis is not None:
            # δx₀⋆ = Φ z⋆
            increment = self.basis.operg(0.0, control_star)
            # x₀⋆ = μ_x + δx₀⋆
            return self.prior_mean + as_shape(increment, self.prior_mean.shape)
        if parameterization is not None:
            # x₀⋆ = g_dec(θ⋆)
            decoded = parameterization.decode(control_star)
            return as_shape(decoded, self.prior_mean.shape)
        return control_star

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the default adapter
        is [`LaplaceCovariance`][vardax.LaplaceCovariance] with this model's Σ_x and
        Σ_ε, or Σ_x|y = Φ Σ_z|y Φᵀ with a ``basis``.

        Args:
            batch: A ``Batch1D`` holding one window.
            adapter: Optional [`PosteriorAdapter`][vardax.PosteriorAdapter],
                e.g. [`GaussNewtonHessian`][vardax.GaussNewtonHessian].

        Returns:
            The window's posterior: mean and lazy covariance.

        Raises:
            ShapeError: If ``batch`` holds more than one window.
            UnsupportedInputError: For a point batch, or a ``parameterization`` whose
                decode is not affine.
        """
        # lazy: the posterior adapters import the models
        from vardax._src.posterior._precision import model_posterior

        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _ReducedBasisFourDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        Returns:
            An analysis step that accepts single arrays or windowed lists, swaps in the
            supplied forecast and operators, and returns the analysis rolled to the end
            of the window.
        """
        return _ReducedBasisFourDVarAnalysisStep(self)


class _ReducedBasisFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``ReducedBasisFourDVar``.

    Analyses the window around the cycle's forecast, with the cycle's h and Σ_ε when
    given, and returns the analysed state at the end of the window, the state the cycle
    steps forward from (``analyse_window``).

    Attributes:
        model: The model to analyse with.
    """

    model: ReducedBasisFourDVar

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
            forecast: The cycle's forecast over the window: a list from
                ``SmootherCycle``, or one state from ``DACycle``.
            obs: Observations over the window, shaped the same way; non-finite entries
                are missing.
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
