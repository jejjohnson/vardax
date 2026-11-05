"""3D-Var: minimise the variational cost of one time with an iterative minimiser.

μ_x and Σ_x are the prior mean and covariance, Σ_ε the observation-error covariance and
h the observation operator; a linear h reduces it to optimal interpolation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import gaussx
import jax
from jaxtyping import Array, Float
import lineax as lx
import optimistix as optx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.errors import InputTypeError
from vardax._src.minimisers import MinimiseResult, minimise, minimise_with_result
from vardax._src.models._analysis_step import analyse_forecast, centred_on
from vardax._src.models._observations import fill_gaps
from vardax._src.models._rollout import accepts_mask
from vardax._src.obs_operators.point import (
    FieldPointObs,
    check_point_obs_cov,
    grid_point_obs_cost,
    map_point_batch,
    point_obs_cost,
    require_obs_cov,
)
from vardax._src.parameterization import NeuralFieldParameterization
from vardax._src.protocols import Minimiser, StateParameterization
from vardax._src.utils.covariance import (
    as_covariance,
    half_weighted_norm,
)
from vardax._src.utils.shapes import as_shape


class ThreeDVar(eqx.Module):
    """3D variational analysis of a single time.

        𝒥(x) = ½ ‖x − μ_x‖²_Σ_x⁻¹ + ½ ‖r‖²_Σ̃_ε⁻¹,   x⋆ = argmin_x 𝒥(x)
        r = m ⊙ (y − h(x))
        Σ̃_ε = M Σ_ε M + (I − M),   M = diag(m)

    When h is linear this reduces to
    [`OptimalInterpolation`][vardax.OptimalInterpolation], which
    ``tests/test_linear_gaussian_agreement.py`` enforces.

    Algorithm:
        1. For each sample, under ``jax.vmap``, start from the prior mean μ_x, or from
           the initial control of ``parameterization``.
        2. Evaluate the background term ½ δxᵀ Σ_x⁻¹ δx with a ``cov_solver`` solve; the
           penalty ℛ(θ) of ``parameterization`` replaces it.
        3. Evaluate the observation term from the masked residual r and a Σ̃_ε⁻¹ solve;
           a point batch uses its σᵢ-weighted cost instead.
        4. Minimise the sum with ``minimiser`` for at most ``max_steps`` iterations and
           return the minimising state.

    Attributes:
        obs_op: Observation operator h, linear or nonlinear.
        prior_mean: Prior mean μ_x.
        prior_cov_op: Prior covariance Σ_x.
        obs_cov_op: Observation-error covariance Σ_ε.
        minimiser: ``optimistix.AbstractMinimiser`` for the inner iteration, by default
            ``optimistix.BFGS(rtol=1e-6, atol=1e-6)``, or a
            [`Minimiser`][vardax.Minimiser] such as
            [`OptaxMinimiser`][vardax.OptaxMinimiser].
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for differentiating through
            the minimum, used only when the analysis is nested inside a larger training
            loop. The default ``ImplicitAdjoint`` is exact at the optimum.
        max_steps: Iteration cap on the inner solver when ``minimiser`` is an
            ``optimistix`` solver; a vardax `Minimiser` keeps its own.
        cov_solver: ``lineax`` solver for Σ_x⁻¹ and Σ_ε⁻¹ in the cost. The default
            ``lx.CG(atol=1e-6, rtol=1e-6)`` suits large matrix-free operators but is
            inexact on an ill-conditioned covariance (for example a tiny Σ_ε); pass
            ``lx.AutoLinearSolver(well_posed=True)`` or ``lx.Cholesky()`` for an exact
            solve when the operator can be materialised.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization]. The control is
            then θ, with x = g_dec(θ) reshaped to ``prior_mean.shape``, and ℛ(θ)
            replaces the background term, so ``prior_mean`` and ``prior_cov_op`` are not
            used, nor is the ``forecast`` of the analysis-step adapter. Default
            ``None``: the control is x.

    Examples:
        With Σ_x = Σ_ε = I, identity h and everything observed, the analysis agrees with
        the closed-form BLUE, x⋆ = y / 2:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float32))
        >>> three = vardax.ThreeDVar(
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros((1, 3)),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 1, 3)), mask=jnp.ones((1, 1, 3)))
        >>> xa = three(batch)
        >>> xa.shape
        (1, 1, 3)
        >>> bool(jnp.allclose(xa, 0.5, atol=1e-3))
        True

        A nonlinear h(x) = x² through [`CallableObs`][vardax.CallableObs]. Where the
        mask is zero the analysis falls back to μ_x, and ``analyse`` also returns the
        minimiser diagnostics:

        >>> sq = vardax.ThreeDVar(
        ...     obs_op=vardax.CallableObs(fn=lambda x: x**2),
        ...     prior_mean=jnp.ones((1, 3)),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> half = vardax.Batch1D(
        ...     input=jnp.full((1, 1, 3), 4.0),
        ...     mask=jnp.array([[[1.0, 0.0, 0.0]]]),
        ... )
        >>> xa, result = sq.analyse(half)
        >>> xa.shape, result.cost.shape
        ((1, 1, 3), (1,))
        >>> bool(jnp.allclose(xa[0, 0, 1:], 1.0, atol=1e-3))
        True

        Composed with a posterior adapter: the default
        [`LaplaceCovariance`][vardax.LaplaceCovariance] returns
        Σ_x|y = (I + I)⁻¹ for the first example's model:

        >>> post = three.posterior(batch)
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones((1, 3))), 0.5, atol=1e-2))
        True
    """

    obs_op: Any
    prior_mean: Float[Array, "*state"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator | None
    minimiser: optx.AbstractMinimiser | Minimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    cov_solver: lx.AbstractLinearSolver = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)
    parameterization: StateParameterization | None = None

    def __init__(
        self,
        obs_op: Any,
        prior_mean: Float[Array, "*state"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator | None,
        minimiser: optx.AbstractMinimiser | Minimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
        parameterization: StateParameterization | None = None,
        cov_solver: lx.AbstractLinearSolver | None = None,
    ) -> None:
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = as_covariance(prior_cov_op)
        self.obs_cov_op = None if obs_cov_op is None else as_covariance(obs_cov_op)
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps
        self.cov_solver = cov_solver or lx.CG(atol=1e-6, rtol=1e-6)
        self.parameterization = parameterization

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, " batch *state"]:
        """Analyse each sample of the batch: x⋆ = argmin_x 𝒥(x).

        A [`PointObsBatch`][vardax.PointObsBatch] needs a point operator as ``obs_op``:
        a [`GridPointObs`][vardax.GridPointObs] on the state (``time_index`` selects the
        row), or a [`FieldPointObs`][vardax.FieldPointObs] with a
        [`NeuralFieldParameterization`][vardax.NeuralFieldParameterization], which
        evaluates the field at the coordinates directly. The error variances are the
        batch's σᵢ²; ``obs_cov_op`` must be ``None`` or an identity.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] or a
                [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The analysis of every sample, with a leading batch axis.

        Raises:
            ObservationError: For a point batch with a non-trivial ``obs_cov_op``.
            MissingInputError: For a gridded batch when ``obs_cov_op`` is ``None``.
            InputTypeError: For a point batch whose ``obs_op`` is not a point operator,
                or a ``FieldPointObs`` without a ``NeuralFieldParameterization``.
        """
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            return map_point_batch(
                lambda window: self._analyse_points(self.prior_mean, window), batch
            )

        obs_covariance = require_obs_cov(self.obs_cov_op)

        def analyse_sample(
            sample_obs: Float[Array, "*obs"], sample_mask: Float[Array, "*obs"]
        ) -> Float[Array, "*state"]:
            return self._analyse(
                self.prior_mean, sample_obs, sample_mask, self.obs_op, obs_covariance
            )

        return jax.vmap(analyse_sample)(fill_gaps(batch.input, batch.mask), batch.mask)

    def analyse(
        self, batch: Batch1D | PointObsBatch
    ) -> tuple[Float[Array, " batch *state"], MinimiseResult]:
        """The analysis, as ``__call__``, with the minimiser's diagnostics.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D].

        Returns:
            ``(analysis, result)``: the analysis of every sample and a
            [`MinimiseResult`][vardax.MinimiseResult] whose leaves carry a leading
            sample axis: ``cost`` 𝒥 and ``grad_norm`` ‖∇𝒥‖₂ at the minimum,
            ``num_steps`` (``-1`` for a custom minimiser) and ``value``, the minimising
            control (x, or θ under a ``parameterization``).

        Raises:
            InputTypeError: For a ``PointObsBatch``.
            MissingInputError: If ``obs_cov_op`` is ``None``.
        """
        if not isinstance(batch, Batch1D):
            raise InputTypeError(
                f"ThreeDVar.analyse takes a Batch1D; got {type(batch).__name__}."
            )
        obs_covariance = require_obs_cov(self.obs_cov_op)

        def analyse_sample(
            sample_obs: Float[Array, "*obs"], sample_mask: Float[Array, "*obs"]
        ) -> tuple[Float[Array, "*state"], MinimiseResult]:
            return self._analyse(
                self.prior_mean,
                sample_obs,
                sample_mask,
                self.obs_op,
                obs_covariance,
                with_result=True,
            )

        return jax.vmap(analyse_sample)(fill_gaps(batch.input, batch.mask), batch.mask)

    def _analyse(
        self,
        prior_mean: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_covariance: lx.AbstractLinearOperator,
        *,
        with_result: bool = False,
    ) -> Any:
        """Minimise the 3D-Var cost of one sample around the prior mean μ_x."""
        # Σ̃_ε = M Σ_ε M + (I − M), so masked entries drop out of the likelihood
        masked_covariance = gaussx.masked_obs_covariance(obs_covariance, sample_mask)

        def obs_cost(state: Float[Array, "*state"]) -> Float[Array, ""]:
            # h(x)
            predicted_obs = (
                obs_op(state, mask=sample_mask)
                if accepts_mask(obs_op)
                else obs_op(state)
            )
            # r = m ⊙ (y − h(x)); masking both sides stops an operator without a
            # ``mask`` keyword from penalising the analysis at missing entries
            residual = sample_mask * (sample_obs - predicted_obs)
            # 𝒥_o = ½ rᵀ Σ̃_ε⁻¹ r
            return half_weighted_norm(masked_covariance, residual, self.cov_solver)

        return self._solve(prior_mean, obs_cost, with_result=with_result)

    def _analyse_points(
        self, prior_mean: Float[Array, "*state"], batch: PointObsBatch
    ) -> Float[Array, "*state"]:
        """Minimise the 3D-Var cost of one point-observation sample."""
        if isinstance(self.obs_op, FieldPointObs):
            parameterization = self.parameterization
            if not isinstance(parameterization, NeuralFieldParameterization):
                raise InputTypeError(
                    "FieldPointObs evaluates a field: it needs "
                    "parameterization=NeuralFieldParameterization(...)."
                )
            field_op = self.obs_op

            def control_obs_cost(params: Any) -> Float[Array, ""]:
                # x(·; θ)
                field = parameterization.as_field(params)
                # ŷᵢ = x(sᵢ; θ)
                predictions = field_op(field, batch)
                # 𝒥_o = ½ ∑ᵢ mᵢ (yᵢ − ŷᵢ)² / σᵢ²
                return point_obs_cost(predictions, batch)

            return self._solve(prior_mean, None, control_obs_cost=control_obs_cost)

        def obs_cost(state: Float[Array, "*state"]) -> Float[Array, ""]:
            # 𝒥_o = ½ ∑ᵢ ‖rᵢ / σᵢ‖²
            return grid_point_obs_cost(self.obs_op, state, batch)

        return self._solve(prior_mean, obs_cost)

    def _solve(
        self,
        prior_mean: Float[Array, "*state"],
        obs_cost: Callable[[Array], Array] | None,
        *,
        control_obs_cost: Callable[[Any], Array] | None = None,
        with_result: bool = False,
    ) -> Any:
        """Minimise the background term or penalty, plus 𝒥_o, from the prior mean.

        Args:
            prior_mean: The prior mean μ_x, the start of the minimiser and the shape of
                the decoded state.
            obs_cost: The observation term x ↦ 𝒥_o(x), or ``None`` when
                ``control_obs_cost`` is given.
            control_obs_cost: The observation term θ ↦ 𝒥_o(θ) of a field evaluated at
                the observation points directly.
            with_result: Also return the minimiser's diagnostics.

        Returns:
            The analysis, or ``(analysis, MinimiseResult)`` when ``with_result``.
        """
        run = minimise_with_result if with_result else minimise
        parameterization = self.parameterization
        if parameterization is not None:

            def decode(params: Any) -> Float[Array, "*state"]:
                # x = g_dec(θ)
                decoded = parameterization.decode(params)
                return as_shape(decoded, prior_mean.shape)

            def params_cost(params: Any, _args: Any) -> Float[Array, ""]:
                # ℛ(θ)
                penalty = parameterization.penalty(params)
                if control_obs_cost is not None:
                    # 𝒥(θ) = ℛ(θ) + 𝒥_o(θ)
                    return penalty + control_obs_cost(params)
                assert obs_cost is not None
                # x = g_dec(θ)
                state = decode(params)
                # 𝒥(θ) = ℛ(θ) + 𝒥_o(g_dec(θ))
                return penalty + obs_cost(state)

            result = run(
                self.minimiser,
                params_cost,
                parameterization.init_control(),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            if with_result:
                return decode(result.value), result
            return decode(result)

        def cost(state: Float[Array, "*state"], _args: Any) -> Float[Array, ""]:
            # δx = x − μ_x
            increment = state - prior_mean
            # 𝒥_b = ½ δxᵀ Σ_x⁻¹ δx
            background_cost = half_weighted_norm(
                self.prior_cov_op, increment, self.cov_solver
            )
            assert obs_cost is not None
            # 𝒥(x) = 𝒥_b + 𝒥_o(x)
            return background_cost + obs_cost(state)

        result = run(
            self.minimiser,
            cost,
            prior_mean,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )
        if with_result:
            return result.value, result
        return result

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the default adapter
        is [`LaplaceCovariance`][vardax.LaplaceCovariance] with this model's Σ_x and
        Σ_ε.

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

    def as_analysis_step(self) -> _ThreeDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        Returns:
            An analysis step that minimises the cost around a supplied forecast,
            treating ``NaN`` observations as missing.
        """
        return _ThreeDVarAnalysisStep(self)


class _ThreeDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``ThreeDVar``.

    Minimises 𝒥(x) with μ_x = ``forecast``, which is also the starting point of the
    minimiser, so the model's ``prior_mean`` is not used. Σ_x is the model's
    ``prior_cov_op``; h is ``obs_op`` and Σ_ε is ``obs_err_cov`` when given, else the
    model's.

    Attributes:
        model: The model to analyse with.
    """

    model: ThreeDVar

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        """Analyse around ``forecast``.

        Args:
            forecast: One state shaped like ``prior_mean``, or a batch of them.
            obs: Observations y, batched the same way; ``NaN`` entries are missing.
            obs_op: The cycle's observation operator h, or ``None``.
            obs_err_cov: The cycle's Σ_ε: ``None``, a
                ``lineax.AbstractLinearOperator`` on one observation sample, or a
                scalar or array of error variances (a diagonal Σ_ε).

        Returns:
            The analysis x⋆, with the shape of ``forecast``.
        """

        def analyse_one(prior_mean: Array, *args: Any) -> Array:
            return centred_on(self.model, prior_mean)._analyse(prior_mean, *args)

        return analyse_forecast(
            analyse_one,
            state_shape=self.model.prior_mean.shape,
            default_obs_op=self.model.obs_op,
            default_obs_cov_op=self.model.obs_cov_op,
            forecast=forecast,
            obs=obs,
            obs_op=obs_op,
            obs_err_cov=obs_err_cov,
        )
