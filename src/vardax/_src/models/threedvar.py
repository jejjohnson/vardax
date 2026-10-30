r"""3D Variational analysis.

Minimises

$$
J(x) = \tfrac{1}{2} \|x - x_b\|^2_{B^{-1}}
     + \tfrac{1}{2} \|y - H(x)\|^2_{R^{-1}}
$$

over a single timestep. When $H$ is linear and $B$, $R$ Gaussian,
``ThreeDVar`` reduces to [`OptimalInterpolation`][vardax.OptimalInterpolation]
— the canonical correctness gate enforced by the linear-Gaussian
agreement test.

The inner minimisation is delegated to an
``optimistix.AbstractMinimiser`` (``GaussNewton``, ``BFGS``,
``NonlinearCG``, ``LevenbergMarquardt``, …). The choice of minimiser
is the user's; vardax provides a sensible default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx
import optimistix as optx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.errors import InputTypeError
from vardax._src.minimisers import MinimiseResult, minimise, minimise_with_result
from vardax._src.models._analysis_step import analyse_forecast
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
from vardax._src.utils.covariance import as_covariance
from vardax._src.utils.shapes import as_shape


class ThreeDVar(eqx.Module):
    r"""3D variational analysis.

    $$J(x) = \tfrac12 \|x - x_b\|^2_{B^{-1}} + \tfrac12 \|y - H(x)\|^2_{R^{-1}}$$

    with the residual masked by $m$ where observations are missing.

    $$x^* = \arg\min_x J(x)$$

    Algorithm:
        1. For each sample in the batch (``jax.vmap``), start from the
           background $x_b$ (or the parameterization's initial control).
        2. Evaluate the background term
           $\tfrac12 \delta x^\top B^{-1} \delta x$ with a ``cov_solver`` solve; under a
           ``parameterization`` the penalty $\mathrm{pen}(\theta)$
           replaces it.
        3. Evaluate the observation term from the masked residual
           $m \odot (y - H(x))$ and an $R^{-1}$ solve; a point batch uses
           the $\sigma_i$-weighted point cost instead.
        4. Minimise the sum with ``minimiser`` for at most ``max_steps``
           iterations and return the minimising state.

    Attributes:
        obs_op: Observation operator (linear or nonlinear).
        prior_mean: Background $x_b$.
        prior_cov_op: $B$. ``lineax.AbstractLinearOperator``.
        obs_cov_op: $R$. ``lineax.AbstractLinearOperator``.
        minimiser: ``optimistix.AbstractMinimiser`` for the inner
            iteration (default ``optimistix.BFGS(rtol=1e-6, atol=1e-6)``),
            or a [`Minimiser`][vardax.Minimiser] such as
            [`OptaxMinimiser`][vardax.OptaxMinimiser].
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for
            differentiating *through* the minimum (used only if the
            ``ThreeDVar`` analysis is itself nested inside a larger
            training loop). Default ``ImplicitAdjoint`` — exact at the
            optimum.
        max_steps: Iteration cap on the inner solver when ``minimiser``
            is an ``optimistix`` solver; a vardax `Minimiser` keeps its own.
        cov_solver: ``lineax`` solver for the covariance solves in the
            cost ($B^{-1}$, $R^{-1}$). Default
            ``lx.CG(atol=1e-6, rtol=1e-6)``, which suits large
            matrix-free operators but is inexact on an ill-conditioned
            covariance (for example a tiny $R$); pass
            ``lx.AutoLinearSolver(well_posed=True)`` or ``lx.Cholesky()``
            for an exact solve when the operator can be materialised.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization]. When
            given, the control is $\theta$, the state is
            $\psi(\theta)$ (reshaped to ``prior_mean.shape``) and
            $\mathrm{pen}(\theta)$ replaces the background term, so
            ``prior_mean`` and ``prior_cov_op`` are not used — nor is the
            ``forecast`` of the analysis-step adapter. Default ``None``:
            the control is the state itself.

    Examples:
        With $B = R = I$, identity $H$ and everything observed, the
        analysis agrees with the closed-form BLUE, $x^* = y / 2$.

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

        A nonlinear $H(x) = x^2$ through
        [`CallableObs`][vardax.CallableObs]. Where the mask is zero the
        analysis falls back to the background, and ``analyse`` also
        returns the minimiser diagnostics:

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

        Composition with a posterior adapter: the default
        [`LaplaceCovariance`][vardax.LaplaceCovariance] returns
        $P^* = (I + I)^{-1}$ for the first example's model.

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
        r"""3DVar analysis for each sample in the batch.

        A [`PointObsBatch`][vardax.PointObsBatch] needs a point operator as
        ``obs_op``: a [`GridPointObs`][vardax.GridPointObs] on the state
        ``(T, N)`` (``time_index`` selects the row), or a
        [`FieldPointObs`][vardax.FieldPointObs] with a
        [`NeuralFieldParameterization`][vardax.NeuralFieldParameterization],
        which evaluates the field at the coordinates directly. The error
        variances are the batch's $\sigma_i^2$; ``obs_cov_op`` must be
        ``None`` or an identity.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] or a
                [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The analysis of every sample, with a leading batch axis.

        Raises:
            ObservationError: For a point batch with a non-trivial
                ``obs_cov_op``.
            MissingInputError: For a gridded batch when ``obs_cov_op`` is
                ``None``.
            InputTypeError: For a point batch whose ``obs_op`` is not a
                point operator, or a ``FieldPointObs`` without a
                ``NeuralFieldParameterization``.
        """
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            return map_point_batch(
                lambda window: self._analyse_points(self.prior_mean, window), batch
            )

        obs_cov_op = require_obs_cov(self.obs_cov_op)

        def _one(
            sample_obs: Float[Array, "*obs"], sample_mask: Float[Array, "*obs"]
        ) -> Float[Array, "*state"]:
            return self._analyse(
                self.prior_mean, sample_obs, sample_mask, self.obs_op, obs_cov_op
            )

        return jax.vmap(_one)(batch.input, batch.mask)

    def analyse(
        self, batch: Batch1D | PointObsBatch
    ) -> tuple[Float[Array, " batch *state"], MinimiseResult]:
        r"""The analysis, as ``__call__``, with the minimiser's diagnostics.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D].

        Returns:
            ``(analysis, result)``: the analysis ``(B, T, N)`` and a
            [`MinimiseResult`][vardax.MinimiseResult] whose leaves carry a
            leading sample axis: ``cost`` $J$ and ``grad_norm``
            $\lVert \nabla J \rVert_2$ at the minimum, ``num_steps`` (``-1``
            for a custom minimiser) and ``value``, the minimising control
            (the state, or $\theta$ under a ``parameterization``).

        Raises:
            InputTypeError: For a ``PointObsBatch``.
            MissingInputError: If ``obs_cov_op`` is ``None``.
        """
        if not isinstance(batch, Batch1D):
            raise InputTypeError(
                f"ThreeDVar.analyse takes a Batch1D; got {type(batch).__name__}."
            )
        obs_cov_op = require_obs_cov(self.obs_cov_op)

        def _one(
            sample_obs: Float[Array, "*obs"], sample_mask: Float[Array, "*obs"]
        ) -> tuple[Float[Array, "*state"], MinimiseResult]:
            return self._analyse(
                self.prior_mean,
                sample_obs,
                sample_mask,
                self.obs_op,
                obs_cov_op,
                with_result=True,
            )

        return jax.vmap(_one)(batch.input, batch.mask)

    def _analyse(
        self,
        background: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
        *,
        with_result: bool = False,
    ) -> Any:
        """Minimise the 3D-Var cost of one sample around the ``background``."""

        def obs_cost(state: Float[Array, "*state"]) -> Float[Array, ""]:
            # Observation term — apply R^{-1} via lineax solve.
            # Mask predictions AND observations symmetrically: for
            # obs_ops that don't take a `mask` kwarg (e.g.
            # LinearObs), unmasked predictions at missing entries
            # would otherwise penalise the analysis there.
            predicted_obs = (
                obs_op(state, mask=sample_mask)
                if _accepts_mask(obs_op)
                else obs_op(state)
            )
            # r = m ⊙ (y − H(x))
            residual = sample_mask * (sample_obs - predicted_obs)
            # R⁻¹r
            weighted_residual = lx.linear_solve(
                obs_cov_op,
                residual,
                solver=self.cov_solver,
            ).value
            # J_o = ½ rᵀ R⁻¹ r
            return 0.5 * jnp.sum(residual * weighted_residual)

        return self._solve(background, obs_cost, with_result=with_result)

    def _analyse_points(
        self, background: Float[Array, "*state"], batch: PointObsBatch
    ) -> Float[Array, "*state"]:
        """Minimise the 3D-Var cost of one point-observation sample."""
        if isinstance(self.obs_op, FieldPointObs):
            param = self.parameterization
            if not isinstance(param, NeuralFieldParameterization):
                raise InputTypeError(
                    "FieldPointObs evaluates a field: it needs "
                    "parameterization=NeuralFieldParameterization(...)."
                )
            field_op = self.obs_op

            def control_obs_cost(theta: Any) -> Float[Array, ""]:
                return point_obs_cost(field_op(param.as_field(theta), batch), batch)

            return self._solve(background, None, control_obs_cost=control_obs_cost)

        def obs_cost(state: Float[Array, "*state"]) -> Float[Array, ""]:
            return grid_point_obs_cost(self.obs_op, state, batch)

        return self._solve(background, obs_cost)

    def _solve(
        self,
        background: Float[Array, "*state"],
        obs_cost: Callable[[Array], Array] | None,
        *,
        control_obs_cost: Callable[[Any], Array] | None = None,
        with_result: bool = False,
    ) -> Any:
        """Minimise background (or penalty) plus ``obs_cost`` from ``background``.

        Returns the analysis, or ``(analysis, MinimiseResult)`` when
        ``with_result``.
        """
        run = minimise_with_result if with_result else minimise
        param = self.parameterization
        if param is not None:
            # J(θ) = pen(θ) + J_o(ψ(θ)): the penalty replaces the B term.
            def decode(theta: Any) -> Float[Array, "*state"]:
                return as_shape(param.decode(theta), background.shape)

            def param_cost(theta: Any, _args: Any) -> Float[Array, ""]:
                if control_obs_cost is not None:
                    return param.penalty(theta) + control_obs_cost(theta)
                assert obs_cost is not None
                return param.penalty(theta) + obs_cost(decode(theta))

            result = run(
                self.minimiser,
                param_cost,
                param.init_control(),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            if with_result:
                return decode(result.value), result
            return decode(result)

        def cost(state: Float[Array, "*state"], _args: Any) -> Float[Array, ""]:
            # Background term — apply B^{-1} via lineax solve.
            # δx = x − x_b
            increment = state - background
            # B⁻¹δx
            weighted_increment = lx.linear_solve(
                self.prior_cov_op,
                increment,
                solver=self.cov_solver,
            ).value
            # J_b = ½ δxᵀ B⁻¹ δx
            background_cost = 0.5 * jnp.sum(increment * weighted_increment)
            assert obs_cost is not None
            return background_cost + obs_cost(state)

        result = run(
            self.minimiser,
            cost,
            background,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )
        if with_result:
            return result.value, result
        return result

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the
        default adapter is [`LaplaceCovariance`][vardax.LaplaceCovariance]
        with this model's $B$ and $R$.

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

        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _ThreeDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8).

        Returns:
            An analysis step that minimises the cost around a supplied
            forecast, treating ``NaN`` observations as missing.
        """
        return _ThreeDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _ThreeDVarAnalysisStep(eqx.Module):
    r"""``pipekit_cycle.AnalysisStep`` adapter for ``ThreeDVar``.

    Minimises the 3D-Var cost around the forecast,
    $J(x) = \tfrac12 \|x - x_b\|^2_{B^{-1}} + \tfrac12 \|y - H(x)\|^2_{R^{-1}}$
    with $x_b$ = ``forecast`` (also the starting point of the
    minimiser). ``B`` is the model's ``prior_cov_op``; ``H`` is ``obs_op``
    and ``R`` is ``obs_err_cov`` when given, else the model's ``obs_op`` /
    ``obs_cov_op``. The model's ``prior_mean`` is not used.

    ``forecast`` is one state shaped like ``prior_mean`` or a batch of
    them, ``(B, T, N)``, and ``obs`` is batched the same way; ``NaN``
    observations are missing. ``obs_err_cov`` may be ``None``, a
    ``lineax.AbstractLinearOperator`` on one observation sample, or a
    scalar / array of error variances (a diagonal $R$).
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
        return analyse_forecast(
            self.model._analyse,
            state_shape=self.model.prior_mean.shape,
            default_obs_op=self.model.obs_op,
            default_obs_cov_op=self.model.obs_cov_op,
            forecast=forecast,
            obs=obs,
            obs_op=obs_op,
            obs_err_cov=obs_err_cov,
        )
