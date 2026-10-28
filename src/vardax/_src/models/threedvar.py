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


class ThreeDVar(eqx.Module):
    r"""3D variational analysis.

    Attributes:
        obs_op: Observation operator (linear or nonlinear).
        prior_mean: Background $x_b$ of shape ``(T, N)``.
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
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
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
        ``None`` or an identity (``ValueError`` otherwise).
        """
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            return map_point_batch(
                lambda b: self._analyse_points(self.prior_mean, b), batch
            )

        obs_cov_op = require_obs_cov(self.obs_cov_op)

        def _one(
            input_i: Float[Array, "*obs"], mask_i: Float[Array, "*obs"]
        ) -> Float[Array, "*state"]:
            return self._analyse(
                self.prior_mean, input_i, mask_i, self.obs_op, obs_cov_op
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
            TypeError: For a ``PointObsBatch``.
        """
        if not isinstance(batch, Batch1D):
            raise InputTypeError(
                f"ThreeDVar.analyse takes a Batch1D; got {type(batch).__name__}."
            )
        obs_cov_op = require_obs_cov(self.obs_cov_op)

        def _one(
            input_i: Float[Array, "*obs"], mask_i: Float[Array, "*obs"]
        ) -> tuple[Float[Array, "*state"], MinimiseResult]:
            return self._analyse(
                self.prior_mean,
                input_i,
                mask_i,
                self.obs_op,
                obs_cov_op,
                with_result=True,
            )

        return jax.vmap(_one)(batch.input, batch.mask)

    def _analyse(
        self,
        x_b: Float[Array, "*state"],
        input_i: Float[Array, "*obs"],
        mask_i: Float[Array, "*obs"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
        *,
        with_result: bool = False,
    ) -> Any:
        """Minimise the 3D-Var cost of one sample around the background ``x_b``."""

        def obs_cost(x: Float[Array, "*state"]) -> Float[Array, ""]:
            # Observation term — apply R^{-1} via lineax solve.
            # Mask predictions AND observations symmetrically: for
            # obs_ops that don't take a `mask` kwarg (e.g.
            # LinearObs), unmasked predictions at missing entries
            # would otherwise penalise the analysis there.
            y_pred = obs_op(x, mask=mask_i) if _accepts_mask(obs_op) else obs_op(x)
            residual = mask_i * (input_i - y_pred)
            R_inv_r = lx.linear_solve(
                obs_cov_op,
                residual,
                solver=self.cov_solver,
            ).value
            return 0.5 * jnp.sum(residual * R_inv_r)

        return self._solve(x_b, obs_cost, with_result=with_result)

    def _analyse_points(
        self, x_b: Float[Array, "*state"], batch: PointObsBatch
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

            return self._solve(x_b, None, control_obs_cost=control_obs_cost)

        def obs_cost(x: Float[Array, "*state"]) -> Float[Array, ""]:
            return grid_point_obs_cost(self.obs_op, x, batch)

        return self._solve(x_b, obs_cost)

    def _solve(
        self,
        x_b: Float[Array, "*state"],
        obs_cost: Callable[[Array], Array] | None,
        *,
        control_obs_cost: Callable[[Any], Array] | None = None,
        with_result: bool = False,
    ) -> Any:
        """Minimise background (or penalty) plus ``obs_cost`` from ``x_b``.

        Returns the analysis, or ``(analysis, MinimiseResult)`` when
        ``with_result``.
        """
        run = minimise_with_result if with_result else minimise
        param = self.parameterization
        if param is not None:
            # J(θ) = pen(θ) + J_o(ψ(θ)): the penalty replaces the B term.
            def decode(theta: Any) -> Float[Array, "*state"]:
                return param.decode(theta).reshape(x_b.shape)

            def param_cost(theta: Any, _args: Any) -> Float[Array, ""]:
                if control_obs_cost is not None:
                    return param.penalty(theta) + control_obs_cost(theta)
                assert obs_cost is not None
                return param.penalty(theta) + obs_cost(decode(theta))

            out = run(
                self.minimiser,
                param_cost,
                param.init_control(),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            if with_result:
                return decode(out.value), out
            return decode(out)

        def cost(x: Float[Array, "*state"], _args: Any) -> Float[Array, ""]:
            # Background term — apply B^{-1} via lineax solve.
            dx = x - x_b
            B_inv_dx = lx.linear_solve(
                self.prior_cov_op,
                dx,
                solver=self.cov_solver,
            ).value
            j_bg = 0.5 * jnp.sum(dx * B_inv_dx)
            assert obs_cost is not None
            return j_bg + obs_cost(x)

        out = run(
            self.minimiser,
            cost,
            x_b,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )
        if with_result:
            return out.value, out
        return out

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
            ValueError: If ``batch`` holds more than one window.
        """
        from vardax._src.posterior._precision import model_posterior

        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _ThreeDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
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
