r"""Weak-constraint 4DVar.

Multi-time analysis with control variable
$(x_0, \eta_1, \ldots, \eta_T)$ — initial state plus per-step
model-error increments. Cost:

$$
J(x_0, \boldsymbol{\eta}) =
    \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}
  + \tfrac{1}{2} \sum_t \|y_t - H_t(x_t)\|^2_{R_t^{-1}}
  + \tfrac{1}{2} \sum_t \|\eta_t\|^2_{Q_t^{-1}}
$$

where $x_t = M_t(x_{t-1}) + \eta_t$.

The control vector has dimension $(T + 1) \cdot N$ (initial
state + T model-error increments). For long windows this is
computationally heavier than [`StrongFourDVar`][vardax.StrongFourDVar]
— use only when the free model $M_t^\text{free}$ is known to have
biases.
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
from vardax._src.minimisers import minimise
from vardax._src.models._rollout import checkpoint_every, resolve_forward, rollout
from vardax._src.obs_operators.point import (
    check_point_obs_cov,
    grid_point_obs_cost,
    map_point_batch,
    require_obs_cov,
    window_steps,
)
from vardax._src.protocols import Minimiser, StateParameterization


class WeakFourDVar(eqx.Module):
    r"""Weak-constraint 4DVar with augmented control vector.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel`` (the free model
            $M_t^\text{free}$).
        obs_op: Observation operator.
        prior_mean: Background $x_b$ — initial state ``(N,)``.
        prior_cov_op: $B$.
        obs_cov_op: $R$.
        model_err_cov_op: $Q$ — covariance of the per-step model
            error $\eta_t$. Defaults to identity scaled by a
            small variance if not supplied (effectively
            strong-constraint with a tiny relaxation).
        minimiser: As in [`StrongFourDVar`][vardax.StrongFourDVar].
        minimiser_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar].
        max_steps: As in [`StrongFourDVar`][vardax.StrongFourDVar].
        cov_solver: ``lineax`` solver for the covariance solves in the
            cost ($B^{-1}$, $R^{-1}$, $Q^{-1}$). Default
            ``lx.CG(atol=1e-6, rtol=1e-6)``, which suits large
            matrix-free operators but is inexact on an ill-conditioned
            covariance (for example a tiny $R$); pass
            ``lx.AutoLinearSolver(well_posed=True)`` or ``lx.Cholesky()``
            for an exact solve when the operator can be materialised.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization] of the
            initial state. When given, the control is $(\theta, \eta)$,
            $x_0 = \psi(\theta)$ and $\mathrm{pen}(\theta)$ replaces the
            background term (``prior_mean`` and ``prior_cov_op`` are not
            used). Default ``None``: the control is $(x_0, \eta)$.
        forward_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]:
            a ``diffrax.AbstractAdjoint`` for a forward implementing
            [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint].
        checkpoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]:
            ``False``, ``True`` or a block length ``k`` for
            ``jax.checkpoint`` over the rollout.

    Examples:
        Trivial dynamics and a single timestep: no model-error steps
        remain ($T = 0$) and the analysis initial state reduces to the
        3DVar / BLUE answer $x_0^* = y / 2$ for $B = R = I$.

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
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator | None
    model_err_cov_op: lx.AbstractLinearOperator
    minimiser: optx.AbstractMinimiser | Minimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    cov_solver: lx.AbstractLinearSolver = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)
    parameterization: StateParameterization | None = None
    forward_adjoint: Any = eqx.field(static=True, default=None)
    checkpoint: bool | int = eqx.field(static=True, default=False)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, " N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator | None,
        model_err_cov_op: lx.AbstractLinearOperator,
        minimiser: optx.AbstractMinimiser | Minimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
        parameterization: StateParameterization | None = None,
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
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.model_err_cov_op = model_err_cov_op
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps
        self.cov_solver = cov_solver or lx.CG(atol=1e-6, rtol=1e-6)
        self.parameterization = parameterization
        self.forward_adjoint = forward_adjoint
        self.checkpoint = checkpoint

    def _rollout_with_error(
        self,
        x_0: Float[Array, " N"],
        etas: Float[Array, "T N"],
    ) -> Float[Array, "T_plus_1 N"]:
        r"""Roll out with per-step model error.

        Each step applies $x_t = M_t^\text{free}(x_{t-1}) + \eta_t$.
        """
        forward = resolve_forward(self.forward, self.forward_adjoint)
        n_steps = etas.shape[0]
        return rollout(forward, x_0, n_steps, forcing=etas, checkpoint=self.checkpoint)

    def __call__(
        self,
        batch: Batch1D | PointObsBatch,
    ) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch T N"]]:
        """Weak-4DVar analysis.

        Returns ``(x_0_star, eta_star)`` — the analysis initial state
        and the per-step model-error trajectory (one row per
        timestep, total ``T``).

        A [`PointObsBatch`][vardax.PointObsBatch] needs
        ``obs_op=GridPointObs(...)``: observation $i$ is compared with the
        trajectory state at ``time_index[i]`` over a window of ``n_times``
        states (so ``T = n_times - 1``), with error variance
        $\\sigma_i^2$ (``obs_cov_op`` must be ``None`` or an identity,
        else ``ValueError``).
        """
        N = self.prior_mean.shape[0]
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            n_steps = window_steps(batch)

            def _one_points(b: PointObsBatch):
                def traj_obs_cost(
                    trajectory: Float[Array, "T_plus_1 N"],
                ) -> Float[Array, ""]:
                    return grid_point_obs_cost(self.obs_op, trajectory, b)

                return self._solve(traj_obs_cost, n_steps, N)

            return map_point_batch(_one_points, batch)

        obs_cov_op = require_obs_cov(self.obs_cov_op)
        # T_plus_1 timesteps in the batch ⇒ T model-error steps.
        T_plus_1 = batch.input.shape[1]
        T = T_plus_1 - 1
        N = batch.input.shape[2]

        def _one(
            input_i: Float[Array, "T_plus_1 N"], mask_i: Float[Array, "T_plus_1 N"]
        ):
            def traj_obs_cost(
                trajectory: Float[Array, "T_plus_1 N"],
            ) -> Float[Array, ""]:
                def _per_step(x_t, y_t, m_t):
                    y_pred = (
                        self.obs_op(x_t, mask=m_t)
                        if _accepts_mask(self.obs_op)
                        else self.obs_op(x_t)
                    )
                    residual = m_t * (y_t - y_pred)
                    R_inv_r = lx.linear_solve(
                        obs_cov_op,
                        residual,
                        solver=self.cov_solver,
                    ).value
                    return 0.5 * jnp.sum(residual * R_inv_r)

                per_step_costs = jax.vmap(_per_step)(trajectory, input_i, mask_i)
                return jnp.sum(per_step_costs)

            return self._solve(traj_obs_cost, T, N)

        x0_star, eta_star = jax.vmap(_one)(batch.input, batch.mask)
        return x0_star, eta_star

    def _solve(
        self,
        traj_obs_cost: Callable[[Array], Array],
        T: int,
        N: int,
    ) -> tuple[Float[Array, " N"], Float[Array, "T N"]]:
        """Minimise over $(x_0, \\eta)$ (or $(\\theta, \\eta)$) for one sample."""

        def model_and_obs_cost(
            x_0: Float[Array, " N"],
            etas: Float[Array, "T N"],
        ) -> Float[Array, ""]:
            # Model-error term
            def _eta_cost(eta_t):
                Q_inv_eta = lx.linear_solve(
                    self.model_err_cov_op,
                    eta_t,
                    solver=self.cov_solver,
                ).value
                return 0.5 * jnp.sum(eta_t * Q_inv_eta)

            # A one-state window has no model-error steps (vmap over an
            # empty axis would trip lineax's CG on a zero-size operand).
            j_eta = (
                jnp.sum(jax.vmap(_eta_cost)(etas))
                if etas.shape[0] > 0
                else jnp.zeros((), dtype=x_0.dtype)
            )

            # Observation term
            j_obs = traj_obs_cost(self._rollout_with_error(x_0, etas))
            return j_obs + j_eta

        # Initial guess: x_0 at the background, etas at zero.
        init_etas = jnp.zeros((T, N))

        param = self.parameterization
        if param is not None:
            # Control (θ, η); x_0 = ψ(θ) and pen(θ) replaces the B term.
            def decode(theta: Any) -> Float[Array, " N"]:
                return param.decode(theta).reshape(self.prior_mean.shape)

            def param_cost(control: Any, _args: Any) -> Float[Array, ""]:
                theta, etas = control
                return param.penalty(theta) + model_and_obs_cost(decode(theta), etas)

            theta_star, eta_star = minimise(
                self.minimiser,
                param_cost,
                (param.init_control(), init_etas),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            return decode(theta_star), eta_star

        def cost(control: Float[Array, "T_plus_1 N"], _args: Any) -> Float[Array, ""]:
            x_0 = control[0]
            etas = control[1:]

            # Background term
            dx = x_0 - self.prior_mean
            B_inv_dx = lx.linear_solve(
                self.prior_cov_op,
                dx,
                solver=self.cov_solver,
            ).value
            j_bg = 0.5 * jnp.sum(dx * B_inv_dx)
            return j_bg + model_and_obs_cost(x_0, etas)

        y0 = jnp.concatenate([self.prior_mean[None, :], init_etas], axis=0)

        control_star = minimise(
            self.minimiser,
            cost,
            y0,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )
        return control_star[0], control_star[1:]

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the
        default adapter is [`LaplaceCovariance`][vardax.LaplaceCovariance]
        with this model's $B$ and $R$.

        The covariance is that of $x_0$: the $x_0$ block of the inverse
        Gauss–Newton precision over $(x_0, \\eta)$.

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

    def as_analysis_step(self) -> _WeakFourDVarAnalysisStep:
        return _WeakFourDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _WeakFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``WeakFourDVar``.

    Returns the analysis initial state only (the model-error
    trajectory is also available via direct ``model(batch)`` calls).
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
        mask = jnp.where(jnp.isfinite(obs), 1.0, 0.0)
        obs_clean = jnp.nan_to_num(obs)
        batch = Batch1D(input=obs_clean, mask=mask, target=None)
        x0_star, _ = self.model(batch)
        return x0_star
