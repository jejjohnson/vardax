r"""Strong-constraint 4DVar.

Multi-time analysis with the control variable being the initial state
$x_0$. The forward model $M_t$ is treated as exact (no model-error
term). Cost:

$$
J(x_0) = \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}
       + \tfrac{1}{2} \sum_{t=0}^{T} \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}.
$$

Gradients of $J$ w.r.t. $x_0$ flow through the rollout
``trajectory = [forward.step(x, dt) for _ in range(T)]`` and back by
reverse-mode autodiff, which by default stores every intermediate of
every step. Two slots bound that memory:

- ``checkpoint=True`` (or a block length ``k``) wraps blocks of steps in
  ``jax.checkpoint``, so reverse mode recomputes the steps inside each
  block instead of storing them. Works for any per-step forward.
- ``forward_adjoint=`` hands a ``diffrax.AbstractAdjoint`` to a forward
  that integrates with diffrax and exposes ``with_adjoint(adjoint)``
  ([`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint]). A forward
  without that method raises ``ValueError`` at construction.
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
from vardax._src.utils.shapes import as_shape


class StrongFourDVar(eqx.Module):
    r"""Strong-constraint 4DVar.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel`` supplying
            ``step(state, dt) -> state``.
        obs_op: Observation operator.
        prior_mean: Background $x_b$ — initial state of shape
            ``(N,)`` (state vector at $t=0$).
        prior_cov_op: $B$.
        obs_cov_op: $R$.
        minimiser: ``optimistix.AbstractMinimiser`` for the outer
            optimisation over $x_0$ (default ``optx.BFGS(rtol=1e-6,
            atol=1e-6)``), or a [`Minimiser`][vardax.Minimiser] such as
            [`OptaxMinimiser`][vardax.OptaxMinimiser] for a large control.
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for
            differentiating through the minimum.
        forward_adjoint: ``diffrax.AbstractAdjoint`` passed to
            ``forward.with_adjoint(...)`` before the rollout. Requires a
            forward implementing
            [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint];
            any other forward raises ``ValueError``. Default ``None``
            (use the forward's own).
        max_steps: Iteration cap on the outer solver when ``minimiser``
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
            given, the control is $\theta$, $x_0 = \psi(\theta)$ (reshaped
            to ``prior_mean.shape``) and $\mathrm{pen}(\theta)$ replaces
            the background term, so ``prior_mean`` and ``prior_cov_op``
            are not used. Default ``None``: the control is $x_0$.
        checkpoint: Rematerialise the rollout in reverse mode. ``False``
            (default) stores every step's intermediates; ``True`` wraps
            each step in ``jax.checkpoint``; an int ``k`` wraps blocks of
            ``k`` steps. Same gradient, less memory, about one extra
            forward run.

    Examples:
        Trivial dynamics ($M_t(x) = x$) and a single timestep reduce
        strong 4DVar to 3DVar, so with $B = R = I$ the analysis is
        $x_0^* = y / 2$.

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Identity:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
        >>> strong = vardax.StrongFourDVar(
        ...     forward=Identity(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 1, 3)), mask=jnp.ones((1, 1, 3)))
        >>> strong(batch).shape
        (1, 3)
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
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
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
        """Step the forward model ``n_steps`` times, returning ``T + 1`` states."""
        forward = resolve_forward(self.forward, self.forward_adjoint)
        return rollout(forward, initial_state, n_steps, checkpoint=self.checkpoint)

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, "*batch N"]:
        """Strong-4DVar analysis: minimise over $x_0$.

        A [`PointObsBatch`][vardax.PointObsBatch] needs
        ``obs_op=GridPointObs(...)``: observation $i$ is compared with the
        trajectory state $x_{\tau_i}$, $\tau_i$ = ``time_index[i]``, over a
        window of ``n_times`` states, with error variance $\\sigma_i^2$
        (``obs_cov_op`` must be ``None`` or an identity,
        else ``ValueError``).
        """
        if isinstance(batch, PointObsBatch):
            check_point_obs_cov(self.obs_cov_op)
            n_steps = window_steps(batch)

            def _one_points(window: PointObsBatch) -> Float[Array, " N"]:
                def obs_cost(initial_state: Float[Array, " N"]) -> Float[Array, ""]:
                    trajectory = self._rollout(initial_state, n_steps=n_steps)
                    return grid_point_obs_cost(self.obs_op, trajectory, window)

                return self._solve(obs_cost)

            return map_point_batch(_one_points, batch)

        obs_cov_op = require_obs_cov(self.obs_cov_op)
        # rollout steps (input has T+1 timesteps)
        n_steps = batch.input.shape[1] - 1

        def _one(
            window_obs: Float[Array, "T_plus_1 N"],
            window_mask: Float[Array, "T_plus_1 N"],
        ) -> Float[Array, " N"]:
            def obs_cost(initial_state: Float[Array, " N"]) -> Float[Array, ""]:
                # Roll out the trajectory.
                trajectory = self._rollout(initial_state, n_steps=n_steps)

                # Observation term, summed across time.
                def _per_step(state, observation, step_mask):
                    predicted_obs = (
                        self.obs_op(state, mask=step_mask)
                        if _accepts_mask(self.obs_op)
                        else self.obs_op(state)
                    )
                    # r = m ⊙ (y − H(x))
                    residual = step_mask * (observation - predicted_obs)
                    # R⁻¹r
                    weighted_residual = lx.linear_solve(
                        obs_cov_op,
                        residual,
                        solver=self.cov_solver,
                    ).value
                    # J_o,t = ½ rᵀ R⁻¹ r
                    return 0.5 * jnp.sum(residual * weighted_residual)

                per_step_costs = jax.vmap(_per_step)(
                    trajectory, window_obs, window_mask
                )
                return jnp.sum(per_step_costs)

            return self._solve(obs_cost)

        return jax.vmap(_one)(batch.input, batch.mask)

    def _solve(
        self,
        obs_cost: Callable[[Array], Array],
    ) -> Float[Array, " N"]:
        """Minimise background (or penalty) plus ``obs_cost(x_0)``."""
        param = self.parameterization
        if param is not None:
            # x_0 = ψ(θ); J(θ) = pen(θ) + J_o(M(ψ(θ))).
            def decode(theta: Any) -> Float[Array, " N"]:
                return as_shape(param.decode(theta), self.prior_mean.shape)

            def param_cost(theta: Any, _args: Any) -> Float[Array, ""]:
                return param.penalty(theta) + obs_cost(decode(theta))

            theta_star = minimise(
                self.minimiser,
                param_cost,
                param.init_control(),
                adjoint=self.minimiser_adjoint,
                max_steps=self.max_steps,
            )
            return decode(theta_star)

        def cost(initial_state: Float[Array, " N"], _args: Any) -> Float[Array, ""]:
            # Background term
            # δx = x₀ − x_b
            increment = initial_state - self.prior_mean
            # B⁻¹δx
            weighted_increment = lx.linear_solve(
                self.prior_cov_op,
                increment,
                solver=self.cov_solver,
            ).value
            # J_b = ½ δxᵀ B⁻¹ δx
            background_cost = 0.5 * jnp.sum(increment * weighted_increment)
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

    def as_analysis_step(self) -> _StrongFourDVarAnalysisStep:
        return _StrongFourDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _StrongFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``StrongFourDVar``."""

    model: StrongFourDVar

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
