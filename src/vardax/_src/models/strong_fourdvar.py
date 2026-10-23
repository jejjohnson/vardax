r"""Strong-constraint 4DVar.

Multi-time analysis with the control variable being the initial state
$x_0$. The forward model $M_t$ is treated as exact (no model-error
term). Cost:

$$
J(x_0) = \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}
       + \tfrac{1}{2} \sum_{t=0}^{T} \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}.
$$

Gradients of $J$ w.r.t. $x_0$ flow through the rollout
``trajectory = [forward.step(x, dt) for _ in range(T)]`` and back —
the adjoint composition is whatever JAX gives by default. For long
windows where the forward is a diffrax ODE solve, the
``forward_adjoint`` slot threads through to ``diffrax.diffeqsolve(...,
adjoint=...)`` (per-step rollouts use plain reverse-mode autodiff).
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
from vardax._src.obs_operators.point import (
    check_point_obs_cov,
    grid_point_obs_cost,
    map_point_batch,
    window_steps,
)
from vardax._src.protocols import Minimiser, StateParameterization


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
        forward_adjoint: ``diffrax.AbstractAdjoint``-like — currently
            stored as a tag, threaded through if the forward delegates
            to ``diffrax``. Default ``None`` (use forward's own).
        max_steps: Iteration cap on the outer solver when ``minimiser``
            is an ``optimistix`` solver; a vardax `Minimiser` keeps its own.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization]. When
            given, the control is $\theta$, $x_0 = \psi(\theta)$ (reshaped
            to ``prior_mean.shape``) and $\mathrm{pen}(\theta)$ replaces
            the background term, so ``prior_mean`` and ``prior_cov_op``
            are not used. Default ``None``: the control is $x_0$.

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
    prior_mean: Float[Array, N]  # ty:ignore[unresolved-reference]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    minimiser: optx.AbstractMinimiser | Minimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    forward_adjoint: Any = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)
    parameterization: StateParameterization | None = None

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, N],  # ty:ignore[unresolved-reference]
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        minimiser: optx.AbstractMinimiser | Minimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        forward_adjoint: Any = None,
        max_steps: int = 200,
        parameterization: StateParameterization | None = None,
    ) -> None:
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.forward_adjoint = forward_adjoint
        self.max_steps = max_steps
        self.parameterization = parameterization

    def _rollout(self, x_0: Float[Array, N], n_steps: int) -> Float[Array, "T N"]:  # ty:ignore[unresolved-reference]
        """Step the forward model ``n_steps`` times, returning the trajectory."""
        dt = self.forward.dt

        def step_fn(
            x: Float[Array, N],  # ty:ignore[unresolved-reference]
            _: None,
        ) -> tuple[Float[Array, N], Float[Array, N]]:  # ty:ignore[unresolved-reference]
            x_new = self.forward.step(x, dt)
            return x_new, x_new

        _, trajectory = jax.lax.scan(step_fn, x_0, None, length=n_steps)
        # Prepend x_0 so the trajectory has length T+1.
        return jnp.concatenate([x_0[None, :], trajectory], axis=0)

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, "B N"]:
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

            def _one_points(b: PointObsBatch) -> Float[Array, N]:  # ty:ignore[unresolved-reference]
                def obs_cost(x_0: Float[Array, N]) -> Float[Array, ""]:  # ty:ignore[unresolved-reference]
                    trajectory = self._rollout(x_0, n_steps=n_steps)
                    return grid_point_obs_cost(self.obs_op, trajectory, b)

                return self._solve(obs_cost)

            return map_point_batch(_one_points, batch)

        T = batch.input.shape[1] - 1  # rollout steps (input has T+1 timesteps)

        def _one(
            input_i: Float[Array, "T_plus_1 N"], mask_i: Float[Array, "T_plus_1 N"]
        ) -> Float[Array, N]:  # ty:ignore[unresolved-reference]
            def obs_cost(x_0: Float[Array, N]) -> Float[Array, ""]:  # ty:ignore[unresolved-reference]
                # Roll out the trajectory.
                trajectory = self._rollout(x_0, n_steps=T)

                # Observation term, summed across time.
                def _per_step(x_t, y_t, m_t):
                    y_pred = (
                        self.obs_op(x_t, mask=m_t)
                        if _accepts_mask(self.obs_op)
                        else self.obs_op(x_t)
                    )
                    residual = m_t * (y_t - y_pred)
                    R_inv_r = lx.linear_solve(
                        self.obs_cov_op,
                        residual,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    return 0.5 * jnp.sum(residual * R_inv_r)

                per_step_costs = jax.vmap(_per_step)(trajectory, input_i, mask_i)
                return jnp.sum(per_step_costs)

            return self._solve(obs_cost)

        return jax.vmap(_one)(batch.input, batch.mask)

    def _solve(
        self,
        obs_cost: Callable[[Array], Array],
    ) -> Float[Array, N]:  # ty:ignore[unresolved-reference]
        """Minimise background (or penalty) plus ``obs_cost(x_0)``."""
        param = self.parameterization
        if param is not None:
            # x_0 = ψ(θ); J(θ) = pen(θ) + J_o(M(ψ(θ))).
            def decode(theta: Any) -> Float[Array, N]:  # ty:ignore[unresolved-reference]
                return param.decode(theta).reshape(self.prior_mean.shape)

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

        def cost(x_0: Float[Array, N], _args: Any) -> Float[Array, ""]:  # ty:ignore[unresolved-reference]
            # Background term
            dx = x_0 - self.prior_mean
            B_inv_dx = lx.linear_solve(
                self.prior_cov_op,
                dx,
                solver=lx.CG(atol=1e-6, rtol=1e-6),
            ).value
            j_bg = 0.5 * jnp.sum(dx * B_inv_dx)
            return j_bg + obs_cost(x_0)

        return minimise(
            self.minimiser,
            cost,
            self.prior_mean,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )

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
