r"""Reduced-basis 4DVar with optional `ReducedBasis` control parameterisation.

Concrete `vardax.protocols.AnalysisStep` (via ``.as_analysis_step()``).
Distinct from `vardax.IncrementalFourDVar` (the operational GN-CG fast
path): this class minimises the **full nonlinear** 4DVar cost via an
``optimistix.AbstractMinimiser``, optionally parameterising the analysis
increment through a reduced basis (Gaussian RBFs, BM, wavelets) — the
classical MASSH / Le Guillou shape.

Cost (basis-less state-space mode):

$$
J(x_0) = \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}
       + \tfrac{1}{2} \sum_{t=0}^{T} \|m_t \odot
                      (y_t - H(M^t(x_0)))\|^2_{R^{-1}}.
$$

With a `ReducedBasis` $\Phi$, the control becomes the coefficient vector
$X$, the state increment is $\delta x_0 = \Phi X$, the prior is the
separable $\tfrac{1}{2} X^\top Q^{-1} X$, and the forward integrates from
$x_b + \Phi X$:

$$
J(X) = \tfrac{1}{2} X^\top Q^{-1} X
     + \tfrac{1}{2} \sum_t \|m_t \odot
                   (y_t - H(M^t(x_b + \Phi X)))\|^2_{R^{-1}}.
$$

Minimisation is delegated to ``optimistix`` (default
``optx.BFGS(rtol=atol=1e-6)``); differentiating through the analysis is
controlled by ``minimiser_adjoint`` (default ``optx.ImplicitAdjoint``)
— matching `StrongFourDVar` / `WeakFourDVar` / `ThreeDVar`.

References:
- MASSH `mapping/src/tools_4Dvar.py:Variational` and
  `mapping/src/inv.py:Inv_4Dvar` (reduced-basis 4DVar with L-BFGS-B).
- vardax `StrongFourDVar` (state-space conventions, ``optx`` plumbing).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx
import optimistix as optx

from vardax._src._types import Batch1D


class Incremental4DVar(eqx.Module):
    """Reduced-basis 4DVar.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel`` supplying
            ``step(state, dt) -> state``.
        obs_op: Observation operator (linear or nonlinear). Supplies
            ``__call__`` and ``linearize``; ``__call__`` may accept a
            ``mask`` kwarg (`MaskedIdentity` / `InterpObs` style).
        prior_mean: Background $x_b$ of shape ``(N,)``.
        prior_cov_op: $B$. Used only when ``basis is None``.
        obs_cov_op: $R$.
        basis: Optional `ReducedBasis`. When supplied, the control is
            the coefficient vector ``X`` (shape ``(basis.nbasis,)``)
            and ``B^{-1}`` is replaced by ``basis.prior_inv``.
        minimiser: ``optimistix.AbstractMinimiser`` for the outer
            optimisation. Default ``optx.BFGS(rtol=1e-6, atol=1e-6)``.
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for
            differentiating through the minimum. Default
            ``optx.ImplicitAdjoint``.
        max_steps: Iteration cap on the outer solver.
    """

    forward: Any
    obs_op: Any
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    basis: Any
    minimiser: optx.AbstractMinimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)

    def __init__(
        self,
        forward: Any,
        obs_op: Any,
        prior_mean: Float[Array, " N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        *,
        basis: Any = None,
        minimiser: optx.AbstractMinimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
    ) -> None:
        self.forward = forward
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.basis = basis
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps

    def _rollout(
        self,
        x_0: Float[Array, " N"],
        n_steps: int,
    ) -> Float[Array, "T_plus_1 N"]:
        """Step the forward ``n_steps`` times, returning the ``T+1``-long trajectory."""
        dt = self.forward.dt

        def step_fn(x, _):
            x_new = self.forward.step(x, dt)
            return x_new, x_new

        _, trajectory = jax.lax.scan(step_fn, x_0, None, length=n_steps)
        return jnp.concatenate([x_0[None, :], trajectory], axis=0)

    def __call__(self, batch: Batch1D) -> Float[Array, "B N"]:
        """Reduced-basis 4DVar analysis: minimise over ``x_0`` (or ``X``)."""

        T = batch.input.shape[1] - 1

        def _one(
            input_i: Float[Array, "T_plus_1 N"],
            mask_i: Float[Array, "T_plus_1 N"],
        ) -> Float[Array, " N"]:
            def _obs_cost(
                trajectory: Float[Array, "T_plus_1 N"],
            ) -> Float[Array, ""]:
                def _per_step(x_t, y_t, m_t):
                    y_pred = (
                        self.obs_op(x_t, mask=m_t)
                        if _accepts_mask(self.obs_op)
                        else self.obs_op(x_t)
                    )
                    residual = m_t * (y_t - y_pred)
                    # Solve against the marginalized covariance
                    # R~ = M R M + (I - M): for a 0/1 mask its inverse on
                    # the observed block equals (R_oo)^{-1}, so masked
                    # entries drop out of the likelihood exactly. Solving
                    # the full R against a zeroed residual would instead
                    # weight observed entries by the observed block of
                    # R^{-1}, biasing the analysis when R is correlated.
                    mask_op = lx.DiagonalLinearOperator(m_t)
                    r_masked = lx.TaggedLinearOperator(
                        mask_op @ self.obs_cov_op @ mask_op
                        + lx.DiagonalLinearOperator(1.0 - m_t),
                        lx.positive_semidefinite_tag,
                    )
                    R_inv_r = lx.linear_solve(
                        r_masked,
                        residual,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    return 0.5 * jnp.sum(residual * R_inv_r)

                per_step_costs = jax.vmap(_per_step)(trajectory, input_i, mask_i)
                return jnp.sum(per_step_costs)

            if self.basis is None:

                def cost(x_0: Float[Array, " N"], _args: Any) -> Float[Array, ""]:
                    dx = x_0 - self.prior_mean
                    B_inv_dx = lx.linear_solve(
                        self.prior_cov_op,
                        dx,
                        solver=lx.CG(atol=1e-6, rtol=1e-6),
                    ).value
                    j_bg = 0.5 * jnp.sum(dx * B_inv_dx)
                    return j_bg + _obs_cost(self._rollout(x_0, n_steps=T))

                y0 = self.prior_mean
            else:

                def cost(X: Float[Array, " M"], _args: Any) -> Float[Array, ""]:
                    j_bg = 0.5 * jnp.sum(X * self.basis.prior_inv(X))
                    x_0 = self.prior_mean + self.basis.operg(0.0, X).reshape(
                        self.prior_mean.shape
                    )
                    return j_bg + _obs_cost(self._rollout(x_0, n_steps=T))

                y0 = jnp.zeros((self.basis.nbasis,), dtype=self.prior_mean.dtype)

            result = optx.minimise(
                fn=cost,
                solver=self.minimiser,
                y0=y0,
                args=None,
                max_steps=self.max_steps,
                adjoint=self.minimiser_adjoint,
                throw=False,
            )
            if self.basis is None:
                return result.value
            return self.prior_mean + self.basis.operg(0.0, result.value).reshape(
                self.prior_mean.shape
            )

        return jax.vmap(_one)(batch.input, batch.mask)

    def as_analysis_step(self) -> _Incremental4DVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
        return _Incremental4DVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _Incremental4DVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``Incremental4DVar``.

    Handles both calling conventions: ``DACycle`` hands in single
    ``(N,)`` arrays, ``SmootherCycle`` hands in length-``window`` lists
    of ``(N,)`` arrays. Either way the adapter stacks to
    ``(1, T+1, N)`` for the model's batched ``__call__`` and threads
    the supplied ``forecast`` / ``obs_op`` / ``obs_err_cov`` into the
    model via ``eqx.tree_at``, so windowed cycles no longer regularise
    against the stale construction-time fields and runtime operator
    swaps are honoured. For ``SmootherCycle(window > 1)`` the optimised
    initial state is rolled forward to the end of the window before
    being returned as the next carrier.
    """

    model: Incremental4DVar

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        if isinstance(obs, list):
            obs_stack = jnp.stack([jnp.asarray(o) for o in obs])
            window = len(obs)
        else:
            obs_stack = jnp.asarray(obs)[None]
            window = 1
        mask = jnp.where(jnp.isfinite(obs_stack), 1.0, 0.0)
        obs_clean = jnp.nan_to_num(obs_stack)
        batch = Batch1D(input=obs_clean[None], mask=mask[None], target=None)

        bg = (
            jnp.asarray(forecast[0])
            if isinstance(forecast, list)
            else jnp.asarray(forecast)
        )
        swapped = eqx.tree_at(lambda m: m.prior_mean, self.model, bg)
        if obs_op is not None:
            swapped = eqx.tree_at(lambda m: m.obs_op, swapped, obs_op)
        if obs_err_cov is not None:
            swapped = eqx.tree_at(lambda m: m.obs_cov_op, swapped, obs_err_cov)

        analysed = swapped(batch)[0]
        if window > 1:
            dt = swapped.forward.dt

            def _step(x, _):
                return swapped.forward.step(x, dt), None

            final, _ = jax.lax.scan(_step, analysed, None, length=window - 1)
            return final
        return analysed
