r"""Reduced-basis 4DVar with optional `ReducedBasis` control parameterisation.

Concrete `vardax.protocols.AnalysisStep` (via ``.as_analysis_step()``).
Distinct from `vardax.IncrementalFourDVar` (the operational Gauss–Newton
/ CG fast path): this class minimises the **full nonlinear** 4DVar cost
via an ``optimistix.AbstractMinimiser``, with no linearisation, outer
loop or inner CG, optionally parameterising the analysis increment
through a reduced basis (Gaussian RBFs, Fourier modes, wavelets, EOFs) —
the classical MASSH / Le Guillou shape.

This class was called ``Incremental4DVar`` until vardax 0.1.10, after
MASSH's name for the increment $\delta x_0 = \Phi X$. In vardax
"incremental" means Gauss–Newton, so the old name is a deprecated alias.

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

from collections.abc import Callable
from typing import Any

import equinox as eqx
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
from vardax._src.models._rollout import checkpoint_every, resolve_forward, rollout
from vardax._src.obs_operators.point import (
    check_point_obs_cov,
    grid_point_obs_cost,
    map_point_batch,
    require_obs_cov,
    window_steps,
)
from vardax._src.protocols import Minimiser, StateParameterization
from vardax._src.utils.covariance import as_covariance
from vardax._src.utils.shapes import as_shape


class ReducedBasisFourDVar(eqx.Module):
    r"""Reduced-basis 4DVar.

    $$J(X) = \tfrac12 X^\top Q^{-1} X + J_o(x_b + \Phi X)$$

    $$J_o(x_0) = \tfrac12 \sum_t \|m_t \odot (y_t - H(M^t(x_0)))\|^2_{R^{-1}}$$

    Without a basis the control is $x_0$ itself and the first term is
    $\tfrac12 \|x_0 - x_b\|^2_{B^{-1}}$.

    Algorithm:
        1. Pick the control and its map to the initial state: $x_0$ itself,
           the coefficients $X$ with $x_0 = x_b + \Phi X$ (``basis``), or
           $\theta$ with $x_0 = \psi(\theta)$ (``parameterization``).
        2. Roll the forward model out from $x_0$ for the window length.
        3. Add the prior term ($B^{-1}$, ``basis.prior_inv`` or the
           parameterization penalty) to the masked observation misfit
           summed over the trajectory. Masked entries are marginalised out
           of $R$, so a correlated $R$ is not biased by the gaps.
        4. Minimise the full nonlinear cost with ``minimiser`` (no
           linearisation or inner loop) and decode the minimiser to
           $x_0^*$.

    Attributes:
        forward: ``pipekit_cycle.ForwardModel`` supplying
            ``step(state, dt) -> state``.
        obs_op: Observation operator (linear or nonlinear). Supplies
            ``__call__`` and ``linearize``; ``__call__`` may accept a
            ``mask`` kwarg (`MaskedIdentity` / `InterpObs` style).
        prior_mean: Background $x_b$.
        prior_cov_op: $B$. Used only when ``basis is None``.
        obs_cov_op: $R$.
        basis: Optional `ReducedBasis`. When supplied, the control is
            the coefficient vector $X$ (``basis.nbasis`` entries)
            and $B^{-1}$ is replaced by ``basis.prior_inv`` — the same as
            ``parameterization=LinearParameterization(basis, prior_mean)``.
        parameterization: Optional
            [`StateParameterization`][vardax.StateParameterization]; the
            control is $\theta$, $x_0 = \psi(\theta)$ and
            $\mathrm{pen}(\theta)$ replaces the background term. Exclusive
            with ``basis``.
        minimiser: ``optimistix.AbstractMinimiser`` for the outer
            optimisation (default ``optx.BFGS(rtol=1e-6, atol=1e-6)``), or
            a [`Minimiser`][vardax.Minimiser] such as
            [`OptaxMinimiser`][vardax.OptaxMinimiser].
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for
            differentiating through the minimum. Default
            ``optx.ImplicitAdjoint``.
        max_steps: Iteration cap on the outer solver when ``minimiser``
            is an ``optimistix`` solver; a vardax `Minimiser` keeps its own.
        cov_solver: ``lineax`` solver for the covariance solves in the
            cost ($B^{-1}$, $R^{-1}$). Default
            ``lx.CG(atol=1e-6, rtol=1e-6)``, which suits large
            matrix-free operators but is inexact on an ill-conditioned
            covariance (for example a tiny $R$); pass
            ``lx.AutoLinearSolver(well_posed=True)`` or ``lx.Cholesky()``
            for an exact solve when the operator can be materialised.
        forward_adjoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]:
            a ``diffrax.AbstractAdjoint`` for a forward implementing
            [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint].
        checkpoint: As in [`StrongFourDVar`][vardax.StrongFourDVar]:
            ``False``, ``True`` or a block length ``k`` for
            ``jax.checkpoint`` over the rollout.

    Examples:
        Trivial dynamics, one timestep, $B = R = I$ and no basis: the
        cost is that of 3DVar, so $x_0^* = y / 2$.

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

        With a basis the increment is confined to $\mathrm{span}(\Phi)$.
        A single unit-variance constant mode gives
        $X^* = \tfrac{\sum_i y_i}{1 + N}$ and $x_0^* = X^* \mathbf{1}$:

        >>> basis = vardax.linear_basis(jnp.ones((4, 1)))
        >>> xa = vardax.ReducedBasisFourDVar(**common, basis=basis)(batch)
        >>> bool(jnp.allclose(xa, 4.0 / 5.0, atol=1e-3))
        True

        Composed with a Fourier basis from [`fourier_basis`][vardax.fourier_basis],
        the analysis is still a state-space array:

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
        """Step the forward ``n_steps`` times, returning the ``T+1``-long trajectory."""
        forward = resolve_forward(self.forward, self.forward_adjoint)
        return rollout(forward, initial_state, n_steps, checkpoint=self.checkpoint)

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, "*batch N"]:
        r"""Reduced-basis 4DVar analysis: minimise over ``x_0`` (or ``X``).

        A [`PointObsBatch`][vardax.PointObsBatch] needs
        ``obs_op=GridPointObs(...)``: observation $i$ is compared with the
        trajectory state at ``time_index[i]`` over a window of ``n_times``
        states, with error variance $\sigma_i^2$ (``obs_cov_op`` must
        be ``None`` or an identity).

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D] whose time axis
                holds the window, or a
                [`PointObsBatch`][vardax.PointObsBatch].

        Returns:
            The analysed initial state $x_0^*$ of every window.

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
            n_steps = window_steps(batch)

            def _one_points(window: PointObsBatch) -> Float[Array, " N"]:
                def _obs_cost(
                    trajectory: Float[Array, "T_plus_1 N"],
                ) -> Float[Array, ""]:
                    return grid_point_obs_cost(self.obs_op, trajectory, window)

                return self._solve(_obs_cost, n_steps)

            return map_point_batch(_one_points, batch)

        obs_cov_op = require_obs_cov(self.obs_cov_op)
        n_steps = batch.input.shape[1] - 1

        def _one(
            window_obs: Float[Array, "T_plus_1 M"],
            window_mask: Float[Array, "T_plus_1 M"],
        ) -> Float[Array, " N"]:
            def _obs_cost(
                trajectory: Float[Array, "T_plus_1 N"],
            ) -> Float[Array, ""]:
                def _per_step(state, observation, step_mask):
                    predicted_obs = (
                        self.obs_op(state, mask=step_mask)
                        if _accepts_mask(self.obs_op)
                        else self.obs_op(state)
                    )
                    # r = m ⊙ (y − H(x))
                    residual = step_mask * (observation - predicted_obs)
                    # Solve against the marginalized covariance
                    # R~ = M R M + (I - M): for a 0/1 mask its inverse on
                    # the observed block equals (R_oo)^{-1}, so masked
                    # entries drop out of the likelihood exactly. Solving
                    # the full R against a zeroed residual would instead
                    # weight observed entries by the observed block of
                    # R^{-1}, biasing the analysis when R is correlated.
                    mask_op = lx.DiagonalLinearOperator(step_mask)
                    masked_obs_cov = lx.TaggedLinearOperator(
                        mask_op @ obs_cov_op @ mask_op
                        + lx.DiagonalLinearOperator(1.0 - step_mask),
                        lx.positive_semidefinite_tag,
                    )
                    # R̃⁻¹r
                    weighted_residual = lx.linear_solve(
                        masked_obs_cov,
                        residual,
                        solver=self.cov_solver,
                    ).value
                    # J_o,t = ½ rᵀ R̃⁻¹ r
                    return 0.5 * jnp.sum(residual * weighted_residual)

                per_step_costs = jax.vmap(_per_step)(
                    trajectory, window_obs, window_mask
                )
                return jnp.sum(per_step_costs)

            return self._solve(_obs_cost, n_steps)

        return jax.vmap(_one)(fill_gaps(batch.input, batch.mask), batch.mask)

    def _solve(
        self, _obs_cost: Callable[[Array], Array], n_steps: int
    ) -> Float[Array, " N"]:
        """Minimise background (or penalty) plus the trajectory ``_obs_cost``."""
        param = self.parameterization
        if self.basis is not None:
            # The reduced-basis path, unchanged; it equals
            # LinearParameterization(basis, prior_mean) mathematically.
            def cost(coefficients: Float[Array, " M"], _args: Any) -> Float[Array, ""]:
                # J_b = ½ Xᵀ P⁻¹ X
                background_cost = 0.5 * jnp.sum(
                    coefficients * self.basis.prior_inv(coefficients)
                )
                basis_increment = self.basis.operg(0.0, coefficients)
                # x₀ = x_b + Φ X
                initial_state = self.prior_mean + as_shape(
                    basis_increment, self.prior_mean.shape
                )
                return background_cost + _obs_cost(
                    self._rollout(initial_state, n_steps=n_steps)
                )

            initial_control = jnp.zeros(
                (self.basis.nbasis,), dtype=self.prior_mean.dtype
            )
        elif param is not None:

            def cost(theta: Any, _args: Any) -> Float[Array, ""]:
                initial_state = as_shape(param.decode(theta), self.prior_mean.shape)
                return param.penalty(theta) + _obs_cost(
                    self._rollout(initial_state, n_steps=n_steps)
                )

            initial_control = param.init_control()
        else:

            def cost(initial_state: Float[Array, " N"], _args: Any) -> Float[Array, ""]:
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
                return background_cost + _obs_cost(
                    self._rollout(initial_state, n_steps=n_steps)
                )

            initial_control = self.prior_mean

        control_star = minimise(
            self.minimiser,
            cost,
            initial_control,
            adjoint=self.minimiser_adjoint,
            max_steps=self.max_steps,
        )
        if self.basis is not None:
            increment = self.basis.operg(0.0, control_star)
            # x₀* = x_b + Φ X*
            return self.prior_mean + as_shape(increment, self.prior_mean.shape)
        if param is not None:
            return as_shape(param.decode(control_star), self.prior_mean.shape)
        return control_star

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
            UnsupportedInputError: For a point batch, or for a model with
                ``basis=`` or ``parameterization=``, whose prior is not
                $B$.
        """
        from vardax._src.posterior._precision import model_posterior

        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _ReducedBasisFourDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8).

        Returns:
            An analysis step that accepts single arrays or windowed lists,
            swaps in the supplied forecast and operators, and returns the
            analysis (rolled to the window end for a smoother).
        """
        return _ReducedBasisFourDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _ReducedBasisFourDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``ReducedBasisFourDVar``.

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

    model: ReducedBasisFourDVar

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        return analyse_window(
            self.model,
            rollout_end,
            forecast=forecast,
            obs=obs,
            obs_op=obs_op,
            obs_err_cov=obs_err_cov,
        )
