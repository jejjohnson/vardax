r"""ODE-based dynamical priors (ported from ``mfourdvar``).

A *dynamical prior* treats a differential-equation forward model as the
regulariser of a variational assimilation: it penalises state sequences
that are inconsistent with the flow of an ODE ``dx/dt = f(t, x; θ)``.
This is the missing physics-informed ingredient that enables
strong/weak-constraint 4DVar with a (possibly learnable) ODE model.

Ported from ``mfourdvar._src.priors.dynamical`` and adapted to vardax's
conventions:

- Implemented as :class:`equinox.Module` (matching the rest of vardax).
  The ``flax.linen`` note in issue #14 predates the vardax rewrite; the
  legacy code was itself Equinox-based.
- The forward model ``model`` is any diffrax-compatible ODE right-hand
  side — a callable ``f(t, y, args) -> dy`` operating directly on state
  arrays (e.g. [`Lorenz63`][vardax.utils.dynamical_systems.Lorenz63] /
  ``Lorenz96``). The legacy ``init_state`` / ``.array`` indirection is
  dropped in favour of plain arrays.
- ``diffrax`` is already a core dependency.

Two concrete variants are provided:

- [`DynIncrements`][vardax.DynIncrements] — one-step increments. The loss
  compares each state to a *single* ODE step from the previous state:

  $$
  R(u; \theta) = \sum_t \| u_{t+1} - \varphi_{\Delta t}(u_t; \theta) \|^2 .
  $$

- [`DynTrajectory`][vardax.DynTrajectory] — full rollout. The loss
  compares the whole trajectory to a single integration from the initial
  state:

  $$
  R(u; \theta) = \sum_t \| u_t - \varphi_t(u_0; \theta) \|^2 .
  $$
"""

from __future__ import annotations

import functools as ft
import typing as tp

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from vardax._src.utils.patches import time_patches

Solver = dfx.AbstractSolver
StepSizeController = dfx.AbstractStepSizeController
Adjoint = dfx.AbstractAdjoint


class DynamicalPrior(eqx.Module):
    """Base class for ODE-based dynamical priors.

    Wraps a diffrax ``diffeqsolve`` around an ODE right-hand side with
    pluggable solver, step-size controller, and adjoint strategy. The
    adjoint choice trades memory against speed for reverse-mode
    differentiation through the solve (relevant for long windows).

    Concrete subclasses implement ``__call__`` (the integration) and
    ``loss`` (the dynamical residual). Instantiating ``DynamicalPrior``
    directly and calling it raises ``NotImplementedError``.

    Attributes:
        model: Diffrax-compatible ODE right-hand side ``f(t, y, args) -> dy``.
        params: Optional default ``args`` threaded to ``model`` /
            ``diffeqsolve`` (e.g. learnable ODE parameters ``θ``).
            Overridable per call.
        solver: Diffrax solver (default :class:`diffrax.Tsit5`).
        stepsize: Diffrax step-size controller (default adaptive
            :class:`diffrax.PIDController`).
        adjoint: Diffrax adjoint strategy (default
            :class:`diffrax.RecursiveCheckpointAdjoint`).
    """

    model: tp.Callable
    params: PyTree | None
    solver: Solver
    stepsize: StepSizeController
    adjoint: Adjoint

    def __init__(
        self,
        model: tp.Callable,
        params: PyTree | None = None,
        solver: Solver | None = None,
        stepsize: StepSizeController | None = None,
        adjoint: Adjoint | None = None,
    ) -> None:
        self.model = model
        self.params = params
        self.solver = solver if solver is not None else dfx.Tsit5()
        self.stepsize = (
            stepsize
            if stepsize is not None
            else dfx.PIDController(rtol=1e-5, atol=1e-5)
        )
        self.adjoint = (
            adjoint if adjoint is not None else dfx.RecursiveCheckpointAdjoint()
        )

    def _integrate(
        self,
        y0: PyTree,
        ts: Array,
        saveat: dfx.SaveAt,
        dt: float | None,
        params: PyTree | None,
    ) -> PyTree:
        """Integrate ``model`` from ``ts[0]`` to ``ts[-1]`` starting at ``y0``."""
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = ts[1] - ts[0] if dt is None else dt
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(self.model),
            solver=self.solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            adjoint=self.adjoint,
            stepsize_controller=self.stepsize,
            args=params if params is not None else self.params,
        )
        return sol.ys

    def __call__(
        self,
        x: Array,
        ts: Array,
        dt: float | None = None,
        params: PyTree | None = None,
    ) -> PyTree:
        raise NotImplementedError

    def loss(
        self,
        x: Array,
        ts: Array,
        x_gt: Array | None = None,
        params: PyTree | None = None,
    ) -> Array:
        raise NotImplementedError


class DynIncrements(DynamicalPrior):
    r"""One-step-increment dynamical prior.

    The loss integrates each state a *single* step forward and compares it
    to the next observed state, so the dynamics act locally in time:

    $$
    R(u; \theta) = \sum_t \| u_{t+1} - \varphi_{\Delta t}(u_t; \theta) \|^2 .
    $$

    Examples:
        A consistent trajectory (produced by rolling out the same model)
        has (near-)zero increment loss.

        >>> import jax.numpy as jnp
        >>> from vardax import DynIncrements, DynTrajectory
        >>> def decay(t, y, args):
        ...     return -y
        >>> ts = jnp.linspace(0.0, 0.5, 6)
        >>> x0 = jnp.array([1.0, 2.0, -1.0])
        >>> traj = DynTrajectory(model=decay)(x0, ts)
        >>> bool(DynIncrements(model=decay).loss(traj, ts) < 1e-3)
        True
    """

    def __call__(
        self,
        x: Array,
        ts: Array,
        dt: float | None = None,
        params: PyTree | None = None,
    ) -> Array:
        """Integrate ``x`` one step from ``ts[0]`` to ``ts[-1]``.

        Args:
            x: State at ``ts[0]`` of shape ``(*state,)``.
            ts: Integration endpoints of shape ``(2,)`` (or more; only the
                first and last are used).
            dt: Initial step size. Defaults to ``ts[1] - ts[0]``.
            params: ODE ``args`` override (defaults to ``self.params``).

        Returns:
            State at the final time, shape ``(*state,)``.
        """
        saveat = dfx.SaveAt(ts=jnp.asarray([ts[-1]]))
        ys = self._integrate(x, ts, saveat, dt, params)
        return ys[-1]

    def loss(
        self,
        x: Array,
        ts: Array,
        x_gt: Array | None = None,
        params: PyTree | None = None,
    ) -> Array:
        r"""One-step-increment dynamical loss.

        Args:
            x: State sequence of shape ``(T, *state)`` — the sources for the
                one-step predictions.
            ts: Time coordinates of shape ``(T,)``.
            x_gt: Target state sequence of shape ``(T, *state)``. Defaults to
                ``x`` (self-consistency).
            params: ODE ``args`` override.

        Returns:
            Scalar dynamical residual ``Σ_t ||φ(u_t) - u_{t+1}||²``.
        """
        if x_gt is None:
            x_gt = x
        ts_pairs = time_patches(ts)
        if not len(x) - 1 == len(ts_pairs) == len(x_gt) - 1:
            msg = (
                "Size mismatch for DynIncrements.loss: expected "
                f"len(x) - 1 == len(time_patches(ts)) == len(x_gt) - 1, got "
                f"{len(x)} | {len(ts_pairs)} | {len(x_gt)}"
            )
            raise ValueError(msg)
        fn = ft.partial(self, params=params)
        x_pred = jax.vmap(fn, in_axes=(0, 0))(x[:-1], ts_pairs)
        return jnp.sum((x_pred - x_gt[1:]) ** 2)


class DynTrajectory(DynamicalPrior):
    r"""Full-rollout dynamical prior.

    The loss integrates the initial state across the whole window and
    compares the resulting trajectory to the target sequence, enforcing the
    dynamics as a global (strong) constraint:

    $$
    R(u; \theta) = \sum_t \| u_t - \varphi_t(u_0; \theta) \|^2 .
    $$

    This is the propagation used by strong-constraint 4DVar — see
    [`strong_variational_cost`][vardax.strong_variational_cost].
    """

    def __call__(
        self,
        x: Array,
        ts: Array,
        dt: float | None = None,
        params: PyTree | None = None,
    ) -> Array:
        """Integrate the initial state ``x`` across all of ``ts``.

        Args:
            x: Initial condition (state at ``ts[0]``) of shape ``(*state,)``.
            ts: Time coordinates of shape ``(T,)`` at which to save the state.
            dt: Initial step size. Defaults to ``ts[1] - ts[0]``.
            params: ODE ``args`` override.

        Returns:
            State trajectory of shape ``(T, *state)`` (``traj[0] == x``).
        """
        saveat = dfx.SaveAt(ts=ts)
        return self._integrate(x, ts, saveat, dt, params)

    def loss(
        self,
        x: Array,
        ts: Array,
        x_gt: Array | None = None,
        params: PyTree | None = None,
    ) -> Array:
        r"""Full-trajectory dynamical loss.

        Args:
            x: State sequence of shape ``(T, *state)``; the initial condition
                ``x[0]`` seeds the rollout.
            ts: Time coordinates of shape ``(T,)``.
            x_gt: Target trajectory of shape ``(T, *state)``. Defaults to
                ``x``.
            params: ODE ``args`` override.

        Returns:
            Scalar dynamical residual ``Σ_t ||φ_t(u_0) - u_t||²``.
        """
        if x_gt is None:
            x_gt = x
        x_pred = self(x[0], ts, params=params)
        return jnp.sum((x_pred - x_gt) ** 2)
