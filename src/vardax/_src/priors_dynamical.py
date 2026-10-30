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

from vardax._src.errors import ShapeError
from vardax._src.utils.patches import time_patches

Solver = dfx.AbstractSolver
StepSizeController = dfx.AbstractStepSizeController
Adjoint = dfx.AbstractAdjoint


class DynamicalPrior(eqx.Module):
    r"""Base class for ODE-based dynamical priors.

    Wraps a diffrax ``diffeqsolve`` around an ODE right-hand side
    $\mathrm{d}x/\mathrm{d}t = f(t, x; \theta)$ with pluggable solver, step-size
    controller, and adjoint strategy. The adjoint choice trades memory against
    speed for reverse-mode differentiation through the solve (relevant for long
    windows).

    Concrete subclasses implement ``__call__`` (the integration) and
    ``loss`` (the dynamical residual). Instantiating ``DynamicalPrior``
    directly and calling it raises ``NotImplementedError``.

    Args:
        model: Diffrax-compatible ODE right-hand side ``f(t, y, args) -> dy``.
        params: Optional default ``args`` threaded to ``model`` /
            ``diffeqsolve`` (e.g. learnable ODE parameters $\theta$).
            Overridable per call.
        solver: Diffrax solver (default :class:`diffrax.Tsit5`).
        stepsize: Diffrax step-size controller. Defaults to an adaptive
            :class:`diffrax.PIDController` for solvers that provide error
            estimates (:class:`diffrax.AbstractAdaptiveSolver` subclasses),
            and to :class:`diffrax.ConstantStepSize` for fixed-step solvers
            such as :class:`diffrax.Euler`, which cannot drive an adaptive
            controller.
        adjoint: Diffrax adjoint strategy (default
            :class:`diffrax.RecursiveCheckpointAdjoint`).

    Examples:
        The concrete subclasses share the constructor; the step-size controller
        follows the solver:

        >>> import diffrax as dfx
        >>> import vardax as vdx
        >>> def decay(t, y, args):
        ...     return -y
        >>> type(vdx.DynTrajectory(model=decay).stepsize).__name__
        'PIDController'
        >>> euler = vdx.DynTrajectory(model=decay, solver=dfx.Euler())
        >>> type(euler.stepsize).__name__
        'ConstantStepSize'

        The base class itself has no integration to run:

        >>> import jax.numpy as jnp
        >>> vdx.DynamicalPrior(model=decay)(
        ...     jnp.ones(2), jnp.array([0.0, 1.0])
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        NotImplementedError
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
        if stepsize is not None:
            self.stepsize = stepsize
        elif isinstance(self.solver, dfx.AbstractAdaptiveSolver):
            self.stepsize = dfx.PIDController(rtol=1e-5, atol=1e-5)
        else:
            # Fixed-step solvers (e.g. Euler) provide no error estimate and
            # cannot drive an adaptive controller.
            self.stepsize = dfx.ConstantStepSize()
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
        adjoint: Adjoint | None = None,
    ) -> PyTree:
        """Integrate ``model`` from ``ts[0]`` to ``ts[-1]`` starting at ``y0``.

        ``adjoint`` overrides this prior's ``adjoint`` for one call.
        """
        start_time = ts[0]
        end_time = ts[-1]
        initial_step = ts[1] - ts[0] if dt is None else dt
        solution = dfx.diffeqsolve(
            terms=dfx.ODETerm(self.model),
            solver=self.solver,
            t0=start_time,
            t1=end_time,
            dt0=initial_step,
            y0=y0,
            saveat=saveat,
            adjoint=self.adjoint if adjoint is None else adjoint,
            stepsize_controller=self.stepsize,
            args=params if params is not None else self.params,
        )
        return solution.ys

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

    def reconstruct(
        self,
        x: Array,
        ts: Array,
        params: PyTree | None = None,
    ) -> Array:
        r"""Dynamics-consistent reconstruction of a state sequence.

        Returns an array of the *same shape* as ``x`` such that
        $\|x - \mathrm{reconstruct}(x, t_s)\|^2$ is the prior's dynamical residual,
        the temporal analogue of the autoencoder reconstruction used by the
        static [`Prior`][vardax.Prior] seam (Decision D18).

        Args:
            x: State sequence.
            ts: Time coordinates $t_s$ of the sequence.
            params: ODE ``args`` override.

        Returns:
            The dynamics-consistent sequence, with the shape of ``x``.
        """
        raise NotImplementedError

    def bind(
        self,
        ts: Array,
        params: PyTree | None = None,
    ) -> _BoundTemporalPrior:
        """Bind a time grid, yielding a one-argument `Prior` adapter.

        The returned callable satisfies the static
        [`Prior`][vardax.Prior] protocol (``__call__(x) -> x_prior``) by
        closing over ``ts``, so a dynamical prior can be used anywhere a
        static prior is expected, e.g. as the ``prior_fn`` of
        [`variational_cost`][vardax.variational_cost], turning the prior
        term into a weak-constraint dynamical residual.

        Args:
            ts: Time coordinates $t_s$ to close over.
            params: ODE ``args`` override to close over.

        Returns:
            A `Prior`-conforming module wrapping this temporal prior.

        Examples:
            >>> import jax.numpy as jnp
            >>> from vardax import DynTrajectory, IdentityPrior, Prior
            >>> def decay(t, y, args):
            ...     return -y
            >>> ts = jnp.linspace(0.0, 0.5, 6)
            >>> bound = DynTrajectory(model=decay).bind(ts)
            >>> isinstance(bound, Prior)
            True
            >>> traj = DynTrajectory(model=decay)(jnp.ones(3), ts)
            >>> bool(jnp.allclose(bound(traj), traj, atol=1e-4))
            True

            A sequence that violates the dynamics is pulled back onto the flow, so
            its reconstruction differs from it:

            >>> bool(jnp.allclose(bound(traj + 1.0), traj + 1.0, atol=1e-4))
            False
        """
        return _BoundTemporalPrior(prior=self, ts=ts, params=params)

    def as_forward_model(
        self, dt: float, adjoint: Adjoint | None = None
    ) -> _DynamicalForwardModel:
        r"""Adapt this prior's dynamics to ``pipekit_cycle.ForwardModel``.

        The returned adapter exposes ``step(state, dt)``, ``dt``, and
        ``state_signature``, so the wrapped ODE can drive anything that
        consumes the pipekit forward-model seam,
        [`StrongFourDVar`][vardax.StrongFourDVar], ``pipekit_cycle.DACycle``,
        etc. (Decisions D8/D18). ``step`` integrates the ODE over
        $[0, \Delta t]$; the dynamics are therefore treated as autonomous
        (time-shift invariant), which holds for the standard testbeds
        (Lorenz-63/96) and any RHS that ignores ``t``.

        Each step integrates one short interval, so the adapter
        differentiates it with ``diffrax.DirectAdjoint``, which supports
        forward mode (``jax.linearize``, as
        [`IncrementalFourDVar`][vardax.IncrementalFourDVar] needs) as
        well as reverse mode. The prior's own ``adjoint`` (by default
        ``RecursiveCheckpointAdjoint``, reverse mode only) still drives
        its long-window ``__call__``.

        Args:
            dt: Default integration step $\Delta t$ advertised as the adapter's ``dt``.
            adjoint: Diffrax adjoint of each step. Default ``None``:
                ``diffrax.DirectAdjoint()``.

        Returns:
            A ``ForwardModel``-conforming adapter around this prior.

        Examples:
            One step of exponential decay matches the closed form:

            >>> import jax.numpy as jnp
            >>> import vardax as vdx
            >>> def decay(t, y, args):
            ...     return -y
            >>> fwd = vdx.DynTrajectory(model=decay).as_forward_model(0.1)
            >>> fwd.dt, fwd.state_signature
            (0.1, None)
            >>> x = jnp.array([1.0, 2.0])
            >>> bool(jnp.allclose(fwd.step(x, 0.1), x * jnp.exp(-0.1), atol=1e-4))
            True

            A zero step leaves the state alone:

            >>> bool(jnp.allclose(fwd.step(x, 0.0), x))
            True
        """
        return _DynamicalForwardModel(
            prior=self,
            dt=dt,
            adjoint=dfx.DirectAdjoint() if adjoint is None else adjoint,
        )


class DynIncrements(DynamicalPrior):
    r"""One-step-increment dynamical prior.

    The loss integrates each state a *single* step forward and compares it
    to the next observed state, so the dynamics act locally in time:

    $$
    R(u; \theta) = \sum_t \| u_{t+1} - \varphi_{\Delta t}(u_t; \theta) \|^2 .
    $$

    Algorithm:
        1. Split the time grid into consecutive pairs
           $(t_t, t_{t+1})$ with [`time_patches`][vardax.time_patches].
        2. Propagate every state $u_t$ one step with the ODE solver, in
           parallel over $t$.
        3. Sum the squared differences to the next states.

    Args:
        model: Diffrax-compatible ODE right-hand side ``f(t, y, args) -> dy``.
        params: Optional default ``args`` threaded to ``model`` (e.g. $\theta$).
        solver: Diffrax solver (default :class:`diffrax.Tsit5`).
        stepsize: Diffrax step-size controller (adaptive by default for
            adaptive solvers).
        adjoint: Diffrax adjoint strategy.

    Raises:
        ShapeError: From ``loss``, if the lengths of ``x``, ``ts`` and ``x_gt``
            are inconsistent.

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

        A perturbed trajectory is penalised:

        >>> bool(DynIncrements(model=decay).loss(traj + 1.0, ts) > 1e-3)
        True

        A time grid that does not match the sequence is rejected:

        >>> DynIncrements(model=decay).loss(
        ...     traj[:3], ts
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapeError: Size mismatch for DynIncrements.loss: expected ...
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
            x: State at ``ts[0]``.
            ts: Integration endpoints (or more; only the first and last are used).
            dt: Initial step size. Defaults to ``ts[1] - ts[0]``.
            params: ODE ``args`` override (defaults to ``self.params``).

        Returns:
            State at the final time.
        """
        saveat = dfx.SaveAt(ts=jnp.asarray([ts[-1]]))
        saved_states = self._integrate(x, ts, saveat, dt, params)
        return saved_states[-1]

    def loss(
        self,
        x: Array,
        ts: Array,
        x_gt: Array | None = None,
        params: PyTree | None = None,
    ) -> Array:
        r"""One-step-increment dynamical loss.

        Args:
            x: State sequence whose entries are the sources for the one-step
                predictions.
            ts: Time coordinates of the sequence.
            x_gt: Target state sequence. Defaults to ``x`` (self-consistency).
            params: ODE ``args`` override.

        Returns:
            Scalar dynamical residual $\sum_t \|\varphi(u_t) - u_{t+1}\|^2$.

        Raises:
            ShapeError: If ``len(x) - 1``, the number of time pairs and
                ``len(x_gt) - 1`` differ.
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
            raise ShapeError(msg)
        propagate = ft.partial(self, params=params)
        predicted_states = jax.vmap(propagate, in_axes=(0, 0))(x[:-1], ts_pairs)
        # R = Σ_t ‖φ(u_t) − u_{t+1}‖²
        return jnp.sum((predicted_states - x_gt[1:]) ** 2)

    def reconstruct(
        self,
        x: Array,
        ts: Array,
        params: PyTree | None = None,
    ) -> Array:
        r"""One-step-increment reconstruction.

        ``reconstruct(x, ts)[t+1]`` is the one-step propagation
        $\varphi_{\Delta t}(x_t)$ and ``reconstruct(x, ts)[0] = x[0]``, so
        $\|x - \text{reconstruct}(x, t_s)\|^2$ equals the increment
        residual of [`loss`][vardax.DynIncrements.loss] (the $t=0$ term vanishes).

        Args:
            x: State sequence.
            ts: Time coordinates $t_s$ of the sequence.
            params: ODE ``args`` override.

        Returns:
            The sequence $[x_0, \varphi(x_0), \dots, \varphi(x_{T-2})]$.

        Examples:
            The residual of the reconstruction is the increment loss:

            >>> import jax.numpy as jnp
            >>> from vardax import DynIncrements
            >>> def decay(t, y, args):
            ...     return -y
            >>> prior = DynIncrements(model=decay)
            >>> ts = jnp.linspace(0.0, 0.5, 6)
            >>> x = jnp.ones((6, 2)) * jnp.arange(6.0)[:, None]
            >>> rec = prior.reconstruct(x, ts)
            >>> rec.shape == x.shape, bool(jnp.all(rec[0] == x[0]))
            (True, True)
            >>> bool(
            ...     jnp.allclose(jnp.sum((x - rec) ** 2), prior.loss(x, ts), rtol=1e-4)
            ... )
            True
        """
        ts_pairs = time_patches(ts)
        propagate = ft.partial(self, params=params)
        predicted_states = jax.vmap(propagate, in_axes=(0, 0))(x[:-1], ts_pairs)
        # x̂ = [x₀; φ(x₀), …, φ(x_{T−1})]
        # jnp, not einx: a one-state trajectory has no predictions to size
        return jnp.concatenate([x[:1], predicted_states])


class DynTrajectory(DynamicalPrior):
    r"""Full-rollout dynamical prior.

    The loss integrates the initial state across the whole window and
    compares the resulting trajectory to the target sequence, enforcing the
    dynamics as a global (strong) constraint:

    $$
    R(u; \theta) = \sum_t \| u_t - \varphi_t(u_0; \theta) \|^2 .
    $$

    This is the propagation used by strong-constraint 4DVar, see
    [`strong_variational_cost`][vardax.strong_variational_cost].

    Algorithm:
        1. Integrate the ODE from the initial state $u_0$, saving the state at
           every time in ``ts``.
        2. Sum the squared differences between the saved trajectory and the
           target sequence.

    Args:
        model: Diffrax-compatible ODE right-hand side ``f(t, y, args) -> dy``.
        params: Optional default ``args`` threaded to ``model`` (e.g. $\theta$).
        solver: Diffrax solver (default :class:`diffrax.Tsit5`).
        stepsize: Diffrax step-size controller (adaptive by default for
            adaptive solvers).
        adjoint: Diffrax adjoint strategy.

    Examples:
        Roll out exponential decay from an initial condition:

        >>> import jax.numpy as jnp
        >>> from vardax import DynTrajectory
        >>> def decay(t, y, args):
        ...     return -y
        >>> ts = jnp.linspace(0.0, 0.5, 6)
        >>> prior = DynTrajectory(model=decay)
        >>> traj = prior(jnp.array([1.0, 2.0]), ts)
        >>> traj.shape
        (6, 2)
        >>> bool(
        ...     jnp.allclose(traj[-1], jnp.array([1.0, 2.0]) * jnp.exp(-0.5), atol=1e-3)
        ... )
        True

        A trajectory generated by the model has (near-)zero loss, and a shifted
        one does not:

        >>> bool(prior.loss(traj, ts) < 1e-6)
        True
        >>> bool(prior.loss(traj + 1.0, ts, x_gt=traj) > 1.0)
        True

        Through [`bind`][vardax.DynamicalPrior.bind] it plugs into the static
        prior seam:

        >>> bool(jnp.allclose(prior.bind(ts)(traj), traj, atol=1e-4))
        True
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
            x: Initial condition (state at ``ts[0]``).
            ts: Time coordinates at which to save the state.
            dt: Initial step size. Defaults to ``ts[1] - ts[0]``.
            params: ODE ``args`` override.

        Returns:
            State trajectory with ``traj[0] == x``, one row per entry of ``ts``.
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
            x: State sequence; the initial condition ``x[0]`` seeds the rollout.
            ts: Time coordinates of the sequence.
            x_gt: Target trajectory. Defaults to ``x``.
            params: ODE ``args`` override.

        Returns:
            Scalar dynamical residual $\sum_t \|\varphi_t(u_0) - u_t\|^2$.
        """
        if x_gt is None:
            x_gt = x
        predicted_states = self(x[0], ts, params=params)
        # R = Σ_t ‖φ_t(u_0) − u_t‖²
        return jnp.sum((predicted_states - x_gt) ** 2)

    def reconstruct(
        self,
        x: Array,
        ts: Array,
        params: PyTree | None = None,
    ) -> Array:
        r"""Full-rollout reconstruction: the trajectory from ``x[0]``.

        $\|x - \text{reconstruct}(x, t_s)\|^2$ equals the trajectory
        residual of [`loss`][vardax.DynTrajectory.loss].

        Args:
            x: State sequence; ``x[0]`` seeds the rollout.
            ts: Time coordinates $t_s$ of the sequence.
            params: ODE ``args`` override.

        Returns:
            The rolled-out trajectory, same shape as ``x``.
        """
        return self(x[0], ts, params=params)


# ---------------------------------------------------------------------------
# Adapters (Decision D18)
# ---------------------------------------------------------------------------


class _BoundTemporalPrior(eqx.Module):
    """A temporal prior with its time grid bound: satisfies `Prior`.

    Returned by :meth:`DynamicalPrior.bind`; not constructed directly.
    """

    prior: DynamicalPrior
    ts: Array
    params: PyTree | None

    def __call__(self, x: Array) -> Array:
        """One-argument reconstruction, per the static `Prior` seam."""
        return self.prior.reconstruct(x, self.ts, params=self.params)


class _DynamicalForwardModel(eqx.Module):
    """``pipekit_cycle.ForwardModel`` adapter over a `DynamicalPrior`.

    Returned by :meth:`DynamicalPrior.as_forward_model`; not constructed
    directly. ``step`` integrates the wrapped ODE over ``[0, dt]``
    (autonomous dynamics).
    """

    prior: DynamicalPrior
    dt: float
    adjoint: Adjoint

    def step(self, state: Array, dt: float) -> Array:
        """Advance ``state`` by ``dt`` through the wrapped ODE."""
        ts = jnp.asarray([0.0, dt])
        saveat = dfx.SaveAt(t1=True)
        saved_states = self.prior._integrate(
            state, ts, saveat, dt, None, adjoint=self.adjoint
        )
        return saved_states[-1]

    @property
    def state_signature(self) -> None:
        """No named-dimension tracking (see ``pipekit.Signature``)."""
        return None
