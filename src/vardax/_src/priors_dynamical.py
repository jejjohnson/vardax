"""ODE-based dynamical priors: the flow Ψ_Δt of ẋ = f(t, x; θ) as a regulariser.

The right-hand side is a diffrax-compatible callable ``f(t, y, args) -> dy`` acting
directly on state arrays, and the priors score a state sequence by its mismatch
with the flow, either one step at a time (``DynIncrements``) or as a full rollout
(``DynTrajectory``).
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
    """Base class for ODE-based dynamical priors.

    Wraps a diffrax ``diffeqsolve`` around the ODE

        ẋ = f(t, x; θ),   Ψ_Δt(x; θ) = x(t + Δt) from x(t) = x

    with pluggable solver, step-size controller, and adjoint strategy. The
    adjoint choice trades memory against speed for reverse-mode differentiation
    through the solve (relevant for long windows).

    Concrete subclasses implement ``__call__`` (the integration) and
    ``loss`` (the dynamical residual). Instantiating ``DynamicalPrior``
    directly and calling it raises ``NotImplementedError``.

    Args:
        model: Diffrax-compatible ODE right-hand side ``f(t, y, args) -> dy``.
        params: Optional default ``args`` threaded to ``model`` /
            ``diffeqsolve`` (e.g. learnable ODE parameters θ).
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
        # t₀ and t₁ of the flow Ψ_{t₁−t₀}
        start_time = ts[0]
        end_time = ts[-1]
        # first step Δt₀ = t₁ − t₀ of the window unless ``dt`` is given
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
        """Integrate the dynamics; implemented by the subclasses."""
        raise NotImplementedError

    def loss(
        self,
        x: Array,
        ts: Array,
        x_gt: Array | None = None,
        params: PyTree | None = None,
    ) -> Array:
        """Dynamical residual 𝒥_p; implemented by the subclasses."""
        raise NotImplementedError

    def reconstruct(
        self,
        x: Array,
        ts: Array,
        params: PyTree | None = None,
    ) -> Array:
        """Dynamics-consistent reconstruction x̂ of a state sequence.

        Returns an array of the *same shape* as ``x`` such that
        ‖x − x̂‖² is the prior's dynamical residual, the temporal analogue of the
        autoencoder reconstruction x̂ = g_ae(x; θ) of the static
        [`Prior`][vardax.Prior] seam.

        Args:
            x: State sequence x.
            ts: Time coordinates of the sequence.
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
            ts: Time coordinates to close over.
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
        """Adapt this prior's dynamics to ``pipekit_cycle.ForwardModel``.

            step(x, Δt) = Ψ_Δt(x; θ)

        The returned adapter exposes ``step(state, dt)``, ``dt``, and
        ``state_signature``, so the wrapped ODE can drive anything that
        consumes the pipekit forward-model seam,
        [`StrongFourDVar`][vardax.StrongFourDVar], ``pipekit_cycle.DACycle``,
        etc. ``step`` integrates the ODE over [0, Δt]; the dynamics are therefore
        treated as autonomous (time-shift invariant), which holds for the standard
        testbeds (Lorenz-63/96) and any RHS that ignores ``t``.

        Each step integrates one short interval, so the adapter
        differentiates it with ``diffrax.DirectAdjoint``, which supports
        forward mode (``jax.linearize``, as
        [`IncrementalFourDVar`][vardax.IncrementalFourDVar] needs) as
        well as reverse mode. The prior's own ``adjoint`` (by default
        ``RecursiveCheckpointAdjoint``, reverse mode only) still drives
        its long-window ``__call__``.

        Args:
            dt: Default integration step Δt advertised as the adapter's ``dt``.
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
    """One-step-increment dynamical prior.

    The loss integrates each state a *single* step forward and compares it
    to the next observed state, so the dynamics act locally in time. It is a sum
    of squares, not a mean:

        𝒥_p(x; θ) = ∑ₜ ‖x_{t+1} − Ψ_Δt(xₜ; θ)‖²

    Algorithm:
        1. Split the time grid into consecutive pairs (tₜ, t_{t+1}) with
           [`time_patches`][vardax.time_patches].
        2. Propagate every state xₜ one step with the ODE solver, in
           parallel over t.
        3. Sum the squared differences to the next states.

    Args:
        model: Diffrax-compatible ODE right-hand side ``f(t, y, args) -> dy``.
        params: Optional default ``args`` threaded to ``model`` (e.g. θ).
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
        """Integrate ``x`` one step from ``ts[0]`` to ``ts[-1]``: Ψ_Δt(x; θ).

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
        """One-step-increment dynamical loss.

            𝒥_p = ∑ₜ ‖Ψ_Δt(xₜ; θ) − x_gt,t+1‖²

        Args:
            x: State sequence whose entries xₜ are the sources for the one-step
                predictions.
            ts: Time coordinates of the sequence.
            x_gt: Target state sequence. Defaults to ``x`` (self-consistency).
            params: ODE ``args`` override, the θ of the flow.

        Returns:
            Scalar dynamical residual 𝒥_p.

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
        # Ψ_Δt(xₜ; θ) for every pair (tₜ, t_{t+1}), in parallel over t
        predicted_states = jax.vmap(propagate, in_axes=(0, 0))(x[:-1], ts_pairs)
        # Ψ_Δt(xₜ; θ) − x_gt,t+1
        residual = predicted_states - x_gt[1:]
        # 𝒥_p = ∑ₜ ‖Ψ_Δt(xₜ; θ) − x_gt,t+1‖²
        return jnp.sum(residual**2)

    def reconstruct(
        self,
        x: Array,
        ts: Array,
        params: PyTree | None = None,
    ) -> Array:
        """One-step-increment reconstruction.

            x̂ = [x₀, Ψ_Δt(x₀; θ), …, Ψ_Δt(x_{T−2}; θ)]

        ``reconstruct(x, ts)[t+1]`` is the one-step propagation Ψ_Δt(xₜ; θ) and
        ``reconstruct(x, ts)[0] = x[0]``, so ‖x − x̂‖² equals the increment
        residual of [`loss`][vardax.DynIncrements.loss] (the t = 0 term vanishes).

        Args:
            x: State sequence x.
            ts: Time coordinates of the sequence.
            params: ODE ``args`` override, the θ of the flow.

        Returns:
            The reconstruction x̂, with the shape of ``x``.

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
        # Ψ_Δt(xₜ; θ) for every pair (tₜ, t_{t+1}), in parallel over t
        predicted_states = jax.vmap(propagate, in_axes=(0, 0))(x[:-1], ts_pairs)
        # x̂ = [x₀, Ψ_Δt(x₀; θ), …, Ψ_Δt(x_{T−2}; θ)]
        # jnp, not einx: a one-state trajectory has no predictions to size
        return jnp.concatenate([x[:1], predicted_states])


class DynTrajectory(DynamicalPrior):
    """Full-rollout dynamical prior.

    The loss integrates the initial state across the whole window and
    compares the resulting trajectory to the target sequence, enforcing the
    dynamics as a global (strong) constraint. It is a sum of squares, not a mean:

        𝒥_p(x; θ) = ∑ₜ ‖xₜ − Ψ_{tₜ−t₀}(x₀; θ)‖²

    This is the propagation used by strong-constraint 4DVar, see
    [`strong_variational_cost`][vardax.strong_variational_cost].

    Algorithm:
        1. Integrate the ODE from the initial state x₀, saving the state at
           every time in ``ts``.
        2. Sum the squared differences between the saved trajectory and the
           target sequence.

    Args:
        model: Diffrax-compatible ODE right-hand side ``f(t, y, args) -> dy``.
        params: Optional default ``args`` threaded to ``model`` (e.g. θ).
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
        """Integrate the initial state ``x`` across all of ``ts``: Ψ_{tₜ−t₀}(x₀; θ).

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
        """Full-trajectory dynamical loss.

            𝒥_p = ∑ₜ ‖Ψ_{tₜ−t₀}(x₀; θ) − x_gt,t‖²

        Args:
            x: State sequence; the initial condition ``x[0]`` seeds the rollout.
            ts: Time coordinates of the sequence.
            x_gt: Target trajectory. Defaults to ``x``.
            params: ODE ``args`` override, the θ of the flow.

        Returns:
            Scalar dynamical residual 𝒥_p.
        """
        if x_gt is None:
            x_gt = x
        # Ψ_{tₜ−t₀}(x₀; θ): only the initial state seeds the rollout
        predicted_states = self(x[0], ts, params=params)
        # Ψ_{tₜ−t₀}(x₀; θ) − x_gt,t
        residual = predicted_states - x_gt
        # 𝒥_p = ∑ₜ ‖Ψ_{tₜ−t₀}(x₀; θ) − x_gt,t‖²
        return jnp.sum(residual**2)

    def reconstruct(
        self,
        x: Array,
        ts: Array,
        params: PyTree | None = None,
    ) -> Array:
        """Full-rollout reconstruction: the trajectory from ``x[0]``.

            x̂ₜ = Ψ_{tₜ−t₀}(x₀; θ)

        ‖x − x̂‖² equals the trajectory residual of
        [`loss`][vardax.DynTrajectory.loss].

        Args:
            x: State sequence; ``x[0]`` seeds the rollout.
            ts: Time coordinates of the sequence.
            params: ODE ``args`` override.

        Returns:
            The rolled-out trajectory, same shape as ``x``.
        """
        # x̂ₜ = Ψ_{tₜ−t₀}(x₀; θ)
        return self(x[0], ts, params=params)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _BoundTemporalPrior(eqx.Module):
    """A temporal prior with its time grid bound: satisfies `Prior`.

    Returned by :meth:`DynamicalPrior.bind`; not constructed directly.
    """

    prior: DynamicalPrior
    ts: Array
    params: PyTree | None

    def __call__(self, x: Array) -> Array:
        """One-argument reconstruction x̂ = reconstruct(x, ts), per the `Prior` seam."""
        # x̂ with the time grid ts and parameters θ closed over
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
        """Advance ``state`` by ``dt`` through the wrapped ODE: Ψ_dt(x; θ)."""
        # autonomous dynamics: integrate over [0, Δt] whatever the true time
        ts = jnp.asarray([0.0, dt])
        saveat = dfx.SaveAt(t1=True)
        saved_states = self.prior._integrate(
            state, ts, saveat, dt, None, adjoint=self.adjoint
        )
        return saved_states[-1]

    @property
    def state_signature(self) -> None:
        """No named-dimension tracking (see ``pipekit_ops.Signature``)."""
        return None
