r"""Runtime-checkable protocols used across vardax.

vardax re-exports the three core ``pipekit_cycle`` protocols
(`ForwardModel`, `ObservationOperator`, `AnalysisStep`) and adds eight
vardax-specific protocols that ``pipekit-cycle`` doesn't name:

- [`Prior`][vardax.Prior] — $\varphi: x \mapsto x_\text{prior}$
- [`TemporalPrior`][vardax.TemporalPrior] —
  $\varphi: (x, t_s) \mapsto x_\text{pred}$ (Decision D18)
- [`GradModulator`][vardax.GradModulator] —
  $\Phi: (\nabla J, h) \mapsto (\Delta x, h')$
- [`CostFunction`][vardax.CostFunction] —
  $J: (x, \text{batch}) \mapsto \text{scalar}$
- [`PosteriorAdapter`][vardax.PosteriorAdapter] — analysis output →
  [`Posterior`][vardax.Posterior]
- [`Minimiser`][vardax.Minimiser] — minimises a cost over any pytree
  control (``optimistix`` solvers or first-order ``optax`` optimisers)
- [`ReducedBasis`][vardax.ReducedBasis] —
  $X \mapsto \Phi(t) X$ reduced-basis control-vector operator
- [`StateParameterization`][vardax.StateParameterization] —
  $\theta \mapsto \psi(\theta)$, a (possibly nonlinear) control-to-state
  map with its own penalty
- [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint] — a
  ``ForwardModel`` that accepts a ``diffrax`` adjoint for its rollout

All protocols are ``@runtime_checkable`` so ``isinstance(obj, Protocol)``
works for structural conformance checking — used by
``tests/test_pipekit_protocols.py``.

The vardax-specific protocols are imported by Layer 1 components.
``pipekit_cycle`` re-exports are imported by Layer 2 models that
expose ``.as_analysis_step()``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array, Float

# Re-export pipekit-cycle protocols — vardax models satisfy them structurally
# (Decision D8: no parallel Abstract* hierarchy).
from pipekit_cycle import (
    AnalysisStep,
    ForwardModel,
    ObservationOperator,
)


@runtime_checkable
class Prior(Protocol):
    r"""Prior model: maps state to its regularised reconstruction.

    For an autoencoder prior $\varphi_\theta$, the variational
    cost includes ``||x - φ(x)||^2``. For a dynamical prior wrapping a
    `ForwardModel`, ``φ(x)`` is the forward integration. For the
    identity prior, ``φ(x) = x``.

    Members:
        ``__call__(x) -> x_prior`` — apply the prior model.
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]: ...


@runtime_checkable
class TemporalPrior(Protocol):
    r"""Time-dependent prior: dynamics-aware regularisation over a window.

    Unlike the static [`Prior`][vardax.Prior] seam ($\varphi: x \mapsto
    x_\text{prior}$), a temporal prior needs the time coordinates of the
    assimilation window: $\varphi: (x, t_s) \mapsto x_\text{pred}$, with a
    ``loss`` scoring dynamical consistency of a state sequence (Decision
    D18). Implementations: [`DynIncrements`][vardax.DynIncrements],
    [`DynTrajectory`][vardax.DynTrajectory].

    A temporal prior is adapted to contexts expecting the one-argument
    `Prior` seam by binding the time grid:
    ``prior.bind(ts)`` returns a `Prior`-conforming callable.

    Members:
        ``__call__(x, ts) -> x_pred`` — propagate through the dynamics.
        ``loss(x, ts, x_gt=None, params=None) -> scalar`` — dynamical
        residual of the sequence ``x`` against ``x_gt``.
    """

    def __call__(
        self,
        x: Float[Array, ...],
        ts: Float[Array, ...],
    ) -> Float[Array, ...]: ...

    def loss(
        self,
        x: Float[Array, ...],
        ts: Float[Array, ...],
        x_gt: Float[Array, ...] | None = None,
        params: Any = None,
    ) -> Float[Array, ""]: ...


@runtime_checkable
class GradModulator(Protocol):
    """Learned gradient modulator for the FourDVarNet inner solver.

    Takes the current variational-cost gradient and the modulator's
    own carry state, returns a state update and the new carry. Used
    only by `FourDVarNet`; the classical analysis methods use
    ``optimistix.AbstractMinimiser`` instead.

    Members:
        ``__call__(grad, state, carry) -> (update, new_carry)``
    """

    def __call__(
        self,
        grad: Float[Array, ...],
        state: Float[Array, ...],
        carry: Any,
    ) -> tuple[Float[Array, ...], Any]: ...


@runtime_checkable
class CostFunction(Protocol):
    """Variational cost function ``J(x, batch, **kwargs) -> scalar``.

    Implementations include `vardax.costs.ThreeDVarCost`,
    `StrongConstraintCost`, `WeakConstraintCost`, `IncrementalCost`,
    and `FourDVarNetCost`.

    Members:
        ``__call__(x, batch, **kwargs) -> scalar``
    """

    def __call__(
        self,
        x: Float[Array, ...],
        batch: Any,
        **kwargs: Any,
    ) -> Float[Array, ""]: ...


@runtime_checkable
class PosteriorAdapter(Protocol):
    """Turns an analysis output into a `Posterior` container.

    Implementations: `LaplaceCovariance`, `GaussNewtonHessian`,
    `EnsembleCovariance`. Each computes the posterior covariance via a
    different approximation; the contract returned is the same.

    Members:
        ``__call__(analysis, model, batch) -> Posterior``
    """

    def __call__(
        self,
        analysis: Float[Array, ...],
        model: AnalysisStep,
        batch: Any,
    ) -> Any: ...


@runtime_checkable
class Minimiser(Protocol):
    """Minimises a cost ``cost_fn(x, batch)`` from an initial guess.

    A `Minimiser` knows how to minimise a `CostFunction` from an
    initial guess ``x0`` (any pytree) against a batch. Implementations:
    [`OptimistixMinimiser`][vardax.OptimistixMinimiser] (optimistix
    solvers such as BFGS, GaussNewton, NonlinearCG) and
    [`OptaxMinimiser`][vardax.OptaxMinimiser] (first-order optax
    optimisers with early stopping, for large controls). The classical
    models accept either as ``minimiser=``.

    Members:
        ``__call__(cost_fn, x0, batch) -> x_star``
    """

    def __call__(
        self,
        cost_fn: CostFunction,
        x0: Float[Array, ...],
        batch: Any,
    ) -> Float[Array, ...]: ...


@runtime_checkable
class ReducedBasis(Protocol):
    r"""Reduced-basis control-vector operator.

    Maps a coefficient vector $X \in \mathbb{R}^M$ to a state increment
    $\Phi(t) X$ on the grid, with a separable Gaussian background term
    $\tfrac{1}{2} X^\top Q^{-1} X$.

    Implementations: `LinearBasis` (any linear family — built via the
    `rbf_basis` / `fourier_basis` / `wavelet_basis` / `eof_basis` /
    `linear_basis` constructors) and `CompositeBasis`. Used by
    variational analysis steps to parameterise the increment in a
    low-dimensional space.

    This protocol is defined in vardax for now; the plan is to upstream
    it into ``pipekit_cycle.protocols`` so other algorithm libraries
    can satisfy the same contract.

    Members:
        ``operg(t, X, state=None) -> increment``
        ``prior_inv(X) -> Q⁻¹ X``
        ``nbasis: int`` property
    """

    def operg(self, t: float, X: Any, state: Any = None) -> Any: ...

    def prior_inv(self, X: Any) -> Any: ...

    @property
    def nbasis(self) -> int: ...


@runtime_checkable
class StateParameterization(Protocol):
    r"""Control-to-state map with its own penalty.

    Generalises [`ReducedBasis`][vardax.ReducedBasis] from a linear
    $x = x_b + \Phi X$ to any $x = \psi(\theta)$ with $\theta$ a pytree.
    A model given ``parameterization=`` minimises over $\theta$, with
    $\mathrm{pen}(\theta)$ in place of the background term:
    $J(\theta) = \mathrm{pen}(\theta) + J_o(\psi(\theta))$.

    Implementations: [`LinearParameterization`][vardax.LinearParameterization]
    (adapts any `ReducedBasis`) and
    [`NeuralFieldParameterization`][vardax.NeuralFieldParameterization]
    (a coordinate network on the grid).

    Members:
        ``decode(theta, t=None) -> state`` — the state $\psi(\theta)$, at
        time ``t`` for a time-dependent parameterization.
        ``penalty(theta) -> scalar`` — $\mathrm{pen}(\theta)$, e.g.
        $\tfrac12 X^\top Q^{-1} X$.
        ``init_control(*, key=None) -> theta`` — the starting control;
        ``key`` is for parameterizations whose start is random.
    """

    def decode(self, theta: Any, t: float | None = None) -> Float[Array, ...]: ...

    def penalty(self, theta: Any) -> Float[Array, ""]: ...

    def init_control(self, *, key: Any = None) -> Any: ...


__all__ = [
    # Re-exports from pipekit-cycle
    "AnalysisStep",
    "ForwardModel",
    "ObservationOperator",
    # Vardax-specific
    "CostFunction",
    "GradModulator",
    "Minimiser",
    "PosteriorAdapter",
    "Prior",
    "ReducedBasis",
    "StateParameterization",
    "TemporalPrior",
]


@runtime_checkable
class SupportsForwardAdjoint(Protocol):
    r"""A ``ForwardModel`` whose integrator can take a ``diffrax`` adjoint.

    The 4DVar models' ``forward_adjoint=`` slot calls
    ``forward.with_adjoint(adjoint)`` and rolls out the returned forward.
    A forward that integrates with ``diffrax.diffeqsolve`` implements it
    by storing the adjoint and passing it as ``diffeqsolve(...,
    adjoint=adjoint)``. A per-step forward cannot, and a 4DVar model given
    ``forward_adjoint=`` with such a forward raises ``ValueError``; bound
    its memory with ``checkpoint=`` instead.

    Members:
        ``dt`` — step length.
        ``step(state, dt) -> state`` — the ``ForwardModel`` contract.
        ``with_adjoint(adjoint) -> forward`` — a copy that integrates with
        ``adjoint``.

    Examples:
        >>> import dataclasses
        >>> import vardax
        >>> @dataclasses.dataclass(frozen=True)
        ... class Decay:
        ...     dt: float = 1.0
        ...     adjoint: object = None
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        ...
        ...     def with_adjoint(self, adjoint):
        ...         return dataclasses.replace(self, adjoint=adjoint)
        >>> isinstance(Decay(), vardax.SupportsForwardAdjoint)
        True
        >>> Decay().with_adjoint("backsolve").adjoint
        'backsolve'
    """

    dt: float

    def step(self, state: Any, dt: float) -> Any: ...

    def with_adjoint(self, adjoint: Any) -> Any: ...
