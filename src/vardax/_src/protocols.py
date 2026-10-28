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

    Examples:
        A callable that shrinks the state towards zero conforms.

        >>> import jax.numpy as jnp, vardax
        >>> class Shrink:
        ...     def __call__(self, x):
        ...         return 0.9 * x
        >>> isinstance(Shrink(), vardax.Prior)
        True

        The built-in identity prior conforms too.

        >>> isinstance(vardax.IdentityPrior(), vardax.Prior)
        True
        >>> float(vardax.IdentityPrior()(jnp.ones(3)).sum())
        3.0
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]: ...


@runtime_checkable
class TemporalPrior(Protocol):
    r"""Time-dependent prior: dynamics-aware regularisation over a window.

    Unlike the static [`Prior`][vardax.Prior] seam
    ($\varphi: x \mapsto x_\text{prior}$), a temporal prior needs the time
    coordinates of the assimilation window,
    $\varphi: (x, t_s) \mapsto x_\text{pred}$, with a
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

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> class Persistence:
        ...     def __call__(self, x, ts):
        ...         return x
        ...
        ...     def loss(self, x, ts, x_gt=None, params=None):
        ...         return jnp.mean((x[1:] - x[:-1]) ** 2)
        >>> prior = Persistence()
        >>> isinstance(prior, vardax.TemporalPrior)
        True
        >>> float(prior.loss(jnp.ones((4, 2)), jnp.arange(4.0)))
        0.0

        An object with only ``__call__`` is not a temporal prior.

        >>> isinstance(vardax.IdentityPrior(), vardax.TemporalPrior)
        False
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

    Examples:
        A modulator that scales the gradient and carries nothing.

        >>> import jax.numpy as jnp, vardax
        >>> class Scale:
        ...     def __call__(self, grad, state, carry):
        ...         return -0.1 * grad, carry
        >>> isinstance(Scale(), vardax.GradModulator)
        True
        >>> update, carry = Scale()(jnp.ones(3), jnp.zeros(3), None)
        >>> update.shape, carry is None
        ((3,), True)
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

    Examples:
        A quadratic cost around the batch.

        >>> import jax.numpy as jnp, vardax
        >>> class Quadratic:
        ...     def __call__(self, x, batch, **kwargs):
        ...         return 0.5 * jnp.sum((x - batch) ** 2)
        >>> cost = Quadratic()
        >>> isinstance(cost, vardax.CostFunction)
        True
        >>> float(cost(jnp.ones(3), jnp.zeros(3)))
        1.5

        It is the argument a [`Minimiser`][vardax.Minimiser] works on.

        >>> x = vardax.OptimistixMinimiser()(cost, jnp.zeros(3), jnp.ones(3))
        >>> bool(jnp.allclose(x, 1.0, atol=1e-5))
        True
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

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> class Passthrough:
        ...     def __call__(self, analysis, model, batch):
        ...         return analysis
        >>> isinstance(Passthrough(), vardax.PosteriorAdapter)
        True

        The built-in adapters conform as well.

        >>> isinstance(vardax.LaplaceCovariance(), vardax.PosteriorAdapter)
        True
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

    Examples:
        A minimiser that takes one gradient step.

        >>> import jax, jax.numpy as jnp, vardax
        >>> class OneStep:
        ...     def __call__(self, cost_fn, x0, batch):
        ...         return x0 - jax.grad(cost_fn)(x0, batch)
        >>> isinstance(OneStep(), vardax.Minimiser)
        True
        >>> cost = lambda x, m: 0.5 * jnp.sum((x - m) ** 2)
        >>> x = OneStep()(cost, jnp.zeros(2), jnp.ones(2))
        >>> bool(jnp.allclose(x, 1.0))
        True

        The built-in minimisers conform.

        >>> isinstance(vardax.OptaxMinimiser(), vardax.Minimiser)
        True
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

    Examples:
        The basis built by ``linear_basis`` conforms.

        >>> import jax.numpy as jnp, vardax
        >>> basis = vardax.linear_basis(jnp.eye(4)[:, :2], variance=1.0)
        >>> isinstance(basis, vardax.ReducedBasis)
        True
        >>> basis.nbasis
        2

        ``prior_inv`` applies $Q^{-1}$ to a coefficient vector.

        >>> basis.prior_inv(jnp.ones(2)).shape
        (2,)
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

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> basis = vardax.linear_basis(jnp.eye(4)[:, :2], variance=1.0)
        >>> param = vardax.LinearParameterization(basis, x_b=jnp.zeros(4))
        >>> isinstance(param, vardax.StateParameterization)
        True
        >>> param.decode(param.init_control()).shape
        (4,)

        A minimal custom parameterization that uses the state itself as
        the control.

        >>> class Direct:
        ...     def decode(self, theta, t=None):
        ...         return theta
        ...
        ...     def penalty(self, theta):
        ...         return 0.5 * jnp.sum(theta**2)
        ...
        ...     def init_control(self, *, key=None):
        ...         return jnp.zeros(4)
        >>> isinstance(Direct(), vardax.StateParameterization)
        True
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
    ``forward_adjoint=`` with such a forward raises
    [`UnsupportedInputError`][vardax.UnsupportedInputError]; bound its
    memory with ``checkpoint=`` instead.

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
