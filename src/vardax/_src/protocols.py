"""Runtime-checkable protocols that vardax components satisfy structurally.

Re-exports the core ``pipekit_cycle`` protocols (``ForwardModel``,
``ObservationOperator``, ``AnalysisStep``) and adds the vardax-specific seams that
``pipekit-cycle`` does not name; ``isinstance(obj, Protocol)`` checks conformance.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array, Float

# vardax models satisfy these structurally, so there is no parallel Abstract*
# hierarchy
from pipekit_cycle import (
    AnalysisStep,
    ForwardModel,
    ObservationOperator,
)


@runtime_checkable
class Prior(Protocol):
    """Prior model: maps a state to its regularised reconstruction.

        φ: x ↦ x_prior

    For an autoencoder prior the variational cost includes ‖x − φ(x)‖². For a
    dynamical prior wrapping a `ForwardModel`, φ(x) is the forward integration.
    For the identity prior, φ(x) = x.

    Members:
        ``__call__(x) -> x_prior``: apply the prior model φ.

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

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Apply the prior: x_prior = φ(x).

        Args:
            x: State x.

        Returns:
            The regularised reconstruction x_prior.
        """


@runtime_checkable
class TemporalPrior(Protocol):
    """Time-dependent prior: dynamics-aware regularisation over a window.

        φ: (x, t) ↦ x_pred

    Unlike the static [`Prior`][vardax.Prior] seam, φ: x ↦ x_prior, a temporal
    prior needs the time coordinates t of the assimilation window, with a
    ``loss`` scoring the dynamical consistency of a state sequence.
    Implementations: [`DynIncrements`][vardax.DynIncrements],
    [`DynTrajectory`][vardax.DynTrajectory].

    A temporal prior is adapted to contexts expecting the one-argument `Prior`
    seam by binding the time grid: ``prior.bind(ts)`` returns a `Prior`-conforming
    callable.

    Members:
        ``__call__(x, ts) -> x_pred``: propagate through the dynamics.
        ``loss(x, ts, x_gt=None, params=None) -> scalar``: dynamical residual of
        the sequence ``x`` against ``x_gt``.

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
    ) -> Float[Array, ...]:
        """Propagate through the dynamics: x_pred = φ(x, t).

        Args:
            x: State sequence x.
            ts: Time coordinates t of the window.

        Returns:
            The propagated state x_pred.
        """

    def loss(
        self,
        x: Float[Array, ...],
        ts: Float[Array, ...],
        x_gt: Float[Array, ...] | None = None,
        params: Any = None,
    ) -> Float[Array, ""]:
        """Score the dynamical consistency of a state sequence.

        Args:
            x: State sequence x.
            ts: Time coordinates t of the window.
            x_gt: Optional ground-truth sequence x_gt to score against.
            params: Optional parameters of the implementation.

        Returns:
            The dynamical residual of ``x`` against ``x_gt``, a scalar.
        """


@runtime_checkable
class GradModulator(Protocol):
    """Learned gradient modulator for the FourDVarNet inner solver.

        g_mod: (∇ₓ𝒥, x, c) ↦ (Δx, c′)

    Takes the current variational-cost gradient ∇ₓ𝒥, the state x and the
    modulator's own carry c, and returns a state update Δx and the new carry c′.
    Used only by `FourDVarNet`; the classical analysis methods use
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
    ) -> tuple[Float[Array, ...], Any]:
        """Turn a cost gradient into a state update: (Δx, c′) = g_mod(∇ₓ𝒥, x, c).

        Args:
            grad: Variational-cost gradient ∇ₓ𝒥.
            state: Current state x.
            carry: The modulator's carry c.

        Returns:
            The state update Δx and the new carry c′.
        """


@runtime_checkable
class CostFunction(Protocol):
    """Variational cost function over the state.

        𝒥: (x, batch) ↦ ℝ

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
    ) -> Float[Array, ""]:
        """Evaluate the cost 𝒥(x; batch).

        Args:
            x: State x.
            batch: Data the cost compares the state with.
            **kwargs: Options of the implementation.

        Returns:
            The cost, a scalar.
        """


@runtime_checkable
class PosteriorAdapter(Protocol):
    """Turns an analysis output into a `Posterior` container.

    Implementations: `LaplaceCovariance`, `GaussNewtonHessian`,
    `EnsembleCovariance`. Each computes the posterior covariance Σ_x|y via a
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
    ) -> Any:
        """Build the posterior around an analysis x⋆.

        Args:
            analysis: The analysis x⋆.
            model: The model that produced it.
            batch: The batch it was analysed from.

        Returns:
            The [`Posterior`][vardax.Posterior], with mean and covariance Σ_x|y.
        """


@runtime_checkable
class Minimiser(Protocol):
    """Minimises a cost ``cost_fn(x, batch)`` from an initial guess.

        x⋆ = argmin_x 𝒥(x; batch)

    A `Minimiser` knows how to minimise a `CostFunction` from an initial guess
    ``x0`` (any pytree) against a batch. Implementations:
    [`OptimistixMinimiser`][vardax.OptimistixMinimiser] (optimistix solvers such
    as BFGS, GaussNewton, NonlinearCG) and
    [`OptaxMinimiser`][vardax.OptaxMinimiser] (first-order optax optimisers with
    early stopping, for large controls). The classical models accept either as
    ``minimiser=``.

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
    ) -> Float[Array, ...]:
        """Minimise ``cost_fn`` from ``x0``: x⋆ = argmin_x 𝒥(x; batch).

        Args:
            cost_fn: The cost 𝒥.
            x0: Initial guess, any pytree.
            batch: Data passed through to ``cost_fn``.

        Returns:
            The minimiser x⋆.
        """


@runtime_checkable
class ReducedBasis(Protocol):
    """Reduced-basis control-vector operator.

        δx = Φ(t) z,   𝒥_b(z) = ½ zᵀ Σ_z⁻¹ z

    Maps a coefficient vector z to a state increment δx on the grid, with a
    separable Gaussian background term of covariance Σ_z over the coefficients.

    Implementations: `LinearBasis` (any linear family, built via the `rbf_basis`,
    `fourier_basis`, `wavelet_basis`, `eof_basis` or `linear_basis` constructors)
    and `CompositeBasis`. Used by variational analysis steps to parameterise the
    increment in a low-dimensional space.

    This protocol is defined in vardax for now; the plan is to upstream it into
    ``pipekit_cycle.protocols`` so other algorithm libraries can satisfy the same
    contract.

    Members:
        ``operg(t, coefficients, state=None) -> increment``: the increment Φ(t) z.
        ``prior_inv(coefficients) -> Σ_z⁻¹ coefficients``: the prior precision.
        ``nbasis: int`` property: the number of coefficients.

    Examples:
        The basis built by ``linear_basis`` conforms.

        >>> import jax.numpy as jnp, vardax
        >>> basis = vardax.linear_basis(jnp.eye(4)[:, :2], variance=1.0)
        >>> isinstance(basis, vardax.ReducedBasis)
        True
        >>> basis.nbasis
        2

        ``prior_inv`` applies Σ_z⁻¹ to a coefficient vector.

        >>> basis.prior_inv(jnp.ones(2)).shape
        (2,)
    """

    def operg(self, t: float, coefficients: Any, state: Any = None) -> Any:
        """Map coefficients to a state increment: δx = Φ(t) z.

        Args:
            t: Time t.
            coefficients: Coefficient vector z.
            state: Optional state, for a basis that depends on it.

        Returns:
            The state increment δx.
        """

    def prior_inv(self, coefficients: Any) -> Any:
        """Apply the prior precision: Σ_z⁻¹ z.

        Args:
            coefficients: Coefficient vector z.

        Returns:
            Σ_z⁻¹ z.
        """

    @property
    def nbasis(self) -> int:
        """Number of coefficients, the length of z."""


@runtime_checkable
class StateParameterization(Protocol):
    """Control-to-state map with its own penalty.

        x = g_dec(θ),   𝒥(θ) = ℛ(θ) + 𝒥_o(g_dec(θ))

    Generalises [`ReducedBasis`][vardax.ReducedBasis] from a linear
    x = μ_x + Φ z to any x = g_dec(θ) with θ a pytree. A model given
    ``parameterization=`` minimises over θ, with the penalty ℛ(θ) in place of the
    background term.

    Implementations: [`LinearParameterization`][vardax.LinearParameterization]
    (adapts any `ReducedBasis`) and
    [`NeuralFieldParameterization`][vardax.NeuralFieldParameterization] (a
    coordinate network on the grid).

    Members:
        ``decode(theta, t=None) -> state``: the state g_dec(θ), at time ``t`` for a
        time-dependent parameterization.
        ``penalty(theta) -> scalar``: ℛ(θ), e.g. ½ zᵀ Σ_z⁻¹ z.
        ``init_control(*, key=None) -> theta``: the starting control; ``key`` is
        for parameterizations whose start is random.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> basis = vardax.linear_basis(jnp.eye(4)[:, :2], variance=1.0)
        >>> param = vardax.LinearParameterization(basis, background=jnp.zeros(4))
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

    def decode(self, theta: Any, t: float | None = None) -> Float[Array, ...]:
        """Decode the control into a state: x = g_dec(θ).

        Args:
            theta: Control θ.
            t: Time, for a time-dependent parameterization.

        Returns:
            The state x.
        """

    def penalty(self, theta: Any) -> Float[Array, ""]:
        """Evaluate the penalty ℛ(θ).

        Args:
            theta: Control θ.

        Returns:
            The penalty, a scalar.
        """

    def init_control(self, *, key: Any = None) -> Any:
        """Return the starting control θ.

        Args:
            key: PRNG key, for a parameterization whose start is random.

        Returns:
            The starting control θ.
        """


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
    """A ``ForwardModel`` whose integrator can take a ``diffrax`` adjoint.

    The 4DVar models' ``forward_adjoint=`` slot calls
    ``forward.with_adjoint(adjoint)`` and rolls out the returned forward.
    A forward that integrates with ``diffrax.diffeqsolve`` implements it
    by storing the adjoint and passing it as ``diffeqsolve(...,
    adjoint=adjoint)``. A per-step forward cannot, and a 4DVar model given
    ``forward_adjoint=`` with such a forward raises
    [`UnsupportedInputError`][vardax.UnsupportedInputError]; bound its
    memory with ``checkpoint=`` instead.

    Members:
        ``dt``: step length.
        ``step(state, dt) -> state``: the ``ForwardModel`` contract.
        ``with_adjoint(adjoint) -> forward``: a copy that integrates with
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

    def step(self, state: Any, dt: float) -> Any:
        """Advance the state one step: xₜ = f(xₜ₋₁).

        Args:
            state: State xₜ₋₁.
            dt: Step length.

        Returns:
            The state xₜ.
        """

    def with_adjoint(self, adjoint: Any) -> Any:
        """Return a copy of the forward that integrates with ``adjoint``.

        Args:
            adjoint: A ``diffrax.AbstractAdjoint``.

        Returns:
            The forward model f with that adjoint.
        """
