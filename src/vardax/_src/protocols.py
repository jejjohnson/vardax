r"""Runtime-checkable protocols used across vardax.

vardax re-exports the three core ``pipekit_cycle`` protocols
(`ForwardModel`, `ObservationOperator`, `AnalysisStep`) and adds six
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
- [`Minimiser`][vardax.Minimiser] — wraps
  ``optimistix.AbstractMinimiser`` over vardax's cost-function
  interface

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
    """Wrapper protocol around ``optimistix.AbstractMinimiser``.

    A `Minimiser` knows how to minimise a `CostFunction` from an
    initial guess ``x0`` against a batch. Implementations adapt
    optimistix solvers (GaussNewton, BFGS, NonlinearCG, …) to vardax's
    cost-function calling convention.

    Members:
        ``__call__(cost_fn, x0, batch) -> x_star``
    """

    def __call__(
        self,
        cost_fn: CostFunction,
        x0: Float[Array, ...],
        batch: Any,
    ) -> Float[Array, ...]: ...


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
    "TemporalPrior",
]
