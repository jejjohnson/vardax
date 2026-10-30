---
status: draft
version: 0.4.0
---

# Layer 1 — Components

`eqx.Module` operators that compose Layer 0 primitives. Protocols define
extension points; concrete implementations provide baselines.

Per Decision D8, components that map onto pipekit-cycle concepts satisfy
those protocols directly — no parallel `Abstract*` hierarchy.

---

## Protocols

### Re-exports from `pipekit_cycle`

```{code-block} python
:caption: Re-exports the pipekit-cycle protocols for the forward model Mₜ of [](#eq-rollout), the observation operator H of [](#eq-obs-model) and the analysis step.
:label: code-design-api-components-pipekit-protocols
import pipekit_cycle as pc

# Shapes: no arrays (protocol re-exports)

# today (pipekit_cycle)
ForwardModel = pc.ForwardModel
# today (pipekit_cycle)
ObservationOperator = pc.ObservationOperator
# today (pipekit_cycle)
AnalysisStep = pc.AnalysisStep
```

| Protocol | Signature |
|---|---|
| `ForwardModel` | `step(state, dt) → state`; `dt` property; `state_signature` property |
| `ObservationOperator` | `__call__(state) → obs`; `linearize(state) → AbstractLinearOperator` |
| `AnalysisStep` | `__call__(forecast, obs, *, obs_op, obs_err_cov) → analysis` |

### Vardax-specific protocols

```{code-block} python
:caption: Declares the vardax protocols: the prior φ of [](#eq-4dvarnet-cost), the modulator Φ of [](#eq-4dvarnet-update), a cost J such as [](#eq-3dvar-cost), its minimiser, and the posterior adapter.
:label: code-design-api-components-vardax-protocols
from typing import Any, Protocol, runtime_checkable

import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size (any trailing shape)

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today (vardax)
@runtime_checkable
class Prior(Protocol):
    """φ: state → regularised state."""

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]: ...


# today* (vardax): shipped __call__ is (grad, state, carry)
@runtime_checkable
class GradModulator(Protocol):
    """Φ: (gradient, carry) → (update, new_carry). FourDVarNet only."""

    def __call__(
        self, grad: Float[Array, " N"], carry: Any
    ) -> tuple[Float[Array, " N"], Any]: ...


# today (vardax)
@runtime_checkable
class CostFunction(Protocol):
    """J: (state, batch, …) → scalar."""

    def __call__(
        self, x: Float[Array, " N"], batch: Batch, **kwargs: Any
    ) -> Float[Array, ""]: ...


# today (vardax)
@runtime_checkable
class PosteriorAdapter(Protocol):
    """Maps inference output → mean + cov + provenance."""

    def __call__(
        self,
        analysis: Float[Array, " N"],
        model: pc.AnalysisStep,
        batch: Batch,
    ) -> vdx.Posterior: ...


# today (vardax)
@runtime_checkable
class Minimiser(Protocol):
    """Wraps optimistix.AbstractMinimiser with vardax's cost-function interface."""

    def __call__(
        self, cost_fn: CostFunction, x0: Float[Array, " N"], batch: Batch
    ) -> Float[Array, " N"]: ...
```

---

## Priors (`vardax.priors`)

### Learned autoencoder priors

```{code-block} python
:caption: Declares the learned autoencoder priors φ_θ that enter the 4DVarNet cost [](#eq-4dvarnet-cost).
:label: code-design-api-components-ae-priors
import equinox as eqx
from jaxtyping import Array, Float

# Shapes: B = batch; T = time; N = state size; C = channels; H, W = grid


# today (vardax)
class BilinAEPrior1D(eqx.Module):
    """φ(x) = decoder(ReLU(A·x) ⊙ tanh(B·x))."""

    state_dim: int
    latent_dim: int
    n_time: int

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]: ...


# today (vardax): acts on (B, T, H, W)
class BilinAEPrior2D(eqx.Module): ...


# today (vardax): acts on (B, T, C, H, W)
class BilinAEPrior2DMultivar(eqx.Module): ...


# today (vardax)
class ConvAEPrior1D(eqx.Module): ...


# today (vardax)
class MLPAEPrior1D(eqx.Module): ...
```

### Identity / classical baseline

```{code-block} python
:caption: Declares the identity prior φ(x) = x, the zero-parameter baseline for the prior term of [](#eq-4dvarnet-cost).
:label: code-design-api-components-identity-prior
import equinox as eqx
from jaxtyping import Array, Float

# Shapes: N = state size (any shape)


# today (vardax)
class IdentityPrior(eqx.Module):
    """φ(x) = x. Zero-parameter."""

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]: ...
```

### Dynamical prior (wraps any `ForwardModel`)

```{code-block} python
:caption: Declares `DynamicalPrior`, which wraps a forward model Mₜ of [](#eq-rollout) as a prior φ and differentiates it through a diffrax adjoint.
:label: code-design-api-components-dynamical-prior
import diffrax
import equinox as eqx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N = state size


# today* (vardax): ships with fields (model, params, solver, stepsize, adjoint)
# and __call__(x, ts, dt=None, params=None)
class DynamicalPrior(eqx.Module):
    """Wrap any pipekit_cycle.ForwardModel as a Prior."""

    # today (pipekit_cycle)
    forward: pc.ForwardModel
    n_steps: int = eqx.field(static=True)
    # today (diffrax)
    forward_adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        # x_n = Mⁿ(x); integrates forward n_steps, gradient via forward_adjoint
        ...
```

When `DynamicalPrior` appears as `FourDVarNet.prior`, the
`forward_adjoint` setting flows through to gradient computation during
training.

---

## Observation operators

See [`observation_operators.md`](observation_operators.md) for the full
family. Summary:

| Class | Purpose |
|---|---|
| `MaskedIdentity` | $H(x) = m \odot x$ |
| `LinearObs` | $H(x) = H_\text{mat} \cdot x$ |
| `AveragingKernel` | $H(x) = A(h \cdot x + (1-h)x_a)$ — RTM L2 product |
| `MultiInstrumentFusion` | Per-instrument composition at the likelihood level |
| `InstrumentRegistry` | `dict[instrument_id, InstrumentSpec]` |

`MaskedIdentity`, `LinearObs`, and `AveragingKernel` satisfy
`pipekit_cycle.ObservationOperator` directly. `MultiInstrumentFusion`
returns `dict[str, Array]` natively and exposes a
`.to_observation_operator()` adapter (block-diagonal flattening) for
strict-protocol contexts.

---

## Gradient modulators (`vardax.grad_mod` — `FourDVarNet` only)

```{code-block} python
:caption: Declares the gradient modulators Φ_φ of the 4DVarNet update [](#eq-4dvarnet-update), with the fixed-step identity as the classical baseline.
:label: code-design-api-components-grad-mod
import equinox as eqx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; T = time; N = state size


# today* (vardax): shipped __call__ is (grad, state, lstm_state)
class ConvLSTMGradMod1D(eqx.Module):
    """1D ConvLSTM gradient modulator. For FourDVarNet over (B, T, N)."""

    state_channels: int
    hidden_dim: int
    kernel_size: int = eqx.field(static=True, default=3)

    def __call__(
        self, grad: Float[Array, "B T N"], carry: vdx.LSTMState1D
    ) -> tuple[Float[Array, "B T N"], vdx.LSTMState1D]: ...


# today (vardax)
class ConvLSTMGradMod2D(eqx.Module): ...


# proposed (components.md)
class MLPGradMod(eqx.Module):
    """Dense MLP gradient modulator. Dimension-agnostic via flatten."""


# proposed (Epic 6)
class AttentionGradMod(eqx.Module):
    """Self-attention over spatial axis (planned, Epic 6)."""


# proposed (components.md)
class IdentityGradMod(eqx.Module):
    """update = -α · grad. The classical 4DVar inner step.

    FourDVarNet with IdentityGradMod and IdentityPrior is mathematically
    equivalent to fixed-step gradient descent on the variational cost,
    the linear-Gaussian baseline.
    """

    alpha: float = eqx.field(static=True, default=0.2)

    def __call__(
        self, grad: Float[Array, "B T N"], carry: None
    ) -> tuple[Float[Array, "B T N"], None]:
        # Φ(∇J, h) = −α ∇J
        update = -self.alpha * grad  # (B, T, N)
        return update, carry
```

The grad modulator family is `FourDVarNet`-specific. Classical methods
(`OptimalInterpolation`, `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`,
`IncrementalFourDVar`) use `optimistix.AbstractMinimiser` instead of a
learned inner step.

---

## Minimiser adapters (`vardax.minimisers` — classical methods)

```{code-block} python
:caption: Wraps an optimistix minimiser to solve x* = argmin J(x) for a cost such as [](#eq-3dvar-cost), differentiating the solution by the implicit function theorem [](#eq-implicit-grad).
:label: code-design-api-components-minimiser
import equinox as eqx
import optimistix as optx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): ships as the Minimiser protocol plus
# OptimistixMinimiser(solver, *, max_steps, throw), with no adjoint field
class Minimiser(eqx.Module):
    """Wraps an optimistix.AbstractMinimiser for CostFunction interface."""

    # today (optimistix)
    minimiser: optx.AbstractMinimiser
    # today (optimistix)
    adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()

    def __call__(
        self, cost_fn: vdx.CostFunction, x0: Float[Array, " N"], batch: Batch
    ) -> Float[Array, " N"]:
        # today (optimistix)
        # x* = argmin_x J(x; batch)
        result = optx.minimise(
            fn=lambda x, args: cost_fn(x, args),
            solver=self.minimiser,
            y0=x0,
            args=batch,
            adjoint=self.adjoint,
        )
        return result.value
```

Typical instantiations:

```{code-block} python
:caption: Instantiates the minimiser above with three optimistix solvers for costs such as [](#eq-3dvar-cost).
:label: code-design-api-components-minimiser-instances
import optimistix as optx

# Shapes: no arrays (configuration)

# today* (vardax): the Minimiser module above
# today* (optimistix): GaussNewton is a least-squares solver, not an AbstractMinimiser
gauss_newton = Minimiser(optx.GaussNewton(rtol=1e-5, atol=1e-5))
# today* (vardax)
# today (optimistix)
bfgs = Minimiser(optx.BFGS(rtol=1e-5, atol=1e-5))
# today* (vardax)
# today (optimistix)
ncg = Minimiser(optx.NonlinearCG(rtol=1e-5, atol=1e-5))
```

---

## Cost functions (`vardax.costs`)

```{code-block} python
:caption: Declares the planned cost modules for BLUE [](#eq-blue-mean), 3DVar [](#eq-3dvar-cost), strong [](#eq-strong4dvar-cost) and weak [](#eq-weak4dvar-cost) 4DVar, the incremental cost [](#eq-incremental-cost) and the 4DVarNet cost [](#eq-4dvarnet-cost).
:label: code-design-api-components-costs
import diffrax
import equinox as eqx
import lineax as lx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size; T = window length

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# proposed (D16)
class BLUECost(eqx.Module):
    """Closed-form linear-Gaussian. Not iterated; consumed by OptimalInterpolation."""

    # See blue_analysis primitive (Layer 0).


# proposed (D14)
class ThreeDVarCost(eqx.Module):
    """J = ½‖x - x_b‖²_{B⁻¹} + ½‖y - H(x)‖²_{R⁻¹}."""

    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    prior_mean: Float[Array, " N"]
    B_inv_op: lx.AbstractLinearOperator
    R_inv_op: lx.AbstractLinearOperator

    def __call__(self, x: Float[Array, " N"], batch: Batch) -> Float[Array, ""]: ...


# proposed (D14)
class StrongConstraintCost(eqx.Module):
    """J = J_b(x_0) + Σ_t ‖y_t - H_t(M_t(x_0))‖²_{R⁻¹}."""

    # today (pipekit_cycle)
    forward: pc.ForwardModel
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    prior_mean: Float[Array, " N"]
    B_inv_op: lx.AbstractLinearOperator
    R_inv_op: lx.AbstractLinearOperator
    # today (diffrax)
    forward_adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

    def __call__(self, x_0: Float[Array, " N"], batch: Batch) -> Float[Array, ""]: ...


# proposed (D14)
class WeakConstraintCost(eqx.Module):
    """J = J_b(x_0) + Σ_t ‖y_t - H_t(x_t)‖²_{R⁻¹} + Σ_t ‖η_t‖²_{Q⁻¹}."""

    # today (pipekit_cycle)
    forward: pc.ForwardModel
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    prior_mean: Float[Array, " N"]
    B_inv_op: lx.AbstractLinearOperator
    R_inv_op: lx.AbstractLinearOperator
    Q_inv_op: lx.AbstractLinearOperator
    # today (diffrax)
    forward_adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

    def __call__(
        self, x_0: Float[Array, " N"], eta: Float[Array, "T N"], batch: Batch
    ) -> Float[Array, ""]: ...


# proposed (D14)
class IncrementalCost(eqx.Module):
    """Linearised J for the incremental 4DVar inner loop."""

    forward_lin: lx.AbstractLinearOperator
    obs_op_lin: lx.AbstractLinearOperator
    x_b: Float[Array, " N"]
    B_inv_op: lx.AbstractLinearOperator
    R_inv_op: lx.AbstractLinearOperator

    def __call__(self, dx: Float[Array, " N"], batch: Batch) -> Float[Array, ""]: ...


# proposed (D3)
class FourDVarNetCost(eqx.Module):
    """Learned-prior variational cost for FourDVarNet."""

    # today (vardax)
    prior: vdx.Prior
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    alpha_obs: float = 1.0
    alpha_prior: float = 1.0

    def __call__(self, x: Float[Array, " N"], batch: Batch) -> Float[Array, ""]: ...
```

---

## Solver configs (`vardax._src._types`)

```{code-block} python
:caption: Declares the configs for the 4DVarNet inner loop [](#eq-4dvarnet-update), the incremental Gauss-Newton loop [](#eq-gn-system) and the amortized head [](#eq-amortized-objective).
:label: code-design-api-components-configs
from typing import Literal

import equinox as eqx

# Shapes: no arrays (configuration)


# proposed (components.md): FourDVarNet1D/2D ship these as fields
# n_solver_steps, alpha and prior_weight instead
class SolverConfig(eqx.Module):
    """Config for FourDVarNet inner loop."""

    n_steps: int = eqx.field(static=True)
    alpha: float = 0.2
    prior_weight: float = 1.0


# today* (vardax): ships with n_inner=30, cg_atol=cg_rtol=1e-6 and covariance_tol
class IncrementalConfig(eqx.Module):
    """Config for IncrementalFourDVar."""

    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=20)
    cg_atol: float = 1e-5
    cg_rtol: float = 1e-5
    cvt: bool = eqx.field(static=True, default=True)


# today* (vardax): ships with head_type default "regression" and no temperature
class AmortizedConfig(eqx.Module):
    head_type: Literal["flow", "score", "regression"] = eqx.field(
        static=True, default="flow"
    )
    n_samples: int = eqx.field(static=True, default=64)
    temperature: float = 1.0
```

Removed in v0.4: `GradMode` / `grad_mode` field. Gradient strategy comes
from the adjoint slots on the model class (Decision D15).

---

## Adjoint composition (Decision D15)

Vardax uses upstream adjoint types directly. No vardax-owned grad-mode
enum.

```{code-block} python
:caption: Lists the adjoints that differentiate the rollout [](#eq-rollout) and the inner minimisation, by checkpointed backprop, the continuous adjoint [](#eq-adjoint-ode) or the implicit function theorem [](#eq-implicit-grad).
:label: code-design-api-components-adjoints
import diffrax
import optimistix as optx
import vardax as vdx

# Shapes: K = number of checkpoints

# diffrax adjoints (for dynamics)
# today (diffrax): default
diffrax.RecursiveCheckpointAdjoint(checkpoints=K)
# today (diffrax): continuous adjoint
diffrax.BacksolveAdjoint()
# today (diffrax): forward sensitivity
diffrax.ForwardMode()
# today (diffrax): straight reverse-mode
diffrax.DirectAdjoint()

# optimistix adjoints (for minimisers)
# today (optimistix): default for FourDVarNet solver
optx.RecursiveCheckpointAdjoint()
# today (optimistix): default for classical minimisers
optx.ImplicitAdjoint()
# proposed (upstream optimistix): not in optimistix 0.1.0
optx.DirectAdjoint()

# vardax-owned (targeting upstream contribution)
# today (vardax): Bolte et al. 2023
vdx.adjoints.OneStepAdjoint()
```

A typical 4DVar configuration:

```{code-block} python
:caption: Configures `StrongFourDVar` for the strong-constraint cost [](#eq-strong4dvar-cost), with an implicit minimiser adjoint and a backsolve forward adjoint.
:label: code-design-api-components-strong4dvar-config
import diffrax
import optimistix as optx
import vardax as vdx

# Shapes: N = state size

# today (vardax)
model = vdx.StrongFourDVar(
    forward=somax_model,
    # today* (vardax)
    obs_op=vdx.AveragingKernel(...),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    # today* (optimistix): a least-squares solver, not an AbstractMinimiser
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    # today (optimistix)
    minimiser_adjoint=optx.ImplicitAdjoint(),
    # today (diffrax): memory-efficient
    forward_adjoint=diffrax.BacksolveAdjoint(),
)
```

---

## Posterior adapters (`vardax.posterior`)

```{code-block} python
:caption: Declares the posterior adapters for the Laplace [](#eq-laplace-cov), Gauss-Newton [](#eq-gn-cov) and ensemble [](#eq-ensemble-cov) covariances, and the mark likelihood built from a posterior.
:label: code-design-api-components-posterior-adapters
import equinox as eqx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size; M = ensemble members

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): also ships prior_cov_op, obs_cov_op and solver fields
class LaplaceCovariance(eqx.Module):
    """P* = (Hᵀ R⁻¹ H + B⁻¹)⁻¹ at MAP."""

    def __call__(
        self, analysis: Float[Array, " N"], model: eqx.Module, batch: Batch
    ) -> vdx.Posterior: ...


# today* (vardax): also ships prior_cov_op, obs_cov_op and tol fields
class GaussNewtonHessian(eqx.Module):
    """Krylov / Lanczos inversion of J''(x*) via lineax."""

    n_krylov: int = eqx.field(static=True, default=50)

    def __call__(
        self, analysis: Float[Array, " N"], model: eqx.Module, batch: Batch
    ) -> vdx.Posterior: ...


# today* (vardax): also ships an inflation field
class EnsembleCovariance(eqx.Module):
    """Posterior from ensemble of analyses (delegates to filterax)."""

    n_members: int = eqx.field(static=True)

    def __call__(
        self, analyses: Float[Array, "M N"], model: eqx.Module, batch: Batch
    ) -> vdx.Posterior: ...


# today (vardax)
class GaussianMarkLikelihood(eqx.Module):
    """Posterior → mark-likelihood for population models."""

    posterior: vdx.Posterior
    event_metadata: dict

    def to_dict(self) -> dict: ...
```

`OptimalInterpolation.posterior(batch)` and
`IncrementalFourDVar.posterior(batch)` are closed-form / reused-Hessian
shortcuts that skip the explicit adapter call.

---

## Data types

### `Batch*`

```{code-block} python
:caption: Declares the batch carriers that hold y, the mask m and the optional truth entering the observation cost [](#eq-obs-cost).
:label: code-design-api-components-batch
import equinox as eqx
from jaxtyping import Array, Float, Int

# Shapes: B = batch; T = time; N = state size


# today* (vardax): ships input, mask and target only;
# instrument and obs_err are proposed (D9)
class Batch1D(eqx.Module):
    input: Float[Array, "B T N"]
    mask: Float[Array, "B T N"]
    target: Float[Array, "B T N"] | None = None
    # per-pixel instrument_id
    instrument: Int[Array, "B T N"] | None = None
    # heteroscedastic σ
    obs_err: Float[Array, "B T N"] | None = None


# today (vardax)
class Batch2D(eqx.Module): ...


# today (vardax)
class Batch2DMultivar(eqx.Module): ...


# proposed (components.md): planned
class Batch3D(eqx.Module): ...
```

`instrument` and `obs_err` are `None` for single-instrument /
homoscedastic cases. `MultiInstrumentFusion` requires them.

### `Posterior`

```{code-block} python
:caption: Declares the `Posterior` container for the mean x* and covariance P* of [](#eq-oi-posterior).
:label: code-design-api-components-posterior
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

# Shapes: B = batch; M = samples; N = state size


# today (vardax)
class Posterior(eqx.Module):
    mean: Float[Array, "B N"]
    # gaussx / lineax operator
    cov: lx.AbstractLinearOperator | None
    samples: Float[Array, "B M N"] | None
    # forward_model_id, n_iter, J_star, …
    provenance: dict
```
