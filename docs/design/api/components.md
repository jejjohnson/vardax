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

```python
from pipekit_cycle import ForwardModel, ObservationOperator, AnalysisStep
```

| Protocol | Signature |
|---|---|
| `ForwardModel` | `step(state, dt) → state`; `dt` property; `state_signature` property |
| `ObservationOperator` | `__call__(state) → obs`; `linearize(state) → AbstractLinearOperator` |
| `AnalysisStep` | `__call__(forecast, obs, *, obs_op, obs_err_cov) → analysis` |

### Vardax-specific protocols

```python
@runtime_checkable
class Prior(Protocol):
    """φ: state → regularised state."""
    def __call__(self, x: Array) -> Array: ...


@runtime_checkable
class GradModulator(Protocol):
    """Φ: (gradient, carry) → (update, new_carry). FourDVarNet only."""
    def __call__(self, grad: Array, carry: Any) -> tuple[Array, Any]: ...


@runtime_checkable
class CostFunction(Protocol):
    """J: (state, batch, …) → scalar."""
    def __call__(self, x: Array, batch: Batch, **kwargs) -> Float[Array, ""]: ...


@runtime_checkable
class PosteriorAdapter(Protocol):
    """Maps inference output → mean + cov + provenance."""
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...


@runtime_checkable
class Minimiser(Protocol):
    """Wraps optimistix.AbstractMinimiser with vardax's cost-function interface."""
    def __call__(self, cost_fn: CostFunction, x0: Array, batch: Batch) -> Array: ...
```

---

## Priors (`vardax.priors`)

### Learned autoencoder priors

```python
class BilinAEPrior1D(eqx.Module):
    """φ(x) = decoder(ReLU(A·x) ⊙ tanh(B·x))."""
    state_dim: int; latent_dim: int; n_time: int
    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]: ...

class BilinAEPrior2D(eqx.Module): ...           # (B, T, H, W)
class BilinAEPrior2DMultivar(eqx.Module): ...   # (B, T, C, H, W)
class ConvAEPrior1D(eqx.Module): ...
class MLPAEPrior1D(eqx.Module): ...
```

### Identity / classical baseline

```python
class IdentityPrior(eqx.Module):
    """φ(x) = x. Zero-parameter."""
    def __call__(self, x: Array) -> Array: ...
```

### Dynamical prior (wraps any `ForwardModel`)

```python
class DynamicalPrior(eqx.Module):
    """Wrap any pipekit_cycle.ForwardModel as a Prior."""
    forward: ForwardModel
    n_steps: int = eqx.field(static=True)
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, x: Array) -> Array:
        # Integrates forward n_steps; gradient via forward_adjoint
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

```python
class ConvLSTMGradMod1D(eqx.Module):
    """1D ConvLSTM gradient modulator. For FourDVarNet over (B, T, N)."""
    state_channels: int
    hidden_dim: int
    kernel_size: int = eqx.field(static=True, default=3)
    def __call__(self, grad, carry) -> tuple[Array, LSTMState1D]: ...

class ConvLSTMGradMod2D(eqx.Module): ...

class MLPGradMod(eqx.Module):
    """Dense MLP gradient modulator. Dimension-agnostic via flatten."""

class AttentionGradMod(eqx.Module):
    """Self-attention over spatial axis (planned, Epic 6)."""

class IdentityGradMod(eqx.Module):
    """update = -α · grad. The classical 4DVar inner step.

    FourDVarNet with IdentityGradMod and IdentityPrior is mathematically
    equivalent to fixed-step gradient descent on the variational cost —
    the linear-Gaussian baseline.
    """
    alpha: float = eqx.field(static=True, default=0.2)
    def __call__(self, grad, carry):
        return -self.alpha * grad, carry
```

The grad modulator family is `FourDVarNet`-specific. Classical methods
(`OptimalInterpolation`, `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`,
`IncrementalFourDVar`) use `optimistix.AbstractMinimiser` instead of a
learned inner step.

---

## Minimiser adapters (`vardax.minimisers` — classical methods)

```python
class Minimiser(eqx.Module):
    """Wraps an optimistix.AbstractMinimiser for CostFunction interface."""
    minimiser: optimistix.AbstractMinimiser
    adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()

    def __call__(self, cost_fn: CostFunction, x0: Array, batch: Batch) -> Array:
        result = optimistix.minimise(
            fn=lambda x, args: cost_fn(x, args),
            solver=self.minimiser,
            y0=x0, args=batch,
            adjoint=self.adjoint,
        )
        return result.value
```

Typical instantiations:

```python
gauss_newton = Minimiser(optimistix.GaussNewton(rtol=1e-5, atol=1e-5))
bfgs = Minimiser(optimistix.BFGS(rtol=1e-5, atol=1e-5))
ncg = Minimiser(optimistix.NonlinearCG(rtol=1e-5, atol=1e-5))
```

---

## Cost functions (`vardax.costs`)

```python
class BLUECost(eqx.Module):
    """Closed-form linear-Gaussian. Not iterated; consumed by OptimalInterpolation."""
    # See blue_analysis primitive (Layer 0).

class ThreeDVarCost(eqx.Module):
    """J = ½‖x - x_b‖²_{B⁻¹} + ½‖y - H(x)‖²_{R⁻¹}."""
    obs_op: ObservationOperator
    prior_mean: Array
    B_inv_op: AbstractLinearOperator
    R_inv_op: AbstractLinearOperator
    def __call__(self, x: Array, batch: Batch) -> Scalar: ...

class StrongConstraintCost(eqx.Module):
    """J = J_b(x_0) + Σ_t ‖y_t - H_t(M_t(x_0))‖²_{R⁻¹}."""
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    B_inv_op: AbstractLinearOperator
    R_inv_op: AbstractLinearOperator
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()
    def __call__(self, x_0: Array, batch: Batch) -> Scalar: ...

class WeakConstraintCost(eqx.Module):
    """J = J_b(x_0) + Σ_t ‖y_t - H_t(x_t)‖²_{R⁻¹} + Σ_t ‖η_t‖²_{Q⁻¹}."""
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    B_inv_op: AbstractLinearOperator
    R_inv_op: AbstractLinearOperator
    Q_inv_op: AbstractLinearOperator
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()
    def __call__(self, x_0: Array, eta: Array, batch: Batch) -> Scalar: ...

class IncrementalCost(eqx.Module):
    """Linearised J for the incremental 4DVar inner loop."""
    forward_lin: AbstractLinearOperator
    obs_op_lin: AbstractLinearOperator
    x_b: Array
    B_inv_op: AbstractLinearOperator
    R_inv_op: AbstractLinearOperator
    def __call__(self, dx: Array, batch: Batch) -> Scalar: ...

class FourDVarNetCost(eqx.Module):
    """Learned-prior variational cost for FourDVarNet."""
    prior: Prior
    obs_op: ObservationOperator
    alpha_obs: float = 1.0
    alpha_prior: float = 1.0
    def __call__(self, x: Array, batch: Batch) -> Scalar: ...
```

---

## Solver configs (`vardax._src._types`)

```python
class SolverConfig(eqx.Module):
    """Config for FourDVarNet inner loop."""
    n_steps: int = eqx.field(static=True)
    alpha: float = 0.2
    prior_weight: float = 1.0


class IncrementalConfig(eqx.Module):
    """Config for IncrementalFourDVar."""
    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=20)
    cg_atol: float = 1e-5
    cg_rtol: float = 1e-5
    cvt: bool = eqx.field(static=True, default=True)


class AmortizedConfig(eqx.Module):
    head_type: Literal["flow", "score", "regression"] = eqx.field(static=True, default="flow")
    n_samples: int = eqx.field(static=True, default=64)
    temperature: float = 1.0
```

Removed in v0.4: `GradMode` / `grad_mode` field. Gradient strategy comes
from the adjoint slots on the model class (Decision D15).

---

## Adjoint composition (Decision D15)

Vardax uses upstream adjoint types directly. No vardax-owned grad-mode
enum.

```python
# diffrax adjoints (for dynamics)
diffrax.RecursiveCheckpointAdjoint(checkpoints=N)   # default
diffrax.BacksolveAdjoint()                          # continuous adjoint
diffrax.ForwardMode()                               # forward sensitivity
diffrax.DirectAdjoint()                             # straight reverse-mode

# optimistix adjoints (for minimisers)
optimistix.RecursiveCheckpointAdjoint()             # default for FourDVarNet solver
optimistix.ImplicitAdjoint()                        # default for classical minimisers
optimistix.DirectAdjoint()

# vardax-owned (targeting upstream contribution)
vardax.adjoints.OneStepAdjoint()                    # Bolte et al. 2023
```

A typical 4DVar configuration:

```python
model = StrongFourDVar(
    forward=somax_model,
    obs_op=AveragingKernel(...),
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=optimistix.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optimistix.ImplicitAdjoint(),
    forward_adjoint=diffrax.BacksolveAdjoint(),     # memory-efficient
)
```

---

## Posterior adapters (`vardax.posterior`)

```python
class LaplaceCovariance(eqx.Module):
    """P* = (Hᵀ R⁻¹ H + B⁻¹)⁻¹ at MAP."""
    def __call__(self, analysis, model, batch) -> Posterior: ...

class GaussNewtonHessian(eqx.Module):
    """Krylov / Lanczos inversion of J''(x*) via lineax."""
    n_krylov: int = eqx.field(static=True, default=50)
    def __call__(self, analysis, model, batch) -> Posterior: ...

class EnsembleCovariance(eqx.Module):
    """Posterior from ensemble of analyses (delegates to filterax)."""
    n_members: int = eqx.field(static=True)
    def __call__(self, analyses, model, batch) -> Posterior: ...

class GaussianMarkLikelihood(eqx.Module):
    """Posterior → mark-likelihood for population models."""
    posterior: Posterior
    event_metadata: dict
    def to_dict(self) -> dict: ...
```

`OptimalInterpolation.posterior(batch)` and
`IncrementalFourDVar.posterior(batch)` are closed-form / reused-Hessian
shortcuts that skip the explicit adapter call.

---

## Data types

### `Batch*`

```python
class Batch1D(eqx.Module):
    input: Float[Array, "B T N"]
    mask: Float[Array, "B T N"]
    target: Float[Array, "B T N"] | None = None
    instrument: Int[Array, "B T N"] | None = None     # per-pixel instrument_id
    obs_err: Float[Array, "B T N"] | None = None      # heteroscedastic σ

class Batch2D(eqx.Module): ...
class Batch2DMultivar(eqx.Module): ...
class Batch3D(eqx.Module): ...   # planned
```

`instrument` and `obs_err` are `None` for single-instrument /
homoscedastic cases. `MultiInstrumentFusion` requires them.

### `Posterior`

```python
class Posterior(eqx.Module):
    mean: Array
    cov: AbstractLinearOperator | None       # gaussx / lineax operator
    samples: Array | None                     # (B, M, ...)
    provenance: dict                          # forward_model_id, n_iter, J_star, …
```
