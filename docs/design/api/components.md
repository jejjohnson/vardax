---
status: draft
version: 0.3.0
---

# Layer 1 — Components

`eqx.Module` operators that compose Layer 0 primitives. Protocols define
extension points; concrete implementations provide baselines.

Per Decision D8, vardax components that map onto pipekit-cycle concepts
satisfy those protocols directly — no parallel `Abstract*` hierarchy.

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
# vardax/protocols.py

@runtime_checkable
class Prior(Protocol):
    """φ: state → regularised state."""
    def __call__(self, x: Array) -> Array: ...


@runtime_checkable
class GradModulator(Protocol):
    """Φ: (gradient, carry) → (update, new_carry)."""
    def __call__(self, grad: Array, carry: Any) -> tuple[Array, Any]: ...


@runtime_checkable
class CostFunction(Protocol):
    """J: (state, batch, …) → scalar."""
    def __call__(self, x: Array, batch: Batch, **kwargs) -> Float[Array, ""]: ...


@runtime_checkable
class PosteriorAdapter(Protocol):
    """Maps inference output → mean + cov + provenance."""
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...
```

---

## Priors (`vardax.priors`)

### Learned autoencoder priors

```python
class BilinAEPrior1D(eqx.Module):
    """φ(x) = ReLU(A·x) ⊙ tanh(B·x) → linear decode."""
    state_dim: int; latent_dim: int; n_time: int
    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]: ...

class BilinAEPrior2D(eqx.Module): ...           # (B, T, H, W)
class BilinAEPrior2DMultivar(eqx.Module): ...   # (B, T, C, H, W)
class ConvAEPrior1D(eqx.Module): ...            # periodic 1D conv AE
class MLPAEPrior1D(eqx.Module): ...
```

### Identity / classical baseline

```python
class IdentityPrior(eqx.Module):
    """φ(x) = x. Zero-parameter. Use for pure obs-driven baselines."""
    def __call__(self, x: Array) -> Array: ...
```

### Dynamical prior (wraps any `ForwardModel`)

```python
class DynamicalPrior(eqx.Module):
    """Wrap any pipekit_cycle.ForwardModel as φ(x)."""
    forward: ForwardModel
    n_steps: int = eqx.field(static=True)

    def __call__(self, x: Array) -> Array:
        state = x
        for _ in range(self.n_steps):
            state = self.forward.step(state, self.forward.dt)
        return state
```

This is how `somax.ShallowWaterModel`, `plumax.GaussianPlumeForward`, etc.
become vardax priors — no per-library adapter.

### Score-based / diffusion prior (planned, Epic 8)

```python
class DiffusionPrior(eqx.Module):
    """φ(x) via learned score s_θ(x, t). For AmortizedVarDA."""
    score_net: eqx.Module
    ...
```

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

All satisfy `pipekit_cycle.ObservationOperator` (Decision D8): `__call__`
plus `linearize` returning a `lineax`/`gaussx` linear operator.

---

## Gradient modulators (`vardax.grad_mod`)

```python
class ConvLSTMGradMod1D(eqx.Module):
    """1D ConvLSTM gradient modulator."""
    state_channels: int  # typically T
    hidden_dim: int
    kernel_size: int = eqx.field(static=True, default=3)
    def __call__(self, grad, carry) -> tuple[Array, LSTMState1D]: ...

class ConvLSTMGradMod2D(eqx.Module): ...

class MLPGradMod(eqx.Module):
    """Dense MLP gradient modulator. Dimension-agnostic via flatten."""
    ...

class AttentionGradMod(eqx.Module):
    """Self-attention over spatial axis (planned, Epic 3)."""
    ...

class IdentityGradMod(eqx.Module):
    """update = -α · grad. Classical 4DVar baseline."""
    alpha: float = eqx.field(static=True, default=0.2)
    def __call__(self, grad, carry):
        return -self.alpha * grad, carry
```

---

## Cost functions (`vardax.costs`)

```python
class WeakConstraintCost(eqx.Module):
    """Standard J = α_obs · J_obs + α_prior · J_prior."""
    prior: Prior
    obs_op: ObservationOperator
    alpha_obs: float = 1.0
    alpha_prior: float = 1.0
    def __call__(self, x: Array, batch: Batch) -> Scalar: ...


class StrongConstraintCost(eqx.Module):
    """J = ||x_0 - x_b||²_B + Σ_t ||y_t - H_t(M_t(x_0))||²_R.

    Forward model M is rolled out from x_0; only the initial condition is
    the control variable.
    """
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    B_inv_op: AbstractLinearOperator
    R_inv_op: AbstractLinearOperator
    def __call__(self, x_0: Array, batch: Batch) -> Scalar: ...


class IncrementalCost(eqx.Module):
    """Linearised J for incremental 4DVar inner loop (Decision D11)."""
    forward_lin: AbstractLinearOperator
    obs_op_lin: AbstractLinearOperator
    x_b: Array
    B_inv_op: AbstractLinearOperator
    R_inv_op: AbstractLinearOperator
    def __call__(self, dx: Array, batch: Batch) -> Scalar: ...
```

---

## Solver configs (`vardax._src._types`)

```python
class SolverConfig(eqx.Module):
    """Config for VarDANet (learned 4DVarNet) inner loop."""
    n_steps: int = eqx.field(static=True)
    alpha: float = 0.2
    prior_weight: float = 1.0
    grad_mode: GradMode = eqx.field(static=True, default="one_step")


class IncrementalConfig(eqx.Module):
    """Config for IncrementalVarDA (operational 4DVar)."""
    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=20)
    cg_atol: float = 1e-5
    cg_rtol: float = 1e-5
    cvt: bool = eqx.field(static=True, default=True)


class AmortizedConfig(eqx.Module):
    """Config for AmortizedVarDA (direct head)."""
    head_type: Literal["flow", "score", "regression"] = eqx.field(static=True, default="flow")
    n_samples: int = eqx.field(static=True, default=64)
    temperature: float = 1.0
```

---

## Optimistix integration

For `grad_mode="implicit"`, `VarDANet*` delegates to
`optimistix.FixedPointIteration`:

```python
import optimistix as optx

solver = optx.FixedPointIteration(rtol=1e-5, atol=1e-5)
sol = optx.fixed_point(prior_projection_fn, solver, x0, args=batch)
# sol.value has correct gradients via IFT — no jaxopt
```

For incremental 4DVar inner loop, `IncrementalVarDA*` uses `lineax.CG`:

```python
import lineax as lx

solver = lx.CG(atol=1e-5, rtol=1e-5, max_steps=config.n_inner)
sol = lx.linear_solve(hessian_op, gradient, solver)
dx_star = sol.value
```

---

## Posterior adapters (`vardax.posterior`)

```python
class LaplaceCovariance(eqx.Module):
    """P* = (Hᵀ R⁻¹ H + B⁻¹)⁻¹ at MAP. Cheap; Gaussian-likelihood-only."""
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...

class GaussNewtonHessian(eqx.Module):
    """Krylov / Lanczos inversion of J''(x*) via lineax. Mid-cost; exact at MAP."""
    n_krylov: int = eqx.field(static=True, default=50)
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...

class EnsembleCovariance(eqx.Module):
    """Posterior from ensemble of analyses (delegates to filterax)."""
    n_members: int = eqx.field(static=True)
    def __call__(self, analyses: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...

class GaussianMarkLikelihood(eqx.Module):
    """Posterior → mark-likelihood for population models (Tier V, Decision D10)."""
    posterior: Posterior
    event_metadata: dict
    def to_dict(self) -> dict: ...
```

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

class Batch2D(eqx.Module):
    input: Float[Array, "B T H W"]
    mask: Float[Array, "B T H W"]
    target: Float[Array, "B T H W"] | None = None
    instrument: Int[Array, "B T H W"] | None = None
    obs_err: Float[Array, "B T H W"] | None = None

class Batch2DMultivar(eqx.Module):
    input: Float[Array, "B T C H W"]
    mask: Float[Array, "B T C H W"]
    target: Float[Array, "B T C H W"] | None = None
    instrument: Int[Array, "B T H W"] | None = None
    obs_err: Float[Array, "B T C H W"] | None = None
```

`instrument` and `obs_err` are `None` for single-instrument or
homoscedastic cases — `MultiInstrumentFusion` requires them.

### `Posterior`

```python
class Posterior(eqx.Module):
    """Output of every PosteriorAdapter."""
    mean: Array
    cov: AbstractLinearOperator | None       # gaussx / lineax operator
    samples: Array | None                     # (B, M, ...)
    provenance: dict                          # forward_model_id, n_iter, J_star, …
```
