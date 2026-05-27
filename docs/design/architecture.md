---
status: draft
version: 0.3.0
---

# vardax — Architecture

## Three-Layer Stack

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Models                                                         │
│  VarDANet1D/2D/3D    ┐                                                    │
│  IncrementalVarDA*   ├─ each satisfies pipekit_cycle.AnalysisStep         │
│  AmortizedVarDA*     ┘                                                    │
│  train_step, eval_step, fit (example only)                                │
├───────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Components (eqx.Module operators)                              │
│  Prior protocol + impls (BilinAE, ConvAE, MLP, somax-wrap, diffusion)     │
│  ObservationOperator protocol + impls (Masked, AveragingKernel, MultiInst)│
│  GradModulator protocol + impls (ConvLSTM, MLP, Attention, Identity)      │
│  CostFunction protocol + impls (weak, strong, incremental)                │
│  PosteriorAdapter (Laplace, GN-Hessian, ensemble)                         │
│  SolverConfig, Batch*, InstrumentRegistry                                 │
├───────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives (pure JAX)                                          │
│  obs_cost, prior_cost, variational_cost, incremental_cost                 │
│  solver_step, unrolled_solve, one_step_solve, implicit_solve              │
│  cvt_transform (Matérn B^{-1/2} via gaussx), tangent_linear (jax.linearize)│
│  laplace_covariance, gauss_newton_hessian                                 │
│  train_loss, train_step                                                   │
└───────────────────────────────────────────────────────────────────────────┘

Adapters & integration:
┌──────────────────────────────────────────────────────────────────────────┐
│  vardax satisfies pipekit-cycle protocols **directly** (Decision D8)     │
│                                                                          │
│  ObservationOperator   ← satisfied by every vardax obs operator          │
│  ForwardModel          ← satisfied by DynamicalPrior wrappers            │
│  AnalysisStep          ← satisfied by VarDANet*, IncrementalVarDA*,      │
│                          AmortizedVarDA* (via .as_analysis_step())       │
└──────────────────────────────────────────────────────────────────────────┘

Foundation (required dependencies):
┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ equinox  │ │ optimistix │ │  optax   │ │   jax    │ │ pipekit +    │
│ (modules)│ │  (Gauss-   │ │ (outer   │ │ (autodiff│ │ pipekit-cycle│
│          │ │   Newton,  │ │  optim)  │ │   vmap)  │ │ (protocols + │
│          │ │   fp,      │ │          │ │          │ │  DACycle)    │
│          │ │   BFGS)    │ │          │ │          │ │              │
└──────────┘ └────────────┘ └──────────┘ └──────────┘ └──────────────┘
┌──────────┐ ┌──────────┐ ┌──────────┐
│  lineax  │ │  gaussx  │ │ diffrax  │
│ (CG inner│ │ (Matérn  │ │ (ODE for │
│  loop)   │ │  B^{1/2})│ │ dyn prior)│
└──────────┘ └──────────┘ └──────────┘
```

## Core Design Principles

1. **Equinox-native.** All components are `eqx.Module` pytrees. No mutable
   state, no side effects. Compatible with `jax.jit`, `jax.grad`,
   `eqx.filter_vmap`, and the equinox ecosystem (optimistix, lineax, diffrax).

2. **Protocol-driven, pipekit-aligned.** Vardax operators directly satisfy
   `pipekit_cycle.ObservationOperator`, `ForwardModel`, and `AnalysisStep`
   protocols — no parallel `Abstract*` hierarchy where pipekit already names
   the contract. Vardax adds protocols that pipekit doesn't: `Prior`,
   `GradModulator`, `CostFunction`, `PosteriorAdapter`.

3. **Three inference families, one core.** `VarDANet*` (learned 4DVarNet),
   `IncrementalVarDA*` (operational 4DVar with control-variable transform),
   `AmortizedVarDA*` (direct $q_\phi(x \mid y)$). All compose the same Layer 1
   operators and Layer 0 primitives.

4. **Dimensional inheritance.** Base `VarDANet` / `IncrementalVarDA` /
   `AmortizedVarDA` classes hold the algorithm. `*1D`, `*2D`, `*3D`
   subclasses set dimension-specific defaults (conv kernels, ConvLSTM shape).

5. **Composable with the ecosystem.** somax priors plug in as `Prior`;
   plumax transports plug in as `ForwardModel`; gaussx factorisations of $B$
   plug into the CVT; filterax ensembles plug in for EnVar; pipekit-cycle
   orchestrates the DA cycle.

6. **Implicit differentiation as a first-class option.** Three gradient modes
   (unrolled, one-step, implicit via `optimistix.FixedPointIteration`).

7. **Library, not framework.** Ships building blocks. `train_step` and
   `eval_step` are the thinnest convenience layer. `fit()` moves to example
   notebooks. Production training composes vardax primitives with
   `pipekit-train` callbacks.

8. **Posterior is a first-class output.** Every analysis emits a
   `PosteriorAdapter` (mean + covariance + provenance) compatible with
   downstream population models — not just a point estimate.

9. **Multi-instrument observation is the default.** `MultiInstrumentFusion`
   composes per-instrument `(A, x_a, h, mask, R)` — no pre-regridding.
   `MaskedIdentity` is a degenerate case, not the default.

## Target Architecture

### Protocols

```python
# vardax/_src/protocols.py — direct re-exports + vardax-specific additions

from pipekit_cycle import ForwardModel, ObservationOperator, AnalysisStep

# Re-exported, NOT shadowed. Vardax classes satisfy these structurally
# via runtime_checkable Protocol — no `Abstract*` parallel hierarchy.

# Vardax-specific protocols (pipekit-cycle has no equivalent):

@runtime_checkable
class Prior(Protocol):
    """φ: state → regularized state."""
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
    """Maps inference output → mean + covariance + provenance."""
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...
```

### State containers

```python
# vardax/_src/_types.py

class Batch1D(eqx.Module):
    input: Float[Array, "B T N"]
    mask: Float[Array, "B T N"]
    target: Float[Array, "B T N"] | None = None
    instrument: Int[Array, "B T N"] | None = None  # per-pixel instrument id
    obs_err: Float[Array, "B T N"] | None = None   # heteroscedastic σ

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

class SolverConfig(eqx.Module):
    n_steps: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    prior_weight: float = eqx.field(static=True)
    grad_mode: GradMode = eqx.field(static=True)  # "unrolled" | "one_step" | "implicit"

class IncrementalConfig(eqx.Module):
    """Incremental 4DVar config: outer Gauss-Newton + inner CG/Lanczos."""
    n_outer: int = eqx.field(static=True, default=3)
    n_inner: int = eqx.field(static=True, default=20)
    cg_atol: float = eqx.field(static=True, default=1e-5)
    cg_rtol: float = eqx.field(static=True, default=1e-5)
    cvt: bool = eqx.field(static=True, default=True)  # apply control-variable transform

class Posterior(eqx.Module):
    """Output of every AnalysisStep.posterior(batch)."""
    mean: Array
    cov: AbstractLinearOperator | None        # gaussx operator, may be None for amortized samples
    samples: Array | None                      # ensemble or flow samples (B, M, ...)
    provenance: dict                           # instruments used, met source, n_iter, J*, convergence flag
```

### Model classes (Layer 2)

```python
# vardax/_src/models/vardanet.py

class VarDANet(eqx.Module):
    """Base learned 4DVarNet. Satisfies AnalysisStep via .as_analysis_step()."""
    prior: Prior
    obs_op: ObservationOperator
    grad_mod: GradModulator
    solver_config: SolverConfig

    def __call__(self, batch: Batch) -> Array:
        """Training interface: batch → x_reconstructed."""
        ...

    def as_analysis_step(self) -> AnalysisStep:
        """Adapt to pipekit_cycle.AnalysisStep signature."""
        return _VarDANetAnalysisStep(self)


class VarDANet1D(VarDANet): ...
class VarDANet2D(VarDANet): ...
class VarDANet3D(VarDANet): ...


# vardax/_src/models/incremental.py

class IncrementalVarDA(eqx.Module):
    """Operational 4DVar via Gauss-Newton outer + CG inner. CVT optional."""
    forward: ForwardModel              # tangent-linear via jax.linearize
    obs_op: ObservationOperator
    prior_mean: Array                  # x_b
    prior_cov_op: AbstractLinearOperator  # gaussx Matérn factorisation
    obs_cov_op: AbstractLinearOperator    # gaussx diagonal / block-diag
    config: IncrementalConfig

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...


class IncrementalVarDA2D(IncrementalVarDA): ...
class IncrementalVarDA3D(IncrementalVarDA): ...


# vardax/_src/models/amortized.py

class AmortizedVarDA(eqx.Module):
    """Direct q_φ(x | y) head (conditional flow / diffusion / regression)."""
    encoder: eqx.Module     # y, mask → conditioning context
    head: eqx.Module        # context → posterior parameters or samples
    config: AmortizedConfig

    def __call__(self, batch: Batch) -> Array: ...                # MAP / mode
    def sample(self, batch: Batch, key, n: int) -> Array: ...    # posterior samples
    def as_analysis_step(self) -> AnalysisStep: ...
```

### Observation operators (Layer 1)

```python
# vardax/_src/obs_operators/

class MaskedIdentity(eqx.Module):
    """H(x) = mask ⊙ x. ObservationOperator-compliant."""
    def __call__(self, x: Array, mask: Array | None = None) -> Array: ...
    def linearize(self, x: Array) -> JacobianLinearOperator: ...


class AveragingKernel(eqx.Module):
    """RTM averaging kernel: ŷ = A(h·x + (1-h)·x_a). Per-instrument."""
    A: AbstractLinearOperator          # averaging kernel matrix (gaussx)
    x_a: Array                          # retrieval prior
    h: Array                            # weighting vector
    def __call__(self, x: Array) -> Array: ...
    def linearize(self, x: Array) -> AbstractLinearOperator: ...


class MultiInstrumentFusion(eqx.Module):
    """Compose per-instrument ObservationOperators into a single H.
    Fuses at the likelihood level — no pre-regridding."""
    registry: dict[str, ObservationOperator]
    def __call__(self, x: Array, batch: Batch) -> dict[str, Array]: ...


class InstrumentRegistry(eqx.Module):
    """Per-instrument (A, x_a, h, mask, R) lookup. Keyed by instrument_id."""
    entries: dict[str, InstrumentSpec]
```

### Posterior adapter (Layer 1)

```python
# vardax/_src/posterior/

class LaplaceCovariance(eqx.Module):
    """At MAP: P* = (Hᵀ R⁻¹ H + B⁻¹)⁻¹. Cheap; assumes Gaussian likelihood."""
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...

class GaussNewtonHessian(eqx.Module):
    """Krylov inversion of (J''(x*)) via lineax. Mid-cost; exact at MAP."""
    n_krylov: int = eqx.field(static=True, default=50)
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...

class EnsembleCovariance(eqx.Module):
    """Posterior from an ensemble of analyses (delegates to filterax)."""
    n_members: int = eqx.field(static=True)
    def __call__(self, analyses: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...

class GaussianMarkLikelihood(eqx.Module):
    """Posterior → mark-likelihood serialisation for population models (Tier V)."""
    posterior: Posterior
    event_metadata: dict
    def to_dict(self) -> dict: ...
```

## Package Structure (target)

```
vardax/
├── __init__.py                        # Public API re-exports
├── _src/
│   ├── _types.py                      # Batch*, SolverConfig, IncrementalConfig, Posterior
│   ├── protocols.py                   # Prior, GradModulator, CostFunction, PosteriorAdapter
│   │                                  # + re-exports of pipekit_cycle protocols
│   ├── costs/
│   │   ├── weak.py                    # obs_cost, prior_cost, variational_cost
│   │   ├── strong.py                  # strong-constraint cost (forward-model rollout)
│   │   └── incremental.py             # incremental_cost, gauss_newton_hessian
│   ├── obs_operators/
│   │   ├── masked.py                  # MaskedIdentity
│   │   ├── averaging_kernel.py        # AveragingKernel
│   │   ├── multi_instrument.py        # MultiInstrumentFusion, InstrumentRegistry
│   │   └── linear.py                  # LinearObs (matrix-vec)
│   ├── priors/
│   │   ├── autoencoders.py            # BilinAE, ConvAE, MLP, BilinAE2D*
│   │   ├── identity.py                # IdentityPrior
│   │   ├── dynamical.py               # somax-wrap: any ForwardModel as a Prior
│   │   └── diffusion.py               # score-based prior (planned)
│   ├── grad_mod/
│   │   ├── conv_lstm.py               # ConvLSTMGradMod1D/2D
│   │   ├── mlp.py                     # MLPGradMod
│   │   ├── attention.py               # AttentionGradMod (planned)
│   │   └── identity.py                # IdentityGradMod (classical 4DVar)
│   ├── solver/
│   │   ├── steps.py                   # solver_step_1d/2d, init_solver_state
│   │   ├── unrolled.py                # solve_4dvarnet (lax.scan)
│   │   ├── one_step.py                # one_step_solve (Bolte 2023)
│   │   ├── implicit.py                # fixed-point via optimistix.FixedPointIteration
│   │   └── incremental.py             # Gauss-Newton outer + CG inner via lineax
│   ├── cvt.py                         # Control-variable transform (gaussx B^{1/2})
│   ├── posterior/
│   │   ├── laplace.py                 # LaplaceCovariance
│   │   ├── gauss_newton.py            # GaussNewtonHessian (Krylov via lineax)
│   │   ├── ensemble.py                # EnsembleCovariance (filterax bridge)
│   │   └── adapter.py                 # GaussianMarkLikelihood + serialisation
│   ├── models/
│   │   ├── vardanet.py                # VarDANet + 1D/2D/3D
│   │   ├── incremental.py             # IncrementalVarDA + 2D/3D
│   │   └── amortized.py               # AmortizedVarDA (flow / diffusion / regression)
│   ├── cycle.py                       # AnalysisStep adapters: ._as_analysis_step impls
│   ├── training.py                    # train_step, eval_step, reconstruction_loss
│   └── utils/                         # Demo utilities (not library API)
│       ├── dynamical_systems.py       # L63/L96 simulators (via diffrax)
│       ├── masks.py                   # random / regular / feature masks
│       ├── noise.py                   # add_gaussian_noise
│       ├── standardize.py             # standardisation helpers
│       ├── preprocessing.py           # xr_to_batch, train_test_split
│       └── viz.py                     # Plotting helpers
├── docs/                              # Mathematical reference (16 chapters)
├── notebooks/                         # Jupytext tutorials
└── tests/
```

## Dependency Stack (target)

**Required:**

```
jax              >= 0.5
jaxlib           >= 0.5
equinox          >= 0.11
optax            >= 0.2
jaxtyping        >= 0.2.28
beartype         >= 0.18
diffrax          >= 0.5
optimistix       >= 0.1
lineax           >= 0.1
gaussx           >= 0.1     # Matérn factorisation, structured B/R operators
einops           >= 0.8
pipekit          >= 0.1     # Operator base; carrier-agnostic composition
pipekit-cycle    >= 0.1     # ForwardModel, ObservationOperator, AnalysisStep, DACycle
```

**Optional (declared as extras):**

```
pipekit-jax         # [persist]  — JaxModelOp for weight serialisation
pipekit-experiment  # [persist]  — ModelRegistry (LocalModelRegistry, S3ModelRegistry)
pipekit-train       # [train]    — Loss/Callback/MetricWriter protocols
filterax            # [ensemble] — EnVar / En4DVar / EnKI
coordax             # [coords]   — coordinate-aware Batch construction
numpyro             # [mcmc]     — full Bayesian inference fallback
xarray              # [data]     — utility scripts
matplotlib          # [viz]      — utility plots
```

**Removed:** `flax` (replaced by equinox), `jaxopt` (replaced by optimistix).

**Build system:** hatchling (PEP 621).
**Python:** ≥ 3.12, < 3.14. **License:** MIT.

## pipekit-cycle Integration (Direct Satisfaction)

vardax satisfies `pipekit-cycle` protocols **by construction** (Decision D8):

| Protocol | Satisfied by |
|---|---|
| `pipekit_cycle.ObservationOperator` | All classes in `vardax.obs_operators.*` |
| `pipekit_cycle.ForwardModel` | `DynamicalPrior` wrapper around any somax / plumax forward |
| `pipekit_cycle.AnalysisStep` | `VarDANet*.as_analysis_step()`, `IncrementalVarDA*.as_analysis_step()`, `AmortizedVarDA*.as_analysis_step()` |

This means users compose vardax with `pipekit_cycle.DACycle` /
`SmootherCycle` / `EnsembleDACycle` directly — no `vardax.adapters.pipekit`
shim module.

See [`pipekit_composition.md`](pipekit_composition.md) for protocol satisfaction
patterns and orchestration examples.

## CI / Quality Gates

| Check | Command | Scope |
|---|---|---|
| Tests | `uv run pytest tests -x` | Full suite |
| Lint | `uv run ruff check .` | Entire repo |
| Format | `uv run ruff format --check .` | Entire repo |
| Typecheck | `uv run ty check vardax` | Package only |
| Protocol conformance | `uv run pytest tests/test_pipekit_protocols.py` | All Layer 2 models |

All must pass before merge. GitHub Actions on push/PR.
Conventional commits required (`feat:`, `fix:`, `docs:`, `test:`, …).
