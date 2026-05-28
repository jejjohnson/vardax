---
status: draft
version: 0.4.0
---

# vardax — Architecture

## Three-Layer Stack

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Models  (each satisfies pipekit_cycle.AnalysisStep)            │
│                                                                           │
│  Classical:                                                               │
│    OptimalInterpolation   — BLUE / OI, closed-form linear-Gaussian        │
│    ThreeDVar              — 3D variational, nonlinear H                   │
│    StrongFourDVar         — 4DVar, control = x_0                          │
│    WeakFourDVar           — 4DVar, control = (x_0, η_1, …, η_T)           │
│    IncrementalFourDVar    — operational GN+CG+CVT form of strong-4DVar    │
│                                                                           │
│  Learned:                                                                 │
│    FourDVarNet            — learned φ_θ + learned Φ_φ                     │
│    AmortizedPosterior     — direct q_φ(x | y) head                        │
│                                                                           │
│  Latent (v0.5+, D17):                                                     │
│    LatentThreeDVar        — 3DVar in z                                    │
│    LatentStrongFourDVar   — 4DVar with latent M_z                         │
│    LatentHybridFourDVar   — physics in x, control + update in z           │
│                                                                           │
│  + train_step, eval_step (Layer 0 primitives used by all learned models)  │
├───────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Components  (eqx.Module operators)                             │
│                                                                           │
│  Prior protocol + impls:  BilinAE, ConvAE, MLP, DynamicalPrior, Diffusion │
│  ObservationOperator:     MaskedIdentity, LinearObs, AveragingKernel,     │
│                           MultiInstrumentFusion, InstrumentRegistry       │
│  GradModulator (4DVarNet only): ConvLSTM, MLP, Attention, Identity        │
│  CostFunction:            WeakConstraint, StrongConstraint, Incremental,  │
│                           ThreeDVarCost, BLUECost                         │
│  Minimiser (optimistix wrapper):  GaussNewton, BFGS, NonlinearCG, …       │
│  PosteriorAdapter:        Laplace, GaussNewtonHessian, EnsembleCovariance │
│  SolverConfig, IncrementalConfig, AmortizedConfig, Batch*                 │
├───────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives  (pure JAX)                                         │
│                                                                           │
│  Cost terms:        obs_cost, prior_cost, model_error_cost                │
│                     variational_cost, incremental_cost                    │
│  Closed-form:       blue_analysis (linear-Gaussian)                       │
│  CVT:               cvt_transform, cvt_inverse (gaussx Matérn factor)     │
│  Posterior:         laplace_covariance, gauss_newton_hessian              │
│  Adjoint wiring:    diffrax.AbstractAdjoint + optimistix.AbstractAdjoint  │
│                     passthrough; no vardax-owned grad_mode enum           │
│  Training:          train_loss, train_step                                │
└───────────────────────────────────────────────────────────────────────────┘

Foundation (required dependencies):
┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐
│ equinox  │ │ optimistix │ │  optax   │ │   jax    │ │ pipekit +        │
│ (modules)│ │ (minimisers│ │ (outer   │ │ (autodiff│ │ pipekit-cycle    │
│          │ │  +adjoints)│ │  optim)  │ │   vmap)  │ │ (protocols +     │
│          │ │            │ │          │ │          │ │  DACycle, etc.)  │
└──────────┘ └────────────┘ └──────────┘ └──────────┘ └──────────────────┘
┌──────────┐ ┌──────────┐ ┌──────────────────────┐
│  lineax  │ │  gaussx  │ │ diffrax              │
│ (CG inner│ │ (Matérn  │ │ (ODE + Backsolve /   │
│  loop)   │ │  B^{1/2})│ │  Recursive /         │
│          │ │          │ │  ForwardMode adjoint)│
└──────────┘ └──────────┘ └──────────────────────┘
```

## Design Principles

1. **DA hierarchy is horizontal.** Seven peer analysis classes, one
   protocol (`pipekit_cycle.AnalysisStep`). No parent–child relationships
   between methods. (Decision [D14](decisions.md#d14-da-hierarchy-as-horizontal-peer-classes))

2. **Equinox-native.** All components are `eqx.Module` pytrees.
   Compatible with `jax.jit`, `jax.grad`, `eqx.filter_vmap`, and the
   equinox ecosystem (optimistix, lineax, diffrax). (D1)

3. **Protocol satisfaction, not duplication.** Vardax classes directly
   satisfy `pipekit_cycle.ForwardModel`, `ObservationOperator`, and
   `AnalysisStep`. Vardax-specific protocols (`Prior`, `GradModulator`,
   `CostFunction`, `PosteriorAdapter`) exist only where pipekit-cycle
   has no equivalent. (D2, D8)

4. **Adjoints come from upstream.** Gradients through dynamics use
   `diffrax.AbstractAdjoint`; gradients through inner minimisation use
   `optimistix.AbstractAdjoint`. Vardax owns no `grad_mode` enum.
   (Decision [D15](decisions.md#d15-lean-on-optimistix-diffrax-adjoints-not-in-house-grad-modes))

5. **BLUE / OI is a first-class method.** The closed-form linear-Gaussian
   analysis is not folded into 3DVar — it's its own `AnalysisStep` with
   its own fast path. (Decision [D16](decisions.md#d16-blue-oi-as-a-first-class-method))

6. **Dimensional inheritance.** Each method's algorithm is
   dimension-agnostic; `*1D`, `*2D`, `*3D` subclasses set
   dimension-specific defaults. (D3)

7. **Nested module configuration.** Configuration is `eqx.Module` —
   serialisable, JIT-friendly, JaxModelOp-compatible. (D4)

8. **Library, not framework.** Ships `train_step` and `eval_step`;
   `fit()` is example code. Production training composes with
   `pipekit-train`. (D5)

9. **Forward models live elsewhere.** `somax` / `plumax` own the
   physics. L63 / L96 in `_src/utils` are demo utilities. (D7)

10. **Posterior is a first-class output.** Every analysis emits a
    `Posterior` container (mean + cov + samples + provenance) with a
    `GaussianMarkLikelihood` export adapter for downstream population
    models. (D10)

## Target Architecture

### Protocols

```python
# vardax/protocols.py

# Re-exports from pipekit-cycle — vardax satisfies these directly.
from pipekit_cycle import ForwardModel, ObservationOperator, AnalysisStep

# Vardax-specific protocols (no pipekit-cycle equivalent):

@runtime_checkable
class Prior(Protocol):
    """φ: state → regularised state.

    For Gaussian priors with mean `m` and covariance op `B`, φ is the
    affine map x ↦ m + B^{1/2}(B^{-1/2}(x - m)) — i.e. identity. For
    learned autoencoder priors, φ is the encode-decode map. For
    dynamical priors, φ integrates the forward model n steps.
    """
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
    """Maps inference output → mean + covariance + provenance."""
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...


@runtime_checkable
class Minimiser(Protocol):
    """Wrapper around optimistix.AbstractMinimiser exposing vardax's
    cost-function interface. Carries its own minimiser_adjoint."""
    def __call__(self, cost_fn: CostFunction, x0: Array, batch: Batch) -> Array: ...
```

### Configuration types

```python
class SolverConfig(eqx.Module):
    """Config for FourDVarNet inner loop."""
    n_steps: int = eqx.field(static=True)
    alpha: float = 0.2
    prior_weight: float = 1.0
    # No grad_mode enum — adjoints come from minimiser_adjoint slot.


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

### Layer 2 model classes — sketch

```python
# vardax/models/optimal_interpolation.py

class OptimalInterpolation(eqx.Module):
    """BLUE: x* = x_b + BH^T(HBH^T + R)^{-1}(y - Hx_b). Closed-form."""
    obs_op: ObservationOperator        # must have a linear `linearize(x_b) -> H`
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...


# vardax/models/threedvar.py

class ThreeDVar(eqx.Module):
    """3D variational: minimise ‖x-x_b‖²_B + ‖y-H(x)‖²_R over x."""
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...


# vardax/models/strong_fourdvar.py

class StrongFourDVar(eqx.Module):
    """4DVar, strong constraint. Control = x_0."""
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...


# vardax/models/weak_fourdvar.py

class WeakFourDVar(eqx.Module):
    """4DVar, weak constraint. Control = (x_0, η_1, …, η_T)."""
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    model_err_cov_op: AbstractLinearOperator   # Q
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...


# vardax/models/incremental_fourdvar.py

class IncrementalFourDVar(eqx.Module):
    """Operational 4DVar: GN outer + CG inner + CVT.

    Functionally equivalent to StrongFourDVar but with a specialised
    inner solver. Use this for production work."""
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    config: IncrementalConfig
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...


# vardax/models/fourdvarnet.py

class FourDVarNet(eqx.Module):
    """Learned 4DVar: learned prior φ_θ + learned grad modulator Φ_φ."""
    prior: Prior
    obs_op: ObservationOperator
    grad_mod: GradModulator
    config: SolverConfig
    # Adjoint through the unrolled solver:
    solver_adjoint: optimistix.AbstractAdjoint = RecursiveCheckpointAdjoint()
    # Adjoint through any dynamical prior:
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...


# vardax/models/amortized.py

class AmortizedPosterior(eqx.Module):
    """Direct q_φ(x | y) head — flow / score / regression."""
    encoder: eqx.Module
    head: eqx.Module
    config: AmortizedConfig

    def __call__(self, batch: Batch) -> Array: ...
    def sample(self, batch: Batch, key, n: int) -> Array: ...
    def log_prob(self, x: Array, batch: Batch) -> Scalar: ...
    def as_analysis_step(self) -> AnalysisStep: ...
```

### Data containers

```python
class Batch1D(eqx.Module):
    input: Float[Array, "B T N"]
    mask: Float[Array, "B T N"]
    target: Float[Array, "B T N"] | None = None
    instrument: Int[Array, "B T N"] | None = None
    obs_err: Float[Array, "B T N"] | None = None

class Batch2D(eqx.Module):
    input: Float[Array, "B T H W"]
    mask: Float[Array, "B T H W"]
    target: Float[Array, "B T H W"] | None = None
    instrument: Int[Array, "B T H W"] | None = None
    obs_err: Float[Array, "B T H W"] | None = None

# Same shape conventions as v0.3. Multivariate variant supplied via
# Batch2DMultivar; 3D via Batch3D.

class Posterior(eqx.Module):
    mean: Array
    cov: AbstractLinearOperator | None
    samples: Array | None
    provenance: dict
```

## Package Structure (target)

```
vardax/
├── __init__.py                        # Public API re-exports
├── _src/
│   ├── _types.py                      # Batch*, Posterior, configs
│   ├── protocols.py                   # Prior, GradModulator, CostFunction,
│   │                                  # PosteriorAdapter, Minimiser
│   │                                  # + re-exports of pipekit_cycle protocols
│   ├── costs/
│   │   ├── obs.py                     # obs_cost (with R^{-1})
│   │   ├── prior.py                   # prior_cost (with B^{-1})
│   │   ├── weak.py                    # variational_cost (3D / strong-4D)
│   │   ├── strong.py                  # strong-constraint cost
│   │   ├── weak_constraint.py         # weak-constraint cost (with η)
│   │   ├── incremental.py             # linearised incremental cost
│   │   ├── threedvar.py               # 3DVar cost
│   │   └── blue.py                    # closed-form BLUE
│   ├── obs_operators/
│   │   ├── masked.py                  # MaskedIdentity
│   │   ├── linear.py                  # LinearObs
│   │   ├── averaging_kernel.py        # AveragingKernel
│   │   └── multi_instrument.py        # MultiInstrumentFusion, InstrumentRegistry
│   ├── priors/
│   │   ├── autoencoders.py            # BilinAE, ConvAE, MLP, BilinAE2D*
│   │   ├── identity.py                # IdentityPrior
│   │   ├── dynamical.py               # DynamicalPrior — wraps any ForwardModel
│   │   └── diffusion.py               # score-based prior (planned)
│   ├── grad_mod/
│   │   ├── conv_lstm.py               # ConvLSTMGradMod1D/2D
│   │   ├── mlp.py                     # MLPGradMod
│   │   ├── attention.py               # AttentionGradMod (planned)
│   │   └── identity.py                # IdentityGradMod (= classical 4DVar baseline)
│   ├── minimisers/
│   │   └── adapters.py                # optimistix.AbstractMinimiser wrappers
│   ├── adjoints/
│   │   └── one_step.py                # Bolte 2023 as optimistix.AbstractAdjoint
│   ├── cvt.py                         # Control-variable transform (gaussx)
│   ├── posterior/
│   │   ├── laplace.py
│   │   ├── gauss_newton.py
│   │   ├── ensemble.py
│   │   └── adapter.py                 # GaussianMarkLikelihood
│   ├── models/
│   │   ├── optimal_interpolation.py   # OptimalInterpolation
│   │   ├── threedvar.py               # ThreeDVar
│   │   ├── strong_fourdvar.py         # StrongFourDVar
│   │   ├── weak_fourdvar.py           # WeakFourDVar
│   │   ├── incremental_fourdvar.py    # IncrementalFourDVar
│   │   ├── fourdvarnet.py             # FourDVarNet
│   │   └── amortized.py               # AmortizedPosterior
│   ├── cycle.py                       # .as_analysis_step() adapters
│   ├── training.py                    # train_step, eval_step, reconstruction_loss
│   └── utils/                         # Demo utilities
│       ├── dynamical_systems.py       # L63 / L96 simulators
│       ├── masks.py
│       ├── noise.py
│       ├── standardize.py
│       ├── preprocessing.py
│       ├── validation.py              # Six-step cycle gates (D12)
│       └── viz.py
├── docs/                              # Math reference + design docs (this directory)
├── notebooks/                         # Jupytext tutorials
└── tests/
    └── test_pipekit_protocols.py      # Conformance suite (Epic 1)
```

## Dependency Stack

**Required:**

```
jax              >= 0.5
jaxlib           >= 0.5
equinox          >= 0.11
optax            >= 0.2
optimistix       >= 0.1
diffrax          >= 0.5
lineax           >= 0.1
gaussx           >= 0.1
jaxtyping        >= 0.2.28
beartype         >= 0.18
einops           >= 0.8
pipekit          >= 0.1
pipekit-cycle    >= 0.1
```

**Optional extras:**

```
pipekit-jax         # [persist]  — JaxModelOp
pipekit-experiment  # [persist]  — ModelRegistry
pipekit-train       # [train]    — Loss / Callback / MetricWriter
filterax            # [ensemble] — EnKF / EnKS / EnKI
coordax             # [coords]   — coordinate-aware Batch construction
numpyro             # [mcmc]     — full Bayesian fallback
xarray              # [data]     — utility scripts
matplotlib          # [viz]      — utility plots
```

**Removed:** `flax` (replaced by equinox), `jaxopt` (replaced by
optimistix).

Build system: `hatchling` (PEP 621). Python ≥ 3.12, < 3.14. MIT license.

## pipekit-cycle Protocol Map (Decision D8)

| Protocol | Satisfied by |
|---|---|
| `pipekit_cycle.ForwardModel` | `DynamicalPrior` wrapper around any `somax` / `plumax` forward; somax / plumax forwards directly |
| `pipekit_cycle.ObservationOperator` | `MaskedIdentity`, `LinearObs`, `AveragingKernel` directly; `MultiInstrumentFusion` via `.to_observation_operator()` adapter |
| `pipekit_cycle.AnalysisStep` | All seven Layer 2 model classes via `.as_analysis_step()` |

See [`pipekit_composition.md`](pipekit_composition.md) for satisfaction
patterns and orchestration recipes.

## CI / Quality Gates

| Check | Command | Scope |
|---|---|---|
| Tests | `uv run pytest tests -x` | Full suite |
| Lint | `uv run ruff check .` | Entire repo |
| Format | `uv run ruff format --check .` | Entire repo |
| Typecheck | `uv run ty check vardax` | Package only |
| Protocol conformance (planned, Epic 1) | `uv run pytest tests/test_pipekit_protocols.py` | All Layer 2 models, all obs operators |

Conventional commits required. The protocol conformance suite is
delivered as part of Epic 1; it is not yet wired into CI on the v0.1.x
codebase.
