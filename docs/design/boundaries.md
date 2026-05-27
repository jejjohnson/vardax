---
status: draft
version: 0.3.0
---

# vardax — Boundaries

## What vardax does NOT do

- **Define forward models.** Geophysics → `somax` (shallow water, QG, MQG, SWM,
  spherical). Atmospheric transport / methane → `plumax` (Gaussian plume,
  Lagrangian Markov-1, Eulerian FV, RTM). vardax accepts any of them via
  `pipekit_cycle.ForwardModel`.
- **Own spatial operators.** Finite-volume → `finitevolX`. Spectral →
  `spectraldiffx`. Used inside `somax` / `plumax`, not by vardax directly.
- **Own structured linear algebra.** Matérn factorisations, Kronecker /
  LowRank / BlockDiag operators → `gaussx`. vardax composes them.
- **Own ensemble methods.** EnKF, EnKS, EnKI → `filterax`. vardax exposes
  ensemble-variational hooks (`EnsembleCovariance` posterior adapter) but
  filterax owns the propagation.
- **Own data I/O.** Satellite L1/L2 reading → `georeader`. Labelled arrays →
  `coordax` + `xarray`. vardax consumes `Batch*` containers; how they're
  populated is upstream.
- **Own experiment orchestration.** Cycles → `pipekit-cycle`. Training
  callbacks → `pipekit-train`. Model storage → `pipekit-experiment`. Vardax
  satisfies the protocols and composes.
- **Provide an opinionated training loop.** Ships `train_step` / `eval_step`
  as library code (encodes correct differentiation). `fit()` is an example
  notebook.
- **Provide production observation operators for specific instruments.**
  `AveragingKernel` is generic — instrument-specific `(A, x_a, h, mask, R)`
  comes from the user (or `plumax.instruments` for methane).

## Ownership Map

| Concern | Owner |
|---|---|
| Variational cost functions (weak / strong / incremental) | **vardax** |
| `Prior` protocol + AE / diffusion implementations | **vardax** |
| `pipekit_cycle.ObservationOperator` satisfaction + masked/AK/multi-instrument impls | **vardax** |
| `GradModulator` protocol + ConvLSTM / MLP / Attention / Identity impls | **vardax** |
| `pipekit_cycle.AnalysisStep` satisfaction (VarDANet, Incremental, Amortized) | **vardax** |
| `PosteriorAdapter` protocol + Laplace / GN-Hessian / Ensemble impls | **vardax** |
| Control-variable transform machinery | **vardax** (composes `gaussx`) |
| `train_step` / `eval_step` | **vardax** (thin) |
| `fit()` training loop | **user code** / examples |
| Geophysical forward models (SWM, QG, MQG, primitive eq.) | **somax** |
| Atmospheric transport forward models (Gaussian plume, Lagrangian, Eulerian) | **plumax** |
| Radiative transfer (HAPI LUTs, neural RTM) | **plumax** (RTM stack) |
| Spatial operators (FV, spectral) | **finitevolX** / **spectraldiffx** |
| ODE / SDE integration | **diffrax** |
| Optimisers (BFGS, GN, fixed-point) | **optimistix** |
| Linear solvers (CG, GMRES, Lanczos) | **lineax** |
| Structured operators (Matérn, Kronecker, LowRank) | **gaussx** |
| Ensemble methods (EnKF, EnKS, EnKI) | **filterax** |
| MCMC / NumPyro models | **user code** (vardax provides priors / costs as building blocks) |
| Cycle orchestration (DACycle, SmootherCycle) | **pipekit-cycle** |
| Run tracking (W&B, MLflow, DVC) | **pipekit-experiment** + adapters |
| Trained model persistence | **pipekit-jax** (`JaxModelOp`) + **pipekit-experiment** |
| Sensor data I/O (L1, L2, footprints) | **georeader** |
| Coordinate-aware arrays | **coordax** |
| Geospatial catalogs | **GeoCatalog** (geotoolz) |

## The "where does X go" test

| Question | If yes | If no |
|---|:---:|---|
| Is it the inference algorithm (cost, solver, posterior)? | → **vardax** | |
| Could a non-DA code use it as a forward model? | | → somax / plumax |
| Is it a spatial operator (diff, interp, advection)? | | → finitevolX |
| Is it a spectral transform or filter? | | → spectraldiffx |
| Is it an ODE/SDE integrator? | | → diffrax |
| Is it an optimisation algorithm? | | → optimistix |
| Is it a linear solver primitive? | | → lineax |
| Is it a structured matrix factorisation? | | → gaussx |
| Is it ensemble propagation / Kalman update? | | → filterax |
| Is it a cycle orchestrator (forecast/analysis loop)? | | → pipekit-cycle |
| Is it about persisting / versioning trained models? | | → pipekit-jax + pipekit-experiment |

## How somax / plumax models become priors and forward models

```python
import somax, plumax, vardax as vdx

# somax / plumax forward model satisfies pipekit_cycle.ForwardModel directly.
# vardax wraps it as a Prior when used as φ(x) in the variational cost.

class DynamicalPrior(eqx.Module, vdx.Prior):
    """Wrap any pipekit_cycle.ForwardModel as φ(x)."""
    forward: ForwardModel
    n_steps: int = eqx.field(static=True)

    def __call__(self, x):
        state = x
        for _ in range(self.n_steps):
            state = self.forward.step(state, self.forward.dt)
        return state

# Geophysics use case:
swm = somax.ShallowWaterModel(grid=grid, params=params)
prior = DynamicalPrior(forward=swm, n_steps=10)

# Methane use case (plumax provides the forward):
plume = plumax.GaussianPlumeForward(met=met_field, dispersion="MO")
# The plume forward is the prior on concentration field given source params.
```

## Dependency graph

```
                georeader ─→ coordax ─→ pipekit (Carrier-agnostic core)
                                            ↓
                                       pipekit-cycle ─→ DACycle, SmootherCycle
                                            ↑
finitevolX ──→ somax  ┐                     │
spectraldiffx ─────┘  │                     │
                      │   ┌──→ gaussx ──┐   │
                      ↓   │             ↓   │
                   plumax │           vardax ─→ pipekit-jax ─→ pipekit-experiment
                          │             ↑   │                     ↑
                          │           lineax │                     │
                          ↓           optimistix                   │
                       diffrax ──────────────┘                     │
                                                                   │
                                                              pipekit-train
                                                                   │
                                                              filterax (optional)
```

## Roadmap — Epics 0 through 10

### Epic 0: Equinox Migration (foundational)

The foundational migration from Flax NNX to Equinox. Blocks every other epic.

| Task | Description |
|---|---|
| `nnx.Module` → `eqx.Module` (priors, grad mods, models) | Type migration |
| `nnx.Linear/Conv` → `eqx.nn.Linear/Conv1d/Conv2d` | Layer migration |
| `nnx.Optimizer` → `optax` + `eqx.filter_value_and_grad` | Training migration |
| `NamedTuple` → `eqx.Module` for `Batch*`, `LSTMState*`, `SolverState*` | Type migration |
| Introduce `SolverConfig`, `IncrementalConfig`, `AmortizedConfig` as `eqx.Module` | Config |
| Remove `flax`; add `optimistix`, `lineax`, `gaussx` | pyproject.toml |

### Epic 1: Protocol Alignment (direct satisfaction, Decision D8)

Vardax classes satisfy `pipekit-cycle` protocols directly. No `Abstract*`
parallel hierarchy.

| Task | Description |
|---|---|
| `vardax.protocols` re-exports `ForwardModel`, `ObservationOperator`, `AnalysisStep` from pipekit-cycle | API surface |
| Add vardax-specific `Prior`, `GradModulator`, `CostFunction`, `PosteriorAdapter` protocols (runtime-checkable) | API surface |
| Every Layer 2 model exposes `.as_analysis_step()` returning an `AnalysisStep` | Adapter pattern |
| Every Layer 1 obs operator satisfies `pipekit_cycle.ObservationOperator` (`__call__` + `linearize`) | API surface |
| `tests/test_pipekit_protocols.py` enforces conformance | Conformance |

### Epic 2: Legacy Port from mvardax

| Task | Description |
|---|---|
| `DynamicalPrior` wrapping any `ForwardModel` | Physics priors |
| `StrongVarCost` (background term + forward-model rollout) | Strong-constraint 4DVar |
| NaN-safe observation operators | Robustness |
| Strong-constraint vs weak-constraint switch | Cost function variant |

### Epic 3: Model Architecture (three families)

| Task | Description |
|---|---|
| Base `VarDANet` + `1D/2D/3D` subclasses (learned 4DVarNet, retained from v0.1.x) | Existing → migrated |
| `IncrementalVarDA` + `2D/3D` (operational incremental 4DVar) | **New** |
| `AmortizedVarDA` (conditional flow / score head / regression) | **New** |
| `IdentityGradMod` (classical 4DVar with hand-tuned step size) | New |
| `MLPGradMod`, `AttentionGradMod` | New |
| Ensemble batch dim support across all three families | New |

### Epic 4: Observation Operators (multi-instrument first)

| Task | Description |
|---|---|
| `MaskedIdentity`, `LinearObs` | Baseline |
| `AveragingKernel(A, x_a, h)` | **First-class, day-one (Decision D9)** |
| `MultiInstrumentFusion(registry)` | Composition at likelihood level |
| `InstrumentRegistry` schema (`(A, x_a, h, mask, R)` per instrument_id) | Data contract |
| `linearize()` method via `lineax.JacobianLinearOperator` for all obs ops | TLM/adjoint |

### Epic 5: Solver Integration (incremental + optimistix)

| Task | Description |
|---|---|
| Incremental cost (tangent-linear via `jax.linearize`) | Operational 4DVar |
| Gauss-Newton outer + CG inner via `lineax.CG` | Operational solver |
| Control-variable transform via `gaussx` Matérn factorisation | Preconditioning (Decision D11) |
| `optimistix.FixedPointIteration` for implicit grad mode | Existing → migrated |
| Custom `optimistix.AbstractMinimiser` wrapping the learned ConvLSTM step | Future upstream |

### Epic 6: Posterior Adapters (Decision D10)

| Task | Description |
|---|---|
| `LaplaceCovariance` at MAP | Cheap UQ |
| `GaussNewtonHessian` via Krylov / lineax | Mid-cost UQ |
| `EnsembleCovariance` (delegates to `filterax`) | Ensemble UQ |
| `Posterior` container (mean, cov, samples, provenance) | Output contract |
| `GaussianMarkLikelihood` serialiser → mark-likelihood for population models | Tier V hand-off |

### Epic 7: pipekit Integration (Decision D8)

Vardax exposes everything as `pipekit.Operator`s; `AnalysisStep` satisfaction
is built-in. Optional extras for persistence / training callbacks / cycles.

| Task | Description |
|---|---|
| `vardax.cycle.VarDACycle(forward, obs_op, model)` constructor that returns a configured `pipekit_cycle.DACycle` | Orchestration sugar |
| `vardax.cycle.SmootherDACycle` for retrospective analysis | Sliding-window 4DVar |
| `JaxModelOp` wrappers for `VarDANet*`, `AmortizedVarDA*` (for registry persistence) | `[persist]` extra |
| `pipekit-train` `Loss` / `Callback` adapters around `train_step` | `[train]` extra |
| `vardax.experiment.LocalModelRegistry` shortcut | `[persist]` extra |

### Epic 8: Amortized Inference (Decision D12)

| Task | Description |
|---|---|
| `AmortizedVarDA` with conditional-flow head (`gauss_flows`) | Direct $q_\phi(x \mid y)$ |
| Score-based posterior head | Diffusion variant |
| Simulation-based amortized training (forward + prior → (x, y) pairs) | Training pipeline |
| Six-step cycle validation: amortized vs MAP vs MCMC oracle | Decision D12 gate |
| Adjoint-calibration test (amortized gradient ≈ physics gradient) | Hard gate |

### Epic 9: Hybrid Ensemble-Variational (Decision D12)

| Task | Description |
|---|---|
| `EnVarDA` hybrid: ensemble cov + variational solve | Depends on filterax |
| Per-instrument bias as joint state element | Multi-instrument fusion |
| Ornstein-Uhlenbeck process prior on $Q(t)$ (or analogous time-varying source) | Temporal coupling |

### Epic 10: Documentation & Tutorials

| Issue | Title |
|---|---|
| #20 | Physical models & ODE prior docs |
| #21 | Uncertainty quantification docs (Laplace, GN-Hessian, ensemble) |
| #22 | OceanBench SSH interpolation walkthrough |
| #23 | Parameter estimation tutorial (joint state + parameters) |
| #24 | Bilevel optimisation tutorial |
| #25 | Methane single-overpass walkthrough (plumax + vardax) |
| #26 | Multi-instrument fusion tutorial (TROPOMI + EMIT + GHGSat) |
| #27 | Incremental 4DVar tutorial (CVT, GN outer, CG inner) |
| #28 | Amortized inference tutorial (conditional flow head) |

### Dependency graph

```
Epic 0 (equinox migration)
  ↓
Epic 1 (protocol alignment) ──→ Epic 2 (legacy port)
  ↓                                ↓
Epic 3 (model architecture) ────→ Epic 4 (obs operators)
  ↓                                ↓
Epic 5 (solver integration) ←─── Epic 6 (posterior adapters)
  ↓                                ↓
Epic 7 (pipekit integration) ─→ Epic 8 (amortized) ─→ Epic 9 (hybrid EnVar)
  ↓
Epic 10 (docs & tutorials, continuous)
```

### Rough timeline

| Phase | Focus | Order |
|---|---|---|
| Phase 1 | Equinox migration + protocol alignment (Epics 0, 1) | First |
| Phase 2 | Architecture + obs operators (Epics 2, 3, 4) | Second |
| Phase 3 | Incremental 4DVar + posterior + pipekit (Epics 5, 6, 7) | Third |
| Phase 4 | Amortized + hybrid EnVar (Epics 8, 9) | Research |
| Phase 5 | Tutorials + real-world examples (Epic 10) | Continuous |

## Open Questions

1. **`coordax` adoption in `Batch*`.** Should `Batch*` carry `coordax.Field`
   instead of `Array`? Better provenance / coordinate-aware operations, but
   couples vardax to coordax. Defer to Epic 7.

2. **3D support depth.** True volumetric `VarDANet3D` vs multilayer-2D via
   `eqx.filter_vmap` over a leading axis. Decision deferred until first 3D
   use case (likely Eulerian methane in plumax Tier III).

3. **`numpyro` integration depth.** Should vardax priors / costs expose
   `dist.Distribution` interfaces for NumPyro sampling, or stay JAX-array
   only? Lean toward staying JAX-only; users wrap costs in NumPyro outside
   vardax.

4. **Package rename `vardax` → `4dvarX`.** Defer indefinitely. Keep `vardax`
   as the canonical name.

5. **`gaussx` maturity gate.** Incremental 4DVar with CVT depends on
   `gaussx.MaternLinearOperator` and structured solves. If gaussx isn't ready,
   fall back to `lineax`-only CG with identity preconditioner. Document the
   fallback path.

6. **Posterior provenance schema.** What metadata does `Posterior.provenance`
   carry? Proposal: `{forward_model_id, obs_ops_used, n_iter, J_star,
   converged, gaussx_op_hash, model_hash}`. Refine in Epic 6.

## Testing Strategy

### Test organization

- **9 test modules** covering all components plus pipekit conformance
- One test class per component per dimension
- Fixtures in `conftest.py` for batch construction (single + multi-instrument)

### Test categories

| Category | What's tested | Module |
|---|---|---|
| Types | `Batch*`, `SolverState*`, `Posterior` shape validation | `test_types.py` |
| Costs | obs / prior / weak-variational / incremental | `test_costs.py` |
| Obs operators | `MaskedIdentity`, `AveragingKernel`, `MultiInstrumentFusion` (+ `linearize()`) | `test_obs_operators.py` |
| Priors | All prior architectures + `DynamicalPrior` wrap of toy forward | `test_priors.py` |
| Grad mods | ConvLSTM / MLP / Attention / Identity forward + state | `test_grad_mod.py` |
| Solver | Steps, unrolled scan, one-step, implicit, incremental GN+CG | `test_solver.py` |
| Posterior | Laplace / GN-Hessian / Ensemble adapters | `test_posterior.py` |
| Models | `VarDANet*`, `IncrementalVarDA*`, `AmortizedVarDA*` end-to-end | `test_models.py` |
| Training | `train_step`, `eval_step`, loss computation | `test_training.py` |
| **Pipekit conformance** | Every Layer 2 model passes `isinstance(model.as_analysis_step(), AnalysisStep)`. Every obs op passes `isinstance(..., ObservationOperator)`. | `test_pipekit_protocols.py` |
| Utils | Dynamical systems, masks, preprocessing | `test_utils/` |

### Test priorities for migration

1. **Protocol conformance** — every Prior / ObsOp / GradMod / AnalysisStep satisfies its protocol
2. **JAX transform compatibility** — all components work under `jax.jit`, `jax.grad`, `eqx.filter_vmap`
3. **Gradient mode equivalence** — unrolled / one-step / implicit produce similar results (up to tolerance)
4. **Dimensional consistency** — 1D / 2D / 3D produce correct output shapes
5. **Six-step cycle validation gates** — emulator MAP ≈ physics MAP within tolerance (Decision D12)
6. **Adjoint correctness** — `linearize().T @ v ≈ jax.vjp(H)(x, v)` for all obs ops

## Relationship to Downstream Libraries

| Library | Role | Coupling |
|---|---|---|
| **somax** | Geophysical forward models, dynamical priors | Optional — accepts via `Prior` / `ForwardModel` protocols |
| **plumax** | Atmospheric transport + RTM forwards (Tier I-IV) | Optional — accepts via `ForwardModel` |
| **finitevolX / spectraldiffx** | Spatial operators inside somax/plumax | Indirect (via somax/plumax) |
| **diffrax** | ODE integration | Required (toy demo priors) |
| **optimistix** | Optimisers (Gauss-Newton, BFGS, FixedPoint) | Required |
| **lineax** | Linear solvers (CG, GMRES) | Required (incremental 4DVar) |
| **gaussx** | Structured operators (Matérn, Kronecker, LowRank) | Required (CVT) |
| **filterax** | Ensemble methods | Optional (`[ensemble]` extra) |
| **pipekit / pipekit-cycle** | Operator composition + cycle protocols | Required |
| **pipekit-jax / -experiment / -train** | Persistence, registry, training callbacks | Optional (`[persist]`, `[train]`) |
| **georeader / coordax** | Data I/O, labelled arrays | Optional (`[coords]`) |

### Key contract: JAX + pipekit transform compatibility

vardax guarantees:
- `jax.jit` — no Python-level side effects in operator `__call__`
- `jax.grad` / `eqx.filter_value_and_grad` — differentiable w.r.t. array params
- `eqx.filter_vmap` — batch over leading dims
- `pipekit_cycle.ObservationOperator` — `__call__(state) → obs`, `linearize(state) → LinearOp`
- `pipekit_cycle.ForwardModel` — `step(state, dt) → state`, `dt` property
- `pipekit_cycle.AnalysisStep` — `__call__(forecast, obs, *, obs_op, obs_err_cov) → analysis`

## Version History & Milestones

### Completed milestones

| Version | Milestone |
|---|---|
| 0.0.1–0.1.0 | Initial VarDANet (Flax NNX) |
| 0.1.1–0.1.3 | 1D + 2D models, BilinAE / ConvAE / MLP priors |
| 0.1.4 | Fixed-point solver + one-step differentiation (Bolte et al. 2023) |
| 0.1.5 | L63 / L96 dynamical system demos |
| 0.1.6 | Multivariate 2D, 8 tutorial notebooks, 14 math docs |

### Current: v0.1.6 (Flax NNX)

- Full VarDANet framework (1D + 2D)
- Three gradient modes (unrolled, implicit, one-step)
- 6 prior architectures + IdentityPrior
- ConvLSTM gradient modulators
- Comprehensive test suite

### Upcoming (v0.2.0+ — equinox-native, pipekit-aligned)

| Priority | Epic | Key work |
|---|---|---|
| P0 | Epic 0 | Equinox migration (Flax NNX → equinox, optax, optimistix) |
| P0 | Epic 1 | Pipekit-cycle protocol alignment |
| P0 | Epic 4 | Averaging kernel + multi-instrument obs operators |
| P1 | Epic 3 | `IncrementalVarDA` + `AmortizedVarDA` model families |
| P1 | Epic 5 | Incremental 4DVar solver (GN outer / CG inner / CVT) |
| P1 | Epic 6 | Posterior adapters (Laplace, GN-Hessian, ensemble) |
| P2 | Epic 7 | pipekit-jax + pipekit-experiment + pipekit-train integration |
| P2 | Epic 8 | Amortized inference (conditional flow head) |
| P3 | Epic 9 | Hybrid ensemble-variational (via filterax) |
| P3 | Epic 10 | Tutorials (methane, multi-instrument, incremental, amortized) |

### References

- Fablet, R. et al. (2021). "Learning Variational Data Assimilation Models and Solvers." *JAMES*.
- Fablet, R. et al. (2023). "Multimodal 4DVarNets for the reconstruction of sea surface dynamics." *IEEE TGRS*.
- Bolte, J. et al. (2023). "One-step differentiation of iterative algorithms." *NeurIPS*.
- Courtier, P. et al. (1994). "A strategy for operational implementation of 4D-Var, using an incremental approach." *QJRMS*.
- Carrassi, A. et al. (2018). "Data assimilation in the geosciences: An overview of methods, issues, and perspectives." *WIRES Climate Change*.
- Cohen, S. et al. (2023). "Score-based diffusion meets annealed importance sampling." *NeurIPS*.
- Predecessor: [mvardax](https://github.com/jejjohnson/mvardax) (deprecated).
- Reference: [CIA-Oceanix/4dvarnet-starter](https://github.com/CIA-Oceanix/4dvarnet-starter).
