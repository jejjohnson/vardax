---
status: draft
version: 0.4.0
---

# vardax — Boundaries

## What vardax does NOT do

- **Define forward models.** Geophysics → `somax` (shallow water, QG,
  MQG, SWM, spherical). Atmospheric transport / methane → `plumax`
  (Gaussian plume, Lagrangian Markov-1, Eulerian FV, RTM). Vardax
  accepts any of them via `pipekit_cycle.ForwardModel`.
- **Own spatial operators.** Finite-volume → `finitevolX`. Spectral →
  `spectraldiffx`. Used inside `somax` / `plumax`, not by vardax.
- **Own structured linear algebra.** Matérn factorisations, Kronecker /
  LowRank / BlockDiag operators → `gaussx`. Vardax composes them.
- **Own optimisers.** Gauss-Newton, BFGS, NonlinearCG, fixed-point
  iteration → `optimistix`. Vardax wraps them as `Minimiser` for the
  `CostFunction` interface but does not reimplement them.
- **Own ODE solvers or adjoints.** `diffrax` provides the integrators
  and the `RecursiveCheckpointAdjoint` / `BacksolveAdjoint` /
  `ForwardMode` / `DirectAdjoint` family. Vardax passes them through as
  constructor slots (Decision D15).
- **Own ensemble methods.** EnKF, EnKS, EnKI → `filterax`. Vardax
  exposes `EnsembleCovariance` for posterior assembly and accepts
  ensemble batches, but propagation lives in filterax.
- **Own data I/O.** Sensor data → `georeader`. Labelled arrays →
  `coordax` + `xarray`. Vardax consumes `Batch*` containers.
- **Own experiment orchestration.** Cycles → `pipekit-cycle`. Training
  callbacks → `pipekit-train`. Model storage → `pipekit-experiment`.
- **Provide an opinionated training loop.** `train_step` / `eval_step`
  ship as library code. `fit()` is example-only.

## Ownership Map

| Concern | Owner |
|---|---|
| Variational cost functions (weak / strong / incremental / 3DVar / BLUE) | **vardax** |
| `Prior` protocol + AE / diffusion / dynamical impls | **vardax** |
| `ObservationOperator` impls (masked, AK, multi-instrument) | **vardax** |
| `GradModulator` protocol + impls (`FourDVarNet` only) | **vardax** |
| `AnalysisStep` impls (the seven Layer 2 classes) | **vardax** |
| `PosteriorAdapter` impls (Laplace / GN-Hessian / Ensemble) | **vardax** |
| Control-variable transform machinery | **vardax** (composes `gaussx`) |
| `train_step` / `eval_step` | **vardax** (thin) |
| `fit()` training loop | **user code / examples** |
| Geophysical forward models (SWM, QG, MQG, primitive eq.) | **somax** |
| Atmospheric transport forward models | **plumax** |
| Radiative transfer (HAPI LUTs, neural RTM) | **plumax** (RTM stack) |
| Spatial operators (FV, spectral) | **finitevolX / spectraldiffx** |
| ODE / SDE integration + adjoints | **diffrax** |
| Optimisers + adjoints | **optimistix** |
| Linear solvers (CG, GMRES, Lanczos) | **lineax** |
| Structured operators (Matérn, Kronecker, LowRank) | **gaussx** |
| Ensemble propagation (EnKF / EnKS / EnKI) | **filterax** |
| MCMC sampling | **user code** (vardax provides costs / priors as building blocks for NumPyro) |
| Cycle orchestration (`DACycle`, `SmootherCycle`) | **pipekit-cycle** |
| Run tracking | **pipekit-experiment** |
| Trained model persistence | **pipekit-jax** + **pipekit-experiment** |
| Sensor data I/O | **georeader** |
| Coordinate-aware arrays | **coordax** |
| Geospatial catalogs | **GeoCatalog** (geotoolz) |

## The "where does X go" test

| Question | Answer |
|---|:---|
| Is it a DA analysis algorithm (cost, minimiser wiring, adjoint composition)? | → **vardax** |
| Could a non-DA code use it as a forward model? | → somax / plumax |
| Is it a spatial differential operator? | → finitevolX / spectraldiffx |
| Is it an ODE / SDE integrator? | → diffrax |
| Is it an adjoint method for ODE integration? | → diffrax |
| Is it an optimisation algorithm? | → optimistix |
| Is it an adjoint method for optimisation? | → optimistix |
| Is it a linear solver primitive (CG, GMRES, Lanczos)? | → lineax |
| Is it a structured matrix factorisation (Matérn, Kronecker, LowRank)? | → gaussx |
| Is it ensemble propagation / Kalman update? | → filterax |
| Is it a forecast/analysis cycle orchestrator? | → pipekit-cycle |
| Is it about persisting / versioning trained models? | → pipekit-jax + pipekit-experiment |

## Dependency graph

```
                georeader ─→ coordax ─→ pipekit (carrier-agnostic core)
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
                                                              pipekit-train
                                                                   │
                                                              filterax (optional)
```

## Roadmap — Epics 0 through 13

### Epic 0: Equinox Migration (foundational)

Blocks every other epic. Replace Flax NNX with equinox; replace
`nnx.Optimizer` with `optax` + `eqx.filter_value_and_grad`; convert
`NamedTuple` types to `eqx.Module`; introduce `SolverConfig`,
`IncrementalConfig`, `AmortizedConfig` as `eqx.Module`. Remove `flax`
and `jaxopt`. Add `optimistix`, `lineax`, `gaussx`, `pipekit`,
`pipekit-cycle` as required.

### Epic 1: Protocol Alignment (Decision D8)

Vardax classes satisfy `pipekit-cycle` protocols directly.
`vardax.protocols` re-exports `ForwardModel`, `ObservationOperator`,
`AnalysisStep`. Vardax-specific protocols added for `Prior`,
`GradModulator`, `CostFunction`, `PosteriorAdapter`, `Minimiser`.
Conformance test suite `tests/test_pipekit_protocols.py` added.

### Epic 2: Observation Operators (Decision D9)

`MaskedIdentity`, `LinearObs`, `AveragingKernel(A, x_a, h)`,
`MultiInstrumentFusion(registry)`, `InstrumentRegistry`. All expose
`linearize()` via `lineax.JacobianLinearOperator` or a structured
override.

### Epic 2.5: Classical DA Methods (**New in v0.4 — Decisions D14, D16**)

The classical DA hierarchy as first-class peer classes:

- `OptimalInterpolation` — closed-form BLUE / OI via `gaussx`. Layer 0
  primitive `blue_analysis(x_b, y, B_op, R_op, H_op)`.
- `ThreeDVar` — `optimistix.GaussNewton` / `BFGS` over the nonlinear
  cost.
- `StrongFourDVar` — control = $x_0$, dynamics rollout via diffrax,
  adjoint via `forward_adjoint` slot.
- `WeakFourDVar` — augmented control, model-error cost term, separate
  `model_err_cov_op` (Q).

This epic establishes the foundation. `FourDVarNet` (Epic 6) and
`AmortizedPosterior` (Epic 8) are learned variants of these classical
methods.

### Epic 3: Adjoint Composition (**New in v0.4 — Decision D15**)

Drop `grad_mode` enum. Add `forward_adjoint: diffrax.AbstractAdjoint`
and `minimiser_adjoint: optimistix.AbstractAdjoint` constructor slots.
Validate default selections (`RecursiveCheckpointAdjoint` for both) via
the conformance suite. Add `BacksolveAdjoint` example for long-window
4DVar.

Implement `vardax._src.adjoints.one_step.OneStepAdjoint(
optimistix.AbstractAdjoint)` as the Bolte 2023 method; goal: contribute
upstream once stable (Decision D6).

### Epic 4: Incremental 4DVar (Decision D11)

`IncrementalFourDVar` as the operational fast path. Tangent-linear via
`jax.linearize`, Gauss-Newton outer, CG inner via `lineax.CG`,
control-variable transform via `gaussx.MaternLinearOperator.half()`.
Layer 0 primitives: `incremental_cost`, `cvt_transform`,
`gauss_newton_inner`, `incremental_outer`.

### Epic 5: Posterior Adapters (Decision D10)

`LaplaceCovariance`, `GaussNewtonHessian` (Krylov via `lineax`),
`EnsembleCovariance` (filterax bridge). `Posterior` container.
`GaussianMarkLikelihood` serialiser → mark-likelihood for downstream
population models.

### Epic 6: `FourDVarNet`

Learned variant of strong-constraint 4DVar. Learned prior $\varphi_\theta$
(AE family) + learned gradient modulator $\Phi_\phi$ (ConvLSTM / MLP /
Attention). The inner solver is itself a learned iteration; the adjoint
choice (`RecursiveCheckpointAdjoint` / `OneStepAdjoint` / `ImplicitAdjoint`)
comes from the `minimiser_adjoint` slot (Epic 3).

### Epic 7: pipekit Integration (Decision D8)

`vardax.cycle.VarDACycle(forward, obs_op, model)` constructor returning
a configured `pipekit_cycle.DACycle`. `JaxModelOp` wrappers for
`FourDVarNet`, `AmortizedPosterior`. `pipekit-train` `Loss` / `Callback`
adapters around `train_step`.

### Epic 8: Amortized Inference (Decision D12)

`AmortizedPosterior` with conditional-flow head (`gauss_flows`),
score-based head, and regression head. Simulation-based training loop.
Six-step cycle validation gates as part of
`tests/test_six_step_validation.py`.

### Epic 9: Hybrid Ensemble-Variational

`EnVarFourDVar` hybrid: ensemble cov + variational solve. Depends on
filterax. Per-instrument bias as joint state element. Ornstein-Uhlenbeck
process prior on time-varying source (or analogous).

### Epic 10: Documentation & Math Reference

The 17-chapter math reference (v0.4 rewrite — see
[`../index.md`](../index.md) for TOC). Includes the seven Layer 2
methods as separate chapters (4–10), shared foundation chapters (1–3),
and concrete example chapters (15–17).

### Epic 11: Real-World Tutorials

OceanBench SSH interpolation walkthrough; methane single-overpass with
plumax; multi-instrument fusion (TROPOMI + EMIT + GHGSat); incremental
4DVar tutorial; amortized inference tutorial.

### Epic 12: Performance Benchmarking

Per-method benchmarks on Lorenz / SSH / methane synthetic problems.
Adjoint-method memory/time tradeoffs (RecursiveCheckpoint vs Backsolve
vs ForwardMode). Comparison of FourDVarNet adjoint choices.

### Epic 13: Operational Deployment Patterns

FastAPI handler example; persistent GeoCatalog integration; real-time
alerting demo. The "research → operations arc" worked through
end-to-end.

### Dependency graph

```
Epic 0 (equinox migration)
  ↓
Epic 1 (protocol alignment)  ──→  Epic 2 (obs operators)
  ↓                                  ↓
Epic 2.5 (classical DA)  ──────→ Epic 3 (adjoint composition)
  ↓                                  ↓
Epic 4 (incremental 4DVar)  ←── Epic 5 (posterior adapters)
  ↓
Epic 6 (FourDVarNet)
  ↓
Epic 7 (pipekit integration)  ──→ Epic 8 (amortized) ──→ Epic 9 (hybrid EnVar)
  ↓
Epic 10 (docs)  ──→  Epic 11 (tutorials)  ──→  Epic 12 (benchmarks)  ──→  Epic 13 (ops)
```

### Rough timeline

| Phase | Focus | Order |
|---|---|---|
| Phase 1 | Equinox migration + protocols + obs operators (Epics 0–2) | First |
| Phase 2 | **Classical DA + adjoint composition (Epics 2.5, 3)** | **Second — foundation for everything learned** |
| Phase 3 | Incremental + posterior + FourDVarNet (Epics 4–6) | Third |
| Phase 4 | pipekit + amortized + hybrid (Epics 7–9) | Fourth |
| Phase 5 | Docs + tutorials + benchmarks + ops (Epics 10–13) | Continuous |

## Open Questions

1. **`coordax` adoption in `Batch*`.** Should batches carry
   `coordax.Field` instead of `Array`? Better provenance, but couples
   vardax to coordax. Defer to Epic 7.

2. **3D support depth.** True volumetric `*3D` classes vs multilayer-2D
   via `eqx.filter_vmap` over a leading axis. Decision deferred until
   first 3D use case (likely Eulerian methane in plumax Tier III).

3. **`numpyro` integration depth.** Vardax priors / costs as
   `dist.Distribution` objects for NumPyro sampling, or stay JAX-array
   only? Lean toward JAX-only; users wrap outside vardax.

4. **`gaussx` maturity gate.** Incremental 4DVar with CVT depends on
   `gaussx.MaternLinearOperator.half()`. If gaussx isn't ready, fall
   back to `lineax`-only CG with identity preconditioner. Document the
   fallback path.

5. **`OneStepAdjoint` upstreaming.** When does
   `vardax._src.adjoints.one_step.OneStepAdjoint` become
   `optimistix.OneStepAdjoint`? Depends on optimistix maintainer
   appetite. Track via Epic 3.

6. **Posterior provenance schema.** Open from v0.3 — refine in Epic 5.

7. **Hybrid analysis: which class owns `EnVarFourDVar`?** Could be in
   vardax (as the eighth model) or in filterax (as the variational
   variant of EnKF). Defer to Epic 9.

## Testing Strategy

### Test organisation

- One test module per Layer 2 class
- One test module per major Layer 1 family (obs operators, priors, grad
  mods, posteriors, minimisers)
- Conformance suite: `tests/test_pipekit_protocols.py`
- Six-step cycle validation: `tests/test_six_step_validation.py`

### Test categories

| Category | What's tested | Module |
|---|---|---|
| Types | `Batch*`, `Posterior` shape validation | `test_types.py` |
| Costs | obs / prior / weak / strong / incremental / 3DVar / BLUE | `test_costs.py` |
| Obs operators | `MaskedIdentity`, `LinearObs`, `AveragingKernel`, `MultiInstrumentFusion` (+ `linearize()` adjoint test) | `test_obs_operators.py` |
| Priors | All AE archs + `DynamicalPrior` wrap | `test_priors.py` |
| Grad mods | ConvLSTM / MLP / Attention / Identity (FourDVarNet only) | `test_grad_mod.py` |
| Minimisers | optimistix wrappers — GaussNewton, BFGS, NonlinearCG | `test_minimisers.py` |
| Adjoints | `RecursiveCheckpointAdjoint`, `BacksolveAdjoint`, `OneStepAdjoint` correctness | `test_adjoints.py` |
| Posterior | Laplace / GN-Hessian / Ensemble adapters | `test_posterior.py` |
| Models — classical | `OptimalInterpolation`, `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`, `IncrementalFourDVar` | `test_classical_models.py` |
| Models — learned | `FourDVarNet`, `AmortizedPosterior` | `test_learned_models.py` |
| Training | `train_step`, `eval_step`, reconstruction loss | `test_training.py` |
| **Pipekit conformance** | All seven Layer 2 models pass `isinstance(model.as_analysis_step(), AnalysisStep)`. All obs ops (or their `.to_observation_operator()` adapters) pass `isinstance(..., ObservationOperator)`. | `test_pipekit_protocols.py` |
| Six-step cycle gates | Posterior agreement, adjoint calibration, SBC | `test_six_step_validation.py` |
| Utils | Dynamical systems, masks, preprocessing | `test_utils/` |

### Test priorities

1. **Protocol conformance** — every Layer 2 class satisfies
   `AnalysisStep`; every obs op satisfies `ObservationOperator`.
2. **Linear-Gaussian baseline** — `OptimalInterpolation` agrees with
   `ThreeDVar(linear obs op)`, agrees with `StrongFourDVar(T=0,
   linear)`, agrees with `FourDVarNet(IdentityPrior, n_steps=∞)` in
   the linear-Gaussian limit. All four converge to the same posterior.
3. **JAX transform compatibility** — every component works under
   `jax.jit`, `jax.grad`, `eqx.filter_vmap`.
4. **Adjoint correctness** — every adjoint variant produces the same
   gradient up to floating-point tolerance.
5. **Dimensional consistency** — 1D / 2D / 3D subclasses produce correct
   output shapes.
6. **Six-step cycle gates** — emulator MAP ≈ physics MAP within
   tolerance (D12).

## Relationship to Downstream Libraries

| Library | Role | Coupling |
|---|---|---|
| `somax` | Geophysical forward models, dynamical priors | Optional — accepts via `Prior` / `ForwardModel` protocols |
| `plumax` | Atmospheric transport + RTM forwards | Optional — accepts via `ForwardModel` |
| `finitevolX / spectraldiffx` | Spatial operators inside somax / plumax | Indirect |
| `diffrax` | ODE integration + adjoints | **Required** (Decision D15) |
| `optimistix` | Optimisers + adjoints | **Required** (Decision D15) |
| `lineax` | Linear solvers | Required (incremental 4DVar, AbstractLinearOperator) |
| `gaussx` | Structured operators | Required (CVT, BLUE) |
| `filterax` | Ensemble methods | Optional (`[ensemble]` extra) |
| `pipekit / pipekit-cycle` | Operator composition + cycle protocols | **Required** |
| `pipekit-jax / -experiment / -train` | Persistence, registry, training callbacks | Optional |
| `georeader / coordax` | Data I/O, labelled arrays | Optional |

### Key contract: JAX + pipekit transform compatibility

Vardax guarantees:

- `jax.jit` — no Python-level side effects in operator `__call__`
- `jax.grad` / `eqx.filter_value_and_grad` — differentiable w.r.t. array
  params
- `eqx.filter_vmap` — batches over leading dimensions
- `pipekit_cycle.ObservationOperator` — `__call__(state) → obs`,
  `linearize(state) → AbstractLinearOperator`
- `pipekit_cycle.ForwardModel` — `step(state, dt) → state`, `dt`
  property
- `pipekit_cycle.AnalysisStep` — `__call__(forecast, obs, *, obs_op,
  obs_err_cov) → analysis`

## Version History

| Version | Milestone |
|---|---|
| 0.0.1–0.1.0 | Initial VarDANet implementation (Flax NNX) |
| 0.1.1–0.1.3 | 1D + 2D models, BilinAE / ConvAE / MLP priors |
| 0.1.4 | Fixed-point solver + one-step differentiation (Bolte 2023) |
| 0.1.5 | L63 / L96 dynamical system demos |
| 0.1.6 | Multivariate 2D, 8 tutorial notebooks, 11 math docs |
| 0.3.0 | Design doc v0.3: pipekit-cycle integration, AK + multi-instrument as first-class, three learned model families, six-step cycle methodology. Math chapters 12–16 added. |
| **0.4.0** | **Design doc v0.4: DA hierarchy as horizontal peer classes, optimistix/diffrax adjoint composition, BLUE/OI as first-class method, math reference fully rewritten to DA-textbook style (17 chapters).** |

### Current: v0.1.6 (Flax NNX implementation)

The package code is still v0.1.6 with Flax NNX. The v0.3 and v0.4 design
doc revisions target the equinox migration roadmap (Epics 0–13). Math
chapters describe the API documented in `docs/design/`, which is not yet
implemented in `src/vardax/_src/`.

### Upcoming (v0.2.0+ — equinox-native, pipekit-aligned)

| Priority | Epic | Key work |
|---|---|---|
| P0 | Epic 0 | Equinox migration |
| P0 | Epic 1 | Pipekit-cycle protocol alignment |
| P0 | Epic 2 | Averaging kernel + multi-instrument obs operators |
| **P0** | **Epic 2.5** | **Classical DA: OptimalInterpolation, ThreeDVar, StrongFourDVar, WeakFourDVar** |
| **P0** | **Epic 3** | **Adjoint composition via optimistix + diffrax** |
| P1 | Epic 4 | Incremental 4DVar with CVT |
| P1 | Epic 5 | Posterior adapters |
| P1 | Epic 6 | FourDVarNet (learned variant of strong-4DVar) |
| **P2** | **Epic 7** | **pipekit-cycle integration (`VarDACycle`, `VarSmootherCycle`) — landed in v0.5** |
| **P2** | **Epic 8** | **Amortized inference (`AmortizedPosterior`, regression head, six-step gates) — landed in v0.5; flow/score heads ship as stubs pending `gauss_flows`** |
| P3 | Epic 9 | Hybrid ensemble-variational |
| P3 | Epic 10 | Math reference (17 chapters) — landed in v0.4 |
| P3 | Epic 11 | Tutorials |
| P3 | Epic 12 | Benchmarks |
| P3 | Epic 13 | Operational deployment |

## References

- Talagrand, O., & Courtier, P. (1987). *Variational assimilation of
  meteorological observations with the adjoint vorticity equation.*
  QJRMS 113(478).
- Lorenc, A. (1981). *A global three-dimensional multivariate
  statistical interpolation scheme.* MWR 109(4).
- Courtier, P., Thépaut, J.-N., & Hollingsworth, A. (1994). *A strategy
  for operational implementation of 4D-Var, using an incremental
  approach.* QJRMS 120(519).
- Trémolet, Y. (2006). *Accounting for an imperfect model in 4D-Var.*
  QJRMS 132(621).
- Fablet, R., et al. (2021). *Learning Variational Data Assimilation
  Models and Solvers.* JAMES 13(10).
- Bolte, J., Pauwels, E., & Vaiter, S. (2023). *One-step differentiation
  of iterative algorithms.* NeurIPS.
- Carrassi, A., et al. (2018). *Data assimilation in the geosciences: An
  overview of methods, issues, and perspectives.* WIREs CC 9(5).
- Bannister, R. N. (2017). *A review of operational methods of
  variational and ensemble-variational data assimilation.* QJRMS
  143(703).
- Reference: [CIA-Oceanix/4dvarnet-starter](https://github.com/CIA-Oceanix/4dvarnet-starter).
- Predecessor: [mvardax](https://github.com/jejjohnson/mvardax) (deprecated).
