---
status: draft
version: 0.3.0
---

# vardax Design Doc

**JAX-native variational and amortized inference for data assimilation.**

*Formerly fourdvarjax — renamed to vardax.*

vardax provides the variational inference layer that sits above forward-model
libraries (`somax`, `plumax`) and below orchestration (`pipekit-cycle`). It
implements 4DVarNet, incremental 4DVar, and amortized inference heads on a
shared protocol-driven core that satisfies the pipekit-cycle `ForwardModel` /
`ObservationOperator` / `AnalysisStep` contracts directly.

## Structure

```
vardax/
├── README.md                       # This file
├── vision.md                       # Identity, six-step inference cycle, scope
├── architecture.md                 # Three-layer stack, package layout, pipekit integration
├── boundaries.md                   # Ownership, roadmap (Epics 0–10)
├── decisions.md                    # Design decisions with rationale (D1–D13)
├── pipekit_composition.md          # How vardax satisfies pipekit-cycle protocols
├── posterior.md                    # Posterior export adapter (Laplace / GN / ensemble)
├── api/
│   ├── README.md                   # Surface inventory, data types, conventions
│   ├── primitives.md               # Layer 0 — costs, solvers, CVT, Laplace
│   ├── components.md               # Layer 1 — priors, grad mods, obs operators
│   ├── observation_operators.md    # AveragingKernel, MultiInstrumentFusion, registry
│   └── models.md                   # Layer 2 — VarDANet, IncrementalVarDA, AmortizedVarDA
└── examples/
    ├── README.md                   # Index and reading order
    ├── primitives.md               # Layer 0 — cost & solver patterns
    ├── components.md               # Layer 1 — protocol implementation patterns
    ├── models.md                   # Layer 2 — training workflows
    ├── integration.md              # somax, plumax, gaussx, filterax, pipekit, coordax
    └── use_cases.md                # Methane single-overpass + SSH 4DVarNet walkthroughs
```

## Reading Order

1. **[vision.md](vision.md)** — identity, six-step cycle, what vardax IS and IS NOT
2. **[architecture.md](architecture.md)** — three-layer stack, pipekit integration, package layout
3. **[boundaries.md](boundaries.md)** — ownership map, roadmap (Epics 0–10)
4. **[decisions.md](decisions.md)** — design decisions D1–D13
5. **[pipekit_composition.md](pipekit_composition.md)** — protocol satisfaction pattern
6. **[posterior.md](posterior.md)** — uncertainty quantification + export contract
7. **[api/README.md](api/README.md)** → **primitives.md** → **components.md** → **observation_operators.md** → **models.md**
8. **[examples/](examples/)** — patterns and end-to-end walkthroughs

## What changed in v0.3.0

- **pipekit-cycle is now a required dependency.** vardax classes directly
  satisfy `pipekit_cycle.ForwardModel`, `ObservationOperator`, and
  `AnalysisStep` — no `Abstract*` parallel hierarchy.
- **Averaging kernel + multi-instrument fusion are first-class.** Required
  day-one for any satellite work.
- **Three concrete model families:** `VarDANet*` (learned 4DVarNet),
  `IncrementalVarDA*` (operational 4DVar with control-variable transform),
  `AmortizedVarDA*` (direct observation → posterior heads).
- **Posterior export adapter.** Per-event posteriors export to
  `GaussianMarkLikelihood` for downstream population models.
- **Six-step inference cycle** (physics → MAP → emulator → faster → amortized
  → improve) is the testing and validation scaffold.
- **Forward operators belong to `plumax` / `somax`.** vardax has no built-in
  methane / SWM / QG forwards beyond Lorenz demos.
