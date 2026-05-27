---
status: draft
version: 0.4.0
---

# vardax Design Doc

**JAX-native data assimilation inference.**

*Formerly fourdvarjax — renamed to vardax.*

vardax provides the seven classical and modern DA analysis methods as
peer `pipekit_cycle.AnalysisStep` implementations:

| Class | Method |
|---|---|
| `OptimalInterpolation` | BLUE / OI — closed-form linear-Gaussian |
| `ThreeDVar` | 3D variational |
| `StrongFourDVar` | Strong-constraint 4DVar |
| `WeakFourDVar` | Weak-constraint 4DVar |
| `IncrementalFourDVar` | Operational fast path of strong-4DVar |
| `FourDVarNet` | Learned 4DVar (prior + grad modulator) |
| `AmortizedPosterior` | Direct $q_\phi(x \mid y)$ head |

Gradients through dynamics and the inner minimiser are composed via
`diffrax.AbstractAdjoint` and `optimistix.AbstractAdjoint` — vardax owns
no in-house grad-mode enum.

## Structure

```
design/
├── README.md                       # This file
├── vision.md                       # Identity, DA hierarchy, six-step cycle
├── architecture.md                 # Three-layer stack, class sketches, package layout
├── boundaries.md                   # Ownership map, Epics 0–13 roadmap
├── decisions.md                    # Design decisions D1–D16
├── pipekit_composition.md          # Protocol satisfaction patterns
├── posterior.md                    # Posterior adapter family
├── api/
│   ├── README.md                   # Surface inventory, conventions
│   ├── primitives.md               # Layer 0 — costs, CVT, Laplace, adjoint wiring
│   ├── components.md               # Layer 1 — priors, grad mods, costs, minimisers
│   ├── observation_operators.md    # AveragingKernel, MultiInstrumentFusion, registry
│   └── models.md                   # Layer 2 — the seven peer classes
└── examples/
    ├── README.md                   # Reading order
    ├── primitives.md               # L0 cost & solver patterns
    ├── components.md               # L1 protocol implementation patterns
    ├── models.md                   # L2 training workflows by method
    ├── integration.md              # somax, plumax, gaussx, filterax, pipekit, coordax
    └── use_cases.md                # Methane single-overpass + SSH walkthroughs
```

## Reading order

1. **[vision.md](vision.md)** — what vardax is, the seven peer methods
2. **[architecture.md](architecture.md)** — three-layer stack, class
   sketches, package layout
3. **[boundaries.md](boundaries.md)** — ownership map, roadmap
4. **[decisions.md](decisions.md)** — design decisions D1–D16
5. **[pipekit_composition.md](pipekit_composition.md)** — protocol
   satisfaction patterns
6. **[posterior.md](posterior.md)** — UQ + export contract
7. **[api/](api/)** — surface inventory and contracts
8. **[examples/](examples/)** — patterns and use case walkthroughs

## What changed in v0.4.0

- **Seven peer `AnalysisStep` classes** (Decision D14). The v0.3 three-family
  grouping (`VarDANet*`, `IncrementalVarDA*`, `AmortizedVarDA*`) is
  replaced by horizontal sibling classes. Classical DA methods (BLUE / OI,
  3DVar, strong / weak 4DVar) get their own first-class implementations
  rather than being treated as legacy.
- **`OptimalInterpolation` as a first-class method** (Decision D16). BLUE
  / OI is the right tool when the regime allows it; it gets a dedicated
  class with a closed-form fast path rather than being folded into 3DVar.
- **Adjoints via `optimistix` / `diffrax`** (Decision D15). The
  `grad_mode` enum is dropped. Models carry `forward_adjoint:
  diffrax.AbstractAdjoint` and `minimiser_adjoint:
  optimistix.AbstractAdjoint` constructor slots; vardax delegates
  gradient computation to upstream libraries. The Bolte 2023 one-step
  method becomes a custom `optimistix.AbstractAdjoint` (targeting
  upstream contribution).
- **Math reference rewritten** to DA-textbook style (Epic 10). 17
  chapters: shared foundation (1–3), seven methods as peer chapters
  (4–10), then observation operators (11), incremental + posterior (12–13),
  six-step cycle (14), and concrete examples (15–17).
- **`IncrementalFourDVar` is a peer, not a subclass.** It is functionally
  equivalent to `StrongFourDVar` but uses a specialised inner solver
  (GN outer + CG inner + CVT). The choice between them is
  performance-driven, not semantic.

## What carried over from v0.3.0

- pipekit-cycle integration (`ForwardModel`, `ObservationOperator`,
  `AnalysisStep` satisfied directly)
- Averaging kernel + multi-instrument fusion as first-class operators
- Posterior export adapter pattern (`Posterior` container,
  `GaussianMarkLikelihood` serialiser)
- Six-step inference cycle as testing scaffold
- `pipekit-jax.JaxModelOp` + `pipekit-experiment.ModelRegistry` for
  persistence
- Boundaries: plumax owns methane forwards, filterax owns ensembles,
  gaussx owns structured linear algebra
