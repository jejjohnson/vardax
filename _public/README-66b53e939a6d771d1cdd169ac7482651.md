---
status: draft
version: 0.4.0
---

# vardax — Examples

Usage patterns organised by API layer plus end-to-end walkthroughs.

## Structure

```
examples/
├── README.md          # This file
├── primitives.md      # Layer 0 — cost, CVT, BLUE, adjoint wiring
├── components.md      # Layer 1 — protocol implementation patterns
├── models.md          # Layer 2 — workflows for each of the seven peer classes
├── integration.md     # Ecosystem composition (somax, plumax, gaussx, filterax, pipekit)
└── use_cases.md       # Walkthroughs: methane single-overpass + SSH 4DVarNet
```

## Reading order

1. **[primitives.md](primitives.md)** — Layer 0: cost terms, closed-form
   BLUE, CVT, adjoint composition via `optimistix` / `diffrax`.
2. **[components.md](components.md)** — Layer 1: implementing `Prior`,
   `ObservationOperator`, `GradModulator`, `PosteriorAdapter`,
   `Minimiser`.
3. **[models.md](models.md)** — Layer 2: workflows for each of the seven
   peer analysis classes (`OptimalInterpolation` through
   `AmortizedPosterior`).
4. **[integration.md](integration.md)** — composing vardax with somax,
   plumax, gaussx, filterax, pipekit-cycle, geostack catalogs.
5. **[use_cases.md](use_cases.md)** — full walkthroughs anchoring the
   abstractions on real workflows.
