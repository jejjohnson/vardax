# vardax — Documentation

> **Status — v0.4.0 design reference (forward-looking).** These docs
> describe the target API after the equinox migration roadmap (Epics
> 0–13, see [`design/boundaries.md`](design/boundaries.md)). The
> shipped package implements 4DVarNet only (single learned method,
> built on Flax NNX); the seven-method DA hierarchy plus
> pipekit-cycle protocol satisfaction is the design target.
> References to `vardax.models.*`, `vardax.obs_operators.*`,
> `vardax.adjoints.*`, the `pipekit_cycle` protocols,
> `tests/test_pipekit_protocols.py`, and `vardax._src.utils.validation`
> describe the design target — they are not yet present in the current
> package. *(The package was previously published as `fourdvarjax`
> v0.1.x; `vardax` is now the canonical name.)*

This directory contains documentation for the vardax library, in two
layers:

- **Mathematical reference** (this directory, 17 chapters) — the
  equations, algorithms, and pseudocode behind every analysis method.
- **Design docs** ([`design/`](design/)) — architecture, API
  contracts, ecosystem boundaries, decision log.

## Mathematical reference

### Foundation (chapters 1–3)

1. [Problem Setting](01_problem_setting.md) — Bayesian state
   estimation, the seven analysis methods
2. [Observation Model](02_observation_model.md) — likelihood, $H$,
   $R$, the `ObservationOperator` protocol
3. [Dynamical Model](03_dynamical_model.md) — $M_t$, the
   `ForwardModel` protocol, `diffrax` adjoint composition

### The seven analysis methods (chapters 4–10)

4. [Optimal Interpolation / BLUE](04_oi_blue.md) — closed-form
   linear-Gaussian
5. [3DVar](05_threedvar.md) — nonlinear $H$, single time
6. [Strong-constraint 4DVar](06_strong_4dvar.md) — control = $x_0$
7. [Weak-constraint 4DVar](07_weak_4dvar.md) — control = $(x_0, \boldsymbol{\eta})$
8. [Incremental 4DVar with CVT](08_incremental_4dvar.md) — operational
   fast path
9. [4DVarNet — Learned 4DVar](09_4dvarnet.md) — learned prior +
   learned grad modulator
10. [Amortized Inference](10_amortized_inference.md) — direct
    $q_\phi(x \mid y)$ heads

### Cross-cutting (chapters 11–14)

11. [Observation Operators](11_observation_operators.md) — masked,
    linear, averaging kernel, multi-instrument fusion
12. [Adjoint Methods](12_adjoint_methods.md) — `diffrax` + `optimistix`
    adjoint composition, Bolte 2023 one-step
13. [Posterior Covariance](13_posterior_covariance.md) — closed form,
    Laplace, GN-Hessian, ensemble; `GaussianMarkLikelihood` export
14. [Six-Step Inference Cycle](14_six_step_cycle.md) — methodology for
    the research-to-operations arc

### End-to-end examples (chapters 15–17)

15. [Lorenz Examples](15_lorenz_examples.md) — L63 / L96 walkthroughs
    with all seven methods
16. [SSH Reconstruction](16_ssh_example.md) — OceanBench-style example
    with `OI`, `IncrementalFourDVar`, `FourDVarNet` side-by-side
17. [Methane Single-Overpass](17_methane_example.md) — multi-instrument
    inversion with `plumax` + `IncrementalFourDVar`

### Reference

- [Notation](notation.md)
- [References](references.md)

## Design docs

See [`design/README.md`](design/README.md) for the full table of
contents and reading order. Recommended entry points:

- [`design/vision.md`](design/vision.md) — identity, DA hierarchy,
  six-step cycle
- [`design/architecture.md`](design/architecture.md) — three-layer
  stack, package layout, pipekit integration
- [`design/boundaries.md`](design/boundaries.md) — ownership map,
  Epics 0–13 roadmap
- [`design/decisions.md`](design/decisions.md) — design decisions
  D1–D16
- [`design/api/`](design/api/) — protocol + class contracts by layer
- [`design/examples/`](design/examples/) — patterns and use case
  walkthroughs

## What changed in v0.4.0

- **DA hierarchy as horizontal peer classes** (Decision D14). Seven
  Layer 2 classes — `OptimalInterpolation`, `ThreeDVar`,
  `StrongFourDVar`, `WeakFourDVar`, `IncrementalFourDVar`,
  `FourDVarNet`, `AmortizedPosterior` — all implementing
  `pipekit_cycle.AnalysisStep`. No parent–child relationships.
- **BLUE / OI as a first-class method** (Decision D16). Closed-form
  linear-Gaussian analysis is the first method to reach for when the
  regime allows it — not folded into `ThreeDVar`.
- **Adjoints via `optimistix` / `diffrax`** (Decision D15). The
  `grad_mode` enum is gone; models carry `forward_adjoint:
  diffrax.AbstractAdjoint` and `minimiser_adjoint:
  optimistix.AbstractAdjoint` constructor slots. Bolte 2023 one-step
  becomes `vardax.adjoints.OneStepAdjoint(optimistix.AbstractAdjoint)`
  targeting upstream contribution.
- **Math reference rewritten** to DA-textbook style. 17 chapters
  organised around the DA hierarchy rather than a 4DVarNet narrative.
