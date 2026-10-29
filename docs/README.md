# vardax — Documentation

> **Status — v0.2.0.** All seven analysis methods ship, and each satisfies `pipekit_cycle.AnalysisStep` through `.as_analysis_step()`. Import them from the package root (`import vardax as vdx; vdx.StrongFourDVar`); there is no `vardax.models` or `vardax.obs_operators` module. The `vardax.adjoints`, `vardax.amortized` and `vardax.cycle` submodules are bound too, and a few symbols live only there. Each code block in the chapters carries a status marker: `# today` for names that ship, `# proposed` for ones that do not yet. *(The package was previously published as `fourdvarjax` v0.1.x; `vardax` is now the canonical name.)*

This directory contains documentation for the vardax library, in two
layers:

- **Mathematical reference** (this directory, 21 chapters) — the
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

### End-to-end examples (chapters 15–18)

15. [Lorenz Examples](15_lorenz_examples.md) — L63 / L96 walkthroughs
    with all seven methods
16. [SSH Reconstruction](16_ssh_example.md) — OceanBench-style example
    with `OI`, `IncrementalFourDVar`, `FourDVarNet` side-by-side
17. [Methane Single-Overpass](17_methane_example.md) — multi-instrument
    inversion with `plumax` + `IncrementalFourDVar`
18. [Latent Variational DA](18_latent_da.md) — assimilation in a
    learned latent space (Decision D17)

### Applications & foundations (chapters 19–21, ported from mfourdvar)

19. [Physical Models & ODE Priors](19_physical_models.md) — the
    testbed-model ladder (L63/L96 → SW/QG → GCM) and the
    `DynamicalPrior` family (Decision D18)
20. [Uncertainty Quantification](20_uncertainty.md) — where
    stochasticity enters (data, model, parameters, estimation) and
    which vardax lever addresses each source
21. [OceanBench SSH Applications](21_oceanbench.md) — the benchmark
    context: data products, OSSE/OSE ladder, interpolation /
    forecasting / surrogate tasks

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
  D1–D19
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

## Documentation build

The site is built by two tools and deployed as one, the same split radtrax, plumax and kernellib use.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose: home, maths reference, tutorials, design docs | [mystmd](https://mystmd.org) | `docs/myst.yml`, `docs/*.md`, `docs/notebooks/`, `docs/design/` | `/` |
| API reference | MkDocs + mkdocstrings | `mkdocs.yml`, `docs/api/` | `/reference/` |

`make docs` runs `scripts/build_docs.py`, which builds the API reference, converts and executes each jupytext tutorial into a gitignored `.ipynb` next to its `.py` source, builds the MyST site, assembles both into `public/` and checks every internal link across the two. `make docs-fast` skips executing the tutorials, and `make docs-serve` serves the result at http://127.0.0.1:8000. The MyST half needs the `myst` CLI (`npm install -g mystmd`).

MyST pages link into the API with the `xref:` protocol and the top-level name, for example ``[`StrongFourDVar`](xref:api#vardax.StrongFourDVar)``; a target missing from the inventory fails the build. API pages link to the prose with absolute URLs under https://jejjohnson.github.io/vardax/. mystmd drops a file's leading number from its URL, so `05_threedvar.md` is served at `/threedvar/` and `09_4dvarnet.md` at `/dvarnet/`.
