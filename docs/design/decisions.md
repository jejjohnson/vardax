---
status: draft
version: 0.3.0
---

# vardax — Design Decisions

This log captures architectural commitments with rationale. Each decision is
referenced by ID throughout the rest of the design docs.

## Index

| ID | Title | Layer |
|---|---|---|
| D1 | Equinox over Flax NNX | Foundation |
| D2 | Protocol-driven extensibility | Layer 1 |
| D3 | Dimensional inheritance over duplication | Layer 2 |
| D4 | Nested module configuration | Layer 1 |
| D5 | Training step as library code, training loop as example | Layer 2 |
| D6 | optimistix for novel solvers | Layer 0 |
| D7 | Demo-quality dynamical priors (forwards live elsewhere) | Boundary |
| D8 | Direct pipekit-cycle protocol satisfaction | Layer 1/2 |
| D9 | Averaging kernel + multi-instrument as first-class | Layer 1 |
| D10 | Posterior export adapter pattern | Layer 1 |
| D11 | Incremental 4DVar with CVT as operational path | Layer 2 |
| D12 | Six-step inference cycle as testing scaffold | Methodology |
| D13 | `pipekit-jax` `JaxModelOp` + `ModelRegistry` for persistence | Layer 1 |

---

## D1: Equinox over Flax NNX

**Decision.** Use `equinox.Module` as the foundation for all components,
replacing Flax NNX `nnx.Module`.

**Why.** Ecosystem consistency (finitevolX, somax, spectraldiffx are all
equinox-based). Direct integration with `lineax`, `optimistix`, and `diffrax`
without adapter layers. Equinox's immutable pytree model is simpler than
Flax NNX's mutable state, and composes cleanly with `pipekit-jax.JaxModelOp`
weight serialisation (Decision D13).

**How to apply.** `nnx.Module` → `eqx.Module`. `nnx.Linear` → `eqx.nn.Linear`.
`nnx.Conv` → `eqx.nn.Conv1d` / `Conv2d`. Training uses `optax` +
`eqx.filter_value_and_grad` instead of `nnx.Optimizer`.

---

## D2: Protocol-driven extensibility

**Decision.** Vardax defines **vardax-specific** runtime-checkable protocols
for `Prior`, `GradModulator`, `CostFunction`, and `PosteriorAdapter`. Where
pipekit-cycle already names the contract (`ForwardModel`,
`ObservationOperator`, `AnalysisStep`), vardax re-exports and satisfies those
directly — no parallel `Abstract*` hierarchy (see Decision D8).

**Why.** DA is inherently modular: cost, prior, obs operator, grad modulator,
posterior adapter are independent choices. Hard-coding any of them limits
reuse. Naming conventions should align with the wider ecosystem so users
don't learn two protocols for the same concept.

**How to apply.** Every new prior / grad mod / posterior adapter satisfies the
relevant vardax `Protocol`. Every observation operator satisfies
`pipekit_cycle.ObservationOperator`. `VarDANet` / `IncrementalVarDA` /
`AmortizedVarDA` accept protocols, not concrete types.

---

## D3: Dimensional inheritance over duplication

**Decision.** Base classes (`VarDANet`, `IncrementalVarDA`, `AmortizedVarDA`)
hold the dimension-agnostic algorithm. `*1D`, `*2D`, `*3D` subclasses set
dimension-specific defaults (conv kernel shape, ConvLSTM layout).

**Why.** The inference algorithm (cost → gradient → update or GN outer / CG
inner) is identical regardless of spatial dimension. Only tensor shapes
differ. Separate classes per dimension led to ~300 lines of duplicated logic
in the v0.1.x codebase.

**How to apply.** Base class implements `__call__`, `_solve_*`,
`.as_analysis_step()`. Subclasses may override defaults (e.g., default grad
modulator for that dimension).

---

## D4: Nested module configuration

**Decision.** Use `SolverConfig(eqx.Module)`, `IncrementalConfig(eqx.Module)`,
`AmortizedConfig(eqx.Module)` for inference parameters rather than flat
constructor arguments.

**Why.** Configuration as a pytree is serialisable (round-trips through
`pipekit-experiment.ModelRegistry`), JIT-friendly, and groups related
settings. It simplifies the model constructor (4 arguments instead of 10+)
and is forward-compatible with swapping the config for an
`optimistix.AbstractMinimiser`.

**How to apply.** Solver-specific settings (`n_steps`, `alpha`,
`prior_weight`, `grad_mode`, `n_outer`, `n_inner`, `cvt`, …) live in the
config object. Model-level components (prior, obs_op, grad_mod, forward)
stay as direct constructor arguments.

---

## D5: Training step as library code, training loop as example

**Decision.** Ship `train_step` and `eval_step` as library functions. `fit()`
moves to example notebooks. Production training composes vardax `train_step`
with `pipekit-train.Loss` / `Callback` / `MetricWriter` protocols.

**Why.** `train_step` encodes the non-obvious part — how to correctly
differentiate through the VarDANet inner solver (especially with implicit
diff or one-step). The outer loop (epochs, logging, checkpointing,
distributed) is always project-specific. Users get the hard parts as library
code; they compose the rest with their preferred tools.

**How to apply.** `from vardax import train_step, eval_step` gives users the
correctness-critical primitive. `pipekit-train.TrainingLoop` wraps it for
production runs (see Epic 7). The `fit()` helper in v0.1.x is moved to an
example notebook.

---

## D6: optimistix for novel solvers

**Decision.** Use existing `optimistix` solvers as-is for standard
minimisation. If vardax develops novel solver strategies (learned gradient
step, hybrid classical-learned warm-start), contribute them upstream as
`optimistix.AbstractMinimiser` subclasses rather than maintaining them
in-tree.

**Why.** Avoids a parallel optimisation library. Novel contributions benefit
the broader JAX community. optimistix already handles implicit
differentiation correctly (Decision D8 protocol satisfaction extends to
optimistix's solver interface).

**How to apply.** Evaluate `optimistix.GradientDescent`, `BFGS`, and
`FixedPointIteration` first. If the learned ConvLSTM step proves useful
beyond `VarDANet`, package it as an optimistix solver.

---

## D7: Demo-quality dynamical priors (forwards live elsewhere)

**Decision.** L63 / L96 priors and simulators are demo utilities under
`vardax._src.utils.dynamical_systems`, **not** library-grade components.
Production dynamical priors come from `somax` (geophysics) or `plumax`
(atmospheric transport / methane).

**Why.** Vardax's job is the **inference** machinery. Maintaining parallel
forward model implementations creates divergence. `somax` and `plumax` are
the authoritative sources.

**How to apply.** L63 / L96 code stays in `utils/` and is imported by
notebooks. It is not part of the core public API. The `Prior` /
`ForwardModel` protocols are what make somax / plumax models work in vardax.

---

## D8: Direct pipekit-cycle protocol satisfaction

**Decision.** Vardax classes **directly satisfy** `pipekit_cycle.ForwardModel`,
`ObservationOperator`, and `AnalysisStep` protocols — without a parallel
`Abstract*` hierarchy or a `vardax.adapters.pipekit` shim module. Where
pipekit-cycle names the contract, vardax uses that name.

**Why.** The user is committed to the pipekit ecosystem for orchestration
(`DACycle`, `SmootherCycle`, `EnsembleDACycle`). An adapter-only pattern
would double the abstraction surface and require users to learn two
hierarchies for the same concept. Direct satisfaction makes vardax a
"first-class citizen" of pipekit's protocol world. `pipekit-cycle` becomes
a required dependency.

**How to apply.**

- `vardax.protocols` re-exports `pipekit_cycle.{ForwardModel,
  ObservationOperator, AnalysisStep}` so users can import from a single
  place. Vardax adds `Prior`, `GradModulator`, `CostFunction`,
  `PosteriorAdapter` for concepts pipekit-cycle doesn't name.
- Every Layer 1 observation operator implements `__call__(state) → obs` and
  `linearize(state) → AbstractLinearOperator` (the `ObservationOperator`
  protocol).
- Every Layer 2 model (`VarDANet*`, `IncrementalVarDA*`, `AmortizedVarDA*`)
  exposes `.as_analysis_step()` returning an `AnalysisStep`-compliant
  callable adapting `(forecast, obs, *, obs_op, obs_err_cov) → analysis`.
- The training interface (`model(batch) → x_recon`) remains unchanged; it
  coexists with the operational `AnalysisStep` interface.
- `tests/test_pipekit_protocols.py` enforces `isinstance(...)` checks on
  every public model and obs operator.

**Trade-off accepted.** `pipekit-cycle` becomes a required dep. Users who
want only the JAX inference code without orchestration still get pipekit in
their environment — pipekit has zero third-party deps, so the cost is
minimal.

---

## D9: Averaging kernel + multi-instrument as first-class

**Decision.** `AveragingKernel(A, x_a, h)` and `MultiInstrumentFusion(registry)`
are part of the day-one `vardax.obs_operators` package — not Epic 6
upgrades, not example-only patterns.

**Why.** Every real satellite inversion (methane, SSH altimetry, soil
moisture, atmospheric chemistry) requires either an averaging kernel
operator (RTM-based L2 products) or multi-instrument fusion (any operational
satellite product). Pushing these to "future work" means vardax can't be
used for its primary use case. Research notes (`methane/1.3_*`,
`methane/2.3_*`, `methane/roadmap/04_rtm_stack.md`) flag this as a hard
day-one requirement.

**How to apply.**

- `AveragingKernel(eqx.Module)` implements `ŷ = A(h·x + (1-h)·x_a)` with
  `A: AbstractLinearOperator` (via gaussx if structured), `x_a: Array`
  (retrieval prior), `h: Array` (weighting). Satisfies
  `pipekit_cycle.ObservationOperator`.
- `MultiInstrumentFusion(registry)` composes per-instrument operators into
  a single H at the **likelihood level**. Fuses without pre-regridding.
- `InstrumentRegistry` carries `(A, x_a, h, mask, R)` per `instrument_id`.
  Per-pixel `instrument` index on `Batch*` selects the operator.
- Per-instrument bias terms are first-class state elements when joint
  inversion is enabled (Epic 9).

---

## D10: Posterior export adapter pattern

**Decision.** Every Layer 2 model emits a `Posterior` object via a
`PosteriorAdapter` (Laplace / Gauss-Newton-Hessian / Ensemble). The
`Posterior` carries mean + covariance + samples + provenance. A
`GaussianMarkLikelihood` serialiser converts `Posterior` to mark-likelihood
form for downstream population models (Tier V TMTPP, hierarchical models).

**Why.** Research notes (`methane/paradox/missing_methane_paradox.md`,
`methane/paradox/mttpp.md`) require that per-event posteriors flow into
population-level models without retraining. A uniform `Posterior` ↔
mark-likelihood adapter keeps the inference layer decoupled from the
population layer; Tier V automatically absorbs improvements to Tier I-IV
forwards.

**How to apply.**

- `vardax.posterior` provides `LaplaceCovariance`, `GaussNewtonHessian`,
  `EnsembleCovariance` — each satisfies `PosteriorAdapter`.
- `Posterior(mean, cov, samples, provenance)` is the standard output
  container. `cov` is an `AbstractLinearOperator` from `gaussx` /
  `lineax` (not necessarily materialised).
- `provenance: dict` carries `{forward_model_id, obs_ops_used, n_iter, J_star,
  converged, gaussx_op_hash, model_hash}`.
- `GaussianMarkLikelihood.to_dict()` serialises to JSON-friendly form for
  storage in catalogs / databases.

---

## D11: Incremental 4DVar with control-variable transform as operational path

**Decision.** Operational 4DVar uses `IncrementalVarDA*` — Gauss-Newton outer
iterations on the full nonlinear cost, CG inner iterations on the
tangent-linear cost, with the control-variable transform $\chi = B^{-1/2}(x - x_b)$
applied via `gaussx.MaternLinearOperator` factorisation by default.

**Why.** Operational DA centres (ECMWF, NCEP, JMA, UKMO) all use incremental
4DVar with CVT as their inner-loop foundation. The transform converts a
Matérn-correlated prior into an identity-Gaussian on $\chi$, which
preconditions the CG solver and makes the inner solve well-conditioned. The
unrolled / one-step / implicit 4DVarNet flavour (`VarDANet*`) is for
research; `IncrementalVarDA*` is for production.

**How to apply.**

- `IncrementalVarDA(forward, obs_op, prior_mean, prior_cov_op, obs_cov_op, config)`
  takes `gaussx` linear operators for $B$ and $R$.
- Tangent-linear model derived via `jax.linearize` on the `ForwardModel`.
- Inner CG solver via `lineax.CG`; outer loop via plain Python `for` (no
  scan — outer iterations are not unrollable due to relinearisation).
- `cvt_transform(x, B_op) → χ` and inverse, exposed in `vardax.cvt`.
- `IncrementalConfig.cvt: bool` toggles the transform (default `True`).
  When `False`, falls back to identity preconditioning.

---

## D12: Six-step inference cycle as testing scaffold

**Decision.** The six-step research-to-operations cycle (physics →
MAP/MCMC → emulator → faster inference → amortized → improve) is the
**validation methodology** for vardax. Step N validates against Step N-1 as
oracle. Hard gates (adjoint calibration, posterior agreement) are part of
the test suite, not just documentation.

**Why.** Research notes (`methane/roadmap/00_prerequisites.md`,
`geotoolz/master_plan/toolz_6_usecases.md`) emphasise that emulators and
amortized predictors are only useful if their output agrees with the
physics-based inference they replace. Without explicit validation gates,
"faster inference" becomes "wrong inference, faster". Vardax codifies the
gates.

**How to apply.**

- Step 2 (model-based) is the oracle for Step 4 (emulator-based). Vardax
  ships `tests/test_six_step_validation.py` with template assertions:
  emulator MAP within $1\sigma_\text{post}$ of physics MAP.
- Step 3 → Step 4 transition requires an adjoint calibration test:
  $\|\partial \text{emulator} / \partial x - \partial \text{physics} / \partial x\|_\text{op} < 5\%$.
- Step 5 (amortized) is validated against Step 2 + Step 4 across a held-out
  set of events.
- `vardax._src.utils.validation` provides `assert_posterior_agreement`,
  `assert_adjoint_calibrated`, `simulation_based_calibration`.
- Notebook tutorials (Epic 10) walk through the cycle end-to-end on Lorenz
  + on a methane case study.

---

## D13: `pipekit-jax` `JaxModelOp` + `ModelRegistry` for persistence

**Decision.** Trained `VarDANet*`, `IncrementalVarDA*`, and `AmortizedVarDA*`
models are persisted via `pipekit-jax.JaxModelOp` (weight serialisation) +
`pipekit-experiment.ModelRegistry` (content-addressed storage). Vardax
provides thin shortcuts (`vardax.persist.save`, `vardax.persist.load`) but
does **not** define its own persistence format.

**Why.** Reusing the pipekit registry means trained vardax models are
discoverable alongside any other pipekit operator (somax forwards, neural
emulators, etc.) in the same registry. Content-addressing avoids name
collisions across projects. `JaxModelOp` handles the split between Python
structure (class skeleton) and weights (serialised bytes) correctly.

**How to apply.**

- Wrap trained model as `pipekit_jax.JaxModelOp(model)` before storing.
- `registry.store(model_op, weights=model_op.serialize_weights())` returns
  a content hash.
- Reload via `template = JaxModelOp(fresh_skeleton); reloaded =
  template.with_weights(registry.load_weights(hash))`.
- `vardax.persist.save(model, registry, tags={...})` is sugar.
- `vardax.persist.load(registry, ref, skeleton_factory)` is sugar.

`pipekit-jax` and `pipekit-experiment` are `[persist]` extras — required
only for users who want to persist models.
