---
status: draft
version: 0.5.0
---

# vardax — Design Decisions

Each decision is referenced by ID throughout the design docs and math
reference. New in v0.5.0: D17 (latent DA peer family). New in v0.4.0:
D14, D15, D16. Revised: D8 (seven peers), D11 (rename), D6 (clarified
upstream adjoint contribution).

## Index

| ID | Title | Layer |
|---|---|---|
| D1 | Equinox over Flax NNX | Foundation |
| D2 | Protocol-driven extensibility | Layer 1 |
| D3 | Dimensional inheritance over duplication | Layer 2 |
| D4 | Nested module configuration | Layer 1 |
| D5 | Training step as library code, training loop as example | Layer 2 |
| D6 | optimistix for novel solvers — contribute upstream | Layer 0 |
| D7 | Demo-quality dynamical priors (forwards live elsewhere) | Boundary |
| D8 | Direct pipekit-cycle protocol satisfaction (seven peer AnalysisStep classes) | Layer 1/2 |
| D9 | Averaging kernel + multi-instrument as first-class | Layer 1 |
| D10 | Posterior export adapter pattern | Layer 1 |
| D11 | `IncrementalFourDVar` as the operational fast path of `StrongFourDVar` | Layer 2 |
| D12 | Six-step inference cycle as testing scaffold | Methodology |
| D13 | `pipekit-jax` `JaxModelOp` + `ModelRegistry` for persistence | Layer 1 |
| D14 | DA hierarchy as horizontal peer classes | Layer 2 |
| D15 | Lean on `optimistix` / `diffrax` adjoints, not in-house grad modes | Layer 0 |
| D16 | BLUE / OI as a first-class method | Layer 2 |
| **D17** | **Latent DA as a first-class peer family** | **Layer 2** |

---

## D1: Equinox over Flax NNX

All components are `eqx.Module`, not `nnx.Module`. The migration enables
direct use of `lineax`, `optimistix`, `diffrax`, and `pipekit-jax`
without adapter layers, and gives a simpler immutable pytree model than
Flax NNX's mutable state.

Apply: `nnx.Module` → `eqx.Module`, `nnx.Linear` → `eqx.nn.Linear`,
`nnx.Optimizer` → `optax` + `eqx.filter_value_and_grad`.

---

## D2: Protocol-driven extensibility

Vardax defines vardax-specific runtime-checkable protocols for `Prior`,
`GradModulator`, `CostFunction`, `PosteriorAdapter`, and `Minimiser`.
Where `pipekit-cycle` already names the contract (`ForwardModel`,
`ObservationOperator`, `AnalysisStep`), vardax re-exports and satisfies
those directly — no parallel `Abstract*` hierarchy (see D8).

---

## D3: Dimensional inheritance over duplication

Each Layer 2 class holds a dimension-agnostic algorithm; `*1D`, `*2D`,
`*3D` subclasses set defaults (conv kernel shape, ConvLSTM layout). The
algorithm is identical regardless of spatial dimension; only tensor
shapes differ.

---

## D4: Nested module configuration

`SolverConfig`, `IncrementalConfig`, `AmortizedConfig` are themselves
`eqx.Module`. Serialisable, JIT-friendly, round-trip through
`pipekit-experiment.ModelRegistry`.

---

## D5: Training step as library code, training loop as example

`train_step` and `eval_step` ship as library functions — they encode the
correctness-critical differentiation through the inner solver. `fit()`
is example-only. Production training composes vardax `train_step` with
`pipekit-train.Loss` / `Callback` / `MetricWriter` protocols.

---

## D6: optimistix for novel solvers — contribute upstream

**Updated in v0.4.0.** Use existing `optimistix.AbstractMinimiser`
subclasses for standard minimisation. The unique contribution from the
4DVarNet line of work — the learned ConvLSTM gradient step, and the
one-step differentiation of Bolte et al. (2023) — should be packaged as
`optimistix.AbstractMinimiser` and `optimistix.AbstractAdjoint`
subclasses respectively, and **contributed upstream** rather than
maintained as parallel implementations in vardax.

Apply: vardax ships `vardax._src.adjoints.one_step.OneStepAdjoint(
optimistix.AbstractAdjoint)` initially, with the goal of upstreaming
once stable. Same for any learned-step minimiser.

---

## D7: Demo-quality dynamical priors (forwards live elsewhere)

L63 / L96 / SWM simulators are demo utilities in
`vardax._src.utils.dynamical_systems`. Production forward models come
from `somax` (geophysics) or `plumax` (atmospheric transport / methane).
Vardax owns inference, not physics.

---

## D8: Direct pipekit-cycle protocol satisfaction (seven peer `AnalysisStep` classes)

**Updated in v0.4.0** with the new class hierarchy.

Vardax classes directly satisfy `pipekit_cycle.{ForwardModel,
ObservationOperator, AnalysisStep}` protocols. **Seven Layer 2 classes
implement `AnalysisStep`** as peers, not a parent–child family:

1. `OptimalInterpolation` (closed-form BLUE / OI)
2. `ThreeDVar`
3. `StrongFourDVar`
4. `WeakFourDVar`
5. `IncrementalFourDVar`
6. `FourDVarNet`
7. `AmortizedPosterior`

Each exposes `.as_analysis_step()` returning a callable matching
`(forecast, obs, *, obs_op, obs_err_cov) → analysis`. The training
interface `model(batch) → x_recon` is preserved on the learned variants
(`FourDVarNet`, `AmortizedPosterior`).

Apply:
- `vardax.protocols` re-exports the three pipekit-cycle protocols.
- Vardax-specific protocols (`Prior`, `GradModulator`, `CostFunction`,
  `PosteriorAdapter`, `Minimiser`) are added for concepts pipekit-cycle
  doesn't name.
- `tests/test_pipekit_protocols.py` enforces `isinstance(...)` checks on
  every public model and obs operator (added as part of Epic 1).

`pipekit-cycle` is a required dependency. It has zero third-party deps,
so the cost is minimal.

---

## D9: Averaging kernel + multi-instrument as first-class

`AveragingKernel(A, x_a, h)` and `MultiInstrumentFusion(registry)` are
part of the day-one `vardax.obs_operators` package. Skipping the
averaging kernel is the most common cause of bias in operational
satellite inversions.

`MultiInstrumentFusion` returns `dict[str, Array]` natively and
satisfies `pipekit_cycle.ObservationOperator` via its
`.to_observation_operator()` adapter (block-diagonal flattening).

---

## D10: Posterior export adapter pattern

Every Layer 2 model emits a `Posterior(mean, cov, samples, provenance)`
via a `PosteriorAdapter`. `GaussianMarkLikelihood` serialises to
JSON-friendly form for downstream population models (Tier V TMTPP,
hierarchical Bayesian inversions).

Adapter selection by inference family:

| Family | Default adapter |
|---|---|
| `OptimalInterpolation` | direct (closed-form posterior cov) |
| `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar` | `LaplaceCovariance` at MAP |
| `IncrementalFourDVar` | `GaussNewtonHessian` (reuses last outer Hessian) |
| `FourDVarNet` | `LaplaceCovariance` |
| `AmortizedPosterior` | direct sampling (`Posterior.samples`) |
| Hybrid `EnVar` (filterax) | `EnsembleCovariance` |

---

## D11: `IncrementalFourDVar` as the operational fast path of `StrongFourDVar`

**Renamed in v0.4.0** (was `IncrementalVarDA`).

`IncrementalFourDVar` is functionally a `StrongFourDVar` with a
specialised inner solver: Gauss-Newton outer iterations on the full
nonlinear cost, CG / Lanczos inner iterations on the linearised
quadratic subproblem, control-variable transform via
`gaussx.MaternLinearOperator.half()` for preconditioning.

For users:
- Use `StrongFourDVar` with `minimiser=optimistix.NonlinearCG(...)` for
  general 4DVar problems.
- Use `IncrementalFourDVar` when you want the operational pattern — fewer
  outer iterations, exact tangent-linear via `jax.linearize`, and CG
  inner with `gaussx`-preconditioned Hessian.

The two share `ForwardModel`, `ObservationOperator`, and `Posterior`
contracts. The choice between them is performance-driven, not
semantic — both solve the same minimisation problem.

---

## D12: Six-step inference cycle as testing scaffold

The cycle (physics → MAP / MCMC → emulator → faster inference →
amortized → improve) is the **validation methodology**. Step N validates
against Step N–1 as oracle. Hard gates (adjoint calibration, posterior
agreement, simulation-based calibration) are part of
`tests/test_six_step_validation.py`, not just documentation.

`vardax._src.utils.validation` exposes `assert_posterior_agreement`,
`assert_adjoint_calibrated`, `simulation_based_calibration`.

---

## D13: `pipekit-jax` `JaxModelOp` + `ModelRegistry` for persistence

Trained models are persisted via `pipekit-jax.JaxModelOp` (weight
serialisation) + `pipekit-experiment.ModelRegistry` (content-addressed
storage). Vardax provides thin shortcuts (`vardax.persist.save`,
`vardax.persist.load`) but does not define its own persistence format.

`pipekit-jax` and `pipekit-experiment` are `[persist]` extras.

---

## D14: DA hierarchy as horizontal peer classes

**New in v0.4.0.**

The previous v0.3 design organised methods into three families —
`VarDANet*` (learned), `IncrementalVarDA*` (operational), `AmortizedVarDA*`
(direct). This was 4DVarNet-centric: it treated the learned method as
the canonical case and the classical methods as historical variants.

The v0.4 design treats **classical and learned DA methods as siblings**.
Seven peer classes, all implementing `pipekit_cycle.AnalysisStep`, none
inheriting from any of the others:

| Method | Use when |
|---|---|
| `OptimalInterpolation` | Linear $H$, Gaussian $B$, $R$. Static field. The right default. |
| `ThreeDVar` | Nonlinear $H$, single time. Snapshot inversion. |
| `StrongFourDVar` | Multi-time, control = $x_0$, model treated as exact. |
| `WeakFourDVar` | Multi-time, control = $(x_0, \eta_1, \ldots, \eta_T)$. Model-error-aware. |
| `IncrementalFourDVar` | Operational fast path: GN outer + CG inner + CVT (= `StrongFourDVar`). |
| `FourDVarNet` | Learned prior + learned grad modulator. Research, data-rich regimes. |
| `AmortizedPosterior` | Direct $q_\phi(x \mid y)$. Real-time / many-event regimes. |

**Why horizontal.**

- BLUE / OI is the right tool when the regime allows it. Forcing it
  through `ThreeDVar` with an "isLinear" branch hides the closed-form
  fast path and confuses the reader about complexity.
- Strong- and weak-constraint 4DVar are different problems (different
  control vectors, different cost terms), not configurations of a
  single class. Burying them under a `mode: Literal["strong", "weak"]`
  flag obscures the structural difference.
- `FourDVarNet` is one variant of 4DVar with learned components — but it
  is **not the parent** of classical 4DVar. Treating it as such was a
  category error inherited from the 4DVarNet-starter codebase.

**Apply.**

- All seven classes live as siblings in `vardax/_src/models/`.
- All seven implement `.as_analysis_step()` returning the same protocol
  shape.
- Naming convention: spelled-out names (`StrongFourDVar`, not
  `VarDA4DStrong` or `FourDVarStrong`). No `VarDA` prefix on the class
  names — `vardax` is the package, the classes are named for the
  method.
- Math reference chapters 4–10 cover one method each, with chapter 1
  (problem setting), chapter 2 (observation model), and chapter 3
  (dynamical model) supplying the shared foundation.

---

## D15: Lean on `optimistix` / `diffrax` adjoints, not in-house grad modes

**New in v0.4.0.**

The v0.3 design defined `grad_mode: Literal["unrolled", "one_step",
"implicit"]` on `FourDVarNet*` and hand-rolled the three corresponding
differentiation paths in `vardax._src.solver`. This was useful as a
proof of concept but is the wrong long-term design:

- `optimistix.AbstractAdjoint` already provides
  `RecursiveCheckpointAdjoint`, `ImplicitAdjoint`, `DirectAdjoint` for
  gradients through minimisers.
- `diffrax.AbstractAdjoint` already provides
  `RecursiveCheckpointAdjoint`, `BacksolveAdjoint` (continuous adjoint),
  `ForwardMode`, `DirectAdjoint` for gradients through ODE integration.

Vardax exposes both as constructor slots and delegates the actual
gradient computation upstream:

```python
class StrongFourDVar(eqx.Module):
    ...
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()
```

A user who wants the operational memory profile:

```python
minimiser_adjoint = optimistix.ImplicitAdjoint()
forward_adjoint = diffrax.BacksolveAdjoint()
```

A user training `FourDVarNet`:

```python
minimiser_adjoint = optimistix.RecursiveCheckpointAdjoint()
# or — once contributed upstream:
minimiser_adjoint = vardax.adjoints.OneStepAdjoint()   # Bolte et al. 2023
```

**Apply.**

- Drop `GradMode` enum from the public API.
- Drop `unrolled_solve`, `one_step_solve`, `implicit_solve` as separate
  Layer 0 primitives. Their logic moves into the appropriate
  `optimistix.AbstractAdjoint` subclasses.
- Bolte 2023 "one-step" differentiation is implemented as
  `vardax._src.adjoints.one_step.OneStepAdjoint(optimistix.AbstractAdjoint)`
  with the goal of upstreaming (continues D6).
- The continuous adjoint via `diffrax.BacksolveAdjoint` becomes the
  recommended default for memory-constrained 4DVar with long
  assimilation windows.

This decision composes with D6: vardax's novel solver / adjoint
contributions become upstream `optimistix` / `diffrax` subclasses, not
parallel in-tree code.

---

## D16: BLUE / OI as a first-class method

**New in v0.4.0.**

Optimal interpolation (BLUE) — the closed-form linear-Gaussian analysis
— is the **first** method vardax users should reach for when the regime
permits. It is not a degenerate case of 3DVar to be invoked via an
auto-detected branch; it is its own `OptimalInterpolation` class with
its own fast path.

The analysis is

$$x^* = x_b + B H^\top (H B H^\top + R)^{-1} (y - H x_b),$$

with posterior covariance

$$P^* = B - B H^\top (H B H^\top + R)^{-1} H B = (B^{-1} + H^\top R^{-1} H)^{-1}.$$

When $B$, $R$, $H$ are structured (Matérn, Kronecker, AveragingKernel),
`gaussx` solves these expressions efficiently without materialising
dense matrices. The whole analysis runs in a handful of structured
mat-vec operations.

**Why first-class.**

- The closed form is exact; no iteration, no convergence criterion, no
  adjoint. When applicable it is faster *and* more accurate than
  3DVar's iterative solution.
- The posterior covariance comes for free in the same expression — no
  separate `PosteriorAdapter` call needed.
- It is the canonical baseline. Every more sophisticated method must
  agree with OI in the linear-Gaussian limit; if it doesn't, something
  is wrong. Making OI a first-class method makes this baseline easy to
  produce.
- The Kalman filter analysis step *is* OI. Code reuse between vardax
  and ensemble libraries (filterax) is direct.

**Apply.**

- `vardax/_src/models/optimal_interpolation.py` implements
  `OptimalInterpolation(eqx.Module)` with no `minimiser` slot.
- `linearize()` is required on the observation operator. If $H$ is
  intrinsically nonlinear, the user picks `ThreeDVar` instead;
  `OptimalInterpolation.__init__` validates linearity and refuses
  nonlinear `obs_op`.
- A new Layer 0 primitive `blue_analysis(x_b, y, B_op, R_op, H_op)`
  implements the closed-form math via `gaussx`.
- Math chapter 4 covers BLUE / OI in detail, including the Sherman–
  Morrison–Woodbury identity used to pick between the $B$-space and
  the $R$-space form.

---

## D17: Latent DA as a first-class peer family

**New in v0.5.0.**

Latent data assimilation — performing the variational analysis in a
learned low-dimensional latent space $\mathcal{Z}$ — is added as a
*peer* family of three new Layer-2 classes alongside the seven
established in D14:

* `LatentThreeDVar` — single-time inversion in $\mathcal{Z}$.
* `LatentStrongFourDVar` — multi-time, latent dynamics $M_z$.
* `LatentHybridFourDVar` — multi-time, physics forecast in $\mathcal{X}$,
  control and background in $\mathcal{Z}$.

**Context.** Vardax already uses learned subspaces in three places —
the AE priors inside `FourDVarNet*`, the observation encoder inside
`AmortizedPosterior`, and the heads of the amortised family. None of
these promote $z$ itself to the control variable; the variational
problem still solves in $\mathcal{X}$. The benchmark literature
(Peyron 2021, Cheng 2023, Fablet 2021) consistently shows that
promoting $z$ to control gives an order-of-magnitude wall-clock win
and a structurally smaller posterior covariance.

**Options.**

(A) Add `space: Literal["x", "z"]` to existing methods and dispatch
    internally.
(B) Add three new peer classes; reuse the AE priors as `LatentMap`s.
(C) Treat latent DA as a configuration of `FourDVarNet`.

**Decision.** Option B.

**Rationale.**

* The control vector ($z$ vs $x$), the cost dimensionality, the Hessian
  object, and the posterior covariance all live in genuinely different
  spaces. The D14 pattern argues we should not hide a structural
  difference behind a flag.
* The AE priors of `FourDVarNet` are *regularisers*, not
  *parameterisations*. Conflating regularisation and parameterisation
  (Option C) confuses the reader and forces users to thread the
  `prior` slot to mean two different things.
* Three peer classes mirror the three flavours formalised in the
  `pipekit_cycle.LatentDACycle` orchestrator: strong (z forecast +
  z update), prior-only (x forecast + x update, AE inside cost),
  hybrid (x forecast + z update). The vardax peers map 1-to-1 to those
  flavours.

**Apply.**

* New module `vardax/_src/models/latent.py` with the three classes.
* New module `vardax/_src/latent.py` with `LatentPrior` and
  `NeuralLatentForwardModel` adapter.
* Existing `BilinAEPrior1D/2D`, `MLPAEPrior1D`, `ConvAEPrior1D` gain
  `latent_dim` and `state_signature` properties so they satisfy
  `pipekit_cycle.LatentMap` structurally. No behaviour change.
* New Layer-0 primitives `variational_cost_latent`,
  `latent_incremental_cost`, plus an `identity_latent_map(N)` helper
  for the regression test in §18.6 of the math reference.
* New posterior adapter `LatentLaplaceCovariance` in
  `vardax/_src/posterior/latent.py`.
* Math chapter 18 covers the formalism; design doc
  `design/latent_da.md` covers the API.

**Consequences.**

* Three more peers in the Layer-2 family (now ten). D14's "no
  parent–child" rule applies: latent peers do not inherit from
  `StrongFourDVar` or each other.
* `LatentMap` is a substrate-neutral pipekit-cycle protocol, not a
  vardax-owned one. Cross-library compatibility with `filterax`
  (`LatentETKF`, `LatentLETKF`) is automatic.
* `WeakLatentFourDVar` and a VAE-aware `LatentMap` are explicitly
  deferred to v0.6 (see design/latent_da.md §13).
