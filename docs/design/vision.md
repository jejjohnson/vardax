---
status: draft
version: 0.4.0
---

# vardax — Vision

## What vardax is

**vardax is the data assimilation inference layer for JAX.** It provides
the seven classical and modern analysis methods that turn observations
into state estimates:

1. **`OptimalInterpolation`** — BLUE / OI, the closed-form linear-Gaussian
   analysis. The right tool when $H$ is linear and $B$, $R$ are Gaussian.
2. **`ThreeDVar`** — 3D variational analysis. Nonlinear observation
   operator, snapshot in time.
3. **`StrongFourDVar`** — strong-constraint 4DVar. Control variable is
   the initial state $x_0$; dynamics $M_t$ are treated as exact.
4. **`WeakFourDVar`** — weak-constraint 4DVar. Augmented control vector
   $(x_0, \eta_1, \ldots, \eta_T)$ admits model error.
5. **`IncrementalFourDVar`** — the operational fast path of strong-constraint
   4DVar. Gauss-Newton outer iterations on the full nonlinear cost, CG
   inner iterations on the linearised quadratic subproblem, control-variable
   transform for preconditioning.
6. **`FourDVarNet`** — the learned variant of 4DVar. Prior $\varphi_\theta$
   replaces the Gaussian $B$; the inner solver becomes a learned gradient
   modulator $\Phi_\phi$.
7. **`AmortizedPosterior`** — the direct head $q_\phi(x \mid y)$. Real-time
   inference via conditional flow, score-based diffusion, or regression.

All seven are siblings, not parent–child relationships. They satisfy the
same `pipekit_cycle.AnalysisStep` protocol and compose with the same
observation operators, forward models, priors, and posterior adapters.
The DA hierarchy is **horizontal**; you pick the method that matches the
regime, not a family.

## The single equation

Every analysis method in vardax is a special case of

$$x^* = \underset{x,\,\boldsymbol{\eta}}{\arg\min}\;
  \underbrace{\tfrac{1}{2}\|x - x_b\|^2_{B^{-1}}}_{\text{background term}}
  + \underbrace{\tfrac{1}{2}\sum_{t=0}^{T} \|y_t - H_t(M_t(x; \boldsymbol{\eta}))\|^2_{R_t^{-1}}}_{\text{observation term}}
  \;[\,+\;\underbrace{\tfrac{1}{2}\sum_{t=1}^{T} \|\eta_t\|^2_{Q_t^{-1}}}_{\text{model-error term}}\,].$$

| Method | $T$ | Prior | Model error $\boldsymbol{\eta}$ | Inner solver |
|---|---|---|---|---|
| `OptimalInterpolation` | 0 | Gaussian $B$, linear $H$ | — | closed-form |
| `ThreeDVar` | 0 | Gaussian $B$, nonlinear $H$ | — | iterative (optimistix) |
| `StrongFourDVar` | $>0$ | Gaussian $B$ | absent | iterative (optimistix) |
| `WeakFourDVar` | $>0$ | Gaussian $B$, $Q$ | active | iterative (optimistix) |
| `IncrementalFourDVar` | $>0$ | Gaussian $B$ | absent | GN outer + CG inner |
| `FourDVarNet` | $>0$ | learned $\varphi_\theta$ | absent | learned $\Phi_\phi$ |
| `AmortizedPosterior` | any | implicit in $q_\phi$ | implicit | none (single forward pass) |

Each method specialises the unified cost differently, but the surface
that user code touches — `model.as_analysis_step()` — is identical.

## Composable gradients

vardax does not re-implement automatic differentiation. Gradients flow
through two upstream libraries:

- **`diffrax`** — gradients through ODE integration via
  `RecursiveCheckpointAdjoint`, `BacksolveAdjoint` (continuous adjoint),
  `ForwardMode` (forward sensitivity), or `DirectAdjoint`. This is how
  4DVar's adjoint of $M_t$ is computed.
- **`optimistix`** — gradients through the inner minimisation via
  `RecursiveCheckpointAdjoint`, `ImplicitAdjoint` (IFT-based), or
  `DirectAdjoint`. This is how training gradients propagate through
  `FourDVarNet`'s solver.

Both are exposed as constructor slots on the Layer 2 classes:

```python
model = StrongFourDVar(
    forward=somax_model,
    obs_op=AveragingKernel(...),
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=optimistix.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optimistix.ImplicitAdjoint(),
    forward_adjoint=diffrax.BacksolveAdjoint(),
)
```

A user who wants memory-efficient operational 4DVar reaches for
`diffrax.BacksolveAdjoint()`. A user training `FourDVarNet` reaches for
`optimistix.RecursiveCheckpointAdjoint()` or a custom one-step adjoint.
Vardax owns the DA algorithm, not the differentiation strategy.

## Three-layer architecture

```
Layer 2  Models (each satisfies pipekit_cycle.AnalysisStep)
         OptimalInterpolation, ThreeDVar, StrongFourDVar, WeakFourDVar,
         IncrementalFourDVar, FourDVarNet, AmortizedPosterior
            ↑
Layer 1  Components (eqx.Module operators)
         priors, observation operators, gradient modulators (4DVarNet only),
         cost functions, posterior adapters, minimiser adapters
            ↑
Layer 0  Primitives (pure JAX)
         cost terms, control-variable transform, Laplace covariance,
         adjoint wiring (diffrax + optimistix passthrough)
```

Users enter at the level appropriate to their task:

- **Layer 0** — integrating vardax algorithms into another framework
- **Layer 1** — building custom DA pipelines (new prior + AK obs op)
- **Layer 2** — operational analysis or training

## What vardax is NOT

(See [`boundaries.md`](boundaries.md) for the full ownership map.)

- It does not define forward models. Use `somax` (geophysics) or
  `plumax` (atmospheric transport / methane). Lorenz-63 / Lorenz-96 demos
  in `vardax._src.utils.dynamical_systems` are toy examples.
- It does not own ensemble methods. Use `filterax`. vardax exposes hooks
  (`EnsembleCovariance` posterior adapter, ensemble batch dimension) but
  EnKF / EnKS / EnKI propagation lives elsewhere.
- It does not own structured linear algebra. Use `gaussx` for Matérn
  factorisations, Kronecker / LowRank / BlockDiag operators.
- It does not own data I/O. Use `georeader` (sensors), `coordax`
  (labelled arrays). vardax consumes `Batch*` containers; how they're
  populated is upstream.
- It does not own optimisers or ODE solvers. Use `optimistix` and
  `diffrax`. vardax composes them via the adjoint slots described above.
- It does not own experiment orchestration. Use `pipekit-cycle` for DA
  cycles, `pipekit-experiment` for run tracking. vardax satisfies the
  protocols.

## The six-step research-to-operations cycle

vardax is engineered around the cycle that turns a new forward model into
operational analysis:

```
(1) Physics forward (somax / plumax)
      → (2) Classical inference: OI / 3DVar / 4DVar               — slow, exact
      → (3) Neural emulator of the forward (trained from Step 1)   — fast surrogate
      → (4) Emulator-based inference: same vardax loop             — 100–1000× faster
      → (5) Amortized predictor: y → posterior directly            — sub-second
      → (6) Improve: swap any block; prior step is the oracle
```

Crucially: **Step 2 uses the same vardax code as Step 4**. The forward
model is swapped via the `pipekit_cycle.ForwardModel` protocol; the
analysis class doesn't know whether $M_t$ is physics or an emulator. This
is what makes the cycle a cycle and not a rewrite.

The validation gates between steps (adjoint calibration, posterior
agreement, simulation-based calibration) are part of the test suite, not
just documentation. See chapter 14 in the math reference.

## Why a JAX-native DA library

The DA community has excellent Fortran/C++ implementations of incremental
4DVar (IFS, GSI, ICON, WRFDA), excellent Python ensemble libraries
(DAPPER, PyOSSE), and excellent research code for 4DVarNet
(`4dvarnet-starter`). What's missing is a single library where:

- **All classical methods are first-class.** BLUE / OI is not a footnote;
  it's the first thing you should reach for when the regime is
  linear-Gaussian. Today this means dropping into NumPy / SciPy or
  rolling your own.
- **The same code transitions research → operations.** A `ThreeDVar`
  trained in a notebook becomes the analysis step in a `pipekit-cycle`
  pipeline without changes to the inversion logic.
- **Gradients are uniform.** `diffrax` adjoints handle the dynamics,
  `optimistix` adjoints handle the inner solver, vardax just composes
  them. No per-method bespoke differentiation.
- **Learned methods coexist with classical ones.** `FourDVarNet` and
  `AmortizedPosterior` are siblings of `OptimalInterpolation`, not a
  replacement. They use the same priors, observation operators, and
  posterior adapters.

This is the gap vardax fills.

## Framework stack

| Package | Role | Required? |
|---|---|---|
| `equinox` | Module system | yes |
| `optax` | Outer-loop training optimiser | yes |
| `optimistix` | Inner-loop minimisers + adjoints | yes |
| `diffrax` | ODE integration + adjoints | yes |
| `lineax` | Linear solvers, `AbstractLinearOperator` | yes |
| `gaussx` | Structured operators (Matérn, Kronecker, LowRank) | yes |
| `pipekit` + `pipekit-cycle` | Operator base + protocol contracts | yes |
| `jaxtyping` | Shape annotations | yes |
| `pipekit-jax` | `JaxModelOp` for weight persistence | optional `[persist]` |
| `pipekit-experiment` | `ModelRegistry`, run tracking | optional `[persist]` |
| `pipekit-train` | `Loss` / `Callback` / `MetricWriter` protocols | optional `[train]` |
| `filterax` | Ensemble methods (EnKF / EnKS / EnKI) | optional `[ensemble]` |
| `coordax` | Coordinate-aware fields | optional `[coords]` |
| `numpyro` | Full Bayesian fallback | optional `[mcmc]` |

`flax` and `jaxopt` are removed by the equinox migration.
