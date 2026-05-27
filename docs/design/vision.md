---
status: draft
version: 0.3.0
---

# vardax — Vision

## Q1: What is vardax?

> What is the core identity and scope of vardax?

### Decision

**vardax is the variational and amortized inference layer for data assimilation
in JAX.** It implements learned 4DVarNet, operational incremental 4DVar with
control-variable transform, and amortized posterior heads on a shared
protocol-driven core that directly satisfies the `pipekit-cycle` contracts
(`ForwardModel`, `ObservationOperator`, `AnalysisStep`).

It is **not** a forward model, a simulation framework, or an experiment runner.
It provides the inference machinery (variational costs, observation operators,
solvers, posterior approximations) that *other* libraries supply the physics
to. Forward models come from [`somax`](https://github.com/jejjohnson/somax)
(geophysical fluid dynamics) and `plumax` (atmospheric transport / methane).
Ensembles come from `filterax`. Structured linear algebra comes from `gaussx`.

### Identity in one sentence

> vardax is to data assimilation inference what somax is to forward modeling —
> a composable library of variational and amortized inference building blocks
> powered by JAX, satisfying pipekit-cycle protocols by construction.

### Core variational cost

Reconstruction is the minimization of

$$
J(x) = \alpha_\text{obs} \|H(x) - y\|^2_{R^{-1}} + \alpha_\text{prior} \|x - \varphi(x)\|^2_{B^{-1}}
$$

with

- $x$ — state estimate
- $y$ — noisy, partial observations (multi-instrument permitted)
- $H$ — observation operator (identity, masked, averaging kernel, multi-instrument fusion, …)
- $R$ — observation-error covariance (per-instrument; often diagonal)
- $\varphi$ — prior (learned autoencoder, learned diffusion, physics integrator from somax)
- $B$ — prior-error covariance (often Matérn-3/2; factorised via `gaussx` for incremental 4DVar)

For 4DVarNet, minimization proceeds via learned gradient steps modulated by a
ConvLSTM. For incremental 4DVar, via Gauss-Newton outer / CG inner iterations
on the control-variable-transformed problem $\chi = B^{-1/2}(x - x_b)$.
Amortized heads bypass minimization entirely by learning $q_\phi(x \mid y)$
directly.

### Three model families

| Family | When | Reference |
|---|---|---|
| **`VarDANet*`** | Learned 4DVarNet with prior + ConvLSTM grad modulator. Three differentiation modes (unrolled / one-step / implicit). Research and benchmarks. | Fablet et al. 2021–2023 |
| **`IncrementalVarDA*`** | Operational 4DVar: tangent-linear via `jax.linearize`, Gauss-Newton outer, CG/Lanczos inner, control-variable transform via `gaussx` Matérn factorisation. | Courtier et al. 1994 |
| **`AmortizedVarDA*`** | Direct $q_\phi(x \mid y)$ head: conditional normalising flow, score-based diffusion, or simulation-based amortized posterior. | Cranmer et al. 2020; Cohen et al. 2023 |

All three satisfy `pipekit_cycle.AnalysisStep` and compose with the same
`ObservationOperator` and `ForwardModel` registries.

### The six-step inference cycle

vardax is engineered around the **six-step research-to-operations cycle** that
runs across all forward-model tiers:

```
(1) Physics forward (somax / plumax)
      → (2) Model-based inference: MAP / MCMC / 4DVarNet                — slow, exact
      → (3) Neural emulator of the forward (trained from Step 1)         — fast surrogate
      → (4) Emulator-based inference: same loop, 100–1000× faster        — same vardax code
      → (5) Amortized predictor: y → posterior directly                  — sub-second
      → (6) Improve: swap any block; the previous step is the oracle    — validation loop
```

vardax's job is to make Steps 2, 4, and 5 use **the same library code**
parameterised only by which `ForwardModel` and which inference family
(`VarDANet*` / `IncrementalVarDA*` / `AmortizedVarDA*`) is plugged in.

### Three-layer architecture

```
Layer 2: Models           VarDANet / IncrementalVarDA / AmortizedVarDA
                          (each satisfies pipekit_cycle.AnalysisStep)
                              ↑
Layer 1: Components       priors (φ), observation operators (H),
                          grad modulators (Φ), cost functions, solver loops
                              ↑
Layer 0: Primitives       pure-JAX cost terms, solver steps, CVT,
                          Laplace covariance, autodiff-based TLM/adjoint
```

Users can enter at any level:
- **Layer 0** when integrating vardax algorithms into another framework
- **Layer 1** when composing custom DA pipelines (e.g. a new prior + AK obs op)
- **Layer 2** when running turnkey training or operational analysis

### Framework choices

vardax is **equinox-native** (deprecating Flax NNX from v0.1.x), and the
required dependency stack is:

| Package | Role |
|---|---|
| **`equinox`** | Module system; all priors / models / operators are `eqx.Module` |
| **`optax`** | Outer-loop training (replaces `nnx.Optimizer`) |
| **`optimistix`** | Inner-loop minimization (Gauss-Newton, BFGS, fixed-point) |
| **`diffrax`** | ODE integration for dynamical priors (wraps somax forward models) |
| **`lineax`** | Linear solves (CG / GMRES) for incremental 4DVar inner loop |
| **`gaussx`** | Structured linear operators; Matérn factorisation of $B$; Kronecker / LowRank for $R$ |
| **`pipekit`** + **`pipekit-cycle`** | Protocol contracts (`ForwardModel`, `ObservationOperator`, `AnalysisStep`) and `DACycle` / `SmootherCycle` orchestration |

Optional extras:

| Package | Use |
|---|---|
| `pipekit-jax` | Persist trained models via `JaxModelOp` + weight serialisation |
| `pipekit-experiment` | Content-addressed `ModelRegistry` for trained heads |
| `pipekit-train` | `Loss` / `Callback` / `MetricWriter` protocols vardax `train_step` plugs into |
| `filterax` | Hybrid ensemble-variational (En4DVar, EnVar) |
| `coordax` | Coordinate-aware batch construction from `xarray` |

### What vardax enables

- **Drop-in tier swaps.** Same vardax inference code accepts a Gaussian-plume
  forward from plumax, a shallow-water forward from somax, or a learned
  emulator — the `ForwardModel` interface is fixed.
- **Multi-instrument satellite inversion.** Per-instrument
  `(A, x_a, h, mask, R)` registries, fused at the likelihood level — no
  pre-regridding.
- **Research → operations arc.** The Jupyter cell that validated methodology
  becomes the backend for the FastAPI handler. Same `DACycle`, same
  `AnalysisStep`, different deployment.
- **Posterior provenance.** Every analysis carries enough metadata
  (`PosteriorAdapter`) to feed downstream population models (TMTPP) without
  retraining.

### What vardax does NOT do

(See [boundaries.md](boundaries.md) for the full ownership map.)

- It does not define forward models. Use `somax` (geophysics) or `plumax`
  (atmospheric transport / methane). Lorenz-63 / Lorenz-96 demos in
  `vardax._src.utils` are toys, not the library API.
- It does not own ensemble methods. Use `filterax`. vardax exposes
  ensemble-variational hooks but EnKF / EnKS proper lives elsewhere.
- It does not own structured linear algebra. Use `gaussx` for Matérn
  factorisations and Kronecker operators.
- It does not own data I/O. Use `georeader` (sensors) and `coordax` (labelled
  arrays). vardax accepts `Batch*` containers; how you fill them is upstream.
- It does not own experiment orchestration. Use `pipekit-cycle` for DA cycles
  and `pipekit-experiment` for run tracking. vardax provides `train_step` /
  `eval_step` / `AnalysisStep` — composing them is the user's job.
