<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="vardax" src="docs/assets/logo-light.svg" width="340">
</picture>

**Variational data assimilation for JAX: seven analysis methods for one cost, from closed-form BLUE to amortized posteriors, on optimistix, optax, lineax, gaussx, argminax, diffrax, integrix, geonnax and pipekit-cycle.**

[![Tests](https://github.com/jejjohnson/vardax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/vardax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/vardax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/vardax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/vardax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/vardax/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/vardax/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/vardax/)
[![codecov](https://codecov.io/gh/jejjohnson/vardax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/vardax)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

📖 [Documentation](https://jejjohnson.github.io/vardax/) ·
🧪 [Tutorials](https://jejjohnson.github.io/vardax/notebooks/) ·
📐 [Maths reference](https://jejjohnson.github.io/vardax/problem-setting/) ·
📚 [API reference](https://jejjohnson.github.io/vardax/reference/) ·
🧭 [Design](https://jejjohnson.github.io/vardax/design/readme/)

</div>

---

A forecast and a set of observations rarely agree, and data assimilation decides how much to trust each. Optimal interpolation, 3DVar, strong- and weak-constraint 4DVar, incremental 4DVar, 4DVarNet and amortized posteriors all answer that question, and they answer it by minimising the same cost under different assumptions. So in vardax we treat the analysis as the minimiser of one cost,

```math
\mathbf{x}_0^\star = \mathrm{argmin}_{\mathbf{x}_0, \boldsymbol{\eta}} \ \mathcal{J}(\mathbf{x}_0, \boldsymbol{\eta}; \mathbf{y}_{0:T})
```

where $`\mathbf{x}_0 \in \mathbb{R}^{D_x}`$ is the state at the start of a window of $`T + 1`$ times, $`\boldsymbol{\eta} = (\boldsymbol{\eta}_1, \ldots, \boldsymbol{\eta}_T)`$ with $`\boldsymbol{\eta}_t \in \mathbb{R}^{D_x}`$ is the model error at each step (zero under the strong constraint), $`\mathbf{y}_{0:T}`$ are the observations, $`\mathbf{y}_t \in \mathbb{R}^{D_y}`$, and $`\mathcal{J}`$ is the scalar cost written out below. The seven methods differ in which terms of $`\mathcal{J}`$ are active and in how they minimise it, so each is a peer module over the same seams (the observation operator, the covariances, the forward model, the minimiser and the adjoint), and switching method is a constructor change, not a rewrite.

```python
# Shapes: N_batch windows; T + 1 times; D_x state elements
# (model and batch as in the quickstart below)

# today (vardax)
# x₀⋆ = argmin 𝒥(x₀), one analysis per window
analysis = model(batch)  # → (N_batch, D_x)
# today (vardax)
# 𝒩(x₀⋆, Σ_x|y), with Σ_x|y a lazy operator
posterior = model.posterior(batch)
# today (vardax)
# the same model as a pipekit-cycle AnalysisStep
step = model.as_analysis_step()
```

None of it is a new framework. Every method is an [`equinox`](https://github.com/patrick-kidger/equinox) module. Its minimiser is an [`optimistix`](https://github.com/patrick-kidger/optimistix) minimiser (BFGS by default) or an [`optax`](https://github.com/google-deepmind/optax) optimiser, and its gradients come from optimistix's adjoints, so nothing is hand-differentiated. The covariances are [`lineax`](https://github.com/patrick-kidger/lineax) operators, with the structured ones (Matérn, Kronecker, low rank) and their square roots from [`gaussx`](https://github.com/jejjohnson/gaussx), so the posterior covariance is a lazy operator rather than a matrix. [`argminax`](https://github.com/jejjohnson/argminax) supplies the Gauss–Newton and dual solves behind incremental 4DVar and 3DVar, the one-step and k-step adjoints of a solver stopped early, and the Laplace curvature. The dynamics integrate with [`diffrax`](https://github.com/patrick-kidger/diffrax), whose adjoints the 4DVars take through `forward_adjoint=`, and [`integrix`](https://github.com/jejjohnson/integrix) checkpoints the rollout. [`geonnax`](https://github.com/jejjohnson/geonnax) gives the reduced bases and the derivatives of physics residuals, and [`pipekit-cycle`](https://github.com/jejjohnson/pipekit) the analysis-step and cycle protocols, so whatever orchestrates pipekit operators orchestrates vardax.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/vardax_overview/vardax_overview.png" alt="vardax in one picture: you supply the observation operator h, the covariances, the forward model f and a minimiser with its adjoint; they define one cost; seven methods minimise it in closed form, iteratively, by learning or by amortization; out come the analysis, its uncertainty and an analysis step for cycling; lineax with gaussx, optimistix with optax, argminax, diffrax with integrix, geonnax and pipekit underneath" width="100%"></p>

## 🌟 Highlights

- 🧮 **Seven peer methods**: `OptimalInterpolation`, `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`, `IncrementalFourDVar`, `FourDVarNet1D` and `FourDVarNet2D`, and `AmortizedPosterior`, each an equinox module behind one constructor.
- 🔁 **One cycling interface**: every method becomes a `pipekit_cycle.AnalysisStep` through `.as_analysis_step()`, and `VarDACycle` and `VarSmootherCycle` cycle it.
- 🧷 **No hand-written adjoints**: diffrax adjoints through `forward_adjoint=`, checkpointed rollouts through `checkpoint=`, optimistix's `ImplicitAdjoint` for the minimiser, and argminax's `OneStep` (Bolte et al., 2023) for a learned solver.
- 🧱 **Operator covariances**: $`\boldsymbol{\Sigma}_x`$, $`\boldsymbol{\Sigma}_\varepsilon`$ and the posterior $`\boldsymbol{\Sigma}_{x \mid y}`$ are lineax operators, so a mat-vec or the variances are computed without forming the matrix.
- 🛰️ **Observation operators** for gappy grids, off-grid and scattered points, retrieval averaging kernels and multi-instrument fusion.
- 📏 **Posteriors for the variational methods**: `LaplaceCovariance`, `GaussNewtonHessian` and `EnsembleCovariance`, exported for population models by `GaussianMarkLikelihood`.
- ✅ **Validation gates** that decide whether a fast method may replace a trusted one.

## 📐 The seven methods

Every method minimises the same cost over a window, with only some of its terms active:

```math
\mathcal{J}(\mathbf{x}_0, \boldsymbol{\eta}) = \frac{1}{2} \Vert \mathbf{x}_0 - \boldsymbol{\mu}_x \Vert^2_{\boldsymbol{\Sigma}_x^{-1}} + \frac{1}{2} \sum_t \Vert \mathbf{m}_t \odot (\mathbf{y}_t - h(\mathbf{x}_t)) \Vert^2_{\boldsymbol{\Sigma}_\varepsilon^{-1}} + \frac{1}{2} \sum_t \Vert \boldsymbol{\eta}_t \Vert^2_{\boldsymbol{\Sigma}_\eta^{-1}}
```

with the rollout $`\mathbf{x}_t = f(\mathbf{x}_{t-1}) + \boldsymbol{\eta}_t`$; the observation sum runs over the window's $`T + 1`$ times and the model-error sum over its $`T`$ steps. Here $`\boldsymbol{\mu}_x \in \mathbb{R}^{D_x}`$ and $`\boldsymbol{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}`$ are the prior (background) mean and covariance, $`h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}`$ is the observation operator, $`\mathbf{m}_t \in \{0, 1\}^{D_y}`$ marks the observed entries, $`\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}`$ is the observation-error covariance, $`f: \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}`$ is the forward model and $`\boldsymbol{\Sigma}_\eta \in \mathbb{R}^{D_x \times D_x}`$ the model-error covariance.

| Class | Method | Control | Reach for it when |
|---|---|---|---|
| `OptimalInterpolation` | BLUE, closed form | $`\mathbf{x}`$ | $`h`$ is linear and the errors Gaussian: the right default |
| `ThreeDVar` | 3D variational | $`\mathbf{x}`$ | one time, with a nonlinear $`h`$ |
| `StrongFourDVar` | strong-constraint 4DVar | $`\mathbf{x}_0`$ | a window of data, the dynamics trusted |
| `WeakFourDVar` | weak-constraint 4DVar | $`(\mathbf{x}_0, \boldsymbol{\eta})`$ | a window of data, the model biased |
| `IncrementalFourDVar` | Gauss–Newton outer loop, CG inner loop | $`\delta \mathbf{x}_0`$ | the operational fast path |
| `FourDVarNet1D`, `FourDVarNet2D` | a learned prior and a learned solver | $`\mathbf{x}`$ | dense training data, no trusted model |
| `AmortizedPosterior` | a direct head $`q(\mathbf{x} \mid \mathbf{y}; \boldsymbol{\nu})`$ | none | real-time use, many events |

Four more entry points are variants of these. `DualThreeDVar` solves 3DVar in observation space, `optimal_estimation` is Rodgers' retrieval form of it, `ReducedBasisFourDVar` is `StrongFourDVar` with the control in a reduced basis, $`\mathbf{x}_0 = \boldsymbol{\mu}_x + \boldsymbol{\Phi} \mathbf{z}`$, and `SpaceTimeFieldVar` fits one continuous neural field to the whole window. The [problem setting](https://jejjohnson.github.io/vardax/problem-setting/) chapter derives the cost.

## 📥 Install

vardax is not on PyPI, and neither are most of the libraries it builds on, so install it with [uv](https://docs.astral.sh/uv/) from git. It brings in JAX, Equinox, einx, optax, diffrax, optimistix, lineax, gaussx, argminax, geonnax, integrix, pipekit-ops and pipekit-cycle, and needs Python 3.12 or 3.13.

uv does not read a dependency's `[tool.uv.sources]`, so a project that depends on vardax maps vardax's git dependencies too. Without these entries uv looks for them on PyPI, where all but `pipekit` are missing and `pipekit` is an unrelated project:

```toml
[project]
dependencies = ["vardax"]

[tool.uv.sources]
vardax = { git = "https://github.com/jejjohnson/vardax", tag = "v0.3.9" }  # x-release-please-version
argminax = { git = "https://github.com/jejjohnson/argminax", tag = "v0.0.4" }
gaussx = { git = "https://github.com/jejjohnson/gaussx", rev = "v0.6.12" }
geonnax = { git = "https://github.com/jejjohnson/geonnax", tag = "v0.0.10" }
functax = { git = "https://github.com/jejjohnson/functax", rev = "v0.0.13" }
integrix = { git = "https://github.com/jejjohnson/integrix", tag = "v0.0.11" }
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.2.0" }
pipekit-cycle = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-cycle", tag = "pipekit-array-v0.2.0" }
```

The extras are `data` (xarray and NumPy for the dataset utilities), `viz` (matplotlib and seaborn), `examples` (matplotlib, loguru and tqdm for the notebook helpers), `jlab` (Jupyter, jupytext and nbconvert, to run and build the tutorials), `exp` (wandb, Hydra, loguru and plotting, for experiments), `persist` (pipekit-jax and pipekit-experiment), `train` (pipekit-train) and `all`. The `persist` and `train` extras need their pipekit packages mapped as above, at the same tag. The tutorials need `viz`:

```toml
[project]
dependencies = ["vardax[viz]"]
```

## 🌪️ Quickstart: 4DVar on Lorenz-63

A window of $`T + 1 = 21`$ noisy observations of the chaotic Lorenz-63 system, half of them missing, and a strong-constraint 4DVar that recovers the initial state. The prior mean $`\boldsymbol{\mu}_x`$ is the true initial state plus noise of standard deviation $`\sigma_x = 2`$, with $`\boldsymbol{\Sigma}_x = \sigma_x^2 \mathbf{I}`$; the observations are the truth plus noise of standard deviation $`\sigma_\varepsilon = 0.5`$, with $`\boldsymbol{\Sigma}_\varepsilon = \sigma_\varepsilon^2 \mathbf{I}`$, observed on every other time ($`\mathbf{m}_t`$); the forward model $`f`$ is the true flow over $`\Delta t = 0.05`$, integrated with diffrax; and $`h`$ is the identity. BFGS minimises the cost, with the gradient from `jax.grad` through the rollout.

```python
import einx
import jax
import jax.numpy as jnp
import lineax as lx

import vardax as vdx

# Shapes: S = 2,000 simulated steps; T + 1 = 21 window times; D = 3 state variables

key_truth, key_obs, key_background = jax.random.split(jax.random.PRNGKey(0), 3)

# today (vardax)
# dx/dt = f(x), sampled every Δt = 0.01 after a burn-in on the attractor
_, trajectory = vdx.simulate_lorenz63(  # → (S, D)
    key_truth, dt=0.01, n_steps=2000, n_burn_in=1000
)
# x_true(t) for t = 0, 0.05, …, 1.0
true_window = trajectory[500 : 500 + 5 * 21 : 5]  # (S, D) → (T+1, D)

# m_t = 1 on every other time, 0 otherwise
mask = jnp.zeros((21, 3)).at[::2].set(1.0)  # → (T+1, D)
# ε_t ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_obs, (21, 3))  # → (T+1, D)
# y_t = x_true(t) + σ_ε ε_t
observations = true_window + 0.5 * obs_noise  # (T+1, D) → (T+1, D)
# ξ ~ 𝒩(0, I)
background_noise = jax.random.normal(key_background, (3,))  # → (D,)
# μ_x = x_true(0) + σ_x ξ
background = true_window[0] + 2.0 * background_noise  # (D,) → (D,)


# user
# Σ = σ² I, tagged positive semi-definite for the CG solve
def diagonal_covariance(variance: float) -> lx.AbstractLinearOperator:
    diagonal = jnp.full(3, variance)
    operator = lx.DiagonalLinearOperator(diagonal)
    return lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)


# today (vardax)
# dx/dt = (σ(x₂ − x₁), x₁(ρ − x₃) − x₂, x₁x₂ − βx₃)
lorenz = vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
# today (vardax)
# f : x_{t−1} ↦ x_t over Δt = 0.05, integrated with diffrax
forward = vdx.DynTrajectory(model=lorenz).as_forward_model(dt=0.05)

# today (vardax)
# x₀⋆ = argmin ½‖x₀ − μ_x‖²_Σx⁻¹ + ½ ∑_t ‖m_t ⊙ (y_t − x_t)‖²_Σε⁻¹
model = vdx.StrongFourDVar(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=diagonal_covariance(2.0**2),
    obs_cov_op=diagonal_covariance(0.5**2),
)

# m ⊙ y
masked_observations = observations * mask  # (T+1, D) → (T+1, D)
# a batch of one window
batch_input, batch_mask, batch_target = einx.id(  # (T+1, D) → (1, T+1, D)
    "t d, t d, t d -> 1 t d, 1 t d, 1 t d",
    masked_observations,
    mask,
    true_window,
)
# today (vardax)
batch = vdx.Batch1D(input=batch_input, mask=batch_mask, target=batch_target)
# today (vardax)
analysis = model(batch)[0]  # (1, T+1, D) → (D,)

# ‖μ_x − x_true(0)‖
background_error = jnp.linalg.norm(background - true_window[0])  # → ()
# ‖x₀⋆ − x_true(0)‖
analysis_error = jnp.linalg.norm(analysis - true_window[0])  # → ()
print(f"background error {background_error:.2f}, analysis error {analysis_error:.2f}")
```

This prints `background error 3.84, analysis error 0.81`. The prior mean starts 3.84 from the true initial state, and the analysis, having fitted the window's eleven observed times through the dynamics, ends 0.81 from it, almost five times closer.

The same code takes any forward model that maps a state to the next one: swap the Lorenz flow for a somax model, `MaskedIdentity` for an averaging kernel, or the diagonal covariances for a gaussx Matérn operator, and the solve is unchanged. The [landing page](https://jejjohnson.github.io/vardax/) has two more quickstarts (optimal interpolation and incremental 4DVar), and the [maths reference](https://jejjohnson.github.io/vardax/problem-setting/) derives every method.

## ♻️ Swap the method, keep the pipeline

Every model exposes `.as_analysis_step()`, so the cycling code never sees which method it is running: `model` can be any of the seven. In the sketch below, $`\mathbf{x}_c^f`$ and $`\mathbf{x}_c^a`$ are the forecast and the analysis at cycle $`c`$, $`f`$ the forward model over one cycle and $`\mathbf{y}_c`$ that cycle's observations; `load_window`, `n_cycles`, `obs_cov` and `initial_analysis` are yours to supply.

```python
import pipekit_cycle as pc
import pipekit_ops

# Shapes: C cycles; D = 3 state variables

# today (pipekit)
# c ↦ y_c, with load_window the user's reader for cycle c
obs_source = pipekit_ops.Lambda(load_window)

# today* (vardax)
# x_c^f = f(x_{c−1}^a),  x_c^a = analysis(x_c^f, y_c)
cycle = vdx.VarDACycle(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    model=model,
    obs_source=obs_source,
    n_steps=n_cycles,
)

# today (pipekit_cycle)
initial_state = pc.DAState(t=0.0, cycle_count=0, obs_err_cov=obs_cov)
final_analysis, final_state = cycle(initial_analysis, initial_state)  # → (D,)
# [(x_c^f, x_c^a)] for c = 1, …, C
forecasts, analyses = zip(*cycle.history, strict=True)  # C pairs of (D,)
```

Each cycle forecasts the carrier one step, hands the forecast and that cycle's observations (`NaN` marks a gap) to the analysis step, and keeps the pair in `cycle.history`. `VarSmootherCycle` does the same over fixed-lag windows. Both return plain `pipekit_cycle` objects, so whatever orchestrates pipekit operators orchestrates vardax. [Tutorial 16](docs/notebooks/16_cycled_assimilation_L63.py) runs the loop for real, including the analysis step that makes each cycle's prior mean $`\boldsymbol{\mu}_x`$ the previous forecast.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/vardax_roles/vardax_roles-dark.svg">
    <img alt="vardax on the GeoStack role ladder: cycles as recipes, every method as an analysis step, solve, train and validation algorithms, the analysis methods and their priors as models, minimisers and uncertainty readers as strategies, cost terms as primitives, and the data, observation operators and controls as structures; pipekit-cycle, the optimistix and argminax adjoints and the setup functions sit beside the ladder" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/vardax_roles/vardax_roles-light.svg" width="100%">
  </picture>
</p>

| Area | Symbols | Notes |
|---|---|---|
| Observation operators | `MaskedIdentity`, `LinearObs`, `InterpObs`, `AveragingKernel`, `MultiInstrumentFusion`, `CallableObs`, `interp_obs_from_coords` | Gappy grids, off-grid interpolation with a passing adjoint test, retrieval averaging kernels, multi-instrument fusion that satisfies the precision-addition identity, and any differentiable function |
| Point observations | `FieldPointObs`, `GridPointObs`, `PointObsBatch`, `point_obs_cost` | Scattered, time-stamped observations, padded to a fixed count |
| Posterior | `LaplaceCovariance`, `GaussNewtonHessian`, `EnsembleCovariance`, `Posterior`, `GaussianMarkLikelihood` | The covariance is a lazy lineax operator; `GaussianMarkLikelihood` serialises it for population models |
| Minimisers | `OptimistixMinimiser`, `OptaxMinimiser`, `MinimiseResult` | Any optimistix minimiser, or an optax optimiser for a neural control, behind one protocol |
| Adjoints | `ImplicitAdjoint`, `RecursiveCheckpointAdjoint`, `to_optimistix_adjoint` | optimistix's adjoints, re-exported; `KStepAdjoint` and `OneStepAdjoint` are deprecated aliases of argminax's `KStep` and `OneStep` |
| Dynamical priors | `DynTrajectory`, `DynIncrements`, `DynamicalPrior` | An ODE prior that doubles as a forward model (`.as_forward_model(dt)`) or a trajectory loss (`.bind(ts)`) |
| Controls | `LinearParameterization`, `NeuralFieldParameterization`, `LinearBasis`, `CompositeBasis`, `ModelErrorBasis`, `eof_basis`, `fourier_basis`, `rbf_basis`, `wavelet_basis` | A reduced basis or a neural field as the control, with its prior penalty |
| Amortized inference | `AmortizedPosterior`, `MLPObsEncoder`, `IdentityObsEncoder`, `RegressionHead` | Train once on simulations, assimilate in a forward pass. `ConditionalFlowHead` and `ScoreDiffusionHead` are declared but raise `NotImplementedError` on construction, pending `gauss_flows` and the reverse-SDE sampler |
| Validation gates | `assert_posterior_agreement`, `assert_adjoint_calibrated`, `simulation_based_calibration` | The go/no-go checks that decide whether a fast method may replace a trusted one |
| Forward adapters | `SoftBoundedForward` | Identity inside a box and a smooth saturation outside, so a line-search trial far off the attractor cannot overflow an explicit ODE step and turn the solve into NaN |

## 📓 Tutorials

Seventeen walkthroughs, all [jupytext](https://jupytext.readthedocs.io/) sources under `docs/notebooks/` that run when the documentation is built, so the figures on the site come from the code in this repository, not from committed snapshots.

| # | Tutorial | What it shows |
|---|---|---|
| 01–02 | [Model-based 4DVar](docs/notebooks/01_model_based_4dvar_L63.py), [unrolling vs fixed point](docs/notebooks/02_unrolling_vs_fixedpoint_L63.py) | The reference 4DVar picture, then the learned solver against it |
| 03–04 | [4DVarNet on L63](docs/notebooks/03_4dvarnet_L63.py), [2-D demo](docs/notebooks/04_4dvarnet_2d_demo.py) | Training the learned variant in 1-D and on spatiotemporal fields |
| 05–07 | [L63 pipeline](docs/notebooks/05_end_to_end_L63.py), [L96 pipeline](docs/notebooks/06_end_to_end_L96.py), [prior pre-training](docs/notebooks/07_prior_pretraining_L63.py) | Simulation, patching, masking and training, end to end |
| 08 | [Classical 4DVar vs 4DVarNet](docs/notebooks/08_classical_4dvar_vs_4dvarnet_L63.py) | Gradient descent on the cost against the learned solver |
| 09–11 | [Parameter estimation](docs/notebooks/09_param_estimation_L63.py), [bilevel optimisation](docs/notebooks/10_bilevel_opt_L63.py), [gradient learning](docs/notebooks/11_gradient_learning_L63.py) | Learning ODE parameters, cost weights, and the update rule itself |
| 12 | [Weak-constraint 4DVar](docs/notebooks/12_weak_constraint_4dvar_L63.py) | A biased model: recovered model-error increments, the $`\boldsymbol{\Sigma}_\eta`$ dial from weak to strong, and the same cost in trajectory space |
| 13 | [Neural closures on two-level L96](docs/notebooks/13_neural_closure_L96.py) | The unresolved coupling learned offline and online through the ODE solve, forecast skill, then used as a 4DVar forward model |
| 14 | [Posterior uncertainty](docs/notebooks/14_posterior_uncertainty_L63.py) | Laplace and Gauss–Newton covariances against the exact Hessian, push-forward along the window, ensemble spread, z-scores and SBC |
| 15 | [Observation operators](docs/notebooks/15_observation_operators.py) | Masks vs selection matrices, off-grid interpolation, averaging-kernel bias, and multi-instrument fusion verified numerically |
| 16 | [Cycled assimilation](docs/notebooks/16_cycled_assimilation_L63.py) | Forecast and analysis loops: spin-up, the error equilibrium and its scalar prediction, tuning $`\boldsymbol{\Sigma}_x`$ by innovations |
| 17 | [Amortized posterior](docs/notebooks/17_amortized_posterior_L63.py) | Simulation-based training against a multi-start 4DVar oracle, wall-clock comparison, and an honest reading of the three gates |

## 📖 Documentation

- **[Maths reference](https://jejjohnson.github.io/vardax/problem-setting/)**: 23 chapters, from the Bayesian foundation (1–3) through each analysis method (4–10, and reduced-order 4DVar in 23), the cross-cutting concerns (11–14) and worked examples on Lorenz, sea-surface height, methane and latent-space assimilation (15–18), to physical models, uncertainty, OceanBench and neural-field 4DVar (19–22).
- **[Tutorials](https://jejjohnson.github.io/vardax/notebooks/)**: the seventeen notebooks above, executed.
- **[API reference](https://jejjohnson.github.io/vardax/reference/)**: every symbol vardax exports, grouped by seam and generated from the docstrings. The two it re-exports from optimistix, `ImplicitAdjoint` and `RecursiveCheckpointAdjoint`, are described there and linked upstream.
- **[Design docs](https://jejjohnson.github.io/vardax/design/readme/)**: the architecture, the ownership boundaries, the decision log and the pipekit composition.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/vardax_ecosystem/vardax_ecosystem-dark.svg">
    <img alt="Packages that import vardax: plumax through its assim extra and sesmax-core through its da extra, both in GeoModels; vardax builds on JAX, Equinox, lineax, gaussx, optimistix, optax, argminax, diffrax, integrix, geonnax and pipekit-cycle" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/vardax_ecosystem/vardax_ecosystem-light.svg" width="100%">
  </picture>
</p>

vardax owns the assimilation methods and nothing else. Forward models, minimisers, ODE solvers, structured linear algebra and ensemble methods are composed in:

| Concern | Owner |
|---|---|
| Modules and pytrees | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Minimisers and their adjoints | [`optimistix`](https://github.com/patrick-kidger/optimistix), [`optax`](https://github.com/google-deepmind/optax) |
| Gauss–Newton and dual solves, adjoints for early stopping, Laplace curvature | [`argminax`](https://github.com/jejjohnson/argminax) |
| Linear solvers (CG, GMRES, Lanczos) | [`lineax`](https://github.com/patrick-kidger/lineax) |
| Structured operators (Matérn, Kronecker, low rank) and their square roots | [`gaussx`](https://github.com/jejjohnson/gaussx) |
| ODE and SDE integration and their adjoints | [`diffrax`](https://github.com/patrick-kidger/diffrax) |
| Checkpointed rollouts | [`integrix`](https://github.com/jejjohnson/integrix) |
| Reduced bases, neural fields and differential operators | [`geonnax`](https://github.com/jejjohnson/geonnax) |
| Analysis-step and cycle protocols, operator composition | [`pipekit`](https://github.com/jejjohnson/pipekit) (`pipekit-ops`, `pipekit-cycle`) |
| Geophysical forward models (shallow water, QG, primitive equations) | [`somax`](https://github.com/jejjohnson/somax) |
| Atmospheric transport and methane retrieval | [`plumax`](https://github.com/jejjohnson/plumax) |
| Ensemble filters and smoothers (EnKF, EnKS, EnKI) | [`filterax`](https://github.com/jejjohnson/filterax) |

Two packages import vardax today, each through an extra: [plumax](https://github.com/jejjohnson/plumax) runs its Lagrangian, Eulerian and coupled source inversions on it (`assim`), and [sesmax-core](https://github.com/jejjohnson/sesmax) assimilates into its Earth-system models with it (`da`). vardax imports neither. See [boundaries](docs/design/boundaries.md) for the full ownership map.

## 🌳 Repository layout

```
src/vardax/
├── __init__.py   ← the public API: every method, operator, posterior and gate
├── adapters/     ← adapters for optional libraries (pipekit-train)
├── adjoints.py   ← vardax.adjoints: the optimistix adjoints and to_optimistix_adjoint
├── amortized.py  ← vardax.amortized: encoders, heads and AmortizedPosterior
├── cycle.py      ← vardax.cycle: VarDACycle and VarSmootherCycle
├── examples.py   ← vardax.examples: demo helpers and reference residuals
└── _src/         ← the implementation, private
docs/             ← 23 maths chapters, the API reference, design docs, 17 tutorials
tests/            ← unit, slow and integration tiers
scripts/          ← build_docs.py, the two-tool docs build
```

## ⚙️ Development

```bash
make install      # uv sync --all-extras, then the pre-commit hooks
make uv-format    # ruff format . and ruff check --fix .
make uv-lint      # ruff format --check ., ruff check --no-fix . and ty check src/vardax
make uv-test      # pytest
make docs         # the MyST site and the MkDocs API reference, tutorials executed
make docs-serve   # the same without executing the tutorials, at http://127.0.0.1:8000
```

Formatting, linting, type checking and the tests must all pass before a commit is pushed. Commits follow Conventional Commits and release-please cuts the versions; committed `.ipynb` files are rejected by a pre-commit hook, so edit the jupytext `.py` source. [`AGENTS.md`](AGENTS.md) has the full workflow.

## 🎓 References

- Fablet et al. (2021). [Learning variational data assimilation models and solvers](https://doi.org/10.1029/2021MS002572). *JAMES* 13(10).
- Fablet, Febvre and Chapron (2023). [Multimodal 4DVarNets for the reconstruction of sea surface dynamics from NADIR and wide-swath altimetry](https://doi.org/10.1109/TGRS.2023.3268006). *IEEE TGRS* 61.
- Bolte, Pauwels and Vaiter (2023). [One-step differentiation of iterative algorithms](https://arxiv.org/abs/2305.13768). *NeurIPS* 36.
- Courtier, Thépaut and Hollingsworth (1994). A strategy for operational implementation of 4D-Var, using an incremental approach. *QJRMS* 120(519).
- Carrassi et al. (2018). Data assimilation in the geosciences: an overview of methods, issues, and perspectives. *WIREs Climate Change* 9(5), e535.
- The full bibliography is in [`docs/references.md`](docs/references.md).

## 📄 License

MIT. See [LICENSE](LICENSE).
