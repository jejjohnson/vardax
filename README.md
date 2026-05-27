# fourdvarjax — Variational Data Assimilation with Learned Components

> Modular, pedagogical JAX/Flax NNX implementation of the 4DVarNet framework

[![CI Tests](https://github.com/jejjohnson/fourdvarjax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/fourdvarjax/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jejjohnson/fourdvarjax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/fourdvarjax)
[![PyPI version](https://img.shields.io/pypi/v/fourdvarjax.svg)](https://pypi.org/project/fourdvarjax/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`fourdvarjax` is a modular JAX/Flax NNX implementation of the **4DVarNet** framework — an end-to-end learnable variational data assimilation approach that combines classical 4D-Var with deep learning.

---

## Overview

4DVarNet frames reconstruction of spatiotemporal fields as minimisation of a variational cost:

$$\mathcal{J}(\mathbf{x}) = \underbrace{\|\mathbf{m} \odot (\mathbf{x} - \mathbf{y})\|^2}_{\mathcal{J}_\text{obs}} + \lambda \underbrace{\|\mathbf{x} - \varphi(\mathbf{x})\|^2}_{\mathcal{J}_\text{prior}}$$

where:
- $\mathbf{y}$ are (noisy, partial) observations and $\mathbf{m}$ is the binary observation mask
- $\varphi$ is a **learned prior** (bilinear autoencoder, ConvAE, or model-based ODE)
- Minimisation uses **learned gradient steps** modulated by a ConvLSTM

The solver iterates:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \left[\alpha_k \cdot \Psi(\nabla\mathcal{J}) + \beta_k \cdot \nabla\mathcal{J}\right]$$

where $\Psi$ is the ConvLSTM gradient modulator. Training supports both **unrolled backprop** and **implicit differentiation** (O(1) memory).

## Supported Tensor Layouts

| Layout | Shape | Use Case | Model |
|--------|-------|----------|-------|
| 1D | `(B, T, N)` | State vectors (e.g. Lorenz-63, N=3) | `FourDVarNet1D` |
| 2D | `(B, T, H, W)` | Spatiotemporal fields (e.g. sea surface height) | `FourDVarNet2D` |
| 2D+ | `(B, T, C, H, W)` | Multivariate 2D fields | `BilinAEPrior2DMultivar` |

## Repository Structure

```
fourdvarjax/
├── pyproject.toml                  ← uv/hatchling build config
├── Makefile                        ← developer workflow targets
├── fourdvarjax/                    ← installable package
│   ├── __init__.py                 ← public API re-exports
│   └── _src/
│       ├── _types.py               ← Batch1D, Batch2D, Batch2DMultivar, LSTMState1D/2D
│       ├── costs.py                ← obs_cost_1d/2d, prior_cost, variational_cost, decomposed_loss
│       ├── priors.py               ← BilinAEPrior1D/2D/2DMultivar, ConvAEPrior1D, L63Prior, L96Prior, MLPAEPrior1D, IdentityPrior
│       ├── grad_mod.py             ← ConvLSTMGradMod1D, ConvLSTMGradMod2D
│       ├── solver.py               ← SolverState1D/2D, solver_step_1d/2d, solve_4dvarnet_1d/2d, fp_solver_step_1d
│       ├── model.py                ← FourDVarNet1D, FourDVarNet2D
│       ├── training.py             ← train_step, eval_step, fit, train_loss_fn, reconstruction_loss
│       └── utils/
│           ├── dynamical_systems.py  ← simulate_lorenz63, simulate_lorenz96 (via Diffrax/Equinox)
│           ├── patches.py            ← trajectory_to_xr_dataset, extract_patches
│           ├── masks.py              ← random_mask, regular_mask, feature_mask
│           ├── noise.py              ← add_gaussian_noise
│           ├── standardize.py        ← compute_scaler_params, apply_standardization, inverse_standardization
│           ├── preprocessing.py      ← xr_to_batch1d, train_test_split, interpolate_initial_condition
│           └── viz.py                ← plot_3d_attractor, plot_state_grid, plot_trajectories, ...
├── docs/                           ← mathematical reference (16 chapters) + design docs
│   ├── README.md                   ← table of contents
│   ├── 01_*.md ... 16_*.md         ← mathematical chapters
│   └── design/                     ← architecture, API contracts, decisions D1–D16
├── tests/                          ← pytest test suite
└── notebooks/                      ← Jupytext percent-format .py tutorials
    ├── 01_model_based_4dvar_L63.py
    ├── 02_unrolling_vs_fixedpoint_L63.py
    ├── 03_4dvarnet_L63.py
    ├── 04_4dvarnet_2d_demo.py
    ├── 05_end_to_end_L63.py
    ├── 06_end_to_end_L96.py
    ├── 07_prior_pretraining_L63.py
    └── 08_classical_4dvar_vs_4dvarnet_L63.py
```

## Installation

```bash
# Install from PyPI
pip install fourdvarjax

# Install with uv (recommended for development)
git clone https://github.com/jejjohnson/fourdvarjax
cd fourdvarjax
uv sync

# Editable install with all extras
pip install -e ".[dev,test,data,viz,jlab]"
```

## Developer Workflow

```bash
make install        # uv sync --all-extras + pre-commit install
make uv-format      # ruff format + ruff check --fix
make uv-lint        # ruff check + ty check
make uv-test        # pytest tests/ -v
make uv-test-cov    # pytest with coverage report
make uv-pre-commit  # run all pre-commit hooks
```

## Quick Start

### Lorenz-63 (1D) Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax

import fourdvarjax
from fourdvarjax._src.utils.dynamical_systems import simulate_lorenz63
from fourdvarjax._src.utils.patches import trajectory_to_xr_dataset, extract_patches
from fourdvarjax._src.utils.masks import random_mask
from fourdvarjax._src.utils.noise import add_gaussian_noise
from fourdvarjax._src.utils.preprocessing import xr_to_batch1d, train_test_split

# ── Simulate L63 and build dataset ──
key = jax.random.PRNGKey(0)
time, states = simulate_lorenz63(key, n_steps=10_000)        # (T,), (T, 3)
ds = trajectory_to_xr_dataset(states, time)
ds = extract_patches(ds, n_patches=2000, n_timesteps=200)
ds = random_mask(ds, missing_rate=0.875)
ds = add_gaussian_noise(ds, sigma=0.1)

ds_train, ds_test = train_test_split(ds, n_train=1600, n_test=400)
batch = xr_to_batch1d(ds_train)  # Batch1D(input, mask, target) — shape (B, T, N)

# ── Build model ──
rngs = nnx.Rngs(42)
model = fourdvarjax.FourDVarNet1D(
    state_dim=3,
    n_time=200,
    latent_dim=32,
    hidden_dim=64,
    n_solver_steps=15,
    rngs=rngs,
)

# ── Train ──
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
history = fourdvarjax.fit(model, optimizer, [batch], n_epochs=50)
```

### 2D Spatiotemporal Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
import fourdvarjax

# ── Build model ──
rngs = nnx.Rngs(0)
model = fourdvarjax.FourDVarNet2D(
    n_time=5,
    hidden_dim=32,
    n_solver_steps=15,
    rngs=rngs,
)

# ── Run forward pass ──
B, T, H, W = 4, 5, 64, 64
key = jax.random.PRNGKey(0)
target = jax.random.normal(key, (B, T, H, W))
mask   = jax.random.bernoulli(jax.random.PRNGKey(1), 0.3, (B, T, H, W)).astype(jnp.float32)
obs    = target * mask

batch  = fourdvarjax.Batch2D(input=obs, mask=mask, target=target)
x_star = model(batch)  # (B, T, H, W) — reconstructed field
```

### Using the Solver Directly

```python
import fourdvarjax

# Compute cost components manually
j_obs  = fourdvarjax.obs_cost_1d(batch, x_hat)
j_prior = fourdvarjax.prior_cost(x_hat, model.prior)
j_total, (j_obs, j_prior) = fourdvarjax.decomposed_loss(batch, x_hat, model.prior)

# Step through the solver
solver_state = fourdvarjax.init_solver_state_1d(batch, hidden_dim=64)
for _ in range(15):
    solver_state = fourdvarjax.solver_step_1d(
        solver_state, batch, model.prior, model.grad_mod
    )
x_hat = solver_state.x
```

## Public API

### Data Types

| Class | Shape | Description |
|-------|-------|-------------|
| `Batch1D` | `(B, T, N)` | 1D spatiotemporal batch (`input`, `mask`, `target`) |
| `Batch2D` | `(B, T, H, W)` | 2D spatiotemporal batch |
| `Batch2DMultivar` | `(B, T, C, H, W)` | Multivariate 2D batch |
| `LSTMState1D` | `(B, H, N)` | ConvLSTM hidden/cell state for 1D |
| `LSTMState2D` | `(B, H, H, W)` | ConvLSTM hidden/cell state for 2D |

### Prior Models

| Class | Input | Architecture | Best For |
|-------|-------|-------------|----------|
| `BilinAEPrior1D` | `(B, T, N)` | Bilinear AE (MLP) | L63, 1D state vectors |
| `BilinAEPrior2D` | `(B, T, H, W)` | Conv2d bilinear AE | Spatiotemporal fields |
| `BilinAEPrior2DMultivar` | `(B, T, C, H, W)` | Conv2d bilinear AE | Multivariate 2D |
| `ConvAEPrior1D` | `(B, T, N)` | Convolutional AE | 1D fields |
| `MLPAEPrior1D` | `(B, T, N)` | MLP AE | Simple baseline |
| `L63Prior` | `(B, T, N)` | RK4 ODE (Diffrax) | Model-based L63 4DVar |
| `L96Prior` | `(B, T, N)` | RK4 ODE (Diffrax) | Model-based L96 4DVar |
| `IdentityPrior` | any | Identity (no prior) | Ablation / pure obs |

### Solver

| Function | Description |
|----------|-------------|
| `init_solver_state_1d(batch, hidden_dim)` | Initialise 1D solver state from masked observations |
| `init_solver_state_2d(batch, hidden_dim)` | Initialise 2D solver state from masked observations |
| `solver_step_1d(state, batch, prior_fn, grad_mod_fn)` | Single unrolled 1D solver step |
| `solver_step_2d(state, batch, prior_fn, grad_mod_fn)` | Single unrolled 2D solver step |
| `fp_solver_step_1d(...)` | Fixed-point solver step for implicit differentiation |
| `solve_4dvarnet_1d(batch, model, ...)` | Full 1D solve loop |
| `solve_4dvarnet_1d_fixedpoint(batch, model, ...)` | Full 1D fixed-point solve (O(1) memory) |
| `solve_4dvarnet_2d(batch, model, ...)` | Full 2D solve loop |

### Costs

| Function | Description |
|----------|-------------|
| `obs_cost_1d(batch, x)` | Observation MSE for 1D |
| `obs_cost_2d(batch, x)` | Observation MSE for 2D |
| `prior_cost(x, prior_fn)` | Prior regularisation cost |
| `variational_cost(batch, x, prior_fn)` | Total variational cost |
| `variational_cost_grad(batch, x, prior_fn)` | Gradient of variational cost |
| `decomposed_loss(batch, x, prior_fn)` | Returns `(total, (obs, prior))` |

### Training

| Function | Description |
|----------|-------------|
| `train_step(model, optimizer, batch)` | Single gradient update |
| `eval_step(model, batch)` | Forward pass (no grad) |
| `reconstruction_loss(model, batch)` | MSE of reconstruction vs target |
| `train_loss_fn(model, batch)` | Combined training loss |
| `fit(model, optimizer, batches, n_epochs)` | Full training loop |

### Utilities

| Module | Key Functions |
|--------|---------------|
| `utils.dynamical_systems` | `simulate_lorenz63`, `simulate_lorenz96` |
| `utils.patches` | `trajectory_to_xr_dataset`, `extract_patches` |
| `utils.masks` | `random_mask`, `regular_mask`, `feature_mask` |
| `utils.noise` | `add_gaussian_noise` |
| `utils.standardize` | `compute_scaler_params`, `apply_standardization`, `inverse_standardization` |
| `utils.preprocessing` | `xr_to_batch1d`, `train_test_split`, `interpolate_initial_condition` |
| `utils.viz` | `plot_3d_attractor`, `plot_state_grid`, `plot_trajectories` |

## Tutorials (Notebooks)

Tutorials are stored as **[Jupytext](https://jupytext.readthedocs.io/) percent-format** `.py` files.

```bash
jupytext --to notebook notebooks/01_model_based_4dvar_L63.py  # convert once
jupyter lab notebooks/01_model_based_4dvar_L63.ipynb           # open
# or open directly in VS Code / JupyterLab with the Jupytext extension
```

| Script | Topic | Prerequisites |
|--------|-------|---------------|
| `01_model_based_4dvar_L63.py` | Classical 4DVar with known L63 dynamics | Python, ODE basics |
| `02_unrolling_vs_fixedpoint_L63.py` | Unrolled backprop vs implicit differentiation | JAX basics |
| `03_4dvarnet_L63.py` | Full 4DVarNet on Lorenz-63 | 01, 02 |
| `04_4dvarnet_2d_demo.py` | 4DVarNet on 2D synthetic data | 03 |
| `05_end_to_end_L63.py` | End-to-end pipeline with xarray utilities | 03 |
| `06_end_to_end_L96.py` | End-to-end pipeline on Lorenz-96 | 05 |
| `07_prior_pretraining_L63.py` | Prior pretraining before full 4DVarNet | 03 |
| `08_classical_4dvar_vs_4dvarnet_L63.py` | Side-by-side comparison of classical vs learned 4DVar | 01, 03 |

## Mathematical Reference

See the [`docs/`](docs/) directory for the full mathematical reference
(17 chapters, rewritten in v0.4 to a DA-textbook style):

**Foundation** (1–3)
- [§1 Problem Setting](docs/01_problem_setting.md)
- [§2 Observation Model](docs/02_observation_model.md)
- [§3 Dynamical Model](docs/03_dynamical_model.md)

**The seven analysis methods** (4–10)
- [§4 Optimal Interpolation / BLUE](docs/04_oi_blue.md)
- [§5 3DVar](docs/05_threedvar.md)
- [§6 Strong-constraint 4DVar](docs/06_strong_4dvar.md)
- [§7 Weak-constraint 4DVar](docs/07_weak_4dvar.md)
- [§8 Incremental 4DVar with CVT](docs/08_incremental_4dvar.md)
- [§9 4DVarNet — Learned 4DVar](docs/09_4dvarnet.md)
- [§10 Amortized Inference](docs/10_amortized_inference.md)

**Cross-cutting** (11–14)
- [§11 Observation Operators](docs/11_observation_operators.md)
- [§12 Adjoint Methods](docs/12_adjoint_methods.md)
- [§13 Posterior Covariance](docs/13_posterior_covariance.md)
- [§14 Six-Step Inference Cycle](docs/14_six_step_cycle.md)

**End-to-end examples** (15–17)
- [§15 Lorenz Examples](docs/15_lorenz_examples.md)
- [§16 SSH Reconstruction](docs/16_ssh_example.md)
- [§17 Methane Single-Overpass](docs/17_methane_example.md)

- [Notation summary](docs/notation.md)
- [References](docs/references.md)

## Design Docs

See [`docs/design/`](docs/design/) for the architecture and decision log
behind v0.4.0:

- [`design/vision.md`](docs/design/vision.md) — identity, DA hierarchy, six-step cycle
- [`design/architecture.md`](docs/design/architecture.md) — three-layer
  stack, pipekit integration, package layout
- [`design/boundaries.md`](docs/design/boundaries.md) — ownership map and
  roadmap (Epics 0–13)
- [`design/decisions.md`](docs/design/decisions.md) — design decisions D1–D16
- [`design/api/`](docs/design/api/) — protocol + class contracts by layer
- [`design/examples/`](docs/design/examples/) — patterns and use case
  walkthroughs

## References

- Fablet et al. (2021). [Learning Variational Data Assimilation Models and Solvers](https://doi.org/10.1029/2021MS002572). *JAMES*.
- [CIA-Oceanix/DLGD2022](https://github.com/CIA-Oceanix/DLGD2022) — Workshop tutorials (PyTorch).
- [CIA-Oceanix/4dvarnet-starter](https://github.com/CIA-Oceanix/4dvarnet-starter) — Production 4DVarNet.

## License

MIT — see [LICENSE](LICENSE).