# vardax

> **JAX-native data assimilation inference** — seven peer analysis methods built on `pipekit-cycle` protocols.

[![CI Tests](https://github.com/jejjohnson/vardax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/vardax/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jejjohnson/vardax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/vardax)
[![PyPI version](https://img.shields.io/pypi/v/vardax.svg)](https://pypi.org/project/vardax/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-jejjohnson.github.io%2Fvardax-blue)](https://jejjohnson.github.io/vardax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`vardax` is the **data assimilation inference layer for JAX**. It provides
seven classical and modern DA analysis methods as peer implementations of
the [`pipekit_cycle.AnalysisStep`](https://github.com/jejjohnson/pipekit)
protocol, with gradients composed from
[`diffrax`](https://github.com/patrick-kidger/diffrax) (ODE adjoints) and
[`optimistix`](https://github.com/patrick-kidger/optimistix) (minimiser
adjoints).

> [!IMPORTANT]
> **Status — v0.1.x ships 4DVarNet only.** The seven-method DA hierarchy
> described in the docs is the **v0.4 design target**, scoped against the
> equinox migration roadmap (Epics 0–13). The current package implements
> the `FourDVarNet1D` / `FourDVarNet2D` learned variant on Flax NNX —
> see [Quickstart (v0.1.x)](#quickstart-v01x---4dvarnet) below for what
> runs today. The seven-method API is documented in
> [`docs/`](https://jejjohnson.github.io/vardax/) so the architecture is
> public while the implementation catches up.
>
> *(The package was previously published as `fourdvarjax` v0.1.x;
> `vardax` is now the canonical name.)*

---

## The seven methods

`vardax` is engineered around the data-assimilation hierarchy: the same
problem expressed seven different ways depending on regime. All seven
satisfy `pipekit_cycle.AnalysisStep` via `.as_analysis_step()`.

| Class | Method | Use when |
|---|---|---|
| `OptimalInterpolation` | BLUE / OI — closed-form linear-Gaussian | Linear $H$, Gaussian $B$ / $R$. The right default. |
| `ThreeDVar` | 3D variational, nonlinear $H$ | Snapshot inversion |
| `StrongFourDVar` | Strong-constraint 4DVar, control = $x_0$ | Multi-time, exact dynamics |
| `WeakFourDVar` | Weak-constraint 4DVar, control = $(x_0, \boldsymbol{\eta})$ | Multi-time, model error active |
| `IncrementalFourDVar` | GN outer + CG inner + CVT | Operational fast path |
| `FourDVarNet` | Learned $\varphi_\theta$ + learned $\Phi_\phi$ | Learned variant of 4DVar |
| `AmortizedPosterior` | Direct $q_\phi(x \mid y)$ head | Real-time / many-event regimes |

Every method is a specialisation of the single unified cost

$$
x^* = \underset{x,\,\boldsymbol{\eta}}{\arg\min}\;
  \tfrac{1}{2}\|x - x_b\|^2_{B^{-1}}
  + \tfrac{1}{2}\sum_{t=0}^{T} \|y_t - H_t(M_t(x; \boldsymbol{\eta}))\|^2_{R_t^{-1}}
  \;[\,+\;\tfrac{1}{2}\sum_{t=1}^{T} \|\eta_t\|^2_{Q_t^{-1}}\,].
$$

The choice of method picks which terms are active and how the
minimisation proceeds (closed form / iterative / learned / amortized).
See [`docs/01_problem_setting.md`](docs/01_problem_setting.md) for the
derivation.

## Composable adjoints

`vardax` does not own adjoint code. Gradients through ODE dynamics come
from `diffrax.AbstractAdjoint`; gradients through inner minimisers come
from `optimistix.AbstractAdjoint`. The Bolte 2023 one-step method is
packaged as `vardax.adjoints.OneStepAdjoint`, an
`optimistix.AbstractAdjoint` subclass targeting upstream contribution.

```python
import diffrax as dfx
import optimistix as optx
from vardax.models import StrongFourDVar

model = StrongFourDVar(
    forward=somax_model,
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),
    prior_mean=x_b,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
    minimiser=optx.NonlinearCG(rtol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),  # IFT at the optimum
    forward_adjoint=dfx.BacksolveAdjoint(),  # constant-memory adjoint ODE
)
```

A user who wants memory-efficient operational 4DVar reaches for
`BacksolveAdjoint`. A user training `FourDVarNet` reaches for
`OneStepAdjoint`. `vardax` owns the DA algorithm; the differentiation
strategy is a slot.

## Cycling any model through `pipekit_cycle.DACycle`

```python
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=AveragingKernel(...),
    analysis_step=model.as_analysis_step(),  # any of the seven
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)

result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

Swap `OptimalInterpolation` for `IncrementalFourDVar` for `FourDVarNet`
by changing the `analysis_step` slot. Orchestration code is unchanged.

---

## Installation

`vardax` is on PyPI as of v0.2.0:

```bash
uv add vardax                       # core
uv add 'vardax[viz,jlab]'           # plus matplotlib + Jupyter
uv add 'vardax[all]'                # everything
```

Or from a workspace checkout:

```bash
git clone https://github.com/jejjohnson/vardax.git
cd vardax
make install                        # uv sync --all-extras + pre-commit
```

Requires Python ≥ 3.12, < 3.14.

## Quickstart (v0.1.x) — 4DVarNet

What the current package implements. Reconstruct masked Lorenz-63
trajectories from noisy partial observations:

```python
import jax, jax.numpy as jnp
import vardax as vdx
from vardax.adjoints import OneStepAdjoint

# Build a 4DVarNet for 1D state vectors (Lorenz-63: N=3)
model = vdx.FourDVarNet1D(
    state_dim=3,
    n_time=20,
    latent_dim=8,
    hidden_dim=16,
    n_solver_steps=15,
    solver_adjoint=OneStepAdjoint(),  # Bolte et al. (2023), O(1) memory
    key=jax.random.PRNGKey(0),
)

# batch.input: (B, T, N) masked observations
# batch.mask:  (B, T, N) binary obs mask
# batch.target: (B, T, N) ground truth (for training)
batch = vdx.Batch1D(input=y_masked, mask=mask, target=y_truth)

# Forward pass — minimises the variational cost via the learned solver
x_recon = model(batch)
```

The full 4DVarNet API today:

| Layout | Shape | Use case | Model |
|---|---|---|---|
| 1D | `(B, T, N)` | State vectors (Lorenz-63, time series) | `FourDVarNet1D` |
| 2D | `(B, T, H, W)` | Spatiotemporal fields (SSH, SST) | `FourDVarNet2D` |
| 2D multivar | `(B, T, C, H, W)` | Multi-channel 2D fields | `BilinAEPrior2DMultivar` |

Priors: `BilinAEPrior1D/2D/2DMultivar`, `ConvAEPrior1D`, `MLPAEPrior1D`,
`IdentityPrior`.

Gradient modes: `"unrolled"` (O(K) memory, standard backprop),
`"one_step"` (O(1) memory, Bolte et al. 2023),
`"implicit"` (O(1) memory, IFT via fixed point).

See [`docs/09_4dvarnet.md`](docs/09_4dvarnet.md) for the math and
[`notebooks/`](notebooks/) for end-to-end tutorials on Lorenz-63 and
Lorenz-96.

## Documentation

- **[Site](https://jejjohnson.github.io/vardax/)** — built with mkdocs
  Material; deployed on push to `main`.
- **[Mathematical reference](docs/)** — 17 chapters covering the
  Bayesian foundation (1–3), each of the seven analysis methods (4–10),
  cross-cutting concerns (observation operators, adjoints, posterior
  covariance, six-step cycle — 11–14), and end-to-end examples on
  Lorenz / SSH / methane (15–17).
- **[Design docs](docs/design/)** — architecture (three-layer stack),
  decisions D1–D16, boundaries / roadmap (Epics 0–13), and integration
  patterns with `somax` / `plumax` / `gaussx` / `filterax` /
  `pipekit-cycle`.

## Ecosystem

`vardax` does not own forward models, optimisers, ODE solvers, ensemble
methods, or structured linear algebra. It composes them:

| Concern | Owner |
|---|---|
| Geophysical forward models (SWM, QG, primitive eq.) | [`somax`](https://github.com/jejjohnson/somax) |
| Atmospheric transport / methane (Gaussian plume, Lagrangian, Eulerian, RTM) | `plumax` |
| ODE / SDE integration + adjoints | [`diffrax`](https://github.com/patrick-kidger/diffrax) |
| Optimisers + adjoints | [`optimistix`](https://github.com/patrick-kidger/optimistix) |
| Linear solvers (CG, GMRES, Lanczos) | [`lineax`](https://github.com/patrick-kidger/lineax) |
| Structured operators (Matérn, Kronecker, LowRank) | [`gaussx`](https://github.com/jejjohnson/gaussx) |
| Ensemble methods (EnKF / EnKS / EnKI) | `filterax` |
| Operator composition + DA cycle protocols | [`pipekit`](https://github.com/jejjohnson/pipekit) + `pipekit-cycle` |
| Coordinate-aware arrays | [`coordax`](https://github.com/neuralgcm/coordax) |

See [`docs/design/boundaries.md`](docs/design/boundaries.md) for the
full ownership map.

## Repository structure

```
vardax/
├── pyproject.toml                  ← uv / hatchling, PEP 735 groups
├── Makefile                        ← developer workflow targets
├── mkdocs.yml                      ← docs site config
├── src/vardax/                     ← installable package
│   ├── __init__.py                 ← public API re-exports
│   └── _src/
│       ├── _types.py               ← Batch1D, Batch2D, Batch2DMultivar, LSTMState1D/2D
│       ├── costs.py                ← obs_cost, prior_cost, variational_cost
│       ├── grad_mod.py             ← ConvLSTM gradient modulator
│       ├── model.py                ← FourDVarNet1D/2D
│       ├── priors.py               ← BilinAE / ConvAE / MLP / Identity priors
│       ├── solver.py               ← solve_vardanet, unrolled / one-step / implicit
│       ├── training.py             ← train_step, eval_step
│       └── utils/                  ← Lorenz-63/96 simulators, masks, plotting
├── docs/                           ← math reference (17 chapters) + design docs
│   ├── *.md                        ← chapters 01–17
│   ├── design/                     ← architecture, API contracts, decisions
│   └── notation.md, references.md
├── notebooks/                      ← jupytext tutorials on Lorenz-63 / 96
├── tests/                          ← pytest suite (147 tests)
└── .github/workflows/              ← CI, lint, format, typecheck, pages, release-please
```

## Developer workflow

The repo uses [PEP 735 dependency groups](https://peps.python.org/pep-0735/)
(installable via `uv sync --group <name>`):

| Group | Contents |
|---|---|
| `dev` | `test` + `lint` + `typecheck` + pre-commit, nbstripout |
| `test` | pytest, pytest-cov, coverage, xarray, numpy, matplotlib |
| `lint` | ruff |
| `typecheck` | ty |
| `docs` | mkdocs Material + mkdocstrings + mkdocs-jupyter + jupytext |

Makefile targets:

```bash
make install         # uv sync --all-extras + pre-commit install
make uv-test         # pytest -v
make uv-test-cov     # pytest with coverage XML
make uv-format       # ruff format + fix
make uv-lint         # ruff check + ty check
make docs-serve      # mkdocs serve at http://127.0.0.1:8000
make docs-deploy     # mkdocs gh-deploy
```

Pre-commit hooks run `ruff format`, `ruff check --fix`, and basic file
hygiene. Conventional commits required (release-please cuts versions).

## License

MIT. See [LICENSE](LICENSE).

## References

- Fablet et al. (2021). [Learning Variational Data Assimilation Models and Solvers](https://doi.org/10.1029/2021MS002572). *JAMES*.
- Fablet et al. (2023). [Multimodal 4DVarNets for the reconstruction of sea surface dynamics from NADIR and wide-swath altimetry](https://doi.org/10.1109/TGRS.2023.3268006). *IEEE TGRS*.
- Bolte, Pauwels & Vaiter (2023). [One-step differentiation of iterative algorithms](https://arxiv.org/abs/2305.13768). *NeurIPS*.
- Courtier, Thépaut & Hollingsworth (1994). *A strategy for operational implementation of 4D-Var, using an incremental approach.* *QJRMS*.
- Carrassi et al. (2018). *Data assimilation in the geosciences: An overview of methods, issues, and perspectives.* *WIREs Climate Change*.
- Full bibliography in [`docs/references.md`](docs/references.md).
