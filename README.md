<div align="center">

# vardax

**Variational data assimilation in JAX — seven analysis methods, one cost function, zero hand-written adjoints.**

[![CI Tests](https://github.com/jejjohnson/vardax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/vardax/actions/workflows/ci.yml)
[![Docs](https://github.com/jejjohnson/vardax/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/vardax/)
[![codecov](https://codecov.io/gh/jejjohnson/vardax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/vardax)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Documentation](https://jejjohnson.github.io/vardax/) ·
[Tutorials](https://jejjohnson.github.io/vardax/notebooks/) ·
[Maths reference](https://jejjohnson.github.io/vardax/01_problem_setting/) ·
[Design docs](https://jejjohnson.github.io/vardax/design/README/)

</div>

---

Optimal interpolation, 3DVar, strong- and weak-constraint 4DVar, incremental
4DVar, 4DVarNet and amortized posteriors are the *same estimation problem*
solved under different assumptions. `vardax` implements all seven as peer
[`equinox`](https://github.com/patrick-kidger/equinox) modules over a shared
set of seams — observation operator, background covariance, forward model,
minimiser, adjoint — so switching method is a constructor change, not a
rewrite.

```python
x_star = model(batch)  # every method
step = model.as_analysis_step()  # every method, cycled identically
```

Gradients are composed, not coded: dynamics differentiate through
[`diffrax`](https://github.com/patrick-kidger/diffrax) adjoints, inner
minimisers through [`optimistix`](https://github.com/patrick-kidger/optimistix)
adjoints, and covariances stay as [`lineax`](https://github.com/patrick-kidger/lineax)
operators so nothing is ever densified unless you ask.

## The seven methods

All seven ship today, and all seven satisfy
[`pipekit_cycle.AnalysisStep`](https://github.com/jejjohnson/pipekit) via
`.as_analysis_step()`.

| Class | Method | Control | Reach for it when |
|---|---|---|---|
| `OptimalInterpolation` | BLUE / OI, closed form | $x$ | Linear $H$, Gaussian $B$/$R$. The right default. |
| `ThreeDVar` | 3D variational | $x$ | Snapshot inversion with nonlinear $H$ |
| `StrongFourDVar` | Strong-constraint 4DVar | $x_0$ | A window of data, dynamics trusted |
| `WeakFourDVar` | Weak-constraint 4DVar | $(x_0, \boldsymbol{\eta})$ | A window of data, model error active |
| `IncrementalFourDVar` | Gauss–Newton outer, CG inner, CVT | $\delta x_0$ | The operational fast path |
| `FourDVarNet1D` / `FourDVarNet2D` | Learned prior $\varphi_\theta$ + learned solver | $x$ | Dense training data, no trusted model |
| `AmortizedPosterior` | Direct head $q_\phi(x \mid y)$ | — | Real-time, many-event regimes |

Each is a specialisation of one cost:

$$
x^* = \underset{x,\,\boldsymbol{\eta}}{\arg\min}\;
  \underbrace{\tfrac{1}{2}\|x - x_b\|^2_{B^{-1}}}_{\text{background}}
  + \underbrace{\tfrac{1}{2}\sum_{t=0}^{T} \|y_t - H_t(M_t(x; \boldsymbol{\eta}))\|^2_{R_t^{-1}}}_{\text{observations}}
  \;\Big[+\;\underbrace{\tfrac{1}{2}\sum_{t=1}^{T} \|\eta_t\|^2_{Q_t^{-1}}}_{\text{model error}}\Big].
$$

Which terms are active, and how the minimisation proceeds (closed form,
iterative, learned, amortized), is the only thing that differs. The
[problem setting](https://jejjohnson.github.io/vardax/01_problem_setting/)
chapter derives it.

## Install

`vardax` is not on PyPI yet. Install from the repository with
[uv](https://docs.astral.sh/uv/), which resolves the pinned git
dependencies (`gaussx`, `geonnax`, `pipekit`) declared in
`[tool.uv.sources]`:

```bash
git clone https://github.com/jejjohnson/vardax.git
cd vardax
make install          # uv sync --all-extras + pre-commit hooks
```

To depend on it from another uv project:

```toml
[project]
dependencies = ["vardax"]

[tool.uv.sources]
vardax = { git = "https://github.com/jejjohnson/vardax" }
```

Requires Python ≥ 3.12, < 3.14. Extras: `data`, `viz`, `exp`, `jlab`,
`examples`, `persist`, `train`, `all`.

## Quickstart — 4DVar on Lorenz-63

A window of noisy, half-missing observations; the true chaotic dynamics as
the forward model; BFGS on the variational cost with the adjoint supplied by
`jax.grad` through `diffrax`.

```python
import jax, jax.numpy as jnp, lineax as lx
import vardax as vdx

# Truth, observations (every other step, σ = 0.5), background (σ = 2.0)
_, states = vdx.simulate_lorenz63(
    jax.random.PRNGKey(0), dt=0.01, n_steps=2000, n_burn_in=1000
)
x_true = states[500 : 500 + 5 * 21 : 5]  # (21, 3) window at dt = 0.05
k_obs, k_bg = jax.random.split(jax.random.PRNGKey(1))
mask = jnp.zeros((21, 3)).at[::2].set(1.0)
y = x_true + 0.5 * jax.random.normal(k_obs, x_true.shape)
x_b = x_true[0] + 2.0 * jax.random.normal(k_bg, (3,))


def pd(variance):  # a PD lineax operator
    op = lx.DiagonalLinearOperator(jnp.full(3, variance))
    return lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)


# The Lorenz-63 vector field as an ODE prior, exposed as a forward model
prior = vdx.DynTrajectory(model=vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0))

model = vdx.StrongFourDVar(
    forward=prior.as_forward_model(dt=0.05),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=x_b,
    prior_cov_op=pd(2.0**2),  # B
    obs_cov_op=pd(0.5**2),  # R
)

batch = vdx.Batch1D(input=(y * mask)[None], mask=mask[None], target=x_true[None])
x0_analysis = model(batch)[0]

print(jnp.linalg.norm(x_b - x_true[0]))  # background error  1.62
print(jnp.linalg.norm(x0_analysis - x_true[0]))  # analysis error    0.41
```

Correlated background errors are a one-line change: swap `pd(...)` for a
`gaussx` Matérn operator and the conjugate-gradient solve adapts.

## Swap the method, keep the pipeline

Every model exposes `.as_analysis_step()`, so the cycling code never sees
which method it is running:

```python
import pipekit, pipekit_cycle as pc

cycle = vdx.VarDACycle(
    forward=prior.as_forward_model(dt=0.05),
    obs_op=vdx.MaskedIdentity(),
    model=model,  # any of the seven
    obs_source=pipekit.Lambda(load_window),  # called with the cycle index
    n_steps=n_cycles,
)

x_final, state = cycle(x_init, pc.DAState(t=0.0, cycle_count=0, obs_err_cov=R))
forecasts, analyses = zip(*cycle.history)
```

Each cycle forecasts the carrier one step, hands the forecast and that
cycle's observations (`NaN` marks a gap) to the analysis step, and keeps the
pair in `cycle.history`. `VarSmootherCycle` does the same over fixed-lag
windows. Both return plain `pipekit_cycle` objects, so whatever orchestrates
pipekit operators orchestrates vardax.
[Tutorial 16](docs/notebooks/16_cycled_assimilation_L63.py) runs the loop
for real, including the forecast-as-background analysis step that makes each
cycle's $x_b$ the previous forecast.

## What else is in the box

| Area | Symbols | Notes |
|---|---|---|
| Observation operators | `MaskedIdentity`, `LinearObs`, `InterpObs`, `AveragingKernel`, `MultiInstrumentFusion`, `interp_obs_from_coords` | Gappy grids, off-grid interpolation with a passing adjoint test, retrieval averaging kernels, and multi-instrument fusion that satisfies the precision-addition identity |
| Posterior | `LaplaceCovariance`, `GaussNewtonHessian`, `EnsembleCovariance`, `Posterior`, `GaussianMarkLikelihood` | Covariance as a `lineax` operator, never a dense matrix you did not ask for |
| Adjoints | `OneStepAdjoint`, `KStepAdjoint`, `ImplicitAdjoint`, `RecursiveCheckpointAdjoint`, `to_optimistix_adjoint` | The Bolte (2023) one-step rule packaged as an `optimistix.AbstractAdjoint`, targeting upstream contribution |
| Dynamical priors | `DynTrajectory`, `DynIncrements`, `DynamicalPrior` | An ODE prior that doubles as a forward model (`.as_forward_model(dt)`) or a trajectory loss (`.bind(ts)`) |
| Reduced bases | `LinearBasis`, `CompositeBasis`, `eof_basis`, `fourier_basis`, `rbf_basis`, `wavelet_basis` | Control-variable transforms for incremental 4DVar |
| Amortized inference | `AmortizedPosterior`, `MLPObsEncoder`, `RegressionHead`, `ConditionalFlowHead`, `ScoreDiffusionHead` | Train once on simulations, assimilate in a forward pass |
| Validation gates | `assert_posterior_agreement`, `assert_adjoint_calibrated`, `simulation_based_calibration` | The go/no-go checks that decide whether a fast method may replace a trusted one |
| Forward adapters | `SoftBoundedForward` | Identity inside a box, smooth saturation outside, so a line-search trial far off the attractor cannot overflow an explicit ODE step and NaN-poison the solve |

## Tutorials

Seventeen executable walkthroughs, all
[jupytext](https://jupytext.readthedocs.io/) sources under
`docs/notebooks/` that are **run at documentation build time** — the
figures on the site are produced by the code in this repository, not
committed snapshots.

| # | Tutorial | What it shows |
|---|---|---|
| 01–02 | [Model-based 4DVar](docs/notebooks/01_model_based_4dvar_L63.py), [unrolling vs fixed point](docs/notebooks/02_unrolling_vs_fixedpoint_L63.py) | The reference 4DVar picture, then the learned solver against it |
| 03–04 | [4DVarNet on L63](docs/notebooks/03_4dvarnet_L63.py), [2-D demo](docs/notebooks/04_4dvarnet_2d_demo.py) | Training the learned variant in 1-D and on spatiotemporal fields |
| 05–07 | [L63 pipeline](docs/notebooks/05_end_to_end_L63.py), [L96 pipeline](docs/notebooks/06_end_to_end_L96.py), [prior pre-training](docs/notebooks/07_prior_pretraining_L63.py) | Simulation → patching → masking → training, end to end |
| 08 | [Classical 4DVar vs 4DVarNet](docs/notebooks/08_classical_4dvar_vs_4dvarnet_L63.py) | Gradient descent on the cost against the learned solver |
| 09–11 | [Parameter estimation](docs/notebooks/09_param_estimation_L63.py), [bilevel optimisation](docs/notebooks/10_bilevel_opt_L63.py), [gradient learning](docs/notebooks/11_gradient_learning_L63.py) | Learning ODE parameters, cost weights, and the update rule itself |
| 12 | [Weak-constraint 4DVar](docs/notebooks/12_weak_constraint_4dvar_L63.py) | A biased model: recovered model-error increments, the $Q$ dial from weak to strong, and the same cost in trajectory space |
| 13 | [Neural closures on two-level L96](docs/notebooks/13_neural_closure_L96.py) | The unresolved coupling learned offline and online through the ODE solve, forecast skill, then used as a 4DVar forward model |
| 14 | [Posterior uncertainty](docs/notebooks/14_posterior_uncertainty_L63.py) | Laplace and Gauss–Newton covariances against the exact Hessian, push-forward along the window, ensemble spread, z-scores and SBC |
| 15 | [Observation operators](docs/notebooks/15_observation_operators.py) | Masks vs selection matrices, off-grid interpolation, averaging-kernel bias, and multi-instrument fusion verified numerically |
| 16 | [Cycled assimilation](docs/notebooks/16_cycled_assimilation_L63.py) | Forecast–analysis loops: spin-up, the error equilibrium and its scalar prediction, tuning $B$ by innovations |
| 17 | [Amortized posterior](docs/notebooks/17_amortized_posterior_L63.py) | Simulation-based training against a multi-start 4DVar oracle, wall-clock comparison, and an honest reading of the three gates |

## Documentation

- **[Maths reference](https://jejjohnson.github.io/vardax/01_problem_setting/)** — 21 chapters: the
  Bayesian foundation (1–3), each analysis method (4–10), cross-cutting
  concerns (11–14), worked examples on Lorenz, sea-surface height, methane
  and latent-space DA (15–18), then physical models, uncertainty
  quantification and OceanBench (19–21).
- **[API reference](https://jejjohnson.github.io/vardax/api/)** — every public symbol, grouped by
  seam, generated from the docstrings.
- **[Design docs](https://jejjohnson.github.io/vardax/design/README/)** — architecture, ownership
  boundaries, the decision log, and the pipekit composition story.

## Ecosystem

`vardax` owns the DA algorithms and nothing else. Forward models,
optimisers, ODE solvers, ensemble methods and structured linear algebra
are composed in:

| Concern | Owner |
|---|---|
| Geophysical forward models (SWM, QG, primitive equations) | [`somax`](https://github.com/jejjohnson/somax) |
| Atmospheric transport and methane retrieval | `plumax` |
| ODE / SDE integration and adjoints | [`diffrax`](https://github.com/patrick-kidger/diffrax) |
| Minimisers and their adjoints | [`optimistix`](https://github.com/patrick-kidger/optimistix) |
| Linear solvers (CG, GMRES, Lanczos) | [`lineax`](https://github.com/patrick-kidger/lineax) |
| Structured operators (Matérn, Kronecker, low rank) | [`gaussx`](https://github.com/jejjohnson/gaussx) |
| Ensemble filters and smoothers (EnKF / EnKS / EnKI) | `filterax` |
| Operator composition and DA cycle protocols | [`pipekit`](https://github.com/jejjohnson/pipekit) + `pipekit-cycle` |
| Neural building blocks | [`geonnax`](https://github.com/jejjohnson/geonnax) |

See [boundaries](docs/design/boundaries.md) for the full ownership map.

## Repository layout

```
src/vardax/
├── __init__.py              ← the public API; everything below is private
└── _src/
    ├── models/              ← OI, 3DVar, strong / weak / incremental 4DVar
    ├── model.py             ← FourDVarNet1D / FourDVarNet2D (equinox)
    ├── amortized/           ← encoders, heads, AmortizedPosterior
    ├── obs_operators/       ← masked, linear, interp, averaging kernel, fusion
    ├── posterior/           ← Laplace, Gauss–Newton, ensemble, container
    ├── adjoints/            ← one-step, k-step, optimistix mapping
    ├── basis/               ← reduced bases and control-variable transforms
    ├── cycle/               ← VarDACycle, VarSmootherCycle
    ├── priors.py            ← autoencoder and Lorenz priors
    ├── priors_dynamical.py  ← DynTrajectory, DynIncrements
    ├── costs.py             ← the variational cost and its decompositions
    ├── solver.py            ← unrolled / one-step / fixed-point solvers
    ├── training.py          ← train and eval steps
    └── utils/               ← Lorenz simulators, SoftBoundedForward, validation gates, plotting
docs/                        ← 21 maths chapters, API pages, design docs, 17 executed tutorials
tests/                       ← 376 tests across 29 modules
```

## Development

```bash
make install        # uv sync --all-extras + pre-commit
make uv-test        # pytest -v
make uv-test-cov    # pytest with coverage XML
make uv-format      # ruff format + ruff check --fix
make uv-lint        # ruff check + ty check
make uv-pre-commit  # all hooks on all files
make docs-serve     # mkdocs at http://127.0.0.1:8000
```

Four checks must pass before a commit lands:

```bash
uv run pytest -q
uv run --group lint ruff check .            # entire repo, not just src/
uv run --group lint ruff format --check .
uv run --group typecheck ty check src/vardax
```

Dependency groups follow [PEP 735](https://peps.python.org/pep-0735/)
(`dev`, `test`, `lint`, `typecheck`, `docs`); activate one with
`uv sync --group <name>`. Commits are conventional — release-please cuts
the versions. Committed `.ipynb` files are rejected by a pre-commit hook:
edit the jupytext `.py` source instead.

## Citation and references

- Fablet et al. (2021). [Learning Variational Data Assimilation Models and Solvers](https://doi.org/10.1029/2021MS002572). *JAMES*.
- Fablet et al. (2023). [Multimodal 4DVarNets for the reconstruction of sea surface dynamics from NADIR and wide-swath altimetry](https://doi.org/10.1109/TGRS.2023.3268006). *IEEE TGRS*.
- Bolte, Pauwels & Vaiter (2023). [One-step differentiation of iterative algorithms](https://arxiv.org/abs/2305.13768). *NeurIPS*.
- Courtier, Thépaut & Hollingsworth (1994). A strategy for operational implementation of 4D-Var, using an incremental approach. *QJRMS*.
- Carrassi et al. (2018). Data assimilation in the geosciences: an overview of methods, issues, and perspectives. *WIREs Climate Change*.
- Full bibliography in [`docs/references.md`](docs/references.md).

## License

MIT. See [LICENSE](LICENSE).
