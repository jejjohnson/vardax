# vardax

**JAX-native data assimilation inference.**

*Formerly `fourdvarjax` — renamed to `vardax`.*

vardax provides the seven classical and modern DA analysis methods as
peer [`pipekit_cycle.AnalysisStep`](design/pipekit_composition.md)
implementations:

| Class | Method | Use when |
|---|---|---|
| `OptimalInterpolation` | BLUE / OI — closed-form linear-Gaussian | Linear $H$, Gaussian $B$ / $R$. The right default. |
| `ThreeDVar` | 3D variational, nonlinear $H$ | Snapshot inversion |
| `StrongFourDVar` | Strong-constraint 4DVar, control = $x_0$ | Multi-time, exact dynamics |
| `WeakFourDVar` | Weak-constraint 4DVar, control = $(x_0, \boldsymbol{\eta})$ | Multi-time, model error active |
| `IncrementalFourDVar` | GN outer + CG inner + CVT | Operational fast path |
| `FourDVarNet` | Learned $\varphi_\theta$ + learned $\Phi_\phi$ | Learned variant of 4DVar |
| `AmortizedPosterior` | Direct $q_\phi(x \mid y)$ head | Real-time / many-event regimes |

Gradients through dynamics and the inner minimiser are composed via
[`diffrax.AbstractAdjoint`](12_adjoint_methods.md) and
[`optimistix.AbstractAdjoint`](12_adjoint_methods.md) — no in-house
adjoint code. The Bolte 2023 one-step method appears as
`vardax.adjoints.OneStepAdjoint`, an `optimistix.AbstractAdjoint`
subclass targeting upstream contribution.

## The single equation

Every analysis method in vardax is a special case of

$$
x^* = \underset{x,\,\boldsymbol{\eta}}{\arg\min}\;
  \underbrace{\tfrac{1}{2}\|x - x_b\|^2_{B^{-1}}}_{\text{background term}}
  + \underbrace{\tfrac{1}{2}\sum_{t=0}^{T} \|y_t - H_t(M_t(x; \boldsymbol{\eta}))\|^2_{R_t^{-1}}}_{\text{observation term}}
  \;[\,+\;\underbrace{\tfrac{1}{2}\sum_{t=1}^{T} \|\eta_t\|^2_{Q_t^{-1}}}_{\text{model-error term}}\,].
$$

Different methods specialise differently:

- $T = 0$ + linear $H$ + Gaussian $B/R$ → `OptimalInterpolation` (closed form)
- $T = 0$ + nonlinear $H$ → `ThreeDVar`
- $T > 0$, model-error term absent → `StrongFourDVar` / `IncrementalFourDVar`
- $T > 0$, model-error term active → `WeakFourDVar`
- Learned $\varphi_\theta$ replacing $\|x - x_b\|^2_{B^{-1}}$ + learned inner solver → `FourDVarNet`
- Direct posterior head $q_\phi(x \mid y)$ → `AmortizedPosterior`

See the [Problem Setting](01_problem_setting.md) chapter for the full derivation.

## Installation

```bash
git clone https://github.com/jejjohnson/vardax.git
cd vardax
uv sync --all-extras
```

vardax is not yet on PyPI; install from the checkout.

## Quickstart — Optimal Interpolation

```python
import gaussx as gx
import lineax as lx
from vardax.models import OptimalInterpolation
from vardax.obs_operators import LinearObs

model = OptimalInterpolation(
    obs_op=LinearObs(H_mat=along_track_op),
    prior_mean=climatology_ssh,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=100.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(altika_variances),
)

# Single forward pass — no iteration, no convergence criterion
x_star = model(batch)
posterior = model.posterior(batch)
```

## Quickstart — Incremental 4DVar with control-variable transform

```python
import diffrax as dfx
from vardax.models import IncrementalFourDVar
from vardax import IncrementalConfig

model = IncrementalFourDVar(
    forward=somax_model,
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),
    prior_mean=x_b,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=10.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(obs_uncertainty),
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
    forward_adjoint=dfx.BacksolveAdjoint(),    # constant memory through dynamics
)

x_star = model(batch)
posterior = model.posterior(batch)
```

## Cycling any model through `pipekit_cycle.DACycle`

All seven methods satisfy `pipekit_cycle.AnalysisStep` via
`.as_analysis_step()` — the orchestration code is identical:

```python
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=AveragingKernel(...),
    analysis_step=model.as_analysis_step(),   # any of the seven
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)

result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

Swap `OptimalInterpolation` for `IncrementalFourDVar` for `FourDVarNet`
by changing the `analysis_step` slot. Nothing else in the pipeline
changes.

## Documentation

This site has two main sections:

- **[Mathematical Reference](01_problem_setting.md)** — 17 chapters
  covering the Bayesian foundation (1–3), each of the seven analysis
  methods (4–10), cross-cutting concerns (11–14), and end-to-end
  examples on Lorenz / SSH / methane (15–17).
- **[Design Docs](design/README.md)** — architecture, API contracts,
  ecosystem boundaries, and the decision log (D1–D16). The "why"
  behind the "what".

## Ecosystem

vardax does not own forward models, optimisers, ODE solvers, ensemble
methods, or structured linear algebra. It composes them:

| Concern | Owner |
|---|---|
| Geophysical forward models | [`somax`](https://github.com/jejjohnson/somax) |
| Atmospheric transport / methane | `plumax` |
| Optimisers + adjoints | [`optimistix`](https://github.com/patrick-kidger/optimistix) |
| ODE integration + adjoints | [`diffrax`](https://github.com/patrick-kidger/diffrax) |
| Linear solvers | [`lineax`](https://github.com/patrick-kidger/lineax) |
| Structured operators | [`gaussx`](https://github.com/jejjohnson/gaussx) |
| Ensemble methods | `filterax` |
| Operator composition + DA cycle protocols | [`pipekit`](https://github.com/jejjohnson/pipekit) + `pipekit-cycle` |

See [Boundaries](design/boundaries.md) for the full ownership map.

## License

MIT.
