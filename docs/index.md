# vardax

**JAX-native data assimilation inference.**

*Formerly `fourdvarjax` — renamed to `vardax`.*

!!! info "Status"
    All seven analysis methods ship and all seven satisfy
    `pipekit_cycle.AnalysisStep` via `.as_analysis_step()`. Import them
    from the package root (`import vardax as vdx; vdx.StrongFourDVar`):
    there is no `vardax.models` or `vardax.obs_operators` module. The
    `vardax.adjoints`, `vardax.amortized` and `vardax.cycle` submodules
    are bound too, and a few symbols live only there — `AbstractAdjoint`
    is `vardax.adjoints.AbstractAdjoint`, not `vardax.AbstractAdjoint`.
    Individual components may still be stubs where a chapter says so
    (the flow and score heads of
    [amortized inference](10_amortized_inference.md) are the current
    ones); see [boundaries](design/boundaries.md) for what vardax will
    and will not own.
    *(The package was previously published as `fourdvarjax` v0.1.x;
    `vardax` is now the canonical name.)*

vardax provides the seven classical and modern DA analysis methods as
peer [`pipekit_cycle.AnalysisStep`](design/pipekit_composition.md)
implementations:

| Class | Method | Use when |
|---|---|---|
| `OptimalInterpolation` | BLUE / OI — closed-form linear-Gaussian | Linear $H$, Gaussian $B$ / $R$. The right default. |
| `ThreeDVar` | 3D variational, nonlinear $H$ | Snapshot inversion |
| `StrongFourDVar` | Strong-constraint 4DVar, control = $x_0$ | Multi-time, exact dynamics |
| `WeakFourDVar` | Weak-constraint 4DVar, control = $(x_0, \boldsymbol{\eta})$ | Multi-time, model error active |
| `IncrementalFourDVar` | Gauss–Newton outer + CG inner | Operational fast path |
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
x^* = \underset{x, \boldsymbol{\eta}}{\arg\min} \quad
\underbrace{\tfrac{1}{2} \Vert x - x_b \Vert^2_{B^{-1}}}_{\text{background term}} +
\underbrace{\tfrac{1}{2}\sum_{t=0}^{T} \Vert y_t - H_t(M_t(x; \boldsymbol{\eta})) \Vert^2_{R_t^{-1}}}_{\text{observation term}}
\quad \Big[ + \underbrace{\tfrac{1}{2}\sum_{t=1}^{T} \Vert \eta_t \Vert^2_{Q_t^{-1}}}_{\text{model-error term}} \Big].
$$

Different methods specialise differently:

- $T = 0$ + linear $H$ + Gaussian $B/R$ → `OptimalInterpolation` (closed form)
- $T = 0$ + nonlinear $H$ → `ThreeDVar`
- $T > 0$, model-error term absent → `StrongFourDVar` / `IncrementalFourDVar`
- $T > 0$, model-error term active → `WeakFourDVar`
- Learned $\varphi_\theta$ replacing $\Vert x - x_b \Vert^2_{B^{-1}}$ + learned inner solver → `FourDVarNet`
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

A single forward pass: no iteration, no convergence criterion. The analysis
is the $B$/$R$-weighted combination of background and observations, so it
beats both.

```python
import jax, jax.numpy as jnp, lineax as lx
import vardax as vdx

N = 64
truth = jnp.sin(jnp.linspace(0.0, 6.0, N))[None]  # (1, N)
x_b = jnp.zeros((1, N))  # flat background
y = truth + 0.3 * jax.random.normal(jax.random.PRNGKey(1), (1, N))


def pd(shape, var):
    op = lx.DiagonalLinearOperator(jnp.full(shape, var))
    return lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)


oi = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=x_b,
    prior_cov_op=pd((1, N), 1.0),  # B
    obs_cov_op=pd((1, N), 0.3**2),  # R
)

batch = vdx.Batch1D(input=y[None], mask=jnp.ones((1, 1, N)), target=truth[None])
analysis = oi(batch)[0]
# RMS error: background 0.72, observations 0.33, analysis 0.31
```

For a correlated background, pass any `lineax` operator — a dense kernel
matrix, or a structured operator from
[`gaussx`](https://github.com/jejjohnson/gaussx) — in place of the diagonal
one. Posterior covariance is a separate concern, handled by the
[posterior adapters](13_posterior_covariance.md) rather than a method on
the model.

## Quickstart — Incremental 4DVar

Gauss–Newton outer loops around a conjugate-gradient inner solve, on a
window of Lorenz-63 observed every other step.

```python
import diffrax as dfx, jax, jax.numpy as jnp, lineax as lx
import vardax as vdx

prior = vdx.DynTrajectory(
    model=vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0),
    adjoint=dfx.DirectAdjoint(),  # the inner solve needs jvp as well as vjp
)
_, states = vdx.simulate_lorenz63(
    jax.random.PRNGKey(0), dt=0.01, n_steps=2000, n_burn_in=1000
)
xs = states[500 : 500 + 5 * 21 : 5]
mask = jnp.zeros((21, 3)).at[::2].set(1.0)
y = xs + 0.5 * jax.random.normal(jax.random.PRNGKey(3), xs.shape)
x_b = xs[0] + jnp.array([1.5, -1.5, 1.5])

model = vdx.IncrementalFourDVar(
    forward=prior.as_forward_model(dt=0.05),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=x_b,
    prior_cov_op=pd((3,), 4.0),  # B, using pd() from the block above
    obs_cov_op=pd((3,), 0.25),  # R
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20),
)

batch = vdx.Batch1D(input=(y * mask)[None], mask=mask[None], target=xs[None])
x_star = model(batch)[0]
# background error 2.60 → analysis error 0.84
```

## Cycling any model through `pipekit_cycle.DACycle`

All seven methods satisfy `pipekit_cycle.AnalysisStep` via
`.as_analysis_step()` — the orchestration code is identical:

```python
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=vdx.AveragingKernel(...),
    analysis_step=model.as_analysis_step(),  # any of the seven
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)

result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

Swap `OptimalInterpolation` for `IncrementalFourDVar` for `FourDVarNet`
by changing the `analysis_step` slot. Nothing else in the pipeline
changes.

What each method's built-in adapter does with the forecast differs,
though, and the shapes have to line up: the 4DVar adapters re-solve
against the background the model was constructed with and expect a
batched observation window, so a cycle that uses each forecast as the
next background supplies a small custom `AnalysisStep`. Notebook
[16](notebooks/16_cycled_assimilation_L63.py) shows that pattern
end to end.

## Documentation

This site has two main sections:

- **[Mathematical Reference](01_problem_setting.md)** — 21 chapters
  covering the Bayesian foundation (1–3), each of the seven analysis
  methods (4–10), cross-cutting concerns (11–14), end-to-end examples
  on Lorenz, SSH, methane and latent-space DA (15–18), then physical
  models, uncertainty quantification and OceanBench (19–21).
- **[Tutorials](notebooks/index.md)** — 17 executable walkthroughs on
  Lorenz-63 and Lorenz-96, run at documentation build time.
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
