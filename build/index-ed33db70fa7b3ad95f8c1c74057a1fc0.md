# vardax

**JAX-native data assimilation inference.**

*Formerly `fourdvarjax` — renamed to `vardax`.*

```{note} Status
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
```

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
| `FourDVarNet1D` / `FourDVarNet2D` | Learned $\varphi_\theta$ + learned $\Phi_\phi$ | Learned variant of 4DVar |
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
- Learned $\varphi_\theta$ replacing $\Vert x - x_b \Vert^2_{B^{-1}}$ + learned inner solver → `FourDVarNet1D` / `FourDVarNet2D`
- Direct posterior head $q_\phi(x \mid y)$ → `AmortizedPosterior`

See the [Problem Setting](01_problem_setting.md) chapter for the full derivation.

## Installation

```bash
git clone https://github.com/jejjohnson/vardax.git
cd vardax
uv sync --all-extras
```

vardax is not yet on PyPI; install from the checkout.

## Quickstart: optimal interpolation

A single forward pass: no iteration, no convergence criterion. The analysis
is the $B$/$R$-weighted combination of background and observations, so it
beats both.

```{code-block} python
:caption: Computes the OI analysis [](#eq-blue-mean) of a noisy sine wave from a flat background.
:label: code-index-oi-quickstart
import einx
import jax
import jax.numpy as jnp
import lineax as lx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 sample; T = 1 time; N = 64 grid cells

key_noise = jax.random.PRNGKey(1)
state_structure = jax.ShapeDtypeStruct((1, 64), jnp.float32)

# sᵢ, cell positions on [0, 6]
positions = jnp.linspace(0.0, 6.0, 64)  # (N,)
# sᵢ laid out as a state
grid = einx.id("n -> 1 n", positions)  # (N,) → (T, N)
# x_true = sin s
truth = jnp.sin(grid)  # (T, N)
# x_b = 0, a flat background
background = jnp.zeros((1, 64))  # (T, N)
# σ_o ε, σ_o = 0.3, ε ~ 𝒩(0, I)
noise = 0.3 * jax.random.normal(key_noise, (1, 64))  # (T, N)
# y = x_true + σ_o ε
obs = truth + noise  # (T, N)

# today (lineax)
# B = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (T N, T N) op
# today (lineax)
# R = σ_o² I; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (T N, T N) op
    0.09 * prior_cov, lx.positive_semidefinite_tag
)
# today (vardax)
# H(x) = m ⊙ x and K = B Hᵀ (H B Hᵀ + R)⁻¹
model = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# today (vardax)
batch = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", obs),  # (T, N) → (S, T, N)
    mask=jnp.ones((1, 1, 64)),
)
# x* = x_b + K (y − H x_b), one pass, no iteration
analysis = model(batch)  # → (S, T, N)


# user
def rms_error(state: Float[Array, "t n"]) -> Float[Array, ""]:
    # (x − x_true)²
    squared_error = (state - truth) ** 2  # (T, N)
    # √(mean (x − x_true)²)
    return jnp.sqrt(einx.mean("t n ->", squared_error))  # → ()


print(
    f"RMS error: background {rms_error(background):.2f}, "
    f"observations {rms_error(obs):.2f}, analysis {rms_error(analysis[0]):.2f}"
)
```

It prints `RMS error: background 0.72, observations 0.33, analysis 0.31`.

For a correlated background, pass any `lineax` operator in place of the identity, such as a dense kernel matrix or a structured operator from [`gaussx`](https://github.com/jejjohnson/gaussx). Posterior covariance is a separate concern, handled by the [posterior adapters](13_posterior_covariance.md) rather than a method on the model.

## Quickstart: incremental 4DVar

Gauss-Newton outer loops around a conjugate-gradient inner solve, on a window of Lorenz-63 observed every other step.

```{code-block} python
:caption: Minimises the incremental cost [](#eq-incremental-cost) through the Gauss-Newton system [](#eq-gn-system) on a Lorenz-63 [](#eq-lorenz63) window.
:label: code-index-incremental-quickstart
import diffrax as dfx

# Shapes: S = 1 window; K = 2,000 steps; T = 21 times; N = 3 state components

key = jax.random.PRNGKey(0)
key_truth, key_noise = jax.random.split(key)
state_structure = jax.ShapeDtypeStruct((3,), jnp.float32)

# today (vardax)
# f(s) = (σ (y − x), x (ρ − z) − y, x y − β z)
lorenz = vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
# today (vardax)
# DirectAdjoint, because the CG inner solve needs jvp as well as vjp (issue #148)
dynamics = vdx.DynTrajectory(model=lorenz, adjoint=dfx.DirectAdjoint())
# today (vardax)
# Mₜ, steps of Δt = 0.05
forward = dynamics.as_forward_model(dt=0.05)
# today (vardax)
# a run at Δt = 0.01 after 1,000 burn-in steps
_, run = vdx.simulate_lorenz63(  # → (K, N)
    key_truth, dt=0.01, n_steps=2000, n_burn_in=1000
)
# xₜ, every fifth state of the run, so Δt = 0.05 apart
truth = run[500:605:5]  # (K, N) → (T, N)
# m, every other time observed
mask = jnp.zeros((21, 3)).at[::2].set(1.0)  # (T, N)
# σ_o ε, σ_o = 0.5, ε ~ 𝒩(0, I)
noise = 0.5 * jax.random.normal(key_noise, (21, 3))  # (T, N)
# xₜ + σ_o ε
noisy_truth = truth + noise  # (T, N)
# yₜ = m ⊙ (xₜ + σ_o ε)
obs = mask * noisy_truth  # (T, N)
# x_b = x_true(t₀) + (1.5, −1.5, 1.5)
background = truth[0] + jnp.array([1.5, -1.5, 1.5])  # (N,)

# today (lineax)
identity = lx.IdentityLinearOperator(state_structure)  # (N, N) op
# today (lineax)
# B = 4 I
prior_cov = lx.TaggedLinearOperator(4.0 * identity, lx.positive_semidefinite_tag)
# today (lineax)
# R = σ_o² I
obs_cov = lx.TaggedLinearOperator(0.25 * identity, lx.positive_semidefinite_tag)
# today (vardax)
config = vdx.IncrementalConfig(n_outer=3, n_inner=20)
# today (vardax)
# δJ(δx) = ½ ‖x⁽ᵏ⁾ + δx − x_b‖²_{B⁻¹} + ½ Σₜ ‖dₜ⁽ᵏ⁾ − H′ₜ M′ₜ δx‖²_{R⁻¹}
model = vdx.IncrementalFourDVar(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=config,
)
# today (vardax)
window = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", obs),  # (T, N) → (S, T, N)
    mask=einx.id("t n -> 1 t n", mask),  # (T, N) → (S, T, N)
)
# 𝐇_GN δx* = −∇J(x⁽ᵏ⁾) by CG, then x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + δx*
analysis = model(window)  # → (S, N)

# ‖x_b − x_true(t₀)‖
background_error = jnp.linalg.norm(background - truth[0])  # → ()
# ‖x₀* − x_true(t₀)‖
analysis_error = jnp.linalg.norm(analysis[0] - truth[0])  # → ()
print(f"background error {background_error:.2f}, analysis error {analysis_error:.2f}")
```

It prints `background error 2.60, analysis error 0.79`.

## Cycling any model through `pipekit_cycle.DACycle`

All seven methods satisfy `pipekit_cycle.AnalysisStep` via `.as_analysis_step()`, so the orchestration code is identical for each of them. The block below continues the incremental quickstart: it observes the whole Lorenz-63 state at every time of the window, and cycles an OI analysis around each forecast.

```{code-block} python
:caption: Cycles the OI analysis [](#eq-blue-mean) around Lorenz-63 [](#eq-lorenz63) forecasts with `pipekit_cycle.DACycle`.
:label: code-index-dacycle
import pipekit as pk
import pipekit_cycle as pc

# Shapes: C = 20 cycles; T = 21 times; N = 3 state components

# yₜ = xₜ + σ_o ε at every time, one per cycle
cycle_obs = truth + noise  # (T, N)
# today (vardax)
# x_a = x_f + K (y − H x_f), with the forecast x_f as the background
oi = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# today (pipekit)
# cycle c forecasts to t = c + 1, so it reads y_{c+1}
obs_source = pk.Lambda(lambda cycle: cycle_obs[cycle + 1], name="cycle_obs")
# today (pipekit_cycle)
da_cycle = pc.DACycle(
    forward_model=forward,
    obs_op=vdx.MaskedIdentity(),
    # any of the seven
    analysis_step=oi.as_analysis_step(),
    obs_source=obs_source,
    n_steps=20,
)
# today (pipekit_cycle)
initial_state = pc.DAState(t=0.0, cycle_count=0)
# x_a at t_C, after C forecast-analysis cycles from x_b
final_analysis, final_state = da_cycle(background, initial_state)  # (N,) → (N,)
# ‖x_a − x_true(t_C)‖
final_error = jnp.linalg.norm(final_analysis - truth[-1])  # → ()
print(f"{final_state.cycle_count} cycles, final error {final_error:.2f}")
```

It prints `20 cycles, final error 0.54`.

Swap `OptimalInterpolation` for `IncrementalFourDVar` for `FourDVarNet1D`
by changing the `analysis_step` slot. Nothing else in the pipeline
changes.

What each method's built-in adapter does with the forecast differs,
though, and the shapes have to line up: the 4DVar adapters re-solve
against the background the model was constructed with and expect a
batched observation window, so a cycle that uses each forecast as the
next background supplies a small custom `AnalysisStep`. Notebook
[16](notebooks/16_cycled_assimilation_L63.ipynb) shows that pattern
end to end.

## Documentation

This site has two main sections:

- **[Mathematical Reference](01_problem_setting.md)** — 22 chapters
  covering the Bayesian foundation (1–3), each of the seven analysis
  methods (4–10), cross-cutting concerns (11–14), end-to-end examples
  on Lorenz, SSH, methane and latent-space DA (15–18), then physical
  models, uncertainty quantification, OceanBench and neural-field
  4D-Var (19–22).
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
