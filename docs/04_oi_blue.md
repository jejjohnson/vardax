---
numbering:
  enumerator: "4.%s"
---

# Optimal Interpolation / BLUE

For linear-Gaussian state estimation, the posterior is Gaussian in
closed form and the analysis is a single matrix expression. This is
the Best Linear Unbiased Estimator (BLUE), known in geosciences as
Optimal Interpolation (OI). It is the first method you should reach
for when the regime allows it (Decision D16).

## Setup

The prior is

$$p(\vect{x}) = \mathcal{N}(\vect{x}; \gvect{\mu}_x, \gmat{\Sigma}_x), \text{ with } \vect{x} \in \mathbb{R}^{D_x}$$ (eq-oi-prior)

where $\gvect{\mu}_x$ is the background mean and $\gmat{\Sigma}_x$ the background-error covariance. The likelihood is the observation model [](#eq-obs-model) with a linear $\mat{H}$,

$$p(\vect{y} \mid \vect{x}) = \mathcal{N}(\vect{y}; \mat{H} \vect{x}, \gmat{\Sigma}_\varepsilon), \text{ with } \vect{y} \in \mathbb{R}^{D_y} \text{ and } \mat{H} \in \mathbb{R}^{D_y \times D_x}$$ (eq-oi-likelihood)

where $\gmat{\Sigma}_\varepsilon$ is the observation-error covariance. By the standard Gaussian conjugacy result, the posterior is

$$p(\vect{x} \mid \vect{y}) = \mathcal{N}(\vect{x}; \vect{x}^\star, \gmat{\Sigma}_{x \mid y})$$ (eq-oi-posterior)

with mean

$$\vect{x}^\star = \gvect{\mu}_x + \mat{G}(\vect{y} - \mat{H} \gvect{\mu}_x), \text{ with } \mat{G} = \gmat{\Sigma}_x \mat{H}^\top (\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1}$$ (eq-blue-mean)

where $\mat{G}$ is the Kalman gain and $\vect{y} - \mat{H} \gvect{\mu}_x$ the innovation, and covariance

$$\gmat{\Sigma}_{x \mid y} = (\mat{I} - \mat{G} \mat{H}) \gmat{\Sigma}_x = (\gmat{\Sigma}_x^{-1} + \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H})^{-1}$$ (eq-blue-cov)

The shapes are $\vect{x}, \gvect{\mu}_x, \vect{x}^\star \in \mathbb{R}^{D_x}$, $\vect{y} \in \mathbb{R}^{D_y}$, $\gmat{\Sigma}_x, \gmat{\Sigma}_{x \mid y} \in \mathbb{R}^{D_x \times D_x}$, $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$ and $\mat{G} \in \mathbb{R}^{D_x \times D_y}$.

The two forms of $\gmat{\Sigma}_{x \mid y}$ are equivalent by Sherman-Morrison-Woodbury, and which one to compute depends on which is cheaper.

## Two equivalent expressions, two computations

The Kalman gain in [](#eq-blue-mean) requires inverting a $D_y \times D_y$ matrix (observation-space form). The second form of [](#eq-blue-cov) instead inverts the posterior precision $\gmat{\Sigma}_x^{-1} + \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}$, a $D_x \times D_x$ matrix (state-space form).

Pick the form whose matrix is smaller:

| Regime | Form | Inversion size |
|---|---|---|
| $D_y \ll D_x$ (sparse observations) | Kalman gain $\mat{G}$ of [](#eq-blue-mean) | $D_y \times D_y$ |
| $D_y \gg D_x$ (dense observations) | precision $\gmat{\Sigma}_{x \mid y}^{-1}$ of [](#eq-blue-cov) | $D_x \times D_x$ |

For SSH altimetry with sparse along-track samples and a 2D grid, $D_y \ll D_x$ and the observation-space form wins. For a satellite imager with one observation per state cell, $D_y \approx D_x$ and either works.

`OptimalInterpolation` solves the observation-space system $(\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon) \vect{v} = \vect{y} - \mat{H} \gvect{\mu}_x$ by default, and the state-space system $(\gmat{\Sigma}_x^{-1} + \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}) \delta\vect{x} = \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} (\vect{y} - \mat{H} \gvect{\mu}_x)$ with `form="state"`. The block below fuses $D_y = 2000$ pixels into $D_x = 3$ unknowns and prints `True`: both forms give the same analysis.

```{code-block} python
:caption: Solves one BLUE in the observation-space and the state-space form and checks they agree.
import jax
import jax.numpy as jnp
import lineax as lx
import vardax as vdx

# Shapes: S = 1 sample; D_x = 3 unknowns; D_y = 2000 pixels

jax.config.update("jax_enable_x64", True)
key_design, key_obs = jax.random.split(jax.random.PRNGKey(0))
# H, a dense design
design = jax.random.normal(key_design, (2000, 3))  # (D_y, D_x)
# y ~ 𝒩(0, I)
pixels = jax.random.normal(key_obs, (1, 2000))  # (S, D_y)
analyses = {}
for form in ("observation", "state"):
    # today (vardax)
    oi = vdx.OptimalInterpolation(
        obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(design)),
        prior_mean=jnp.zeros(3),
        # Σ_x = I
        prior_cov_op=lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float64)),
        # Σ_ε = 10⁻² I
        obs_cov_op=lx.DiagonalLinearOperator(jnp.full(2000, 1e-2)),
        solver=lx.Cholesky(),
        form=form,
    )
    # x⋆ = μ_x + δx
    analyses[form] = oi(vdx.Batch1D(input=pixels, mask=jnp.ones_like(pixels)))  # (S, D_y) → (S, D_x)
print(bool(jnp.allclose(analyses["observation"], analyses["state"], rtol=1e-8)))
```

The observation-space solve factors a $2000 \times 2000$ matrix here, the state-space one a $3 \times 3$ matrix.

## Why the matrices are never materialised

Both forms require solving a linear system, not inverting a matrix
explicitly. `kernellib` and `gaussx` supply structured operators:

- $\gmat{\Sigma}_x$ as a Matérn covariance: `kernellib.ImplicitKernelOperator(matern, coords)` applies it matrix-free with $O(D_x)$ memory, and `gaussx.Kronecker` exploits separable grids. Both can be passed to a vardax model directly when the state is a single $(D_x,)$ vector; for a $(T + 1, D_x)$ state, wrap the mat-vec as `lx.FunctionLinearOperator(apply, structure, lx.positive_semidefinite_tag)`, as the example below does.
- $\gmat{\Sigma}_\varepsilon$ as `lineax.DiagonalLinearOperator(variances)`, which is diagonal and trivial.
- $\mat{H} \gmat{\Sigma}_x \mat{H}^\top$ is never materialised; only $\vect{v} \mapsto \mat{H} \gmat{\Sigma}_x \mat{H}^\top \vect{v}$ is needed, which is one mat-vec each with $\mat{H}^\top$, $\gmat{\Sigma}_x$ and $\mat{H}$.

The innovation system $(\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon) \vect{v} = \vect{y} - \mat{H} \gvect{\mu}_x$ is then solved by `lineax.CG`, returning $\vect{v} \in \mathbb{R}^{D_y}$ in $O(N_\text{iter} \cdot D_y)$ time where $N_\text{iter}$ is the CG iteration count (typically tens). The analysis [](#eq-blue-mean) is then $\vect{x}^\star = \gvect{\mu}_x + \gmat{\Sigma}_x \mat{H}^\top \vect{v}$, one more mat-vec.

CG runs at a fixed tolerance with a step cap, so its analysis is
approximate, and an ill-conditioned $\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon$ (small $\gmat{\Sigma}_\varepsilon$, smooth
$\gmat{\Sigma}_x$) can exhaust the cap. When the innovation system is small enough to
factor, pass a direct solver instead:
`OptimalInterpolation(..., solver=lx.Cholesky())` (or
`lx.AutoLinearSolver(well_posed=True)`) gives the BLUE to machine
precision, and `posterior()` then uses the same solver for its
covariance.

A full OI on a 2D Matérn prior with 50,000 grid cells and 5,000
observations takes seconds on a single GPU. No materialised matrices,
no $O(D_x^3)$ factorisations.

## Implementation in vardax

The example is a toy along-track problem: $D_x = 20$ cells every 10 km on one track, $D_y = 5$ samples, a Matérn-3/2 prior $\gmat{\Sigma}_x$ with length scale $\ell = 50$ km and standard deviation $\sigma_x = 0.1$ m around a zero climatology, and $\gmat{\Sigma}_\varepsilon = \sigma_\varepsilon^2 \mat{I}$ with $\sigma_\varepsilon = 0.02$ m. The truth is a smooth sine wave and the observations add noise to it. The window has $T = 0$ steps, so the state has shape $(T + 1, D_x) = (1, D_x)$, as `posterior` expects.

```{code-block} python
:caption: Computes the OI analysis [](#eq-blue-mean) and posterior covariance [](#eq-blue-cov) of a toy along-track problem and reads two marginal standard deviations off Σ_x|y.
:label: code-oi-blue-along-track
import einx
import jax
import jax.numpy as jnp
import kernellib as kl
import lineax as lx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 sample; T = 0 steps, so T + 1 = 1 time; D_x = 20 grid cells;
# D_y = 5 along-track samples; D_s = 1 coordinate

key_noise = jax.random.PRNGKey(0)
# ℓ = 50 km, σ_x = 0.1 m, σ_ε = 0.02 m
length_scale, prior_std, obs_std = 50.0, 0.1, 0.02
state_structure = jax.ShapeDtypeStruct((1, 20), jnp.float32)


# user
def matern32(cell: Float[Array, " d"], other: Float[Array, " d"]) -> Float[Array, ""]:
    # (sᵢ − sⱼ)²
    squared_difference = (cell - other) ** 2  # (D_s,)
    # ‖sᵢ − sⱼ‖², with a floor so the gradient at r = 0 is finite
    squared_gap = einx.sum("d -> ", squared_difference) + 1e-12  # → ()
    # √3 r, with r = ‖sᵢ − sⱼ‖ / ℓ
    scaled_gap = jnp.sqrt(3.0 * squared_gap) / length_scale  # → ()
    # k(r) = σ_x² (1 + √3 r) exp(−√3 r)
    return prior_std**2 * (1.0 + scaled_gap) * jnp.exp(-scaled_gap)  # → ()


# user
def sample_track(state: Float[Array, "t n"]) -> Float[Array, "t m"]:
    # H x, the cells under the along-track samples
    return einx.dot("t n, m n -> t m", state, along_track)  # (T + 1, D_y)


# user
def apply_prior_cov(state: Float[Array, "t n"]) -> Float[Array, "t n"]:
    # Σ_x xₜ at each time
    return jax.vmap(kernel_op.mv)(state)  # (T + 1, D_x)


# sᵢ, cell centres every 10 km along one track
positions = jnp.linspace(0.0, 190.0, 20)  # (D_x,)
coords = einx.id("n -> n 1", positions)  # (D_x,) → (D_x, D_s)
# rows of I at the sampled cells
along_track = jnp.eye(20)[jnp.array([1, 5, 9, 13, 17])]  # (D_y, D_x)

# not pinned (kernellib)
# Σ_x = [k(sᵢ, sⱼ)]ᵢⱼ, never materialised
kernel_op = kl.ImplicitKernelOperator(matern32, coords)  # (D_x, D_x) op
# today (lineax)
# Σ_x on (T + 1, D_x) states, applied at each time
prior_cov = lx.FunctionLinearOperator(  # ((T + 1) D_x, (T + 1) D_x) PSD op
    apply_prior_cov, state_structure, lx.positive_semidefinite_tag
)
# today (lineax)
along_track_op = lx.FunctionLinearOperator(  # ((T + 1) D_y, (T + 1) D_x) op
    sample_track, state_structure
)
# σ_ε², one per sample
obs_var = jnp.full((1, 5), obs_std**2)  # (T + 1, D_y)
# today (lineax)
# Σ_ε = σ_ε² I
obs_cov = lx.DiagonalLinearOperator(obs_var)  # ((T + 1) D_y, (T + 1) D_y) op

# μ_x = 0, a climatology
climatology_ssh = jnp.zeros((1, 20))  # (T + 1, D_x)
# sᵢ laid out as a state
track_position = einx.id("n 1 -> 1 n", coords)  # (D_x, D_s) → (T + 1, D_x)
# x_true(s) = σ_x sin(2π s / 200 km), a smooth truth
truth = prior_std * jnp.sin(2.0 * jnp.pi * track_position / 200.0)  # (T + 1, D_x)
# y = H x_true + σ_ε ε, ε ~ 𝒩(0, I)
noise = obs_std * jax.random.normal(key_noise, (1, 5))  # (T + 1, D_y)
# user
# H x_true
track_truth = sample_track(truth)  # (T + 1, D_x) → (T + 1, D_y)
obs = track_truth + noise  # (T + 1, D_y)

# today (vardax)
# H x, a linear observation operator
along_track_obs = vdx.LinearObs(operator=along_track_op)
# today (vardax)
model = vdx.OptimalInterpolation(
    obs_op=along_track_obs,
    prior_mean=climatology_ssh,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    solver=lx.Cholesky(),
)
# today (vardax)
batch = vdx.Batch1D(
    input=einx.id("t m -> 1 t m", obs),  # (T + 1, D_y) → (S, T + 1, D_y)
    mask=jnp.ones((1, 1, 5)),
)
# x⋆ = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x), one pass, no iteration
x_star = model(batch)  # → (S, T + 1, D_x)

# today (vardax)
# Σ_x|y = (Σ_x⁻¹ + Hᵀ Σ_ε⁻¹ H)⁻¹, a lazy operator
posterior = model.posterior(batch)
# eᵢ at an observed cell (i = 1) and between samples (i = 3)
observed_cell = jnp.eye(20)[jnp.array([1])]  # (T + 1, D_x)
gap_cell = jnp.eye(20)[jnp.array([3])]  # (T + 1, D_x)
# Σ_x|y eᵢ, one matrix-free product each
observed_column = posterior.cov.mv(observed_cell)  # (T + 1, D_x) → (T + 1, D_x)
gap_column = posterior.cov.mv(gap_cell)  # (T + 1, D_x) → (T + 1, D_x)
# [Σ_x|y]ᵢᵢ = eᵢᵀ Σ_x|y eᵢ
observed_var = einx.dot("t n, t n ->", observed_cell, observed_column)  # → ()
gap_var = einx.dot("t n, t n ->", gap_cell, gap_column)  # → ()

# ‖μ_x − x_true‖
background_error = jnp.linalg.norm(climatology_ssh - truth)  # → ()
# ‖x⋆ − x_true‖
analysis_error = jnp.linalg.norm(x_star[0] - truth)  # → ()
print(f"background error {background_error:.3f}, analysis error {analysis_error:.3f}")
print(f"posterior std {observed_var**0.5:.3f} observed, {gap_var**0.5:.3f} in a gap")
```

It prints `background error 0.316, analysis error 0.095` and `posterior std 0.019 observed, 0.035 in a gap`. The Cholesky solver makes the analysis and the posterior exact; the posterior solves the same innovation system as the analysis and never applies $\gmat{\Sigma}_x^{-1}$.

The `posterior.cov` is an `AbstractLinearOperator` representing $\gmat{\Sigma}_{x \mid y}$.
Sampling from the posterior is a mat-vec in $\gmat{\Sigma}_{x \mid y}^{1/2}$; the diagonal
of $\gmat{\Sigma}_{x \mid y}$ (marginal variances) is one more mat-vec family. Nothing is
ever materialised.

## Validation

`OptimalInterpolation` is the canonical baseline. Every more
sophisticated method must agree with it in the linear-Gaussian limit
(Decision D14 invariant). `tests/test_linear_gaussian_agreement.py` enforces this for `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar` and `IncrementalFourDVar`. The block below checks the first two against the BLUE, with $f$ the identity and a single time for `StrongFourDVar`. It prints `True` twice. Both use a direct `cov_solver` and 500 minimiser steps, because the smooth prior makes $\gmat{\Sigma}_x^{-1}$ ill-conditioned and BFGS needs about 260 steps here.

```{code-block} python
:caption: Checks that ThreeDVar minimising [](#eq-3dvar-cost) and StrongFourDVar minimising [](#eq-strong4dvar-cost) both reach the BLUE [](#eq-blue-mean).
:label: code-oi-blue-agreement
import equinox as eqx

# Shapes: S = 1 sample; T = 0 steps, so T + 1 = 1 time; D_x = 20 grid cells;
# D_y = 5 along-track samples


# user
class StaticForward(eqx.Module):
    """Trivial dynamics, f(x) = x: a one-time window is the 3D problem."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        return state


# today (vardax)
# 𝒥(x) = ½ ‖x − μ_x‖²_Σ_x⁻¹ + ½ ‖y − H x‖²_Σ_ε⁻¹, minimised iteratively
three_dvar = vdx.ThreeDVar(
    obs_op=model.obs_op,
    prior_mean=climatology_ssh,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    cov_solver=lx.AutoLinearSolver(well_posed=True),
    max_steps=500,
)
# StrongFourDVar steps (D_x,) states, so Σ_x, H and Σ_ε act on one time
cell_structure = jax.ShapeDtypeStruct((20,), jnp.float32)
# not pinned (kernellib)
# today (lineax)
# Σ_x on (D_x,) states, applied by the kernellib operator's mv
cell_cov = lx.FunctionLinearOperator(  # (D_x, D_x) PSD op
    kernel_op.mv, cell_structure, lx.positive_semidefinite_tag
)
# today (lineax)
along_track_matrix = lx.MatrixLinearOperator(along_track)  # (D_y, D_x) op
# today (vardax)
# H x on (D_x,) states
cell_obs = vdx.LinearObs(operator=along_track_matrix)
# σ_ε², one per sample
cell_obs_var = jnp.full(5, obs_std**2)  # (D_y,)
# today (lineax)
# Σ_ε = σ_ε² I on (D_y,) observations
cell_obs_cov = lx.DiagonalLinearOperator(cell_obs_var)  # (D_y, D_y) op
# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ‖y − H x₀‖²_Σ_ε⁻¹, with f = I
strong_4dvar = vdx.StrongFourDVar(
    forward=StaticForward(),
    obs_op=cell_obs,
    prior_mean=climatology_ssh[0],
    prior_cov_op=cell_cov,
    obs_cov_op=cell_obs_cov,
    cov_solver=lx.AutoLinearSolver(well_posed=True),
    max_steps=500,
)
# x⋆ = argminₓ 𝒥(x), by 3DVar
iterative_3d = three_dvar(batch)  # → (S, T + 1, D_x)
# x₀⋆ = argmin 𝒥(x₀), by strong 4DVar
iterative_4d = strong_4dvar(batch)  # → (S, D_x)

print(bool(jnp.allclose(x_star, iterative_3d, atol=1e-3)))
print(bool(jnp.allclose(x_star[:, 0], iterative_4d, atol=1e-3)))
```

If a new method disagrees with `OptimalInterpolation` in this limit,
something is wrong. The agreement is the definition of correctness.

## Connection to the Kalman filter

The Kalman filter analysis step is OI applied at each cycle,

$$\begin{aligned} \text{forecast: } & \vect{x}_{t \mid t-1} = \mat{F}_t \vect{x}_{t-1 \mid t-1}, \text{ and } \gmat{\Sigma}_{x,t \mid t-1} = \mat{F}_t \gmat{\Sigma}_{x,t-1 \mid t-1} \mat{F}_t^\top + \gmat{\Sigma}_\eta \\ \text{analysis: } & \mat{G}_t = \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top (\mat{H} \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1} \\ & \vect{x}_{t \mid t} = \vect{x}_{t \mid t-1} + \mat{G}_t (\vect{y}_t - \mat{H} \vect{x}_{t \mid t-1}) \\ & \gmat{\Sigma}_{x,t \mid t} = (\mat{I} - \mat{G}_t \mat{H}) \gmat{\Sigma}_{x,t \mid t-1} \end{aligned}$$ (eq-kalman-filter)

where $\vect{x}_{t \mid t-1} \in \mathbb{R}^{D_x}$ and $\gmat{\Sigma}_{x,t \mid t-1} \in \mathbb{R}^{D_x \times D_x}$ are the forecast mean and covariance, $\vect{x}_{t \mid t}$ and $\gmat{\Sigma}_{x,t \mid t}$ the analysis mean and covariance, $\mat{F}_t \in \mathbb{R}^{D_x \times D_x}$ the linear dynamics, $\gmat{\Sigma}_\eta \in \mathbb{R}^{D_x \times D_x}$ the model-error covariance and $\mat{G}_t \in \mathbb{R}^{D_x \times D_y}$ the gain. The analysis lines are [](#eq-blue-mean) and the first form of [](#eq-blue-cov).

The analysis step is exactly `OptimalInterpolation` with `prior_mean` set to $\vect{x}_{t \mid t-1}$,
`prior_cov_op` to $\gmat{\Sigma}_{x,t \mid t-1}$, `obs_cov_op` to $\gmat{\Sigma}_\varepsilon$ and `obs_op` to $\mat{H}$. The forecast step lives
in `filterax` (it's about propagating uncertainty through dynamics, not
about inference per se).

This means `filterax`'s Kalman filter can be built compositionally:
forecast via `filterax`, analysis via vardax `OptimalInterpolation`,
update covariance via `filterax`. The interfaces line up because both
libraries respect the same protocols.

For a linear-Gaussian state-space model, `filterax.KalmanSmoother` runs the whole recursion exactly: the filter above, then the Rauch–Tung–Striebel smoother backwards, over irregular observation times, with `KalmanSmoother.from_sde` building the transitions from a gaussx SDE kernel.

## When OI is the right answer

Use `OptimalInterpolation` when **all** of:

- $h$ is linear (or the linearisation around $\gvect{\mu}_x$ is good enough that
  the residual nonlinearity is dominated by observation noise)
- $\gmat{\Sigma}_x$ and $\gmat{\Sigma}_\varepsilon$ are Gaussian
- The posterior is unimodal (no symmetry-breaking, no bistable
  regimes)
- A single timestep (no dynamics) or the dynamics are linear

If any fails:

- Nonlinear $h$ → `ThreeDVar` (chapter 5)
- Dynamics → `StrongFourDVar` / `IncrementalFourDVar` (chapters 6, 8)
- Non-Gaussian likelihood → `AmortizedPosterior` with score head (chapter 10)
- Multimodal posterior → ensemble methods via `filterax`

`OptimalInterpolation` does not check that `obs_op` is linear: it linearises a nonlinear operator at $\gvect{\mu}_x$ and returns a single extended-Kalman update, so a nonlinear $h$ belongs in `ThreeDVar`.

## See also

- Chapter 2: the observation model (the linear $\mat{H}$ assumption)
- Chapter 5: `ThreeDVar` (the nonlinear extension)
- Chapter 13: posterior covariance (the closed-form $\gmat{\Sigma}_{x \mid y}$ is the simplest case)
- Design doc: [`design/decisions.md#d16`](design/decisions.md#d16-blue-oi-as-a-first-class-method)

## References

- Lorenc, A. (1981). *A global three-dimensional multivariate
  statistical interpolation scheme.* MWR 109(4).
- Daley, R. (1991). *Atmospheric Data Analysis*. CUP. Ch. 4.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning*. Ch. 2.7.
