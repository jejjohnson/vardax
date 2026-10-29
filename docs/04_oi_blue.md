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

$$p(x) = \mathcal{N}(x; x_b, B), \text{ with } x \in \mathbb{R}^n$$ (eq-oi-prior)

where $x_b$ is the background mean and $B$ the background-error covariance. The likelihood is the observation model [](#eq-obs-model) with a linear $H$,

$$p(y \mid x) = \mathcal{N}(y; H x, R), \text{ with } H \in \mathbb{R}^{m \times n}$$ (eq-oi-likelihood)

where $R$ is the observation-error covariance. By the standard Gaussian conjugacy result, the posterior is

$$p(x \mid y) = \mathcal{N}(x; x^*, P^*)$$ (eq-oi-posterior)

with mean

$$x^* = x_b + K(y - H x_b), \text{ with } K = B H^\top (H B H^\top + R)^{-1}$$ (eq-blue-mean)

where $K$ is the Kalman gain and $y - H x_b$ the innovation, and covariance

$$P^* = (I - K H) B = (B^{-1} + H^\top R^{-1} H)^{-1}$$ (eq-blue-cov)

The two forms of $P^*$ are equivalent by Sherman-Morrison-Woodbury, and which one to compute depends on which is cheaper.

## Two equivalent expressions, two computations

The Kalman gain in [](#eq-blue-mean) requires inverting an $m \times m$ matrix (observation-space form). The second form of [](#eq-blue-cov) instead inverts the posterior precision $B^{-1} + H^\top R^{-1} H$, an $n \times n$ matrix (state-space form).

Pick the form whose matrix is smaller:

| Regime | Form | Inversion size |
|---|---|---|
| $m \ll n$ (sparse observations) | Kalman gain $K$ of [](#eq-blue-mean) | $m \times m$ |
| $m \gg n$ (dense observations) | precision $(P^*)^{-1}$ of [](#eq-blue-cov) | $n \times n$ |

For SSH altimetry with sparse along-track samples and a 2D grid, $m \ll n$ and the observation-space form wins. For a satellite imager with one observation per state cell, $m \approx n$ and either works.

vardax always solves the observation-space system $(H B H^\top + R) v = y - H x_b$; the state-space form is not implemented.

## Why the matrices are never materialised

Both forms require solving a linear system, not inverting a matrix
explicitly. `gaussx` supplies structured factorisations:

- $B$ as a Matérn covariance: `gaussx.ImplicitKernelOperator(matern, coords)` applies it matrix-free with $O(N)$ memory, and `gaussx.Kronecker` exploits separable grids. gaussx operators do not register `lineax.linearise`, so wrap one as `lx.FunctionLinearOperator(op.mv, structure, lx.positive_semidefinite_tag)` before handing it to a vardax model.
- $R$ as `lineax.DiagonalLinearOperator(variances)`, which is diagonal and trivial.
- $H B H^\top$ is never materialised; only $v \mapsto H B H^\top v$ is needed, which is one mat-vec each with $H^\top$, $B$ and $H$.

The innovation system $(H B H^\top + R) v = y - H x_b$ is then solved by `lineax.CG`, returning $v$ in $O(k \cdot m)$ time where $k$ is the CG iteration count (typically tens). The analysis [](#eq-blue-mean) is then $x^* = x_b + B H^\top v$, one more mat-vec.

CG runs at a fixed tolerance with a step cap, so its analysis is
approximate, and an ill-conditioned $H B H^\top + R$ (small $R$, smooth
$B$) can exhaust the cap. When the innovation system is small enough to
factor, pass a direct solver instead:
`OptimalInterpolation(..., solver=lx.Cholesky())` (or
`lx.AutoLinearSolver(well_posed=True)`) gives the BLUE to machine
precision, and `posterior()` then uses the same solver for its Laplace
covariance.

A full OI on a 2D Matérn prior with 50,000 grid cells and 5,000
observations takes seconds on a single GPU. No materialised matrices,
no $O(n^3)$ factorisations.

## Implementation in vardax

The example is a toy along-track problem: $N = 20$ cells every 10 km on one track, $M = 5$ samples, a Matérn-3/2 prior $B$ with length scale $\ell = 50$ km and standard deviation $\sigma_b = 0.1$ m around a zero climatology, and $R = \sigma_o^2 I$ with $\sigma_o = 0.02$ m. The truth is a smooth sine wave and the observations add noise to it. The state has shape $(T, N)$ with $T = 1$, as `posterior` expects. The block computes the analysis [](#eq-blue-mean) and the posterior covariance [](#eq-blue-cov), and reads two marginal standard deviations off $P^*$.

```python
import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 sample; T = 1 time; N = 20 grid cells; M = 5 along-track samples

key_noise = jax.random.PRNGKey(0)
# ℓ = 50 km, σ_b = 0.1 m, σ_o = 0.02 m
length_scale, prior_std, obs_std = 50.0, 0.1, 0.02
state_structure = jax.ShapeDtypeStruct((1, 20), jnp.float32)


# user
def matern32(cell: Float[Array, " d"], other: Float[Array, " d"]) -> Float[Array, ""]:
    # ‖sᵢ − sⱼ‖², with a floor so the gradient at r = 0 is finite
    squared_gap = einx.sum("d -> ", (cell - other) ** 2) + 1e-12  # → ()
    # √3 r, with r = ‖sᵢ − sⱼ‖ / ℓ
    scaled_gap = jnp.sqrt(3.0 * squared_gap) / length_scale  # → ()
    # k(r) = σ_b² (1 + √3 r) exp(−√3 r)
    return prior_std**2 * (1.0 + scaled_gap) * jnp.exp(-scaled_gap)  # → ()


# user
def sample_track(state: Float[Array, "t n"]) -> Float[Array, "t m"]:
    # H x, the cells under the along-track samples
    return einx.dot("t n, m n -> t m", state, along_track)  # (T, M)


# user
def apply_prior_cov(state: Float[Array, "t n"]) -> Float[Array, "t n"]:
    # B xₜ at each time
    return jax.vmap(kernel_op.mv)(state)  # (T, N)


# sᵢ, cell centres every 10 km along one track
coords = einx.id("n -> n 1", jnp.linspace(0.0, 190.0, 20))  # (N, 1)
# rows of I at the sampled cells
along_track = jnp.eye(20)[jnp.array([1, 5, 9, 13, 17])]  # (M, N)

# today (gaussx)
# B = [k(sᵢ, sⱼ)]ᵢⱼ, never materialised
kernel_op = gx.ImplicitKernelOperator(matern32, coords)  # (N, N) op
# today (lineax)
# B on (T, N) states; wrapped because gaussx operators do not register lx.linearise
prior_cov = lx.FunctionLinearOperator(  # (T N, T N) PSD op
    apply_prior_cov, state_structure, lx.positive_semidefinite_tag
)
# today (lineax)
along_track_op = lx.FunctionLinearOperator(  # (T M, T N) op
    sample_track, state_structure
)
# today (lineax)
# R = σ_o² I
obs_cov = lx.DiagonalLinearOperator(jnp.full((1, 5), obs_std**2))  # (T M, T M) op

# x_b = 0, a climatology
climatology_ssh = jnp.zeros((1, 20))  # (T, N)
# sᵢ laid out as a state
track_position = einx.id("n 1 -> 1 n", coords)  # (N, 1) → (T, N)
# x_true(s) = σ_b sin(2π s / 200 km), a smooth truth
truth = prior_std * jnp.sin(2.0 * jnp.pi * track_position / 200.0)  # (T, N)
# y = H x_true + σ_o ε, ε ~ 𝒩(0, I)
noise = obs_std * jax.random.normal(key_noise, (1, 5))  # (T, M)
obs = sample_track(truth) + noise  # (T, M)

# today (vardax)
model = vdx.OptimalInterpolation(
    obs_op=vdx.LinearObs(operator=along_track_op),
    prior_mean=climatology_ssh,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    solver=lx.Cholesky(),
)
# today (vardax)
batch = vdx.Batch1D(
    input=einx.id("t m -> 1 t m", obs),  # (T, M) → (S, T, M)
    mask=jnp.ones((1, 1, 5)),
)
# x* = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b), one pass, no iteration
x_star = model(batch)  # → (S, T, N)

# today (vardax)
# P* = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹, a lazy operator
posterior = model.posterior(batch)
# eᵢ at an observed cell (i = 1) and between samples (i = 3)
observed_cell = jnp.eye(20)[jnp.array([1])]  # (T, N)
gap_cell = jnp.eye(20)[jnp.array([3])]  # (T, N)
# P* eᵢ, one matrix-free product each
observed_column = posterior.cov.mv(observed_cell)  # (T, N) → (T, N)
gap_column = posterior.cov.mv(gap_cell)  # (T, N) → (T, N)
# σᵢ² = eᵢᵀ P* eᵢ
observed_var = einx.dot("t n, t n ->", observed_cell, observed_column)  # → ()
gap_var = einx.dot("t n, t n ->", gap_cell, gap_column)  # → ()

background_error = jnp.linalg.norm(climatology_ssh - truth)
analysis_error = jnp.linalg.norm(x_star[0] - truth)
print(f"background error {background_error:.3f}, analysis error {analysis_error:.3f}")
print(f"posterior std {observed_var**0.5:.3f} observed, {gap_var**0.5:.3f} in a gap")
```

It prints `background error 0.316, analysis error 0.095` and `posterior std 0.019 observed, 0.035 in a gap`. The Cholesky solver is there because the default CG returns non-finite values for $B^{-1}$ in the posterior of this smooth prior.

The `posterior.cov` is an `AbstractLinearOperator` representing $P^*$.
Sampling from the posterior is a mat-vec in $(P^*)^{1/2}$; the diagonal
of $P^*$ (marginal variances) is one more mat-vec family. Nothing is
ever materialised.

## Validation

`OptimalInterpolation` is the canonical baseline. Every more
sophisticated method must agree with it in the linear-Gaussian limit
(Decision D14 invariant). `tests/test_linear_gaussian_agreement.py` enforces this for `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar` and `IncrementalFourDVar`. The block below continues the one above and checks the first two against the BLUE: `ThreeDVar` minimises [](#eq-3dvar-cost) and `StrongFourDVar` minimises [](#eq-strong4dvar-cost) with $M_t$ the identity and a single time. It prints `True` twice. Both use a direct `cov_solver` and 500 minimiser steps, because the smooth prior makes $B^{-1}$ ill-conditioned and BFGS needs about 260 steps here.

```python
import equinox as eqx

# Shapes: S = 1 sample; T = 1 time; N = 20 grid cells; M = 5 along-track samples


# user
class StaticForward(eqx.Module):
    """Trivial dynamics, Mₜ(x) = x: a one-time window is the 3D problem."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        return state


# today (vardax)
# J(x) = ½ ‖x − x_b‖²_{B⁻¹} + ½ ‖y − H x‖²_{R⁻¹}, minimised iteratively
three_dvar = vdx.ThreeDVar(
    obs_op=model.obs_op,
    prior_mean=climatology_ssh,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    cov_solver=lx.AutoLinearSolver(well_posed=True),
    max_steps=500,
)
# StrongFourDVar steps (N,) states, so B, H and R act on one time
# today (lineax)
cell_cov = lx.FunctionLinearOperator(  # (N, N) PSD op
    kernel_op.mv, jax.ShapeDtypeStruct((20,), jnp.float32), lx.positive_semidefinite_tag
)
# today (vardax)
strong_4dvar = vdx.StrongFourDVar(
    forward=StaticForward(),
    # today (vardax)
    obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(along_track)),
    prior_mean=climatology_ssh[0],
    prior_cov_op=cell_cov,
    obs_cov_op=lx.DiagonalLinearOperator(jnp.full(5, obs_std**2)),
    cov_solver=lx.AutoLinearSolver(well_posed=True),
    max_steps=500,
)
iterative_3d = three_dvar(batch)  # → (S, T, N)
iterative_4d = strong_4dvar(batch)  # → (S, N)
print(bool(jnp.allclose(x_star, iterative_3d, atol=1e-3)))
print(bool(jnp.allclose(x_star[:, 0], iterative_4d, atol=1e-3)))
```

If a new method disagrees with `OptimalInterpolation` in this limit,
something is wrong. The agreement is the definition of correctness.

## Connection to the Kalman filter

The Kalman filter analysis step is OI applied at each cycle,

$$\begin{aligned} \text{forecast: } & x^f_t = M_t x^a_{t-1}, \text{ and } P^f_t = M_t P^a_{t-1} M_t^\top + Q \\ \text{analysis: } & K_t = P^f_t H^\top (H P^f_t H^\top + R)^{-1} \\ & x^a_t = x^f_t + K_t (y_t - H x^f_t) \\ & P^a_t = (I - K_t H) P^f_t \end{aligned}$$ (eq-kalman-filter)

where $x^f_t$ and $P^f_t$ are the forecast mean and covariance, $x^a_t$ and $P^a_t$ the analysis mean and covariance, $M_t$ the linear dynamics and $Q$ the model-error covariance. The analysis lines are [](#eq-blue-mean) and the first form of [](#eq-blue-cov).

The analysis step is exactly `OptimalInterpolation(prior_mean=x^f_t,
prior_cov_op=P^f_t, obs_cov_op=R, obs_op=H)`. The forecast step lives
in `filterax` (it's about propagating uncertainty through dynamics, not
about inference per se).

This means `filterax`'s Kalman filter can be built compositionally:
forecast via `filterax`, analysis via vardax `OptimalInterpolation`,
update covariance via `filterax`. The interfaces line up because both
libraries respect the same protocols.

## When OI is the right answer

Use `OptimalInterpolation` when **all** of:

- $H$ is linear (or the linearisation around $x_b$ is good enough that
  the residual nonlinearity is dominated by observation noise)
- $B$ and $R$ are Gaussian
- The posterior is unimodal (no symmetry-breaking, no bistable
  regimes)
- A single timestep (no dynamics) or the dynamics are linear

If any fails:

- Nonlinear $H$ → `ThreeDVar` (chapter 5)
- Dynamics → `StrongFourDVar` / `IncrementalFourDVar` (chapters 6, 8)
- Non-Gaussian likelihood → `AmortizedPosterior` with score head (chapter 10)
- Multimodal posterior → ensemble methods via `filterax`

`OptimalInterpolation` does not check that `obs_op` is linear: it linearises a nonlinear operator at $x_b$ and returns a single extended-Kalman update, so a nonlinear $H$ belongs in `ThreeDVar`.

## See also

- Chapter 2: the observation model (the linear $H$ assumption)
- Chapter 5: `ThreeDVar` (the nonlinear extension)
- Chapter 13: posterior covariance (the closed-form $P^*$ is the simplest case)
- Design doc: [`design/decisions.md#d16`](design/decisions.md#d16-blue-oi-as-a-first-class-method)

## References

- Lorenc, A. (1981). *A global three-dimensional multivariate
  statistical interpolation scheme.* MWR 109(4).
- Daley, R. (1991). *Atmospheric Data Analysis*. CUP. Ch. 4.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning*. Ch. 2.7.
