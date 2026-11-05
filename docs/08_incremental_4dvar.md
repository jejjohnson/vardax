---
numbering:
  enumerator: "8.%s"
---

# Incremental 4DVar with Control-Variable Transform

Operational 4DVar at ECMWF, NCEP, JMA, and UKMO all follow a common
template: Gauss-Newton outer iterations on the full nonlinear cost,
CG (or Lanczos) inner iterations on a linearised quadratic
subproblem, with a control-variable transform that preconditions the
inner solve. `IncrementalFourDVar` packages this pattern.

It solves the same problem as `StrongFourDVar` (chapter 6) and gives the same answer, but the specialised inner solver is much faster when the prior is structured (Matérn, Kronecker) and the window is long.

## The outer / inner decomposition

Strong-constraint 4DVar minimises [](#eq-strong4dvar-cost). Write $\vect{x}^{(i)}$ for the outer iterate, where $i$ counts the Gauss-Newton outer iterations, starting from $\vect{x}^{(0)} = \gvect{\mu}_x$. The background $\gvect{\mu}_x$ stays fixed across outer loops; only $\vect{x}^{(i)}$ moves. Linearise around $\vect{x}^{(i)}$:

$$f_t(\vect{x}^{(i)} + \delta\vect{x}) \approx f_t(\vect{x}^{(i)}) + \mat{F}_{t \leftarrow 0} \delta\vect{x}$$ (eq-tlm-forward)

$$h_t(\vect{x}_t + \delta\vect{x}_t) \approx h_t(\vect{x}_t) + \mat{H}_t \delta\vect{x}_t$$ (eq-tlm-obs)

where $\mat{F}_{t \leftarrow 0} = \partial f_t / \partial \vect{x}$ at $\vect{x}^{(i)}$ comes from
`jax.linearize` and $\mat{H}_t$ is the tangent of $h_t$ at
$f_t(\vect{x}^{(i)})$. With the innovation
$\vect{r}_t^{(i)} = \vect{y}_t - h_t(f_t(\vect{x}^{(i)}))$, the linearised (quadratic)
cost of an increment $\delta\vect{x} \in \mathbb{R}^{D_x}$ is

$$\delta\mathcal{J}(\delta\vect{x}) = \frac{1}{2} \Vert \vect{x}^{(i)} + \delta\vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{r}_t^{(i)} - \mat{H}_t \mat{F}_{t \leftarrow 0} \delta\vect{x} \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}$$ (eq-incremental-cost)

The shapes are those of chapter 6: $\mat{F}_{t \leftarrow 0} \in \mathbb{R}^{D_x \times D_x}$, $\mat{H}_t \in \mathbb{R}^{D_y \times D_x}$, $\vect{r}_t^{(i)} \in \mathbb{R}^{D_y}$, $\gmat{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}$ and $\gmat{\Sigma}_{\varepsilon,t} \in \mathbb{R}^{D_y \times D_y}$.
The background term measures the *total* departure $\vect{x}^{(i)} + \delta\vect{x} - \gvect{\mu}_x$, not $\delta\vect{x}$ alone. The Gauss-Newton Hessian is

$$\nabla^2_{\vect{x}_0} \mathcal{J} \approx \gmat{\Sigma}_x^{-1} + \sum_{t=0}^{T} (\mat{H}_t \mat{F}_{t \leftarrow 0})^\top \gmat{\Sigma}_{\varepsilon,t}^{-1} (\mat{H}_t \mat{F}_{t \leftarrow 0})$$ (eq-gn-hessian)

which is the 4DVar form of the 3DVar Hessian [](#eq-3dvar-gn-hessian). Setting $\nabla \delta\mathcal{J} = \mathbf{0}$ gives the inner system that the CG (or Lanczos) solver solves for $\delta\vect{x}^\star$:

$$\left(\gmat{\Sigma}_x^{-1} + \sum_{t=0}^{T} (\mat{H}_t \mat{F}_{t \leftarrow 0})^\top \gmat{\Sigma}_{\varepsilon,t}^{-1} (\mat{H}_t \mat{F}_{t \leftarrow 0})\right) \delta\vect{x}^\star = -\gmat{\Sigma}_x^{-1}\left(\vect{x}^{(i)} - \gvect{\mu}_x\right) + \sum_{t=0}^{T} (\mat{H}_t \mat{F}_{t \leftarrow 0})^\top \gmat{\Sigma}_{\varepsilon,t}^{-1} \vect{r}_t^{(i)} = -\nabla_{\vect{x}_0} \mathcal{J}\left(\vect{x}^{(i)}\right)$$ (eq-gn-system)

The right-hand side is the negative gradient [](#eq-strong4dvar-grad) of the full nonlinear cost at the current iterate. On the first outer loop $\vect{x}^{(0)} = \gvect{\mu}_x$, so the background term vanishes and only the innovation term is left. On later loops it does not vanish: dropping it would pull every increment toward $\vect{x}^{(i)}$ instead of $\gvect{\mu}_x$, and the iteration would no longer converge to the 4DVar minimum.

Outer update: $\vect{x}^{(i+1)} = \vect{x}^{(i)} + \delta\vect{x}^\star$. Relinearise.

This pattern converges in a handful of outer iterations (typically 3 to 5) regardless of the strength of the nonlinearity, because each outer step sees a fresh tangent linearisation.

## Control-variable transform (CVT)

The Gauss-Newton Hessian inherits the condition number of
$\gmat{\Sigma}_x^{-1}$: a smooth (long-correlation) $\gmat{\Sigma}_x$ has tiny eigenvalues, which
become huge ones of $\gmat{\Sigma}_x^{-1}$. In state space every Hessian product also
needs $\gmat{\Sigma}_x^{-1} \vect{v}$, which `IncrementalFourDVar` would have to compute by a
nested CG solve.

The CVT changes variables. Let $\mat{L}_x \in \mathbb{R}^{D_x \times D_x}$ be a square root of $\gmat{\Sigma}_x$, $\mat{L}_x \mat{L}_x^\top = \gmat{\Sigma}_x$,
and write the initial state through the whitened control $\vect{z} \in \mathbb{R}^{D_x}$ as

$$\vect{x}_0 = \gvect{\mu}_x + \mat{L}_x \vect{z}$$ (eq-cvt)

The background term becomes $\frac{1}{2} \Vert \vect{x}_0 - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} = \frac{1}{2} \Vert \vect{z} \Vert_2^2$, and the inner system at the outer iterate
$\vect{z}^{(i)}$ becomes

$$\left(\mat{I} + \mat{L}_x^\top \mat{H}_{0:T}^\top \gmat{\Sigma}_{\varepsilon,0:T}^{-1} \mat{H}_{0:T} \mat{L}_x\right) \delta\vect{z}^\star = -\nabla_{\vect{z}} \mathcal{J}\left(\vect{z}^{(i)}\right) = -\vect{z}^{(i)} + \mat{L}_x^\top \mat{H}_{0:T}^\top \gmat{\Sigma}_{\varepsilon,0:T}^{-1} \vect{r}^{(i)}$$ (eq-cvt-system)

where $\mat{H}_{0:T} \in \mathbb{R}^{(T+1) D_y \times D_x}$ stacks $\mat{H}_t \mat{F}_{t \leftarrow 0}$ over the window,
$\gmat{\Sigma}_{\varepsilon,0:T}$ is the block-diagonal covariance of the stacked residual and $\vect{r}^{(i)} \in \mathbb{R}^{(T+1) D_y}$ the stacked innovation. Its eigenvalues are all
$\ge 1$, close to 1 in directions the observations barely see, and no
$\gmat{\Sigma}_x^{-1}$ appears anywhere. The outer update is
$\vect{z}^{(i+1)} = \vect{z}^{(i)} + \delta\vect{z}^\star$, and the analysis is
$\vect{x}_0^\star = \gvect{\mu}_x + \mat{L}_x \vect{z}^\star$.

`IncrementalConfig(cvt=True)` (the default) uses it. $\mat{L}_x$ is
`prior_sqrt_op=` when given (a Cholesky factor, a structured
operator's own square root, or `gaussx.sqrt(prior_cov_op, lanczos_order=k)` for a
large unstructured $\gmat{\Sigma}_x$), else `gaussx.sqrt(prior_cov_op)`: closed form
for diagonal, Kronecker and block-diagonal $\gmat{\Sigma}_x$, a dense
eigendecomposition otherwise. `cvt=False` keeps the state-space solve,
with $\gmat{\Sigma}_x^{-1}$ and $\gmat{\Sigma}_\varepsilon^{-1}$ applied by CG at tolerance
`covariance_tol`.

**How much it helps.** A linear problem with 20 states, 4 times, every
third site observed, a Gaussian-kernel $\gmat{\Sigma}_x$ of length scale 3 sites plus
a diagonal jitter that sets its condition number; inner CG iterations to
reach $10^{-4}$ relative error on the exact analysis:

| $\kappa(\gmat{\Sigma}_x)$ | CVT | state space |
|---:|---:|---:|
| $7 \times 10^1$ | 10 | 8 |
| $7 \times 10^2$ | 9 | 14 |
| $7 \times 10^3$ | 9 | 20 |

The CVT count does not grow with $\kappa(\gmat{\Sigma}_x)$, and each state-space
iteration also runs a nested CG for $\gmat{\Sigma}_x^{-1} \vect{v}$. For a well-conditioned
$\gmat{\Sigma}_x$ the two are comparable.

## Implementation in vardax

The example below is a $Y \times X = 4 \times 5$ grid ($D_x = 20$ cells) over a window of $T = 3$ steps, which holds $T + 1 = 4$ times, with a nonlinear forward $f(\vect{x}) = \vect{x} + 0.2 \sin(\vect{x})$ applied cell by cell.
$\gmat{\Sigma}_x = \gmat{\Sigma}_\text{row} \otimes \gmat{\Sigma}_\text{col}$ is separable, the factors $\gmat{\Sigma}_\text{row} \in \mathbb{R}^{Y \times Y}$ and $\gmat{\Sigma}_\text{col} \in \mathbb{R}^{X \times X}$ each a Gaussian correlation of length scale 1.5 cells (plus a $10^{-1}$ jitter), and the background is $\gvect{\mu}_x = \mathbf{0}$.
Every cell is observed through an averaging kernel [](#eq-averaging-kernel) with $\mat{J}_{\hat{x}} = \mat{I}$, $\vect{w} = 0.8 \cdot \mathbf{1}$ and $\gvect{\mu}_\text{ret} = \mathbf{0}$, so $D_y = D_x$, and $\gmat{\Sigma}_\varepsilon = 0.04 \mat{I}$.

```{code-block} python
:caption: Solves the control-variable system [](#eq-cvt-system) in each outer loop of incremental 4DVar and compares the analysis with the direct minimum of [](#eq-strong4dvar-cost).
:label: code-incremental4dvar-analysis
import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 window; T = 3 steps, so T + 1 = 4 times; D_x = D_y = Y X = 20 cells;
# Y = 4 rows; X = 5 columns

key_truth, key_noise = jax.random.split(jax.random.PRNGKey(0))


# user
class SineDrift(eqx.Module):
    """A toy nonlinear forward, f(x) = x + 0.2 sin(x), cell by cell."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # sin(x)
        drift = jnp.sin(state)
        # f(x) = x + 0.2 Δt sin(x)
        return state + 0.2 * dt * drift


# user
def gaussian_corr(n_cells: int) -> lx.MatrixLinearOperator:
    """Gaussian correlation of length scale 1.5 cells, plus a 10⁻¹ jitter."""
    cell = jnp.arange(float(n_cells))  # (n_cells,)
    # dᵢⱼ = i − j
    offset = einx.subtract("i, j -> i j", cell, cell)  # (n_cells, n_cells)
    # exp(−dᵢⱼ² / 2ℓ²), ℓ = 1.5 cells
    correlation = jnp.exp(-0.5 * (offset / 1.5) ** 2)  # (n_cells, n_cells)
    # 10⁻¹ δᵢⱼ
    jitter = 1e-1 * jnp.eye(n_cells)  # (n_cells, n_cells)
    # Cᵢⱼ = exp(−dᵢⱼ² / 2ℓ²) + 10⁻¹ δᵢⱼ
    corr = correlation + jitter  # (n_cells, n_cells)
    return lx.MatrixLinearOperator(corr, lx.positive_semidefinite_tag)


forward = SineDrift()
# Σ_row
row_corr = gaussian_corr(4)  # (Y, Y) PSD op
# Σ_col
col_corr = gaussian_corr(5)  # (X, X) PSD op
# today (gaussx)
# Σ_x = Σ_row ⊗ Σ_col, whose square root gaussx.sqrt computes factor by factor
prior_cov = gx.Kronecker(  # (D_x, D_x) PSD op
    row_corr, col_corr, tags=lx.positive_semidefinite_tag
)
# J_x̂ = I
kernel_matrix = lx.MatrixLinearOperator(jnp.eye(20))  # (D_y, D_x) op
# μ_ret = 0
prior_profile = jnp.zeros(20)  # (D_x,)
# w = 0.8
pressure_weights = jnp.full(20, 0.8)  # (D_x,)
# today (vardax)
# h(x) = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_ret)
obs_op = vdx.AveragingKernel(
    kernel_matrix=kernel_matrix,
    prior_profile=prior_profile,
    pressure_weights=pressure_weights,
)
# diag(Σ_ε) = 0.04
obs_diag = jnp.full(20, 0.04)  # (D_y,)
# Σ_ε = 0.04 I, tagged PSD for the Cholesky solve below
obs_cov = lx.TaggedLinearOperator(  # (D_y, D_y) PSD op
    lx.DiagonalLinearOperator(obs_diag), lx.positive_semidefinite_tag
)
# μ_x = 0
background = jnp.zeros(20)  # (D_x,)
# Σ_x as a dense matrix
prior_matrix = prior_cov.as_matrix()  # (D_x, D_x)
# Σ_x = L_x L_xᵀ
prior_chol = jnp.linalg.cholesky(prior_matrix)  # (D_x, D_x)
# ξ ~ 𝒩(0, I)
truth_noise = jax.random.normal(key_truth, (20,))  # (D_x,)
# x₀ = L_x ξ
truth_start = einx.dot("i j, j -> i", prior_chol, truth_noise)  # (D_x,)


# user
def advance(
    state: Float[Array, " n"], _: None
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """One forward step, in the carry-and-output form of jax.lax.scan."""
    # xₜ₊₁ = f(xₜ)
    next_state = forward.step(state, forward.dt)  # (D_x,)
    return next_state, next_state


# x₁ … x_T
_, truth_later = jax.lax.scan(advance, truth_start, length=3)  # (T, D_x)
# [x₀; x₁ … x_T]
truth = einx.id("n, t n -> (1 + t) n", truth_start, truth_later)  # (T + 1, D_x)
# h(xₜ) at every time
truth_observed = jax.vmap(obs_op)(truth)  # (T + 1, D_y)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (4, 20))  # (T + 1, D_y)
# yₜ = h(xₜ) + 0.2 εₜ
obs = truth_observed + 0.2 * obs_noise  # (T + 1, D_y)

# today (vardax)
# x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ + L_x δz⋆, (I + L_xᵀ (H_0:T)ᵀ Σ_ε⁻¹ H_0:T L_x) δz⋆ = −∇_z 𝒥
model = vdx.IncrementalFourDVar(
    forward=forward,
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)
obs_window = einx.id("t n -> 1 t n", obs)  # (T + 1, D_y) → (S, T + 1, D_y)
# mₜ = 1 everywhere
all_observed = jnp.ones((1, 4, 20))  # (S, T + 1, D_y)
# today (vardax)
batch = vdx.Batch1D(input=obs_window, mask=all_observed)
# today (vardax)
# x₀⋆ after three Gauss-Newton outer loops
analysis = model(batch)  # → (S, D_x)

# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖yₜ − h(fₜ(x₀))‖²_Σ_ε⁻¹, minimised directly
strong = vdx.StrongFourDVar(
    forward=forward,
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.BFGS(rtol=1e-6, atol=1e-6),
    cov_solver=lx.Cholesky(),
)
# today (vardax)
# x₀⋆ = argmin 𝒥(x₀)
strong_analysis = strong(batch)  # → (S, D_x)
# x₀ⁱⁿᶜ − x₀ˢᵗʳᵒⁿᵍ
gap = analysis - strong_analysis  # (S, D_x)
# |x₀ⁱⁿᶜ − x₀ˢᵗʳᵒⁿᵍ|
abs_gap = jnp.abs(gap)  # (S, D_x)
# max |x₀ⁱⁿᶜ − x₀ˢᵗʳᵒⁿᵍ|
largest_gap = einx.max("s n ->", abs_gap)  # → ()
print(f"largest gap to StrongFourDVar {largest_gap:.1e}")

# today (vardax)
# Gauss-Newton posterior at the analysis, with the model's Σ_x and Σ_ε
posterior = model.posterior(batch)
```

It prints `largest gap to StrongFourDVar 2.0e-03`: three outer loops of at most 20 CG steps each land that close to the BFGS minimum of the same cost.

`IncrementalConfig`:

- `n_outer`: number of Gauss-Newton outer iterations (typically 3).
- `n_inner`: maximum CG iterations per outer iteration (typically 20 to 50); the inner solve is truncated there.
- `cg_atol`, `cg_rtol`: inner CG convergence tolerances.
- `cvt`: solve over $\vect{z}$ of [](#eq-cvt) (default `True`).
- `covariance_tol`: tolerance of the CG solves that apply $\gmat{\Sigma}_\varepsilon^{-1}$ (and $\gmat{\Sigma}_x^{-1}$ when `cvt=False`).
- `jac`: `"fwd"` (default) applies $\mat{H}_t \mat{F}_{t \leftarrow 0}$ by `jax.linearize`, matrix-free. `"bwd"` forms the window's Jacobian with `jax.jacrev`, one VJP per observation, and runs CG on that matrix. Pass it when the forward or observation map defines only a reverse-mode rule, such as a `jax.custom_vjp` or a diffrax solve under `RecursiveCheckpointAdjoint`; it suits windows of up to a few thousand observations, and the posterior uses the same mode.

A static control, such as a source field rather than an initial state, is an identity-step forward (`step` returns its input) with a one-time window: the outer loop is then Gauss-Newton 3D-Var through any `obs_op`.

## Algorithm

Written out with the control-variable transform, each outer loop linearises the observed trajectory $\hat{\vect{y}}_{0:T}(\vect{x}_0) = [h_t(f_t(\vect{x}_0))]_t$ once and hands the inner system [](#eq-cvt-system) to CG, then maps the control back through [](#eq-cvt).
The block continues the one above and reproduces `model(batch)`.

```{code-block} python
:caption: Runs the incremental outer loop by hand, solving [](#eq-cvt-system) with truncated CG and mapping the control back through [](#eq-cvt).
:label: code-incremental4dvar-outer-loop
# Shapes: T = 3 steps, so T + 1 = 4 times; D_x = D_y = 20 cells

config = model.config
covariance_solver = lx.CG(atol=config.covariance_tol, rtol=config.covariance_tol)
inner_solver = lx.CG(atol=config.cg_atol, rtol=config.cg_rtol, max_steps=config.n_inner)
state_struct = jax.ShapeDtypeStruct((20,), jnp.float32)
# today (gaussx)
# L_x L_xᵀ = Σ_x, here L_x = L_row ⊗ L_col
prior_sqrt = gx.sqrt(prior_cov)  # (D_x, D_x) op
# L_xᵀ
prior_sqrt_t = prior_sqrt.transpose()  # (D_x, D_x) op


# user
def observed_trajectory(state: Float[Array, " n"]) -> Float[Array, "t n"]:
    """The observed trajectory ŷ_0:T(x₀) = [hₜ(fₜ(x₀))]ₜ."""
    # x₁ … x_T
    _, later_states = jax.lax.scan(advance, state, length=3)  # (T, D_x)
    # [x₀; x₁ … x_T]
    trajectory = einx.id("n, t n -> (1 + t) n", state, later_states)  # (T + 1, D_x)
    # ŷ_0:T(x₀) = [hₜ(fₜ(x₀))]ₜ
    return jax.vmap(obs_op)(trajectory)  # (T + 1, D_y)


# user
def whiten_one(residual: Float[Array, " n"]) -> Float[Array, " n"]:
    """Apply Σ_ε⁻¹ to one time's residual with a lineax solve."""
    # Σ_ε⁻¹ r
    solution = lx.linear_solve(obs_cov, residual, solver=covariance_solver)
    return solution.value  # (D_y,)


# user
def whiten(residual: Float[Array, "t n"]) -> Float[Array, "t n"]:
    """Apply Σ_ε⁻¹ at every time of the window."""
    # Σ_ε⁻¹ rₜ at every time
    return jax.vmap(whiten_one)(residual)  # (T + 1, D_y)


# z⁽⁰⁾ = 0, so x⁽⁰⁾ = μ_x
control = jnp.zeros(20)  # (D_x,)
for _ in range(config.n_outer):
    # L_x z⁽ⁱ⁾
    increment = prior_sqrt.mv(control)  # (D_x,)
    # x⁽ⁱ⁾ = μ_x + L_x z⁽ⁱ⁾
    state = background + increment  # (D_x,)
    # ŷ_0:T(x⁽ⁱ⁾) and v ↦ H_0:T v, the tangent of the observed trajectory
    predicted, tangent = jax.linearize(observed_trajectory, state)  # (T + 1, D_y)
    # w ↦ (H_0:T)ᵀ w, its adjoint
    adjoint = jax.linear_transpose(tangent, state)
    # r⁽ⁱ⁾ = y − ŷ_0:T(x⁽ⁱ⁾)
    innovation = obs - predicted  # (T + 1, D_y)
    # Σ_ε⁻¹ r⁽ⁱ⁾
    weighted_innovation = whiten(innovation)  # (T + 1, D_y)
    # (H_0:T)ᵀ Σ_ε⁻¹ r⁽ⁱ⁾
    (obs_gradient,) = adjoint(weighted_innovation)  # (T + 1, D_y) → (D_x,)
    # L_xᵀ (H_0:T)ᵀ Σ_ε⁻¹ r⁽ⁱ⁾
    control_obs_gradient = prior_sqrt_t.mv(obs_gradient)  # (D_x,)
    # −∇_z 𝒥(z⁽ⁱ⁾) = −z⁽ⁱ⁾ + L_xᵀ (H_0:T)ᵀ Σ_ε⁻¹ r⁽ⁱ⁾
    rhs = -control + control_obs_gradient  # (D_x,)

    # user
    def hessian_mv(direction: Float[Array, " n"]) -> Float[Array, " n"]:
        """The CVT Hessian product (I + L_xᵀ (H_0:T)ᵀ Σ_ε⁻¹ H_0:T L_x) v."""
        # L_x v
        state_direction = prior_sqrt.mv(direction)  # (D_x,)
        # H_0:T L_x v
        observed_direction = tangent(state_direction)  # (T + 1, D_y)
        # Σ_ε⁻¹ H_0:T L_x v
        weighted_direction = whiten(observed_direction)  # (T + 1, D_y)
        # (H_0:T)ᵀ Σ_ε⁻¹ H_0:T L_x v
        (back_projected,) = adjoint(weighted_direction)  # (T + 1, D_y) → (D_x,)
        # L_xᵀ (H_0:T)ᵀ Σ_ε⁻¹ H_0:T L_x v
        control_back_projected = prior_sqrt_t.mv(back_projected)  # (D_x,)
        # (I + L_xᵀ (H_0:T)ᵀ Σ_ε⁻¹ H_0:T L_x) v
        return direction + control_back_projected  # (D_x,)

    # I + L_xᵀ (H_0:T)ᵀ Σ_ε⁻¹ H_0:T L_x, eigenvalues ≥ 1
    inner_hessian = lx.FunctionLinearOperator(  # (D_x, D_x) PSD op
        hessian_mv, state_struct, lx.positive_semidefinite_tag
    )
    # δz⋆ from truncated CG
    step = lx.linear_solve(inner_hessian, rhs, solver=inner_solver, throw=False)
    # z⁽ⁱ⁺¹⁾ = z⁽ⁱ⁾ + δz⋆
    control = control + step.value  # (D_x,)

# L_x z⋆
final_increment = prior_sqrt.mv(control)  # (D_x,)
# x₀⋆ = μ_x + L_x z⋆
analysis_by_hand = background + final_increment  # (D_x,)
print(bool(jnp.allclose(analysis_by_hand, analysis[0], atol=1e-4)))
```

It prints `True`.

With `cvt=False` the inner system is [](#eq-gn-system) in state space instead, with $\gmat{\Sigma}_x^{-1}$ applied by a nested CG solve at every Hessian product.

## Why it's the operational fast path

Three multiplicative speedups over `StrongFourDVar` with `NonlinearCG`:

1. **CVT preconditioning.** CG converges in $O(10)$ iterations instead of $O(\sqrt{\kappa(\gmat{\Sigma}_x)})$, where $\kappa(\gmat{\Sigma}_x)$ is the prior condition number.
2. **Linearisation reuse.** The tangent linear $\mat{F}_{t \leftarrow 0}$ is computed once per outer iteration and reused across all `n_inner` inner CG steps, whereas `NonlinearCG` on the full nonlinear cost recomputes the gradient (a full forward and adjoint pass) at every step.
3. **Cheap posterior.** The posterior precision is the operator the inner loop solves with (see the posterior section below).

For a typical operational problem (a 12-hour window, $D_x = 10^7$ grid
points, $D_y = 10^5$ observations), `IncrementalFourDVar` converges in
minutes; `StrongFourDVar` with `NonlinearCG` takes hours.

## Linear-Gaussian agreement

For $T = 0$ and linear $h$, `IncrementalFourDVar` must agree with `OptimalInterpolation` (the Decision D14 invariant, see [chapter 4](04_oi_blue.md#validation)): the outer iteration converges in one step (no relinearisation needed), and the inner CG produces the BLUE [](#eq-blue-mean). The block continues the one above:

```{code-block} python
:caption: Checks that incremental 4DVar with T = 0 and f = I reaches the BLUE [](#eq-blue-mean).
:label: code-incremental4dvar-oi-agreement
# Shapes: S = 1 window; T = 0 steps, so T + 1 = 1 time; D_x = D_y = 4 cells


# user
class Static(eqx.Module):
    """f(x) = x: a single-time window reduces 4DVar to OI."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # f(x) = x
        return state


# diag(Σ_x) = 2
static_prior_diag = jnp.full(4, 2.0)  # (D_x,)
# diag(Σ_ε) = 1/2
static_obs_diag = jnp.full(4, 0.5)  # (D_y,)
# Σ_x = 2 I, over the (D_x,) state of 4DVar
static_prior_cov = lx.DiagonalLinearOperator(static_prior_diag)  # (D_x, D_x) op
# Σ_ε = I / 2
static_obs_cov = lx.DiagonalLinearOperator(static_obs_diag)  # (D_y, D_y) op
# the same Σ_x and Σ_ε over OI's (T + 1, D_x) state
field_prior_diag = jnp.full((1, 4), 2.0)  # (T + 1, D_x)
field_obs_diag = jnp.full((1, 4), 0.5)  # (T + 1, D_y)
field_prior_cov = lx.DiagonalLinearOperator(field_prior_diag)  # ((T+1) D_x)² op
field_obs_cov = lx.DiagonalLinearOperator(field_obs_diag)  # ((T+1) D_y)² op
# y = (1, 2, 3, 4)
single_obs = jnp.arange(1.0, 5.0)  # (D_y,)
single_window = einx.id("n -> 1 1 n", single_obs)  # (D_y,) → (S, T + 1, D_y)
# m = 1 everywhere
single_mask = jnp.ones((1, 1, 4))  # (S, T + 1, D_y)
# today (vardax)
single_batch = vdx.Batch1D(input=single_window, mask=single_mask)
# today (vardax)
# p(x) = 𝒩(0, Σ_x), p(y ∣ x) = 𝒩(x, Σ_ε)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 4)),
    prior_cov_op=field_prior_cov,
    obs_cov_op=field_obs_cov,
)
# today (vardax)
# (I + L_xᵀ Σ_ε⁻¹ L_x) δz⋆ = −∇_z 𝒥, with h = f = I
static_incremental = vdx.IncrementalFourDVar(
    forward=Static(),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(4),
    prior_cov_op=static_prior_cov,
    obs_cov_op=static_obs_cov,
    config=vdx.IncrementalConfig(n_outer=2, n_inner=50, cvt=True),
)
# today (vardax)
# x⋆ = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x)
closed_form = optimal_interpolation(single_batch)  # → (S, T + 1, D_x)
# today (vardax)
# x₀⋆ after two Gauss-Newton outer loops
gauss_newton = static_incremental(single_batch)  # → (S, D_x)
print(bool(jnp.allclose(closed_form[:, 0], gauss_newton, atol=1e-3)))
```

It prints `True`.

## Strong- vs weak-constraint

`IncrementalFourDVar` is strong-constraint by default. For weak-constraint incremental 4DVar (the augmented control vector with $\gvect{\eta}_t$), use `WeakFourDVar` directly, since the incremental machinery for the augmented problem is heavier and not always worth the complexity.
The strong / weak choice is structural; the inner-solver choice
(general vs incremental) is performance.

## Posterior covariance

The posterior precision at the analysis is the inner loop's Gauss-Newton Hessian [](#eq-gn-hessian), and its inverse is the covariance [](#eq-gn-cov) of chapter 13. `model.posterior(batch)` relinearises the rollout once at the analysis and returns that inverse as a lazy operator. The block continues the one above:

```{code-block} python
:caption: Applies the Gauss-Newton posterior covariance [](#eq-gn-cov), the inverse of [](#eq-gn-hessian) at the analysis, to one vector.
:label: code-incremental4dvar-posterior
# Shapes: D_x = 20 cells

# today (vardax)
# Σ_x|y = (Σ_x⁻¹ + (H_0:T)ᵀ Σ_ε⁻¹ H_0:T)⁻¹ at x₀⋆, a lazy operator
posterior = model.posterior(batch)
# 1
ones = jnp.ones(20)  # (D_x,)
# Σ_x|y 1, one matrix-free product
spread = posterior.cov.mv(ones)  # (D_x,) → (D_x,)
print(posterior.mean.shape, spread.shape, posterior.samples)
```

It prints `(20,) (20,) None`.

Pass `adapter=vardax.GaussNewtonHessian(n_krylov=...)` to set the Krylov
budget of each covariance mat-vec.

## When to switch from `IncrementalFourDVar` to `StrongFourDVar`

Almost never, for operational work. `IncrementalFourDVar` is faster
and produces the same answer. Use `StrongFourDVar` only when:

- $\gmat{\Sigma}_x$ is large, unstructured and has no cheap square root, so
  `gaussx.sqrt` would fall back to a dense eigendecomposition. Pass a
  Lanczos square root (`prior_sqrt_op=gaussx.sqrt(prior_cov_op, lanczos_order=k)`)
  or use `cvt=False` before giving up on the incremental solver.
- You want to swap in a custom optimistix minimiser (e.g.
  `optx.LevenbergMarquardt` for very nonlinear cases). Incremental
  uses a fixed GN+CG; the rest of the optimistix zoo isn't compatible
  with the incremental machinery.

For most users, `IncrementalFourDVar` is the right default.

## See also

- Chapter 6: `StrongFourDVar` (equivalent problem, slower solver)
- Chapter 12: adjoint composition (CVT relies on `gaussx` factorisation)
- Chapter 13: posterior covariance (reuses GN Hessian)
- Design doc: [`design/decisions.md#d11`](design/decisions.md#d11-incrementalfourdvar-as-the-operational-fast-path-of-strongfourdvar)
- Chapter 23: reduced incremental 4DVar, the inner problem in a basis ([](#eq-reduced-incremental-system))

## References

- Courtier, P., Thépaut, J.-N., & Hollingsworth, A. (1994). *A strategy
  for operational implementation of 4D-Var, using an incremental
  approach.* QJRMS 120(519).
- Lorenc, A. C. (1997). *Development of an operational variational
  assimilation scheme.* JMSJ 75(1B).
- Bannister, R. N. (2017). *A review of operational methods of
  variational and ensemble-variational data assimilation.* QJRMS
  143(703).
