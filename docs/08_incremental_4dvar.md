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

Strong-constraint 4DVar minimises [](#eq-strong4dvar-cost). Write $x^{(k)}$ for the outer iterate, starting from $x^{(0)} = x_b$. The background $x_b$ stays fixed across outer loops; only $x^{(k)}$ moves. Linearise around $x^{(k)}$:

$$M_t(x^{(k)} + \delta x) \approx M_t(x^{(k)}) + M'_t \delta x$$ (eq-tlm-forward)

$$H_t(z + \delta z) \approx H_t(z) + H'_t \delta z$$ (eq-tlm-obs)

where $M'_t = \partial M_t / \partial x|_{x^{(k)}}$ from
`jax.linearize` and $H'_t$ is the tangent of $H_t$ at
$M_t(x^{(k)})$. With the innovation
$d_t^{(k)} = y_t - H_t(M_t(x^{(k)}))$, the linearised (quadratic)
cost of an increment $\delta x$ is

$$\delta J(\delta x) = \frac{1}{2} \|x^{(k)} + \delta x - x_b\|^2_{B^{-1}} + \frac{1}{2} \sum_t \|d_t^{(k)} - H'_t M'_t \delta x\|^2_{R_t^{-1}}$$ (eq-incremental-cost)

The background term measures the *total* departure $x^{(k)} + \delta x - x_b$, not $\delta x$ alone. The Gauss-Newton Hessian is

$$\mathbf{H}_\text{GN} = B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t)$$ (eq-gn-hessian)

which is the 4DVar form of the 3DVar Hessian [](#eq-3dvar-gn-hessian). Setting $\nabla \delta J = 0$ gives the inner system that the CG (or Lanczos) solver solves for $\delta x^*$:

$$\mathbf{H}_\text{GN} \delta x^* = -B^{-1}\left(x^{(k)} - x_b\right) + \sum_t (H'_t M'_t)^\top R_t^{-1} d_t^{(k)} = -\nabla J\left(x^{(k)}\right)$$ (eq-gn-system)

The right-hand side is the negative gradient [](#eq-strong4dvar-grad) of the full nonlinear cost at the current iterate. On the first outer loop $x^{(0)} = x_b$, so the background term vanishes and only the innovation term is left. On later loops it does not vanish: dropping it would pull every increment toward $x^{(k)}$ instead of $x_b$, and the iteration would no longer converge to the 4DVar minimum.

Outer update: $x^{(k+1)} = x^{(k)} + \delta x^*$. Relinearise.

This pattern converges in a handful of outer iterations (typically 3 to 5) regardless of the strength of the nonlinearity, because each outer step sees a fresh tangent linearisation.

## Control-variable transform (CVT)

The Hessian $\mathbf{H}_\text{GN}$ inherits the condition number of
$B^{-1}$: a smooth (long-correlation) $B$ has tiny eigenvalues, which
become huge ones of $B^{-1}$. In state space every Hessian product also
needs $B^{-1}v$, which `IncrementalFourDVar` would have to compute by a
nested CG solve.

The CVT changes variables. Let $S$ be a square root of $B$, $S S^\top = B$,
and write the departure from the background as

$$x_0 - x_b = S \chi$$ (eq-cvt)

The background term becomes $\frac{1}{2}\|x_0 - x_b\|^2_{B^{-1}} =
\frac{1}{2}\|\chi\|^2$, and the inner system at the outer iterate
$\chi^{(k)}$ becomes

$$\left(I + S^\top G'^\top R^{-1} G' S\right) \delta\chi^* = -\nabla_\chi J\left(\chi^{(k)}\right) = -\chi^{(k)} + S^\top G'^\top R^{-1} d^{(k)}$$ (eq-cvt-system)

where $G'$ stacks $H'_t M'_t$ over the window. Its eigenvalues are all
$\ge 1$, close to 1 in directions the observations barely see, and no
$B^{-1}$ appears anywhere. The outer update is
$\chi^{(k+1)} = \chi^{(k)} + \delta\chi^*$, and the analysis is
$x_0^* = x_b + S\chi^*$.

`IncrementalConfig(cvt=True)` (the default) uses it. $S$ is
`prior_sqrt_op=` when given (a Cholesky factor, a structured
operator's own square root, or `gaussx.sqrt(B, lanczos_order=k)` for a
large unstructured $B$), else `gaussx.sqrt(prior_cov_op)`: closed form
for diagonal, Kronecker and block-diagonal $B$, a dense
eigendecomposition otherwise. `cvt=False` keeps the state-space solve,
with $B^{-1}$ and $R^{-1}$ applied by CG at tolerance
`covariance_tol`.

**How much it helps.** A linear problem with 20 states, 4 times, every
third site observed, a Gaussian-kernel $B$ of length scale 3 sites plus
a diagonal jitter that sets its condition number; inner CG iterations to
reach $10^{-4}$ relative error on the exact analysis:

| $\kappa(B)$ | CVT | state space |
|---:|---:|---:|
| $7 \times 10^1$ | 10 | 8 |
| $7 \times 10^2$ | 9 | 14 |
| $7 \times 10^3$ | 9 | 20 |

The CVT count does not grow with $\kappa(B)$, and each state-space
iteration also runs a nested CG for $B^{-1}v$. For a well-conditioned
$B$ the two are comparable.

## Implementation in vardax

The example below is a $4 \times 5$ grid ($N = 20$ cells) over a window of $T = 4$ times, with a nonlinear forward $M(x) = x + 0.2 \sin(x)$ applied cell by cell.
$B = B_y \otimes B_x$ is separable, each factor a Gaussian correlation of length scale 1.5 cells (plus a $10^{-1}$ jitter), and the background is $x_b = 0$.
Every cell is observed through an averaging kernel [](#eq-averaging-kernel) with $A = I$, $h = 0.8$ and $x_a = 0$, and $R = 0.04 I$.

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

# Shapes: S = 1 window; T = 4 times; N = Y X = 20 cells; Y = 4 rows; X = 5 columns

key_truth, key_noise = jax.random.split(jax.random.PRNGKey(0))


# user
class SineDrift(eqx.Module):
    """A toy nonlinear forward, M(x) = x + 0.2 sin(x), cell by cell."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # sin(x)
        drift = jnp.sin(state)
        # M(x) = x + 0.2 Δt sin(x)
        return state + 0.2 * dt * drift


# user
def gaussian_corr(n_cells: int) -> lx.MatrixLinearOperator:
    """Gaussian correlation of length scale 1.5 cells, plus a 10⁻¹ jitter."""
    cell = jnp.arange(float(n_cells))  # (n,)
    # dᵢⱼ = i − j
    offset = einx.subtract("i, j -> i j", cell, cell)  # (n, n)
    # exp(−dᵢⱼ² / 2ℓ²), ℓ = 1.5 cells
    correlation = jnp.exp(-0.5 * (offset / 1.5) ** 2)  # (n, n)
    # 10⁻¹ δᵢⱼ
    jitter = 1e-1 * jnp.eye(n_cells)  # (n, n)
    # Cᵢⱼ = exp(−dᵢⱼ² / 2ℓ²) + 10⁻¹ δᵢⱼ
    corr = correlation + jitter  # (n, n)
    return lx.MatrixLinearOperator(corr, lx.positive_semidefinite_tag)


forward = SineDrift()
# B_y
row_corr = gaussian_corr(4)  # (Y, Y) PSD op
# B_x
col_corr = gaussian_corr(5)  # (X, X) PSD op
# today (gaussx)
# B = B_y ⊗ B_x, whose square root gaussx.sqrt computes factor by factor
prior_cov = gx.Kronecker(  # (N, N) PSD op
    row_corr, col_corr, tags=lx.positive_semidefinite_tag
)
# A = I
kernel_matrix = lx.MatrixLinearOperator(jnp.eye(20))  # (N, N) op
# x_a = 0
prior_profile = jnp.zeros(20)  # (N,)
# h = 0.8
pressure_weights = jnp.full(20, 0.8)  # (N,)
# today (vardax)
# H(x) = A (h ⊙ x + (1 − h) ⊙ x_a)
obs_op = vdx.AveragingKernel(
    kernel_matrix=kernel_matrix,
    prior_profile=prior_profile,
    pressure_weights=pressure_weights,
)
# diag(R) = 0.04
obs_diag = jnp.full(20, 0.04)  # (N,)
# R = 0.04 I, tagged PSD for the Cholesky solve below
obs_cov = lx.TaggedLinearOperator(  # (N, N) PSD op
    lx.DiagonalLinearOperator(obs_diag), lx.positive_semidefinite_tag
)
# x_b = 0
background = jnp.zeros(20)  # (N,)
# B as a dense matrix
prior_matrix = prior_cov.as_matrix()  # (N, N)
# B = L Lᵀ
prior_chol = jnp.linalg.cholesky(prior_matrix)  # (N, N)
# ξ ~ 𝒩(0, I)
truth_noise = jax.random.normal(key_truth, (20,))  # (N,)
# x₀ = L ξ
truth_start = einx.dot("i j, j -> i", prior_chol, truth_noise)  # (N,)


# user
def advance(
    state: Float[Array, " n"], _: None
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """One forward step, in the carry-and-output form of jax.lax.scan."""
    # xₜ₊₁ = M(xₜ)
    next_state = forward.step(state, forward.dt)  # (N,)
    return next_state, next_state


# x₁ … x₃
_, truth_later = jax.lax.scan(advance, truth_start, length=3)  # (T − 1, N)
# [x₀; x₁ … x₃]
truth = einx.id("n, t n -> (1 + t) n", truth_start, truth_later)  # (T, N)
# H(xₜ) at every time
truth_observed = jax.vmap(obs_op)(truth)  # (T, N)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (4, 20))  # (T, N)
# yₜ = H(xₜ) + 0.2 εₜ
obs = truth_observed + 0.2 * obs_noise  # (T, N)

# today (vardax)
# x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + δx*, (I + Sᵀ G′ᵀ R⁻¹ G′ S) δχ* = −∇_χ J
model = vdx.IncrementalFourDVar(
    forward=forward,
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)
obs_window = einx.id("t n -> 1 t n", obs)  # (T, N) → (S, T, N)
# mₜ = 1 everywhere
all_observed = jnp.ones((1, 4, 20))  # (S, T, N)
# today (vardax)
batch = vdx.Batch1D(input=obs_window, mask=all_observed)
# today (vardax)
# x₀* after three Gauss-Newton outer loops
analysis = model(batch)  # → (S, N)

# today (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖yₜ − H(Mₜ(x₀))‖²_{R⁻¹}, minimised directly
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
# x₀* = argmin J(x₀)
strong_analysis = strong(batch)  # → (S, N)
# x₀ⁱⁿᶜ − x₀ˢᵗʳᵒⁿᵍ
gap = analysis - strong_analysis  # (S, N)
# |x₀ⁱⁿᶜ − x₀ˢᵗʳᵒⁿᵍ|
abs_gap = jnp.abs(gap)  # (S, N)
# max |x₀ⁱⁿᶜ − x₀ˢᵗʳᵒⁿᵍ|
largest_gap = einx.max("s n ->", abs_gap)  # → ()
print(f"largest gap to StrongFourDVar {largest_gap:.1e}")

# today (vardax)
# Gauss-Newton posterior at the analysis, with the model's B and R
posterior = model.posterior(batch)
```

It prints `largest gap to StrongFourDVar 2.0e-03`: three outer loops of at most 20 CG steps each land that close to the BFGS minimum of the same cost.

`IncrementalConfig`:

- `n_outer`: number of Gauss-Newton outer iterations (typically 3).
- `n_inner`: maximum CG iterations per outer iteration (typically 20 to 50); the inner solve is truncated there.
- `cg_atol`, `cg_rtol`: inner CG convergence tolerances.
- `cvt`: solve over $\chi$ of [](#eq-cvt) (default `True`).
- `covariance_tol`: tolerance of the CG solves that apply $R^{-1}$ (and $B^{-1}$ when `cvt=False`).

## Algorithm

Written out with the control-variable transform, each outer loop linearises the observed trajectory $G(x) = [H_t(M_t(x))]_t$ once and hands the inner system [](#eq-cvt-system) to CG, then maps the control back through [](#eq-cvt).
The block continues the one above and reproduces `model(batch)`.

```{code-block} python
:caption: Runs the incremental outer loop by hand, solving [](#eq-cvt-system) with truncated CG and mapping the control back through [](#eq-cvt).
:label: code-incremental4dvar-outer-loop
# Shapes: T = 4 times; N = 20 cells

config = model.config
covariance_solver = lx.CG(atol=config.covariance_tol, rtol=config.covariance_tol)
inner_solver = lx.CG(atol=config.cg_atol, rtol=config.cg_rtol, max_steps=config.n_inner)
state_struct = jax.ShapeDtypeStruct((20,), jnp.float32)
# today (gaussx)
# S Sᵀ = B, here S = S_y ⊗ S_x
prior_sqrt = gx.sqrt(prior_cov)  # (N, N) op
# Sᵀ
prior_sqrt_t = prior_sqrt.transpose()  # (N, N) op


# user
def observed_trajectory(state: Float[Array, " n"]) -> Float[Array, "t n"]:
    """The observed trajectory G(x) = [Hₜ(Mₜ(x))]ₜ."""
    # x₁ … x₃
    _, later_states = jax.lax.scan(advance, state, length=3)  # (T − 1, N)
    # [x₀; x₁ … x₃]
    trajectory = einx.id("n, t n -> (1 + t) n", state, later_states)  # (T, N)
    # G(x) = [Hₜ(Mₜ(x))]ₜ
    return jax.vmap(obs_op)(trajectory)  # (T, N)


# user
def whiten_one(residual: Float[Array, " n"]) -> Float[Array, " n"]:
    """Apply R⁻¹ to one time's residual with a lineax solve."""
    # R⁻¹ r
    solution = lx.linear_solve(obs_cov, residual, solver=covariance_solver)
    return solution.value  # (N,)


# user
def whiten(residual: Float[Array, "t n"]) -> Float[Array, "t n"]:
    """Apply R⁻¹ at every time of the window."""
    # R⁻¹ rₜ at every time
    return jax.vmap(whiten_one)(residual)  # (T, N)


# χ⁽⁰⁾ = 0, so x⁽⁰⁾ = x_b
control = jnp.zeros(20)  # (N,)
for _ in range(config.n_outer):
    # S χ⁽ᵏ⁾
    increment = prior_sqrt.mv(control)  # (N,)
    # x⁽ᵏ⁾ = x_b + S χ⁽ᵏ⁾
    state = background + increment  # (N,)
    # G(x⁽ᵏ⁾) and v ↦ G′ v, the tangent of the observed trajectory
    predicted, tangent = jax.linearize(observed_trajectory, state)  # (T, N)
    # w ↦ G′ᵀ w, its adjoint
    adjoint = jax.linear_transpose(tangent, state)
    # dᵏ = y − G(x⁽ᵏ⁾)
    innovation = obs - predicted  # (T, N)
    # R⁻¹ dᵏ
    weighted_innovation = whiten(innovation)  # (T, N)
    # G′ᵀ R⁻¹ dᵏ
    (obs_gradient,) = adjoint(weighted_innovation)  # (T, N) → (N,)
    # Sᵀ G′ᵀ R⁻¹ dᵏ
    control_obs_gradient = prior_sqrt_t.mv(obs_gradient)  # (N,)
    # −∇_χ J(χ⁽ᵏ⁾) = −χ⁽ᵏ⁾ + Sᵀ G′ᵀ R⁻¹ dᵏ
    rhs = -control + control_obs_gradient  # (N,)

    # user
    def hessian_mv(direction: Float[Array, " n"]) -> Float[Array, " n"]:
        """The CVT Hessian product (I + Sᵀ G′ᵀ R⁻¹ G′ S) v."""
        # S v
        state_direction = prior_sqrt.mv(direction)  # (N,)
        # G′ S v
        observed_direction = tangent(state_direction)  # (T, N)
        # R⁻¹ G′ S v
        weighted_direction = whiten(observed_direction)  # (T, N)
        # G′ᵀ R⁻¹ G′ S v
        (back_projected,) = adjoint(weighted_direction)  # (T, N) → (N,)
        # Sᵀ G′ᵀ R⁻¹ G′ S v
        control_back_projected = prior_sqrt_t.mv(back_projected)  # (N,)
        # (I + Sᵀ G′ᵀ R⁻¹ G′ S) v
        return direction + control_back_projected  # (N,)

    # I + Sᵀ G′ᵀ R⁻¹ G′ S, eigenvalues ≥ 1
    inner_hessian = lx.FunctionLinearOperator(  # (N, N) PSD op
        hessian_mv, state_struct, lx.positive_semidefinite_tag
    )
    # δχ* from truncated CG
    step = lx.linear_solve(inner_hessian, rhs, solver=inner_solver, throw=False)
    # χ⁽ᵏ⁺¹⁾ = χ⁽ᵏ⁾ + δχ*
    control = control + step.value  # (N,)

# S χ*
final_increment = prior_sqrt.mv(control)  # (N,)
# x₀* = x_b + S χ*
analysis_by_hand = background + final_increment  # (N,)
print(bool(jnp.allclose(analysis_by_hand, analysis[0], atol=1e-4)))
```

It prints `True`.

With `cvt=False` the inner system is [](#eq-gn-system) in state space instead, with $B^{-1}$ applied by a nested CG solve at every Hessian product.

## Why it's the operational fast path

Three multiplicative speedups over `StrongFourDVar` with `NonlinearCG`:

1. **CVT preconditioning.** CG converges in $O(10)$ iterations instead of $O(\sqrt{\kappa(B)})$, where $\kappa(B)$ is the prior condition number.
2. **Linearisation reuse.** The tangent linear $M'$ is computed once per outer iteration and reused across all $n_\text{inner}$ inner CG steps, whereas `NonlinearCG` on the full nonlinear cost recomputes the gradient (a full forward and adjoint pass) at every step.
3. **Cheap posterior.** The posterior precision is the operator the inner loop solves with (see the posterior section below).

For a typical operational problem ($T = 12$ hours, $n = 10^7$ grid
points, $m = 10^5$ observations), `IncrementalFourDVar` converges in
minutes; `StrongFourDVar` with `NonlinearCG` takes hours.

## Linear-Gaussian agreement

For $T = 0$ and linear $H$, `IncrementalFourDVar` must agree with `OptimalInterpolation` (the Decision D14 invariant, see [chapter 4](04_oi_blue.md#validation)): the outer iteration converges in one step (no relinearisation needed), and the inner CG produces the BLUE [](#eq-blue-mean). The block continues the one above:

```{code-block} python
:caption: Checks that incremental 4DVar with T = 0 and M = I reaches the BLUE [](#eq-blue-mean).
:label: code-incremental4dvar-oi-agreement
# Shapes: S = 1 window; T = 1 time; N = 4 cells


# user
class Static(eqx.Module):
    """M(x) = x: a single-time window reduces 4DVar to OI."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # M(x) = x
        return state


# diag(B) = 2
static_prior_diag = jnp.full(4, 2.0)  # (N,)
# diag(R) = 1/2
static_obs_diag = jnp.full(4, 0.5)  # (N,)
# B = 2 I, over the (N,) state of 4DVar
static_prior_cov = lx.DiagonalLinearOperator(static_prior_diag)  # (N, N) op
# R = I / 2
static_obs_cov = lx.DiagonalLinearOperator(static_obs_diag)  # (N, N) op
# the same B and R over OI's (T, N) state
field_prior_diag = jnp.full((1, 4), 2.0)  # (T, N)
field_obs_diag = jnp.full((1, 4), 0.5)  # (T, N)
field_prior_cov = lx.DiagonalLinearOperator(field_prior_diag)  # (T N, T N) op
field_obs_cov = lx.DiagonalLinearOperator(field_obs_diag)  # (T N, T N) op
# y = (1, 2, 3, 4)
single_obs = jnp.arange(1.0, 5.0)  # (N,)
single_window = einx.id("n -> 1 1 n", single_obs)  # (N,) → (S, T, N)
# m = 1 everywhere
single_mask = jnp.ones((1, 1, 4))  # (S, T, N)
# today (vardax)
single_batch = vdx.Batch1D(input=single_window, mask=single_mask)
# today (vardax)
# p(x) = 𝒩(0, B), p(y | x) = 𝒩(x, R)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 4)),
    prior_cov_op=field_prior_cov,
    obs_cov_op=field_obs_cov,
)
# today (vardax)
# (I + Sᵀ R⁻¹ S) δχ* = −∇_χ J, with H = M = I
static_incremental = vdx.IncrementalFourDVar(
    forward=Static(),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(4),
    prior_cov_op=static_prior_cov,
    obs_cov_op=static_obs_cov,
    config=vdx.IncrementalConfig(n_outer=2, n_inner=50, cvt=True),
)
# today (vardax)
# x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
closed_form = optimal_interpolation(single_batch)  # → (S, T, N)
# today (vardax)
# x₀* after two Gauss-Newton outer loops
gauss_newton = static_incremental(single_batch)  # → (S, N)
print(bool(jnp.allclose(closed_form[:, 0], gauss_newton, atol=1e-3)))
```

It prints `True`.

## Strong- vs weak-constraint

`IncrementalFourDVar` is strong-constraint by default. For weak-constraint incremental 4DVar (the augmented control vector with $\eta_t$), use `WeakFourDVar` directly, since the incremental machinery for the augmented problem is heavier and not always worth the complexity.
The strong / weak choice is structural; the inner-solver choice
(general vs incremental) is performance.

## Posterior covariance

The posterior precision at the analysis is the inner loop's Gauss-Newton Hessian [](#eq-gn-hessian), and its inverse is the covariance [](#eq-gn-cov) of chapter 13. `model.posterior(batch)` relinearises the rollout once at the analysis and returns that inverse as a lazy operator. The block continues the one above:

```{code-block} python
:caption: Applies the Gauss-Newton posterior covariance [](#eq-gn-cov), the inverse of [](#eq-gn-hessian) at the analysis, to one vector.
:label: code-incremental4dvar-posterior
# Shapes: N = 20 cells

# today (vardax)
# P* = (B⁻¹ + G′ᵀ R⁻¹ G′)⁻¹ at x₀*, a lazy operator
posterior = model.posterior(batch)
# 1
ones = jnp.ones(20)  # (N,)
# P* 1, one matrix-free product
spread = posterior.cov.mv(ones)  # (N,) → (N,)
print(posterior.mean.shape, spread.shape, posterior.samples)
```

It prints `(20,) (20,) None`.

Pass `adapter=vardax.GaussNewtonHessian(n_krylov=...)` to set the Krylov
budget of each covariance mat-vec.

## When to switch from `IncrementalFourDVar` to `StrongFourDVar`

Almost never, for operational work. `IncrementalFourDVar` is faster
and produces the same answer. Use `StrongFourDVar` only when:

- $B$ is large, unstructured and has no cheap square root, so
  `gaussx.sqrt` would fall back to a dense eigendecomposition. Pass a
  Lanczos square root (`prior_sqrt_op=gaussx.sqrt(B, lanczos_order=k)`)
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

## References

- Courtier, P., Thépaut, J.-N., & Hollingsworth, A. (1994). *A strategy
  for operational implementation of 4D-Var, using an incremental
  approach.* QJRMS 120(519).
- Lorenc, A. C. (1997). *Development of an operational variational
  assimilation scheme.* JMSJ 75(1B).
- Bannister, R. N. (2017). *A review of operational methods of
  variational and ensemble-variational data assimilation.* QJRMS
  143(703).
