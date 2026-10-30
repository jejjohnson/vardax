---
numbering:
  enumerator: "23.%s"
---
# Reduced-order variational DA

Every variational method in this reference is a choice of control and prior: 3DVar estimates one state ([](#eq-3dvar-cost)), strong-constraint 4DVar the initial state of a window ([](#eq-strong4dvar-cost)), and weak-constraint 4DVar the initial state plus a model error after every step ([](#eq-weak4dvar-cost)). Each of them has a reduced form, where we keep only a few coefficients of the correction (of the state, of the initial state, or of the model error) in a basis chosen beforehand. So the control drops from $N$ or $N(T + 1)$ numbers to a few tens, the Hessian becomes small enough to factor, and the prior is far better conditioned. Incremental 4DVar ([](#eq-incremental-cost)) adds no new prior: it solves the strong-constraint pair a second way, by Gauss-Newton, and its reduced form does the same with the inner problem restricted to the basis.

This gives a ladder of eight methods, which we write in the same language as the GeoStack esmax ocean page [Variational DA as programs](https://jejjohnson.github.io/geostack/esmax-ocean/).

| Rung | Control | vardax |
|---|---|---|
| 3DVar | one state, $N$ values | `ThreeDVar` (chapter [5](05_threedvar.md)) |
| Reduced 3DVar | basis coefficients $u$, $P \ll N$ values | `ThreeDVar(parameterization=LinearParameterization(...))` |
| Strong-constraint 4DVar | the initial state, $N$ values | `StrongFourDVar` (chapter [6](06_strong_4dvar.md)) |
| Reduced strong-constraint 4DVar | coefficients of the initial increment, $P$ values | `ReducedBasisFourDVar(basis=...)` |
| Incremental 4DVar | as strong-constraint, solved by Gauss-Newton | `IncrementalFourDVar` (chapter [8](08_incremental_4dvar.md)) |
| Reduced incremental 4DVar | as reduced strong-constraint, solved by Gauss-Newton | `IncrementalFourDVar(prior_sqrt_op=...)` with a rectangular square root |
| Weak-constraint 4DVar | the initial state and every model error, $N(T + 1)$ values | `WeakFourDVar` (chapter [7](07_weak_4dvar.md)) |
| Reduced weak-constraint 4DVar | the initial state and a few model-error coefficients, $N + CP$ values | `WeakFourDVar(model_err_basis=ModelErrorBasis(...))`, composing with `parameterization=` |

## The reduced control

Let $\Phi \in \mathbb{R}^{N \times P}$ be a basis matrix with $P \ll N$ columns and $u \in \mathbb{R}^P$ its coefficients. The reduced control writes the state as the background plus a correction in the span of the basis,

$$x = x_b + \Phi u$$ (eq-reduced-control)

with a Gaussian prior on the coefficients,

$$u \sim \mathcal{N}(0, Q), \text{ so } \mathrm{pen}(u) = \frac{1}{2} u^\top Q^{-1} u$$ (eq-reduced-prior)

where $Q \in \mathbb{R}^{P \times P}$ is the prior covariance of the coefficients, which is diagonal for every built-in basis. The implied state covariance is $\Phi Q \Phi^\top$, which has rank $P$: the reduced methods simply cannot correct anything outside the span of $\Phi$.

A natural basis comes from $B$ itself. If $B = U \Lambda U^\top$ is its eigendecomposition and $U_P$, $\Lambda_P$ keep the $P$ largest eigenpairs, then

$$B \approx G G^\top, \text{ with } G = U_P \Lambda_P^{1/2}$$ (eq-reduced-features)

and if we take $\Phi = G$ with $Q = I$ we get a whitened control: the coefficients are standard normal, the prior term is $\frac{1}{2} \|u\|^2$, and the minimiser no longer has to learn a $B^{-1}$ whose condition number can span several decades. These are the EOFs of the background error. A stationary $B$ on a periodic grid is diagonalised by Fourier modes, so there $G$ is a truncated Fourier series, which is what the esmax page builds with kernellib features.

Every vardax basis carries its own $Q$ ([API](https://jejjohnson.github.io/vardax/reference/costs_priors/)):

| Constructor | Columns of $\Phi$ | $Q$ |
|---|---|---|
| `rbf_basis` | Gaussian or Wendland bumps at chosen centres | `variance` (scalar or per mode) |
| `fourier_basis` | Laplacian eigenfunctions on a box | `variance`, times a kernel's spectral density if given (HSGP) |
| `wavelet_basis` | orthonormal DWT atoms | `variance` per scale |
| `eof_basis` | leading EOFs of a sample matrix | each mode's sample variance (`"explained"`) |
| `linear_basis` | any $(N, P)$ matrix | `variance` |
| `composite_basis` | concatenated families | block-diagonal |

`LinearParameterization(basis, background)` turns a basis into a control for any model that takes `parameterization=`: it decodes $u$ with [](#eq-reduced-control) and adds the penalty [](#eq-reduced-prior).

## Running example

All the rungs below share one twin experiment: Lorenz-96 [](#eq-lorenz96) on $N = 40$ sites with forcing $F = 8$, one RK4 step of $\Delta t = 0.05$ per call, over a window of $T = 11$ times (10 steps). The background is $x_b = x_\text{true} + L\xi$ with $B = LL^\top$ a Gaussian correlation of length scale 2 sites plus a $10^{-3}$ jitter, and every second site is observed at every time with $R = I$. It is the example of chapter [6](06_strong_4dvar.md), and we rebuild it here so the page runs on its own.

```{code-block} python
:caption: Builds the Lorenz-96 twin experiment shared by every rung: the truth, the background x_b with covariance B, and the observations of [](#eq-obs-model).
:label: code-reduced-setup
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 window; T = 11 times; N = 40 sites; P basis modes

key_spinup, key_background, key_noise = jax.random.split(jax.random.PRNGKey(0), 3)


# user
class Lorenz96(eqx.Module):
    """Lorenz-96 with forcing F, one RK4 step of length dt per call."""

    forcing: float = 8.0
    dt: float = 0.05

    def tendency(self, state: Float[Array, " n"]) -> Float[Array, " n"]:
        # xₖ₊₁
        east = jnp.roll(state, -1)
        # xₖ₋₂
        far_west = jnp.roll(state, 2)
        # xₖ₋₁
        west = jnp.roll(state, 1)
        # (xₖ₊₁ − xₖ₋₂) xₖ₋₁
        advection = (east - far_west) * west
        # dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + F
        return advection - state + self.forcing

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # k₁ = f(x)
        k1 = self.tendency(state)
        # k₂ = f(x + Δt/2 k₁)
        k2 = self.tendency(state + 0.5 * dt * k1)
        # k₃ = f(x + Δt/2 k₂)
        k3 = self.tendency(state + 0.5 * dt * k2)
        # k₄ = f(x + Δt k₃)
        k4 = self.tendency(state + dt * k3)
        # x ← x + Δt/6 (k₁ + 2k₂ + 2k₃ + k₄)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


truth_model = Lorenz96(forcing=8.0)


# user
def rollout(state: Float[Array, " n"], n_steps: int) -> Float[Array, "t n"]:
    """Return the truth trajectory x₀, M(x₀), …, Mⁿ(x₀)."""

    def advance(
        carry: Float[Array, " n"], _: None
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        # xₜ₊₁ = M(xₜ)
        next_state = truth_model.step(carry, truth_model.dt)
        return next_state, next_state

    _, later_states = jax.lax.scan(advance, state, length=n_steps)
    # [x₀; x₁ … xₙ]
    return einx.id("n, t n -> (1 + t) n", state, later_states)


site = jnp.arange(40.0)  # (N,)
# i − j
signed_offset = einx.subtract("i, j -> i j", site, site)  # (N, N)
# |i − j|
offset = jnp.abs(signed_offset)  # (N, N)
# dᵢⱼ = min(|i − j|, N − |i − j|), the distance round the ring
distance = jnp.minimum(offset, 40.0 - offset)  # (N, N)
# exp(−dᵢⱼ² / 2ℓ²), ℓ = 2 sites
correlation = jnp.exp(-0.5 * (distance / 2.0) ** 2)  # (N, N)
# 10⁻³ δᵢⱼ
jitter = 1e-3 * jnp.eye(40)  # (N, N)
# Bᵢⱼ = exp(−dᵢⱼ² / 2ℓ²) + 10⁻³ δᵢⱼ
background_corr = correlation + jitter  # (N, N)
# today (lineax)
# B
prior_cov = lx.MatrixLinearOperator(  # (N, N) PSD op
    background_corr, lx.positive_semidefinite_tag
)
# today (lineax)
# R = I
obs_cov = lx.IdentityLinearOperator(  # (N, N) op
    jax.ShapeDtypeStruct((40,), jnp.float32)
)

# δ ~ 𝒩(0, I)
spinup_noise = jax.random.normal(key_spinup, (40,))  # (N,)
# x_true = M²⁰⁰(F + δ): a state on the attractor
spinup = rollout(8.0 + spinup_noise, 200)  # (N,) → (201, N)
truth_start = spinup[-1]  # (N,)
# B = L Lᵀ
background_chol = jnp.linalg.cholesky(background_corr)  # (N, N)
# ξ ~ 𝒩(0, I)
background_noise = jax.random.normal(key_background, (40,))  # (N,)
# L ξ
background_error = einx.dot(  # (N,)
    "i j, j -> i", background_chol, background_noise
)
# x_b = x_true + L ξ
background = truth_start + background_error  # (N,)
# xₜ = Mₜ(x_true), t = 0 … 10
truth_trajectory = rollout(truth_start, 10)  # (N,) → (T, N)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (11, 40))  # (T, N)
# yₜ = Mₜ(x_true) + εₜ
obs = truth_trajectory + obs_noise  # (T, N)
# mₜ = 1 on every second site
mask = jnp.zeros((11, 40)).at[:, ::2].set(1.0)  # (T, N)
# today (vardax)
window = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", obs),  # (T, N) → (S, T, N)
    mask=einx.id("t n -> 1 t n", mask),  # (T, N) → (S, T, N)
)


# user
def rmse(state: Float[Array, " n"]) -> Float[Array, ""]:
    """Root-mean-square error of an initial state against x_true."""
    # (x − x_true)²
    squared_error = (state - truth_start) ** 2  # (N,)
    # mean (x − x_true)²
    mean_squared_error = einx.mean("n ->", squared_error)  # → ()
    return jnp.sqrt(mean_squared_error)  # → ()


print(f"RMSE of x_b {rmse(background):.2f}")
```

It prints `RMSE of x_b 1.44`.

The basis for every rung is [](#eq-reduced-features) with $P = 8$ and $P = 16$.

```{code-block} python
:caption: Builds the whitened EOF basis G of [](#eq-reduced-features) for P = 8 and P = 16, so that Φ = G and Q = I in [](#eq-reduced-prior).
:label: code-reduced-basis
# Shapes: N = 40 sites; P = 8 or 16 modes

# B = U Λ Uᵀ, eigenvalues ascending
eigvals, eigvecs = jnp.linalg.eigh(background_corr)  # (N,), (N, N)
# tr(B)
total_variance = einx.sum("n ->", eigvals)  # → ()
bases = {}
for n_modes in (8, 16):
    # U_P, Λ_P: the leading P eigenpairs
    leading_vecs = eigvecs[:, -n_modes:]  # (N, P)
    leading_vals = eigvals[-n_modes:]  # (P,)
    # Λ_P^{1/2}
    mode_scale = jnp.sqrt(leading_vals)  # (P,)
    # G = U_P Λ_P^{1/2}, so B ≈ G Gᵀ and Q = I
    features = einx.multiply("n p, p -> n p", leading_vecs, mode_scale)  # (N, P)
    # tr(Λ_P) / tr(B)
    trace_share = einx.sum("p ->", leading_vals) / total_variance  # → ()
    # today (vardax)
    # Φ = G with Q = I
    bases[n_modes] = vdx.linear_basis(features, variance=1.0)
    print(f"P = {n_modes}: {trace_share:.0%} of tr B")
```

It prints `P = 8: 79% of tr B` and `P = 16: 99% of tr B`. The block continues the one above.

## Reduced 3DVar

Putting [](#eq-reduced-control) into the 3DVar cost [](#eq-3dvar-cost) gives a cost over $P$ numbers,

$$J(u) = \frac{1}{2} u^\top Q^{-1} u + \frac{1}{2} \|y - H(x_b + \Phi u)\|^2_{R^{-1}}$$ (eq-reduced-3dvar-cost)

whose minimiser $u^*$ gives the analysis $x^* = x_b + \Phi u^*$. In vardax this is `ThreeDVar` with `parameterization=LinearParameterization(basis, background)`; the background term of [](#eq-3dvar-cost) is replaced by the penalty, so `prior_cov_op` is not used.

```{code-block} python
:caption: Minimises the full 3DVar cost [](#eq-3dvar-cost) and the reduced cost [](#eq-reduced-3dvar-cost) on the observations at the initial time.
:label: code-reduced-3dvar
# Shapes: S = 1 window; N = 40 sites; P = 8 or 16 modes

# today (vardax)
# y₀, m₀: the observations at the initial time only
snapshot = vdx.Batch1D(input=window.input[:, 0], mask=window.mask[:, 0])  # (S, N)
# today (vardax)
# x* = argmin J(x), the full control
full_3dvar = vdx.ThreeDVar(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    cov_solver=lx.Cholesky(),
)
full_3dvar_analysis = full_3dvar(snapshot)  # → (S, N)
print(f"3D-Var, full: RMSE {rmse(full_3dvar_analysis[0]):.2f}")
for n_modes, basis in bases.items():
    # today (vardax)
    # x = x_b + Φ u, pen(u) = ½ uᵀ Q⁻¹ u
    reduced_control = vdx.LinearParameterization(basis, background=background)
    # today (vardax)
    # J(u) = ½ ‖u‖² + ½ ‖m ⊙ (y − (x_b + Φ u))‖²_{R⁻¹}
    reduced_3dvar = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=prior_cov,
        obs_cov_op=obs_cov,
        parameterization=reduced_control,
    )
    reduced_3dvar_analysis = reduced_3dvar(snapshot)  # → (S, N)
    print(f"3D-Var, P = {n_modes}: RMSE {rmse(reduced_3dvar_analysis[0]):.2f}")
```

It prints `3D-Var, full: RMSE 0.53`, `3D-Var, P = 8: RMSE 0.57` and `3D-Var, P = 16: RMSE 0.52`. With 99 % of the trace of $B$ in 16 modes, the reduced analysis is as good as the full one; with 8 modes the correction misses the smaller scales.

## Reduced strong-constraint 4DVar

The same substitution in the strong-constraint cost [](#eq-strong4dvar-cost) gives

$$J(u) = \frac{1}{2} u^\top Q^{-1} u + \frac{1}{2} \sum_t \lVert m_t \odot \left(y_t - H_t(M_t(x_b + \Phi u))\right)\rVert^2_{R_t^{-1}}$$ (eq-reduced-basis-cost)

where $m_t$ is the observation mask at time $t$. It is minimised directly, with no linearisation, outer loop or inner CG. The gradient is the state-space adjoint projected onto the basis,

$$\nabla_u J = Q^{-1} u + \Phi^\top \lambda_0$$ (eq-reduced-basis-grad)

where $\lambda_0$ is the adjoint state at $t = 0$ from the backward sweep [](#eq-adjoint-recursion). One cost-and-gradient evaluation costs one forward and one adjoint run, as in state space. The gain is that the minimiser works in $P$ dimensions, and when $P$ is a few tens even forward-mode differentiation with $P$ tangent runs and the full $P \times P$ Hessian become affordable. `ReducedBasisFourDVar(basis=...)` minimises [](#eq-reduced-basis-cost); with `basis=None` it is the same as `StrongFourDVar`, and `basis=` is equivalent to `StrongFourDVar(parameterization=LinearParameterization(basis, x_b))`.

```{code-block} python
:caption: Minimises the full strong-constraint cost [](#eq-strong4dvar-cost) and the reduced cost [](#eq-reduced-basis-cost) with one first-order minimiser.
:label: code-reduced-strong
# Shapes: S = 1 window; T = 11 times; N = 40 sites; P = 8 or 16 modes

# today (vardax)
# one first-order minimiser for every control, so only the control differs
first_order = vdx.OptaxMinimiser(optax.adam(1e-2), max_steps=2000)
# today (vardax)
# x₀* = argmin J(x₀), the full control
full_strong = vdx.StrongFourDVar(
    forward=truth_model,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=first_order,
    cov_solver=lx.Cholesky(),
)
full_strong_analysis = full_strong(window)  # → (S, N)
print(f"strong 4D-Var, full: RMSE {rmse(full_strong_analysis[0]):.2f}")
for n_modes, basis in bases.items():
    # today (vardax)
    # J(u) = ½ ‖u‖² + ½ Σₜ ‖mₜ ⊙ (yₜ − Mₜ(x_b + Φ u))‖²_{R⁻¹}
    reduced_strong = vdx.ReducedBasisFourDVar(
        forward=truth_model,
        obs_op=vdx.MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=prior_cov,
        obs_cov_op=obs_cov,
        basis=basis,
        minimiser=first_order,
        cov_solver=lx.Cholesky(),
    )
    reduced_strong_analysis = reduced_strong(window)  # → (S, N)
    print(f"strong 4D-Var, P = {n_modes}: RMSE {rmse(reduced_strong_analysis[0]):.2f}")
```

It prints `strong 4D-Var, full: RMSE 0.26`, `strong 4D-Var, P = 8: RMSE 0.64` and `strong 4D-Var, P = 16: RMSE 0.27`. Every control uses the same Adam minimiser, so only the control differs. A chaotic forward can return non-finite values on an aggressive quasi-Newton trial step, and the masked $R$ solve raises on them; a first-order minimiser with a small step avoids it on this window. `ReducedBasisFourDVar` was called `Incremental4DVar` until vardax 0.1.10; the old name still works and emits a `DeprecationWarning`.

## Reduced incremental 4DVar

Incremental 4DVar [](#eq-incremental-cost) linearises around the current iterate and solves a quadratic inner problem by CG, and with the control-variable transform [](#eq-cvt) its unknowns are whitened. The reduced form keeps the same outer loop, which still runs the full nonlinear model, but restricts the increment to the basis, $\delta x = \Phi \delta u$. With the whitened basis ($Q = I$) the inner cost at outer iterate $u^{(k)}$ is

$$\delta J(\delta u) = \frac{1}{2} \|u^{(k)} + \delta u\|^2 + \frac{1}{2} \sum_t \|d_t^{(k)} - H'_t M'_t \Phi \delta u\|^2_{R_t^{-1}}$$ (eq-reduced-incremental-cost)

where $d_t^{(k)} = y_t - H_t(M_t(x_b + \Phi u^{(k)}))$ is the innovation of the current trajectory, and $H'_t$, $M'_t$ are the tangent-linear operators of [](#eq-tlm-forward) and [](#eq-tlm-obs). Its normal equations are a $P \times P$ system,

$$\left(I + \sum_t (H'_t M'_t \Phi)^\top R_t^{-1} (H'_t M'_t \Phi)\right) \delta u^* = -u^{(k)} + \sum_t (H'_t M'_t \Phi)^\top R_t^{-1} d_t^{(k)}$$ (eq-reduced-incremental-system)

which is the CVT system [](#eq-cvt-system) with the square root $S$ replaced by the rectangular $\Phi$. Being $P \times P$, it is small enough to factor directly instead of iterating. The outer update is $u^{(k+1)} = u^{(k)} + \delta u^*$, and the analysis is $x_0^* = x_b + \Phi u^*$, the same minimiser as [](#eq-reduced-basis-cost).

In vardax this is `IncrementalFourDVar` with a rectangular `prior_sqrt_op`: the control takes the input size of the square root, so passing $\Phi$ in place of $S$ makes the reduced form a strict generalisation of the CVT. `ReducedBasisFourDVar` minimises the same cost by quasi-Newton or a first-order minimiser instead.

```{code-block} python
:caption: The reduced incremental solve of [](#eq-reduced-incremental-system), with a rectangular square root Φ in place of S in [](#eq-cvt).
:label: code-reduced-incremental
# Shapes: S = 1 window; N = 40 sites; P = 16 modes

# today (lineax)
# Φ = G, the whitened P = 16 basis, as an (N, P) operator
basis_sqrt = lx.MatrixLinearOperator(bases[16].design_matrix)  # (N, P) op
# today (vardax)
# x₀ = x_b + Φ u, u ~ 𝒩(0, I_P): the inner system is P × P
reduced_incremental = vdx.IncrementalFourDVar(
    forward=truth_model,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    prior_sqrt_op=basis_sqrt,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)
# today (vardax)
# x₀* = x_b + Φ u*
reduced_incremental_analysis = reduced_incremental(window)  # → (S, N)
print(f"incremental 4D-Var, P = 16: RMSE {rmse(reduced_incremental_analysis[0]):.2f}")
```

It prints `incremental 4D-Var, P = 16: RMSE 0.28`: three Gauss-Newton outer loops over the same 16 coefficients land close to the first-order minimiser's 0.27.

## Reduced weak-constraint 4DVar

Weak-constraint 4DVar [](#eq-weak4dvar-cost) adds a model error $\eta_k$ after every step, $N T$ more unknowns. If that error has a shape one can name, like a bias, a slow drift or a large-scale pattern, it is much better to estimate the few numbers of that shape. Let $\chi \in \mathbb{R}^{T \times C}$ be a temporal basis, for example Legendre polynomials over the window, $\varphi_j$ the columns of a spatial basis $\Phi \in \mathbb{R}^{N \times P}$, and $W \in \mathbb{R}^{C \times P}$ the coefficients. The model error and its prior are

$$\eta(x_n, t_k) = \sum_{c=1}^{C} \sum_{j=1}^{P} \chi_c(t_k) W_{cj} \varphi_j(x_n), \text{ with } \mathrm{vec}_r W = (\Lambda_t \otimes \Lambda_s)^{1/2} \omega, \text{ and } \omega \sim \mathcal{N}(0, I)$$ (eq-reduced-model-error)

where $\mathrm{vec}_r$ stacks $W$ row by row and $\Lambda_t$, $\Lambda_s$ are the diagonal prior variances of the temporal and spatial coefficients. The implied model-error covariance is $Q = (\chi \otimes \Phi)(\Lambda_t \otimes \Lambda_s)(\chi \otimes \Phi)^\top$, never formed. The cost over the control $(x_0, \omega)$ is

$$J(x_0, \omega) = \frac{1}{2} \|x_0 - x_b\|^2_{B^{-1}} + \frac{1}{2} \|\omega\|^2 + \frac{1}{2} \sum_k \|y_k - H_k(z_k)\|^2_{R_k^{-1}}, \text{ with } z_k = M_k(z_{k-1}) + \eta_k$$ (eq-reduced-weak4dvar-cost)

so the control has $N + CP$ numbers instead of $N(T + 1)$. A constant bias is the smallest case: $C = 1$ and $P = 1$, one coefficient. The initial state can be reduced as well, with [](#eq-reduced-control) for $x_0$.

`WeakFourDVar(parameterization=LinearParameterization(...))` reduces the initial state and `model_err_basis=` the model error. Below we first run the window with a biased model, $F = 7$ against the $F = 8$ truth, and we measure the error at the end of the window, where the bias has built up.

```{code-block} python
:caption: Compares strong-constraint 4DVar [](#eq-strong4dvar-cost) with a biased model against weak-constraint 4DVar [](#eq-weak4dvar-cost) with a full and a reduced [](#eq-reduced-control) initial state.
:label: code-reduced-weak
# Shapes: S = 1 window; T = 11 times; N = 40 sites; P = 16 modes

# M with F = 7: the analysis model is biased against the F = 8 truth
biased_model = Lorenz96(forcing=7.0)


# user
def window_end(
    initial_state: Float[Array, " n"], model_errors: Float[Array, "k n"]
) -> Float[Array, " n"]:
    """Run the biased model over the window, adding ηₖ after each step."""

    def advance(
        carry: Float[Array, " n"], model_error: Float[Array, " n"]
    ) -> tuple[Float[Array, " n"], None]:
        # zₖ = M(zₖ₋₁) + ηₖ
        next_state = biased_model.step(carry, biased_model.dt) + model_error
        return next_state, None

    final_state, _ = jax.lax.scan(advance, initial_state, model_errors)
    return final_state  # (N,)


# user
def end_rmse(state: Float[Array, " n"]) -> Float[Array, ""]:
    """Root-mean-square error against the truth at the end of the window."""
    # (z_T − x_true(t_T))²
    squared_error = (state - truth_trajectory[-1]) ** 2  # (N,)
    # mean (z_T − x_true(t_T))²
    mean_squared_error = einx.mean("n ->", squared_error)  # → ()
    return jnp.sqrt(mean_squared_error)  # → ()


no_errors = jnp.zeros((10, 40))  # (T − 1, N)
# today (vardax)
# the strong constraint trusts the biased model
biased_strong = vdx.StrongFourDVar(
    forward=biased_model,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=first_order,
    cov_solver=lx.Cholesky(),
)
biased_initial = biased_strong(window)[0]  # (S, N) → (N,)
biased_end = window_end(biased_initial, no_errors)  # (N,)
print(f"strong 4D-Var, biased model: end-of-window RMSE {end_rmse(biased_end):.2f}")
# z_T from the exact model and the full strong analysis, for reference
exact_end = rollout(full_strong_analysis[0], 10)[-1]  # (N,)
print(f"strong 4D-Var, exact model: end-of-window RMSE {end_rmse(exact_end):.2f}")
# today (lineax)
# Q = σ_q² I, σ_q = 0.1
model_err_cov = lx.TaggedLinearOperator(0.01 * obs_cov, lx.positive_semidefinite_tag)
# today (vardax)
# x₀ = x_b + Φ u with the P = 16 basis
reduced_initial = vdx.LinearParameterization(bases[16], background=background)
for label, control in (("full x₀", None), ("P = 16 x₀", reduced_initial)):
    # today (vardax)
    # J(x₀, η) with the model errors ηₖ in full
    weak = vdx.WeakFourDVar(
        forward=biased_model,
        obs_op=vdx.MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=prior_cov,
        obs_cov_op=obs_cov,
        model_err_cov_op=model_err_cov,
        parameterization=control,
        minimiser=first_order,
        cov_solver=lx.Cholesky(),
    )
    initial_states, model_errors = weak(window)  # → (S, N), (S, T − 1, N)
    weak_end = window_end(initial_states[0], model_errors[0])  # (N,)
    print(f"weak 4D-Var, {label}: end-of-window RMSE {end_rmse(weak_end):.2f}")
```

It prints `strong 4D-Var, biased model: end-of-window RMSE 0.55`, `strong 4D-Var, exact model: end-of-window RMSE 0.42`, `weak 4D-Var, full x₀: end-of-window RMSE 0.53` and `weak 4D-Var, P = 16 x₀: end-of-window RMSE 0.54`. The weak constraint barely helps here, and changing $\sigma_q$ to 0.05 or 0.32 gives 0.54 and 0.58: with 400 model-error unknowns against 220 observations, an error at every point and time cannot separate the bias from noise. A reduced model error with one constant mode is exactly that separation, and on a longer Lorenz-96 window the esmax page reports it recovering the bias as 1.00 ± 0.04 (truth 1.0). In vardax that is `WeakFourDVar(model_err_basis=...)`, below.

```{code-block} python
:caption: The reduced weak-constraint 4DVar of [](#eq-reduced-weak4dvar-cost), with one constant mode in time and space for the bias in [](#eq-reduced-model-error).
:label: code-reduced-weak-model-error
# Shapes: S = 1 window; T = 11 times; N = 40 sites; C = 1 temporal mode;
#         P = 1 spatial mode

# χ(t) = 1, one constant temporal mode
temporal_basis = jnp.ones((10, 1))  # (T − 1, C)
# φ(x) = 1, one constant spatial mode
spatial_basis = jnp.ones((40, 1))  # (N, P)
# today (vardax)
# η(xₙ, tₖ) = χ(tₖ) W φ(xₙ), W ~ 𝒩(0, Λ_t ⊗ Λ_s)
error_basis = vdx.ModelErrorBasis(
    temporal=temporal_basis,
    spatial=spatial_basis,
    temporal_variance=jnp.ones(1),
    spatial_variance=jnp.ones(1),
)
# today (vardax)
# J(x₀, ω) over N + C P = 41 numbers
reduced_weak = vdx.WeakFourDVar(
    forward=biased_model,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    model_err_basis=error_basis,
    minimiser=first_order,
    cov_solver=lx.Cholesky(),
)
# today (vardax)
# (x₀*, W*): W* is the estimated bias per step
initial_states, error_coefficients = reduced_weak(window)  # → (S, N), (S, C, P)
# η = χ W Φᵀ, the bias added after every step
reduced_errors = temporal_basis @ error_coefficients[0] @ spatial_basis.T  # (T − 1, N)
reduced_end = window_end(initial_states[0], reduced_errors)  # (N,)
print(f"weak 4D-Var, one bias mode: end-of-window RMSE {end_rmse(reduced_end):.2f}")
print(f"estimated bias per step {float(error_coefficients[0, 0, 0]):.3f}")
```

It prints `weak 4D-Var, one bias mode: end-of-window RMSE 0.43` and `estimated bias per step 0.047`. One coefficient brings the biased model to within 0.01 of the exact model's 0.42, and the bias it finds is the forcing gap times the step, $(8 - 7) \times 0.05 = 0.05$, to first order.

## Uncertainty

A reduced control has a reduced posterior: the Laplace precision lives in coefficient space, $P \times P$, and maps to a rank-$P$ state covariance, [](#eq-reduced-basis-cov) in chapter [13](13_posterior_covariance.md). The posterior adapters build it for any linear control, so `posterior(batch)` works for reduced 3DVar and reduced strong-constraint 4DVar. The spread outside the span of $\Phi$ is exactly zero, which says more about the basis than about the data.

## Choosing the basis and its size

- **EOFs of $B$** ([](#eq-reduced-features)) are the best rank-$P$ approximation of the prior, so they are the default when $B$ is known; the trace share $\mathrm{tr}(\Lambda_P) / \mathrm{tr}(B)$ says how much prior variance the control can still correct.
- **EOFs of a model run** (`eof_basis`) adapt the control to the dynamics rather than to the prior, as in reduced-order incremental 4DVar for ocean models.
- **Fourier or RBF bases** (`fourier_basis`, `rbf_basis`) suit a stationary or smooth correction and need no sample; wavelets (`wavelet_basis`) keep local features.
- **Size**: we increase $P$ until the analysis stops improving. On the running example the RMSE goes from 0.64 to 0.27 between $P = 8$ and $P = 16$ (79 % and 99 % of the trace), and the full control gives 0.26.

## See also

- Chapter [5](05_threedvar.md), [6](06_strong_4dvar.md), [7](07_weak_4dvar.md) and [8](08_incremental_4dvar.md): the full methods.
- Chapter [13](13_posterior_covariance.md): the reduced posterior [](#eq-reduced-basis-cov).
- [#154](https://github.com/jejjohnson/vardax/issues/154): the epic for a reduced form of every method.

## References

- Robert, C., Durbiano, S., Blayo, E., Verron, J., Blum, J. and Le Dimet, F.-X. (2005). *A reduced-order strategy for 4D-Var data assimilation.* Journal of Marine Systems 57, 70-82. doi:10.1016/j.jmarsys.2005.04.003.
- Trémolet, Y. (2006). *Accounting for an imperfect model in 4D-Var.* QJRMS 132(621).
