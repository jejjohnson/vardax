---
numbering:
  enumerator: "23.%s"
---
# Reduced-order variational DA

Every variational method in this reference is a choice of control and prior: 3DVar estimates one state ([](#eq-3dvar-cost)), strong-constraint 4DVar the initial state of a window ([](#eq-strong4dvar-cost)), and weak-constraint 4DVar the initial state plus a model error after every step ([](#eq-weak4dvar-cost)). Each of them has a reduced form, where we keep only a few coefficients of the correction (of the state, of the initial state, or of the model error) in a basis chosen beforehand. So the control drops from $D_x$ or $D_x (T + 1)$ numbers to a few tens, the Hessian becomes small enough to factor, and the prior is far better conditioned. Incremental 4DVar ([](#eq-incremental-cost)) adds no new prior: it solves the strong-constraint pair a second way, by Gauss-Newton, and its reduced form does the same with the inner problem restricted to the basis.

This gives a ladder of eight methods, which we write in the same language as the GeoStack esmax ocean page [Variational DA as programs](https://jejjohnson.github.io/geostack/esmax-ocean/).

| Rung | Control | vardax |
|---|---|---|
| 3DVar | one state, $D_x$ values | `ThreeDVar` (chapter [5](05_threedvar.md)) |
| Reduced 3DVar | basis coefficients $\vect{z}$, $D_z \ll D_x$ values | `ThreeDVar(parameterization=LinearParameterization(...))` |
| Strong-constraint 4DVar | the initial state, $D_x$ values | `StrongFourDVar` (chapter [6](06_strong_4dvar.md)) |
| Reduced strong-constraint 4DVar | coefficients of the initial increment, $D_z$ values | `ReducedBasisFourDVar(basis=...)` |
| Incremental 4DVar | as strong-constraint, solved by Gauss-Newton | `IncrementalFourDVar` (chapter [8](08_incremental_4dvar.md)) |
| Reduced incremental 4DVar | as reduced strong-constraint, solved by Gauss-Newton | `IncrementalFourDVar(prior_sqrt_op=...)` with a rectangular square root |
| Weak-constraint 4DVar | the initial state and every model error, $D_x (T + 1)$ values | `WeakFourDVar` (chapter [7](07_weak_4dvar.md)) |
| Reduced weak-constraint 4DVar | the initial state and a few model-error coefficients, $D_x + C P$ values | `WeakFourDVar(model_err_basis=ModelErrorBasis(...))`, composing with `parameterization=` |

## The reduced control

Let $\gmat{\Phi} \in \mathbb{R}^{D_x \times D_z}$ be a basis matrix with $D_z \ll D_x$ columns and $\vect{z} \in \mathbb{R}^{D_z}$ its coefficients. The reduced control writes the state $\vect{x} \in \mathbb{R}^{D_x}$ as the background mean $\gvect{\mu}_x \in \mathbb{R}^{D_x}$ plus a correction in the span of the basis,

$$\vect{x} = \gvect{\mu}_x + \gmat{\Phi} \vect{z}$$ (eq-reduced-control)

with a Gaussian prior on the coefficients,

$$\vect{z} \sim \mathcal{N}(\mathbf{0}, \gmat{\Sigma}_z), \text{ so } \mathcal{R}(\vect{z}) = \frac{1}{2} \vect{z}^\top \gmat{\Sigma}_z^{-1} \vect{z}$$ (eq-reduced-prior)

where $\gmat{\Sigma}_z \in \mathbb{R}^{D_z \times D_z}$ is the prior covariance of the coefficients, which is diagonal for every built-in basis. The implied state covariance is $\gmat{\Phi} \gmat{\Sigma}_z \gmat{\Phi}^\top$, which has rank $D_z$: the reduced methods simply cannot correct anything outside the span of $\gmat{\Phi}$.

A natural basis comes from $\gmat{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}$ itself. If $\gmat{\Sigma}_x = \mat{U} \op{diag}(\gvect{\sigma}^2) \mat{U}^\top$ is its eigendecomposition and $\mat{U}_P \in \mathbb{R}^{D_x \times D_z}$, $\gvect{\sigma}_P^2 \in \mathbb{R}^{D_z}$ keep the $D_z$ largest eigenpairs, then

$$\gmat{\Sigma}_x \approx \mat{G} \mat{G}^\top, \text{ with } \mat{G} = \mat{U}_P \op{diag}(\gvect{\sigma}_P)$$ (eq-reduced-features)

and if we take $\gmat{\Phi} = \mat{G}$ with $\gmat{\Sigma}_z = \mat{I}$ we get a whitened control: the coefficients are standard normal, the prior term is $\frac{1}{2} \Vert \vect{z} \Vert_2^2$, and the minimiser no longer has to learn a $\gmat{\Sigma}_x^{-1}$ whose condition number can span several decades. These are the EOFs of the background error. A stationary $\gmat{\Sigma}_x$ on a periodic grid is diagonalised by Fourier modes, so there $\mat{G}$ is a truncated Fourier series, which is what the esmax page builds with kernellib features.

Every vardax basis carries its own $\gmat{\Sigma}_z$ ([API](https://jejjohnson.github.io/vardax/reference/costs_priors/)):

| Constructor | Columns of $\gmat{\Phi}$ | $\gmat{\Sigma}_z$ |
|---|---|---|
| `rbf_basis` | Gaussian or Wendland bumps at chosen centres | `variance` (scalar or per mode) |
| `fourier_basis` | Laplacian eigenfunctions on a box | `variance`, times a kernel's spectral density if given (HSGP) |
| `wavelet_basis` | orthonormal DWT atoms | `variance` per scale |
| `eof_basis` | leading EOFs of a sample matrix | each mode's sample variance (`"explained"`) |
| `linear_basis` | any $D_x \times D_z$ matrix | `variance` |
| `composite_basis` | concatenated families | block-diagonal |

`LinearParameterization(basis, background)` turns a basis into a control for any model that takes `parameterization=`: it decodes $\vect{z}$ with [](#eq-reduced-control) and adds the penalty [](#eq-reduced-prior).

## Running example

All the rungs below share one twin experiment: Lorenz-96 [](#eq-lorenz96) on $D_x = 40$ sites with forcing $F = 8$, one RK4 step of $\Delta t = 0.05$ per call, over a window of $T = 10$ steps (11 times). The background mean is $\gvect{\mu}_x = \vect{x}_\text{true} + \mat{L} \gvect{\xi}$ with $\gvect{\xi} \sim \mathcal{N}(\mathbf{0}, \mat{I})$ and $\gmat{\Sigma}_x = \mat{L} \mat{L}^\top \in \mathbb{R}^{D_x \times D_x}$ a Gaussian correlation of length scale 2 sites plus a $10^{-3}$ jitter, and every second site is observed at every time with $\gmat{\Sigma}_\varepsilon = \mat{I}$, so $D_y = D_x$. It is the example of chapter [6](06_strong_4dvar.md), and we rebuild it here so the page runs on its own.

```{code-block} python
:caption: Builds the Lorenz-96 twin experiment shared by every rung: the truth, the background mean μ_x with covariance Σ_x, and the observations of [](#eq-obs-model).
:label: code-reduced-setup
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N_batch = 1 window; T = 10 steps, so T + 1 = 11 times; D_x = 40 sites;
#         D_z basis modes

key_spinup, key_background, key_noise = jax.random.split(jax.random.PRNGKey(0), 3)


# user
class Lorenz96(eqx.Module):
    """Lorenz-96 with forcing F, one RK4 step of length dt per call."""

    forcing: float = 8.0
    dt: float = 0.05

    def tendency(self, state: Float[Array, " n"]) -> Float[Array, " n"]:
        # x_d+1
        east = jnp.roll(state, -1)
        # x_d−2
        far_west = jnp.roll(state, 2)
        # x_d−1
        west = jnp.roll(state, 1)
        # (x_d+1 − x_d−2) x_d−1
        advection = (east - far_west) * west
        # ẋ_d = (x_d+1 − x_d−2) x_d−1 − x_d + F
        return advection - state + self.forcing

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # k₁ = f_ode(x)
        k1 = self.tendency(state)
        # k₂ = f_ode(x + Δt/2 k₁)
        k2 = self.tendency(state + 0.5 * dt * k1)
        # k₃ = f_ode(x + Δt/2 k₂)
        k3 = self.tendency(state + 0.5 * dt * k2)
        # k₄ = f_ode(x + Δt k₃)
        k4 = self.tendency(state + dt * k3)
        # x ← x + Δt/6 (k₁ + 2k₂ + 2k₃ + k₄)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


truth_model = Lorenz96(forcing=8.0)


# user
def rollout(state: Float[Array, " n"], n_steps: int) -> Float[Array, "t n"]:
    """Return the truth trajectory x₀, f(x₀), …, fⁿ(x₀)."""

    def advance(
        carry: Float[Array, " n"], _: None
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        # xₜ₊₁ = f(xₜ)
        next_state = truth_model.step(carry, truth_model.dt)
        return next_state, next_state

    _, later_states = jax.lax.scan(advance, state, length=n_steps)
    # [x₀; x₁ … x_T]
    return einx.id("n, t n -> (1 + t) n", state, later_states)


site = jnp.arange(40.0)  # (D_x,)
# i − j
signed_offset = einx.subtract("i, j -> i j", site, site)  # (D_x, D_x)
# |i − j|
offset = jnp.abs(signed_offset)  # (D_x, D_x)
# dᵢⱼ = min(|i − j|, D_x − |i − j|), the distance round the ring
distance = jnp.minimum(offset, 40.0 - offset)  # (D_x, D_x)
# exp(−dᵢⱼ² / 2ℓ²), ℓ = 2 sites
correlation = jnp.exp(-0.5 * (distance / 2.0) ** 2)  # (D_x, D_x)
# 10⁻³ δᵢⱼ
jitter = 1e-3 * jnp.eye(40)  # (D_x, D_x)
# [Σ_x]ᵢⱼ = exp(−dᵢⱼ² / 2ℓ²) + 10⁻³ δᵢⱼ
background_corr = correlation + jitter  # (D_x, D_x)
# today (lineax)
# Σ_x
prior_cov = lx.MatrixLinearOperator(  # (D_x, D_x) PSD op
    background_corr, lx.positive_semidefinite_tag
)
# today (lineax)
# Σ_ε = I
obs_cov = lx.IdentityLinearOperator(  # (D_x, D_x) op
    jax.ShapeDtypeStruct((40,), jnp.float32)
)

# noise ~ 𝒩(0, I)
spinup_noise = jax.random.normal(key_spinup, (40,))  # (D_x,)
# x_true = f²⁰⁰(F + noise): a state on the attractor
spinup = rollout(8.0 + spinup_noise, 200)  # (D_x,) → (201, D_x)
truth_start = spinup[-1]  # (D_x,)
# Σ_x = L Lᵀ
background_chol = jnp.linalg.cholesky(background_corr)  # (D_x, D_x)
# ξ ~ 𝒩(0, I)
background_noise = jax.random.normal(key_background, (40,))  # (D_x,)
# L ξ
background_error = einx.dot(  # (D_x,)
    "i j, j -> i", background_chol, background_noise
)
# μ_x = x_true + L ξ
background = truth_start + background_error  # (D_x,)
# x_true,t = fᵗ(x_true), t = 0, …, T
truth_trajectory = rollout(truth_start, 10)  # (D_x,) → (T + 1, D_x)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (11, 40))  # (T + 1, D_x)
# yₜ = x_true,t + εₜ
obs = truth_trajectory + obs_noise  # (T + 1, D_x)
# mₜ = 1 on every second site
mask = jnp.zeros((11, 40)).at[:, ::2].set(1.0)  # (T + 1, D_x)
# today (vardax)
window = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", obs),  # (T + 1, D_x) → (N_batch, T + 1, D_x)
    mask=einx.id("t n -> 1 t n", mask),  # (T + 1, D_x) → (N_batch, T + 1, D_x)
)


# user
def rmse(state: Float[Array, " n"]) -> Float[Array, ""]:
    """Root-mean-square error of an initial state against x_true."""
    # (x − x_true)²
    squared_error = (state - truth_start) ** 2  # (D_x,)
    # mean (x − x_true)²
    mean_squared_error = einx.mean("n ->", squared_error)  # → ()
    return jnp.sqrt(mean_squared_error)  # → ()


print(f"RMSE of μ_x {rmse(background):.2f}")
```

It prints `RMSE of μ_x 1.44`.

The basis for every rung is [](#eq-reduced-features) with $D_z = 8$ and $D_z = 16$.

```{code-block} python
:caption: Builds the whitened EOF basis G of [](#eq-reduced-features) for D_z = 8 and D_z = 16, so that Φ = G and Σ_z = I in [](#eq-reduced-prior).
:label: code-reduced-basis
# Shapes: D_x = 40 sites; D_z = 8 or 16 modes

# Σ_x = U diag(σ²) Uᵀ, σ² ascending
eigvals, eigvecs = jnp.linalg.eigh(background_corr)  # (D_x,), (D_x, D_x)
# tr(Σ_x)
total_variance = einx.sum("n ->", eigvals)  # → ()
bases = {}
for n_modes in (8, 16):
    # U_P, σ_P²: the leading D_z eigenpairs
    leading_vecs = eigvecs[:, -n_modes:]  # (D_x, D_z)
    leading_vals = eigvals[-n_modes:]  # (D_z,)
    # σ_P
    mode_scale = jnp.sqrt(leading_vals)  # (D_z,)
    # G = U_P diag(σ_P), so Σ_x ≈ G Gᵀ and Σ_z = I
    features = einx.multiply(  # (D_x, D_z)
        "n p, p -> n p", leading_vecs, mode_scale
    )
    # ‖σ_P‖² / tr(Σ_x)
    trace_share = einx.sum("p ->", leading_vals) / total_variance  # → ()
    # today (vardax)
    # Φ = G with Σ_z = I
    bases[n_modes] = vdx.linear_basis(features, variance=1.0)
    print(f"D_z = {n_modes}: {trace_share:.0%} of tr Σ_x")
```

It prints `D_z = 8: 79% of tr Σ_x` and `D_z = 16: 99% of tr Σ_x`. The block continues the one above.

## Reduced 3DVar

Putting [](#eq-reduced-control) into the 3DVar cost [](#eq-3dvar-cost) gives a cost over $D_z$ numbers, with $\vect{y} \in \mathbb{R}^{D_y}$, $h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ and $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$,

$$\mathcal{J}(\vect{z}) = \frac{1}{2} \vect{z}^\top \gmat{\Sigma}_z^{-1} \vect{z} + \frac{1}{2} \Vert \vect{y} - h(\gvect{\mu}_x + \gmat{\Phi} \vect{z}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}$$ (eq-reduced-3dvar-cost)

whose minimiser $\vect{z}^\star$ gives the analysis $\vect{x}^\star = \gvect{\mu}_x + \gmat{\Phi} \vect{z}^\star$. In vardax this is `ThreeDVar` with `parameterization=LinearParameterization(basis, background)`; the background term of [](#eq-3dvar-cost) is replaced by the penalty, so `prior_cov_op` is not used.

```{code-block} python
:caption: Minimises the full 3DVar cost [](#eq-3dvar-cost) and the reduced cost [](#eq-reduced-3dvar-cost) on the observations at the initial time.
:label: code-reduced-3dvar
# Shapes: N_batch = 1 window; D_x = 40 sites; D_z = 8 or 16 modes

# today (vardax)
# y₀, m₀: the observations at the initial time only
snapshot = vdx.Batch1D(  # (N_batch, D_x)
    input=window.input[:, 0], mask=window.mask[:, 0]
)
# today (vardax)
# x⋆ = argmin 𝒥(x), the full control
full_3dvar = vdx.ThreeDVar(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    cov_solver=lx.Cholesky(),
)
full_3dvar_analysis = full_3dvar(snapshot)  # → (N_batch, D_x)
print(f"3D-Var, full: RMSE {rmse(full_3dvar_analysis[0]):.2f}")
for n_modes, basis in bases.items():
    # today (vardax)
    # x = μ_x + Φ z, ℛ(z) = ½ zᵀ Σ_z⁻¹ z
    reduced_control = vdx.LinearParameterization(basis, background=background)
    # today (vardax)
    # 𝒥(z) = ½ ‖z‖² + ½ ‖m ⊙ (y − (μ_x + Φ z))‖²_Σ_ε⁻¹
    reduced_3dvar = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=prior_cov,
        obs_cov_op=obs_cov,
        parameterization=reduced_control,
    )
    reduced_3dvar_analysis = reduced_3dvar(snapshot)  # → (N_batch, D_x)
    error = rmse(reduced_3dvar_analysis[0])
    print(f"3D-Var, D_z = {n_modes}: RMSE {error:.2f}")
```

It prints `3D-Var, full: RMSE 0.53`, `3D-Var, D_z = 8: RMSE 0.57` and `3D-Var, D_z = 16: RMSE 0.52`. With 99 % of the trace of $\gmat{\Sigma}_x$ in 16 modes, the reduced analysis is as good as the full one; with 8 modes the correction misses the smaller scales.

## Reduced strong-constraint 4DVar

The same substitution in the strong-constraint cost [](#eq-strong4dvar-cost) gives

$$\mathcal{J}(\vect{z}) = \frac{1}{2} \vect{z}^\top \gmat{\Sigma}_z^{-1} \vect{z} + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{m}_t \odot \left(\vect{y}_t - h_t(f^t(\gvect{\mu}_x + \gmat{\Phi} \vect{z}))\right) \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}$$ (eq-reduced-basis-cost)

where $f^t$ is the forward model $f$ applied $t$ times, $\vect{y}_t \in \mathbb{R}^{D_y}$, $\vect{m}_t \in \{0, 1\}^{D_y}$ is the observation mask at time $t$, $h_t$ is the observation operator and $\gmat{\Sigma}_{\varepsilon,t} \in \mathbb{R}^{D_y \times D_y}$ the observation-error covariance. It is minimised directly, with no linearisation, outer loop or inner CG. The gradient is the state-space adjoint projected onto the basis,

$$\nabla_{\vect{z}} \mathcal{J} = \gmat{\Sigma}_z^{-1} \vect{z} + \gmat{\Phi}^\top \vect{x}^\dagger_0$$ (eq-reduced-basis-grad)

where $\vect{x}^\dagger_0 \in \mathbb{R}^{D_x}$ is the adjoint state at $t = 0$ from the backward sweep [](#eq-adjoint-recursion). One cost-and-gradient evaluation costs one forward and one adjoint run, as in state space. The gain is that the minimiser works in $D_z$ dimensions, and when $D_z$ is a few tens even forward-mode differentiation with $D_z$ tangent runs and the full $D_z \times D_z$ Hessian become affordable. `ReducedBasisFourDVar(basis=...)` minimises [](#eq-reduced-basis-cost); with `basis=None` it is the same as `StrongFourDVar`, and `basis=` is equivalent to `StrongFourDVar(parameterization=LinearParameterization(basis, background))`.

```{code-block} python
:caption: Minimises the full strong-constraint cost [](#eq-strong4dvar-cost) and the reduced cost [](#eq-reduced-basis-cost) with one first-order minimiser.
:label: code-reduced-strong
# Shapes: N_batch = 1 window; T + 1 = 11 times; D_x = 40 sites; D_z = 8 or 16 modes

# today (vardax)
# one first-order minimiser for every control, so only the control differs
first_order = vdx.OptaxMinimiser(optax.adam(1e-2), max_steps=2000)
# today (vardax)
# x₀⋆ = argmin 𝒥(x₀), the full control
full_strong = vdx.StrongFourDVar(
    forward=truth_model,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=first_order,
    cov_solver=lx.Cholesky(),
)
full_strong_analysis = full_strong(window)  # → (N_batch, D_x)
print(f"strong 4D-Var, full: RMSE {rmse(full_strong_analysis[0]):.2f}")
for n_modes, basis in bases.items():
    # today (vardax)
    # 𝒥(z) = ½ ‖z‖² + ½ ∑ₜ ‖mₜ ⊙ (yₜ − h(fᵗ(μ_x + Φ z)))‖²_Σ_ε⁻¹
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
    reduced_strong_analysis = reduced_strong(window)  # → (N_batch, D_x)
    error = rmse(reduced_strong_analysis[0])
    print(f"strong 4D-Var, D_z = {n_modes}: RMSE {error:.2f}")
```

It prints `strong 4D-Var, full: RMSE 0.26`, `strong 4D-Var, D_z = 8: RMSE 0.64` and `strong 4D-Var, D_z = 16: RMSE 0.27`. Every control uses the same Adam minimiser, so only the control differs. A chaotic forward can return non-finite values on an aggressive quasi-Newton trial step, and the masked $\gmat{\Sigma}_\varepsilon$ solve raises on them; a first-order minimiser with a small step avoids it on this window. `ReducedBasisFourDVar` was called `Incremental4DVar` until vardax 0.1.10; the old name still works and emits a `DeprecationWarning`.

## Reduced incremental 4DVar

Incremental 4DVar [](#eq-incremental-cost) linearises around the current iterate and solves a quadratic inner problem by CG, and with the control-variable transform [](#eq-cvt) its unknowns are whitened. The reduced form keeps the same outer loop, which still runs the full nonlinear model, but restricts the increment to the basis, $\delta\vect{x} = \gmat{\Phi} \delta\vect{z}$. With the whitened basis ($\gmat{\Sigma}_z = \mat{I}$) the inner cost at outer iterate $\vect{z}^{(i)}$ is

$$\delta\mathcal{J}(\delta\vect{z}) = \frac{1}{2} \Vert \vect{z}^{(i)} + \delta\vect{z} \Vert_2^2 + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{r}_t^{(i)} - \mat{H}_t \mat{F}_t \gmat{\Phi} \delta\vect{z} \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}$$ (eq-reduced-incremental-cost)

where $\vect{r}_t^{(i)} = \vect{y}_t - h_t(f^t(\gvect{\mu}_x + \gmat{\Phi} \vect{z}^{(i)})) \in \mathbb{R}^{D_y}$ is the innovation of the current trajectory, and $\mat{F}_t \in \mathbb{R}^{D_x \times D_x}$, $\mat{H}_t \in \mathbb{R}^{D_y \times D_x}$ are the tangent-linear operators of [](#eq-tlm-forward) and [](#eq-tlm-obs). Its normal equations are a $D_z \times D_z$ system,

$$\left(\mat{I} + \sum_{t=0}^{T} (\mat{H}_t \mat{F}_t \gmat{\Phi})^\top \gmat{\Sigma}_{\varepsilon,t}^{-1} (\mat{H}_t \mat{F}_t \gmat{\Phi})\right) \delta\vect{z}^\star = -\vect{z}^{(i)} + \sum_{t=0}^{T} (\mat{H}_t \mat{F}_t \gmat{\Phi})^\top \gmat{\Sigma}_{\varepsilon,t}^{-1} \vect{r}_t^{(i)}$$ (eq-reduced-incremental-system)

which is the CVT system [](#eq-cvt-system) with the square root $\gmat{\Sigma}_x^{1/2}$ replaced by the rectangular $\gmat{\Phi}$. Being $D_z \times D_z$, it is small enough to factor directly instead of iterating. The outer update is $\vect{z}^{(i+1)} = \vect{z}^{(i)} + \delta\vect{z}^\star$, and the analysis is $\vect{x}_0^\star = \gvect{\mu}_x + \gmat{\Phi} \vect{z}^\star$, the same minimiser as [](#eq-reduced-basis-cost).

In vardax this is `IncrementalFourDVar` with a rectangular `prior_sqrt_op`: the control takes the input size of the square root, so passing $\gmat{\Phi}$ in place of $\gmat{\Sigma}_x^{1/2}$ makes the reduced form a strict generalisation of the CVT. `ReducedBasisFourDVar` minimises the same cost by quasi-Newton or a first-order minimiser instead.

```{code-block} python
:caption: The reduced incremental solve of [](#eq-reduced-incremental-system), with a rectangular square root Φ in place of Σ_x^½ in [](#eq-cvt).
:label: code-reduced-incremental
# Shapes: N_batch = 1 window; D_x = 40 sites; D_z = 16 modes

# today (lineax)
# Φ = G, the whitened D_z = 16 basis, as a (D_x, D_z) operator
basis_sqrt = lx.MatrixLinearOperator(bases[16].design_matrix)  # (D_x, D_z) op
# today (vardax)
# x₀ = μ_x + Φ z, z ~ 𝒩(0, I): the inner system is D_z × D_z
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
# x₀⋆ = μ_x + Φ z⋆
reduced_incremental_analysis = reduced_incremental(window)  # → (N_batch, D_x)
error = rmse(reduced_incremental_analysis[0])
print(f"incremental 4D-Var, D_z = 16: RMSE {error:.2f}")
```

It prints `incremental 4D-Var, D_z = 16: RMSE 0.28`: three Gauss-Newton outer loops over the same 16 coefficients land close to the first-order minimiser's 0.27.

## Reduced weak-constraint 4DVar

Weak-constraint 4DVar [](#eq-weak4dvar-cost) adds a model error $\gvect{\eta}_t \in \mathbb{R}^{D_x}$ after every step, $D_x T$ more unknowns. If that error has a shape one can name, like a bias, a slow drift or a large-scale pattern, it is much better to estimate the few numbers of that shape. Let $\gmat{\chi} \in \mathbb{R}^{T \times C}$ be a temporal basis, for example Legendre polynomials over the window, $\phi_d$ the columns of a spatial basis $\gmat{\Phi} \in \mathbb{R}^{D_x \times P}$, and $\mat{W} \in \mathbb{R}^{C \times P}$ the coefficients. The model error and its prior are

$$\eta(s_n, t) = \sum_{c=1}^{C} \sum_{d=1}^{P} \chi_c(t) W_{cd} \phi_d(s_n), \text{ with } \op{vec}_r(\mat{W}) = (\gmat{\Sigma}_\text{time} \otimes \gmat{\Sigma}_\text{space})^{1/2} \gvect{\omega}, \text{ and } \gvect{\omega} \sim \mathcal{N}(\mathbf{0}, \mat{I})$$ (eq-reduced-model-error)

where $\gvect{\omega} \in \mathbb{R}^{C P}$, $\op{vec}_r$ stacks $\mat{W}$ row by row and $\gmat{\Sigma}_\text{time} \in \mathbb{R}^{C \times C}$, $\gmat{\Sigma}_\text{space} \in \mathbb{R}^{P \times P}$ are the diagonal prior variances of the temporal and spatial coefficients. The implied model-error covariance is $\gmat{\Sigma}_\eta = (\gmat{\chi} \otimes \gmat{\Phi})(\gmat{\Sigma}_\text{time} \otimes \gmat{\Sigma}_\text{space})(\gmat{\chi} \otimes \gmat{\Phi})^\top$, never formed. The cost over the control $(\vect{x}_0, \gvect{\omega})$ is

$$\mathcal{J}(\vect{x}_0, \gvect{\omega}) = \frac{1}{2} \Vert \vect{x}_0 - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \Vert \gvect{\omega} \Vert_2^2 + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{y}_t - h_t(\vect{x}_t) \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}, \text{ with } \vect{x}_t = f(\vect{x}_{t-1}) + \gvect{\eta}_t$$ (eq-reduced-weak4dvar-cost)

so the control has $D_x + C P$ numbers instead of $D_x (T + 1)$. A constant bias is the smallest case: $C = 1$ and $P = 1$, one coefficient. The initial state can be reduced as well, with [](#eq-reduced-control) for $\vect{x}_0$.

`WeakFourDVar(parameterization=LinearParameterization(...))` reduces the initial state and `model_err_basis=` the model error. Below we first run the window with a biased model, $F = 7$ against the $F = 8$ truth, and we measure the error at the end of the window, where the bias has built up.

```{code-block} python
:caption: Compares strong-constraint 4DVar [](#eq-strong4dvar-cost) with a biased model against weak-constraint 4DVar [](#eq-weak4dvar-cost) with a full and a reduced [](#eq-reduced-control) initial state.
:label: code-reduced-weak
# Shapes: N_batch = 1 window; T = 10 steps; D_x = 40 sites; D_z = 16 modes

# f with F = 7: the analysis model is biased against the F = 8 truth
biased_model = Lorenz96(forcing=7.0)


# user
def window_end(
    initial_state: Float[Array, " n"], model_errors: Float[Array, "k n"]
) -> Float[Array, " n"]:
    """Run the biased model over the window, adding ηₜ after each step."""

    def advance(
        carry: Float[Array, " n"], model_error: Float[Array, " n"]
    ) -> tuple[Float[Array, " n"], None]:
        # xₜ = f(xₜ₋₁) + ηₜ
        next_state = biased_model.step(carry, biased_model.dt) + model_error
        return next_state, None

    final_state, _ = jax.lax.scan(advance, initial_state, model_errors)
    return final_state  # (D_x,)


# user
def end_rmse(state: Float[Array, " n"]) -> Float[Array, ""]:
    """Root-mean-square error against the truth at the end of the window."""
    # (x_T − x_true,T)²
    squared_error = (state - truth_trajectory[-1]) ** 2  # (D_x,)
    # mean (x_T − x_true,T)²
    mean_squared_error = einx.mean("n ->", squared_error)  # → ()
    return jnp.sqrt(mean_squared_error)  # → ()


no_errors = jnp.zeros((10, 40))  # (T, D_x)
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
biased_initial = biased_strong(window)[0]  # (N_batch, D_x) → (D_x,)
biased_end = window_end(biased_initial, no_errors)  # (D_x,)
print(f"strong 4D-Var, biased model: end-of-window RMSE {end_rmse(biased_end):.2f}")
# x_T from the exact model and the full strong analysis, for reference
exact_end = rollout(full_strong_analysis[0], 10)[-1]  # (D_x,)
print(f"strong 4D-Var, exact model: end-of-window RMSE {end_rmse(exact_end):.2f}")
# today (lineax)
# Σ_η = σ_q² I, σ_q = 0.1
model_err_cov = lx.TaggedLinearOperator(0.01 * obs_cov, lx.positive_semidefinite_tag)
# today (vardax)
# x₀ = μ_x + Φ z with the D_z = 16 basis
reduced_initial = vdx.LinearParameterization(bases[16], background=background)
for label, control in (("full x₀", None), ("D_z = 16 x₀", reduced_initial)):
    # today (vardax)
    # 𝒥(x₀, η₁, …, η_T) with the model errors ηₜ in full
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
    # → (N_batch, D_x), (N_batch, T, D_x)
    initial_states, model_errors = weak(window)
    weak_end = window_end(initial_states[0], model_errors[0])  # (D_x,)
    print(f"weak 4D-Var, {label}: end-of-window RMSE {end_rmse(weak_end):.2f}")
```

It prints `strong 4D-Var, biased model: end-of-window RMSE 0.55`, `strong 4D-Var, exact model: end-of-window RMSE 0.42`, `weak 4D-Var, full x₀: end-of-window RMSE 0.53` and `weak 4D-Var, D_z = 16 x₀: end-of-window RMSE 0.54`. The weak constraint barely helps here, and changing $\sigma_q$ to 0.05 or 0.32 gives 0.54 and 0.58: with 400 model-error unknowns against 220 observations, an error at every point and time cannot separate the bias from noise. A reduced model error with one constant mode is exactly that separation, and on a longer Lorenz-96 window the esmax page reports it recovering the bias as 1.00 ± 0.04 (truth 1.0). In vardax that is `WeakFourDVar(model_err_basis=...)`, below.

```{code-block} python
:caption: The reduced weak-constraint 4DVar of [](#eq-reduced-weak4dvar-cost), with one constant mode in time and space for the bias in [](#eq-reduced-model-error).
:label: code-reduced-weak-model-error
# Shapes: N_batch = 1 window; T = 10 steps; D_x = 40 sites; C = 1 temporal mode;
#         P = 1 spatial mode

# χ_1(t) = 1, one constant temporal mode
temporal_basis = jnp.ones((10, 1))  # (T, C)
# φ_1(s) = 1, one constant spatial mode
spatial_basis = jnp.ones((40, 1))  # (D_x, P)
# today (vardax)
# η(sₙ, t) = ∑_c ∑_d χ_c(t) W_cd φ_d(sₙ), vec_r(W) ~ 𝒩(0, Σ_time ⊗ Σ_space)
error_basis = vdx.ModelErrorBasis(
    temporal=temporal_basis,
    spatial=spatial_basis,
    temporal_variance=jnp.ones(1),
    spatial_variance=jnp.ones(1),
)
# today (vardax)
# 𝒥(x₀, ω) over D_x + C P = 41 numbers
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
# (x₀⋆, W⋆): W⋆ is the estimated bias per step
# → (N_batch, D_x), (N_batch, C, P)
initial_states, error_coefficients = reduced_weak(window)
# η = χ W Φᵀ, the bias added after every step
reduced_errors = einx.dot(  # (T, D_x)
    "t c, c p, n p -> t n", temporal_basis, error_coefficients[0], spatial_basis
)
reduced_end = window_end(initial_states[0], reduced_errors)  # (D_x,)
print(f"weak 4D-Var, one bias mode: end-of-window RMSE {end_rmse(reduced_end):.2f}")
print(f"estimated bias per step {float(error_coefficients[0, 0, 0]):.3f}")
```

It prints `weak 4D-Var, one bias mode: end-of-window RMSE 0.43` and `estimated bias per step 0.047`. One coefficient brings the biased model to within 0.01 of the exact model's 0.42, and the bias it finds is the forcing gap times the step, $(8 - 7) \times 0.05 = 0.05$, to first order.

## Uncertainty

A reduced control has a reduced posterior: the Laplace precision lives in coefficient space, $D_z \times D_z$, and maps to a rank-$D_z$ state covariance, [](#eq-reduced-basis-cov) in chapter [13](13_posterior_covariance.md). The posterior adapters build it for any linear control, so `posterior(batch)` works for reduced 3DVar and reduced strong-constraint 4DVar. The spread outside the span of $\gmat{\Phi}$ is exactly zero, which says more about the basis than about the data.

## Choosing the basis and its size

- **EOFs of $\gmat{\Sigma}_x$** ([](#eq-reduced-features)) are the best rank-$D_z$ approximation of the prior, so they are the default when $\gmat{\Sigma}_x$ is known; the trace share $\Vert \gvect{\sigma}_P \Vert_2^2 / \mathrm{tr}(\gmat{\Sigma}_x)$ says how much prior variance the control can still correct.
- **EOFs of a model run** (`eof_basis`) adapt the control to the dynamics rather than to the prior, as in reduced-order incremental 4DVar for ocean models.
- **Fourier or RBF bases** (`fourier_basis`, `rbf_basis`) suit a stationary or smooth correction and need no sample; wavelets (`wavelet_basis`) keep local features.
- **Size**: we increase $D_z$ until the analysis stops improving. On the running example the RMSE goes from 0.64 to 0.27 between $D_z = 8$ and $D_z = 16$ (79 % and 99 % of the trace), and the full control gives 0.26.

## See also

- Chapter [5](05_threedvar.md), [6](06_strong_4dvar.md), [7](07_weak_4dvar.md) and [8](08_incremental_4dvar.md): the full methods.
- Chapter [13](13_posterior_covariance.md): the reduced posterior [](#eq-reduced-basis-cov).
- [#154](https://github.com/jejjohnson/vardax/issues/154): the epic for a reduced form of every method.

## References

- Robert, C., Durbiano, S., Blayo, E., Verron, J., Blum, J. and Le Dimet, F.-X. (2005). *A reduced-order strategy for 4D-Var data assimilation.* Journal of Marine Systems 57, 70-82. doi:10.1016/j.jmarsys.2005.04.003.
- Trémolet, Y. (2006). *Accounting for an imperfect model in 4D-Var.* QJRMS 132(621).
