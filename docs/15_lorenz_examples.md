---
numbering:
  enumerator: "15.%s"
---

# Lorenz Examples

The Lorenz systems are the canonical small-scale testbeds for DA.
Their chaotic dynamics expose what each method does well or badly in a
setting where ground truth is available, integration is cheap, and the
state dimension is small enough to visualise.

This chapter walks the seven analysis methods through Lorenz-63 (3D
chaotic system) and Lorenz-96 (40D periodic system). The goal is to
make each method's behaviour concrete.

## Lorenz-63

The Lorenz-63 system is

$$\dot{x} = \sigma(y - x), \dot{y} = x(\rho - z) - y, \dot{z} = xy - \beta z,$$ (eq-lorenz63)

where $(x, y, z)$ are the three components of the state $\vect{x}_t \in \mathbb{R}^{D_x}$ with $D_x = 3$, and $\sigma$, $\rho$, $\beta \in \mathbb{R}$ are its three dimensionless parameters, here the canonical $(10, 28, 8/3)$.
These three letters are the physics symbols of Lorenz-63, not a standard deviation, a density or a regularisation weight; the standard deviation of the observation noise is written $\sigma_o$.
The dynamics live on the famous two-winged attractor.

The code on this page runs top to bottom as one script; each block continues the one above.
The keys for every stochastic stage are split once, here.

```{code-block} python
:caption: Splits the page's PRNG keys and integrates the Lorenz-63 system [](#eq-lorenz63).
:label: code-lorenz-l63-simulate
import einx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N_sim = 5,001 saved times; D_x = 3 state components

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 5)
key_l63, key_background, key_net, key_l96, key_background_l96 = keys

# today (vardax)
# ẋ = σ (y − x), ẏ = x (ρ − z) − y, ż = x y − β z, integrated with Tsit5
time, trajectory = vdx.simulate_lorenz63(  # (N_sim,), (N_sim, D_x)
    key_l63, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, n_steps=5000
)
print(trajectory.shape)
```

It prints `(5001, 3)`; the default burn-in of 1,000 steps is dropped.
Here `time` holds the $N_\text{sim}$ time stamps and `trajectory` the state $\vect{x}_t$ at each of them.

### Synthetic observation scenario

Take one window of $T = 100$ steps, which holds $T + 1 = 101$ times (one time unit, about one Lyapunov time), and observe it every tenth step, with Gaussian noise of standard deviation $\sigma_o = 1$: $\vect{y}_t = \vect{x}_t + \gvect{\varepsilon}_t$ with $\gvect{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \gmat{\Sigma}_\varepsilon)$, $\gmat{\Sigma}_\varepsilon = \sigma_o^2 \mat{I}$ and $D_y = D_x$.
The data utilities are private helpers under `vardax._src.utils`, which the tutorial notebooks also use.
`xr_to_batch1d` expects a `patch` dimension, so the window gets one.

```{code-block} python
:caption: Builds the masked, noisy Lorenz-63 window, the observations of [](#eq-obs-model) with the masked identity [](#eq-masked-identity) as $\mat{H}$.
:label: code-lorenz-l63-observations
import vardax._src.utils.masks as masks
import vardax._src.utils.noise as noise
import vardax._src.utils.patches as patches
import vardax._src.utils.preprocessing as preprocessing

# Shapes: S = 1 window; T = 100 steps, so T + 1 = 101 times; D_x = D_y = 3

# today (vardax._src.utils)
dataset = patches.trajectory_to_xr_dataset(trajectory, time)  # (N_sim, D_x)
window = dataset.isel(time=slice(500, 601))  # (N_sim, D_x) → (T + 1, D_x)
window = window.expand_dims("patch")  # (T + 1, D_x) → (S, T + 1, D_x)
# today (vardax._src.utils)
# mₜ = 1 for t = 0, 10, 20, …
window = masks.regular_mask(window, variable="state", obs_interval=10)
# today (vardax._src.utils)
# yₜ = xₜ + σ_o εₜ, σ_o = 1, εₜ ~ 𝒩(0, I)
window = noise.add_gaussian_noise(window, variable="state", sigma=1.0, name="obs")
# today (vardax._src.utils)
batch = preprocessing.xr_to_batch1d(
    window, state_var="state", obs_var="obs", mask_var="mask"
)
truth = batch.target[0]  # (S, T + 1, D_x) → (T + 1, D_x)
```

### `OptimalInterpolation` won't work directly

OI ([chapter 4](04_oi_blue.md)) assumes a static Gaussian field, which the nonlinear Lorenz dynamics violate.
So OI with the climatological mean $\bar{\vect{x}}$ and covariance $\hat{\gmat{\Sigma}}_x$ of the trajectory as its prior gives a poor reconstruction, but it is a useful baseline.
Each time is treated separately, with $\gvect{\mu}_x = \bar{\vect{x}}$, $\gmat{\Sigma}_x = \hat{\gmat{\Sigma}}_x$ and $\gmat{\Sigma}_\varepsilon = \mat{I}$.

```{code-block} python
:caption: Computes the BLUE mean [](#eq-blue-mean) at each time of the window from the climatological prior, with the masked identity [](#eq-masked-identity) as $\mat{H}$.
:label: code-lorenz-l63-oi
# Shapes: N_sim = 5,001; T + 1 = 101 times; D_x = D_y = 3


# user
def rmse(
    estimate: Float[Array, "w n"], reference: Float[Array, "w n"]
) -> Float[Array, ""]:
    # e = x̂ − x
    error = estimate - reference
    # e²
    squared_error = error**2
    # (1/((T + 1) D_x)) ∑ e²
    mean_squared_error = einx.mean("w n ->", squared_error)
    # RMSE = √((1/((T + 1) D_x)) ∑ e²)
    return jnp.sqrt(mean_squared_error)


# x̄ = (1/N_sim) ∑ₜ xₜ
climatology_mean = einx.mean("t n -> n", trajectory)  # (N_sim, D_x) → (D_x,)
# x̃ₜ = xₜ − x̄
anomaly = einx.subtract("t n, n -> t n", trajectory, climatology_mean)  # (N_sim, D_x)
# ∑ₜ x̃ₜ x̃ₜᵀ
scatter = einx.dot("t n, t m -> n m", anomaly, anomaly)  # (N_sim, D_x) → (D_x, D_x)
# Σ̂_x = ∑ₜ x̃ₜ x̃ₜᵀ / (N_sim − 1)
climatology_cov = scatter / (trajectory.shape[0] - 1)  # (D_x, D_x)
window_structure = jax.ShapeDtypeStruct((101, 3), jnp.float32)


# user
def apply_static_cov(state: Float[Array, "w n"]) -> Float[Array, "w n"]:
    # (Σ_x v)ₜ = Σ̂_x vₜ, the same Σ̂_x at every time, no correlation across times
    return einx.dot("w n, n m -> w m", state, climatology_cov)


# today (lineax)
static_cov = lx.FunctionLinearOperator(  # ((T + 1) D_x, (T + 1) D_x) op
    apply_static_cov,
    window_structure,
    tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
)
# today (lineax)
# Σ_ε = I
unit_obs_cov = lx.IdentityLinearOperator(  # ((T + 1) D_x, (T + 1) D_x) op
    window_structure
)
# μ_x = x̄ at every time
static_mean = einx.id("n -> w n", climatology_mean, w=101)  # (D_x,) → (T + 1, D_x)
# today (vardax)
# h(x) = m ⊙ x
masked_identity = vdx.MaskedIdentity()

# today (vardax)
# Treat as static: wrong but instructive
oi = vdx.OptimalInterpolation(
    obs_op=masked_identity,
    prior_mean=static_mean,
    prior_cov_op=static_cov,
    obs_cov_op=unit_obs_cov,
)
# today (vardax)
# x⋆ = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x), at each time separately
oi_analysis = oi(batch)[0]  # (S, T + 1, D_x) → (T + 1, D_x)
# user
oi_rmse = rmse(oi_analysis, truth)  # → ()
print(f"OI trajectory RMSE {oi_rmse:.2f}")
```

It prints `OI trajectory RMSE 6.06`.

### `StrongFourDVar` is the right tool

The background $\gvect{\mu}_x$ is the true initial state $\vect{x}_\text{true}$ of the window plus noise of standard deviation $\sigma_b = 2$, and the prior $\gmat{\Sigma}_x = 100 \mat{I}$ is deliberately weak.
$\mat{H}_t$ is the masked identity and $\gmat{\Sigma}_\varepsilon = \mat{I}$.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) for the Lorenz-63 window and rolls the analysis out with [](#eq-rollout).
:label: code-lorenz-l63-strong4dvar
# Shapes: T = 100 steps, so T + 1 = 101 times; D_x = D_y = 3

# today (vardax)
lorenz63 = vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
# today (vardax)
trajectory_model = vdx.DynTrajectory(model=lorenz63)
# today (vardax)
# f: xₜ ↦ xₜ₊₁, one Δt = 0.01 ODE step
forward = trajectory_model.as_forward_model(dt=0.01)
state_structure = jax.ShapeDtypeStruct((3,), jnp.float32)
# ξ ~ 𝒩(0, I)
background_noise = jax.random.normal(key_background, (3,))  # (D_x,)
# σ_b ξ, σ_b = 2
background_offset = 2.0 * background_noise  # (D_x,)
# μ_x = x_true + σ_b ξ
background = truth[0] + background_offset  # (D_x,)
# today (lineax)
# I
state_identity = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# 100 I
scaled_identity = 100.0 * state_identity  # (D_x, D_x) op
# today (lineax)
# Σ_x = 100 I, tagged PSD so the default CG solve accepts it
weak_prior_cov = lx.TaggedLinearOperator(  # (D_x, D_x) PSD op
    scaled_identity, lx.positive_semidefinite_tag
)
# today (optimistix)
minimiser = optx.NonlinearCG(rtol=1e-6, atol=1e-6)

# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖mₜ ⊙ (yₜ − fₜ(x₀))‖²_Σ_ε⁻¹, Σ_ε = I
strong = vdx.StrongFourDVar(
    forward=forward,
    obs_op=masked_identity,
    prior_mean=background,
    prior_cov_op=weak_prior_cov,
    obs_cov_op=state_identity,
    minimiser=minimiser,
)
# today (vardax)
# x₀⋆ = argmin 𝒥(x₀)
initial_analysis = strong(batch)[0]  # (S, D_x) → (D_x,)
step_index = jnp.arange(101)  # (T + 1,)
# t_k = k Δt, k = 0, …, T
window_times = 0.01 * step_index  # (T + 1,)
# today (vardax)
# xₜ = fₜ(x₀⋆)
strong_analysis = trajectory_model(initial_analysis, window_times)  # (T + 1, D_x)
# user
strong_rmse = rmse(strong_analysis, truth)  # → ()
print(f"strong 4DVar trajectory RMSE {strong_rmse:.2f}")
```

It prints `strong 4DVar trajectory RMSE 0.28`.

Diagnostic: plot the attractor reconstruction. `StrongFourDVar`
recovers the butterfly; `OptimalInterpolation` smears across both
wings because the static prior doesn't know about the dynamics.

### `FourDVarNet` for the learned-prior comparison

Train a small AE prior on short patches of the training trajectory.
Two epochs of `vardax.examples.fit_demo` only show the plumbing; notebook 03 trains it properly.

```{code-block} python
:caption: Trains the learned-prior solver of [](#eq-4dvarnet-cost) and [](#eq-4dvarnet-update) on the loss [](#eq-4dvarnet-loss) for two epochs.
:label: code-lorenz-l63-4dvarnet
# Shapes: N_patch = 32 patches; N_batch = 8 patches per batch;
#         T_patch = 19 steps, so T_patch + 1 = 20 times per patch; D_x = D_y = 3

# today (vardax._src.utils)
train_patches = patches.extract_patches(  # (N_patch, T_patch + 1, D_x)
    dataset, n_patches=32, n_timesteps=20, seed=0
)
# today (vardax._src.utils)
# mₜ = 1 for t = 0, 2, 4, …
train_patches = masks.regular_mask(train_patches, variable="state", obs_interval=2)
# today (vardax._src.utils)
# yₜ = xₜ + σ_o εₜ, σ_o = 1
train_patches = noise.add_gaussian_noise(
    train_patches, variable="state", sigma=1.0, name="obs"
)
train_batches = []
for start in range(0, 32, 8):
    batch_patches = train_patches.isel(  # → (N_batch, T_patch + 1, D_x)
        patch=slice(start, start + 8)
    )
    # today (vardax._src.utils)
    train_batch = preprocessing.xr_to_batch1d(  # → (N_batch, T_patch + 1, D_x)
        batch_patches
    )
    train_batches.append(train_batch)

# today (vardax)
# builds a BilinAEPrior1D prior and a ConvLSTMGradMod1D modulator
net = vdx.FourDVarNet1D(
    state_dim=3,
    n_time=20,
    latent_dim=8,
    hidden_dim=32,
    n_solver_steps=15,
    alpha=0.2,
    solver_adjoint=vdx.OneStepAdjoint(),
    key=key_net,
)
# today (vardax.examples)
# θ⋆ = argmin_θ ∑ ‖x̂(y; θ) − x‖², Adam, two epochs
net, train_losses, _ = vdx.examples.fit_demo(
    net, train_batches, n_epochs=2, verbose=False
)
# today (vardax)
# x̂ = x^(N_iter) after N_iter = 15 unrolled solver steps
net_analysis = net(train_batches[0])  # (N_batch, T_patch + 1, D_x)
print(net_analysis.shape)
```

It prints `(8, 20, 3)`.

`FourDVarNet1D` learns the attractor structure from the training data
and reconstructs without explicit access to the dynamics. The
comparison `strong_analysis` vs `net_analysis` is the canonical demonstration that
learned priors recover similar information to model-based ones.

## Lorenz-96

The Lorenz-96 system is a one-dimensional periodic version:

$$\dot{x}_k = (x_{k+1} - x_{k-2}) x_{k-1} - x_k + u, \text{ for } k = 0, \ldots, D_x - 1,$$ (eq-lorenz96)

where $x_k$ is the component at site $k$ of the state $\vect{x} \in \mathbb{R}^{D_x}$, a periodic ring (indices taken modulo $D_x$), and $u$ is the forcing.
We use $D_x = 40$, which replaces the Lorenz-63 value of 3 until the linear-Gaussian test below, and $u = 8$, which is the chaotic regime.
The state is a periodic 1D field, closer in structure to atmospheric problems than Lorenz-63.

```{code-block} python
:caption: Integrates the Lorenz-96 system [](#eq-lorenz96) on a ring of 40 sites.
:label: code-lorenz-l96-simulate
# Shapes: N_sim = 5,001 saved times; D_x = 40 sites

# today (vardax)
# ẋₖ = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + u
time_l96, trajectory_l96 = vdx.simulate_lorenz96(  # (N_sim,), (N_sim, D_x)
    key_l96, n_sites=40, forcing=8.0, dt=0.01, n_steps=5000
)
print(trajectory_l96.shape)
```

It prints `(5001, 40)`.

### Sparse observation scenario

Observe a random half of the cells and times in one window of $T = 20$ steps, which holds $T + 1 = 21$ times, with noise of standard deviation $\sigma_o = 0.5$:

```{code-block} python
:caption: Builds the sparse, noisy Lorenz-96 window, the observations of [](#eq-obs-model) with the masked identity [](#eq-masked-identity) as $\mat{H}$.
:label: code-lorenz-l96-observations
# Shapes: S = 1 window; T = 20 steps, so T + 1 = 21 times; D_x = D_y = 40 sites

site_names = [f"x{site}" for site in range(40)]
# today (vardax._src.utils)
dataset_l96 = patches.trajectory_to_xr_dataset(  # (N_sim, D_x)
    trajectory_l96, time_l96, feature_names=site_names
)
window_l96 = dataset_l96.isel(time=slice(500, 521))  # (N_sim, D_x) → (T + 1, D_x)
window_l96 = window_l96.expand_dims("patch")  # (T + 1, D_x) → (S, T + 1, D_x)
# today (vardax._src.utils)
# m_{t,k} ~ Bernoulli(½)
window_l96 = masks.random_mask(window_l96, variable="state", missing_rate=0.5)
# today (vardax._src.utils)
# yₜ = xₜ + σ_o εₜ, σ_o = 0.5
window_l96 = noise.add_gaussian_noise(window_l96, variable="state", sigma=0.5)
# today (vardax._src.utils)
batch_l96 = preprocessing.xr_to_batch1d(window_l96)
truth_l96 = batch_l96.target[0]  # (S, T + 1, D_x) → (T + 1, D_x)
```

### `IncrementalFourDVar` with structured Matérn prior

This is the realistic operational scenario: structured spatial prior, incremental solver ([chapter 8](08_incremental_4dvar.md)), AK-style or masked observations.

The Matérn-3/2 covariance on the periodic grid with length scale $\ell = 4$ sites and unit variance depends only on the ring distance $d_{ij} = \min(\vert i - j \vert, D_x - \vert i - j \vert)$ between sites $i$ and $j$, so it is circulant; here it is a dense $D_x \times D_x$ matrix.
`gaussx.Toeplitz` would store it by its first column; the dense matrix keeps the example short.
The forward model is one RK4 step written on the page, which forward- and reverse-mode autodiff both see through, as the Gauss-Newton loop needs.
The background is the true initial state plus noise drawn from the same scale as $\gmat{\Sigma}_x$.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) for Lorenz-96 through the incremental cost [](#eq-incremental-cost) with a periodic Matérn prior, then builds the Gauss-Newton posterior [](#eq-gn-cov).
:label: code-lorenz-l96-incremental
import equinox as eqx

# Shapes: T = 20 steps, so T + 1 = 21 times; D_x = D_y = 40 sites

site_index = jnp.arange(40.0)  # (D_x,)
# i − j
site_difference = einx.subtract("i, j -> i j", site_index, site_index)  # (D_x, D_x)
# |i − j|
offset = jnp.abs(site_difference)  # (D_x, D_x)
# D_x − |i − j|
wrapped_offset = 40.0 - offset  # (D_x, D_x)
# dᵢⱼ = min(|i − j|, D_x − |i − j|)
ring_distance = jnp.minimum(offset, wrapped_offset)  # (D_x, D_x)
# rᵢⱼ = √3 dᵢⱼ / ℓ, ℓ = 4
scaled_distance = jnp.sqrt(3.0) * ring_distance / 4.0  # (D_x, D_x)
# exp(−rᵢⱼ)
decay = jnp.exp(-scaled_distance)  # (D_x, D_x)
# [Σ_x]ᵢⱼ = (1 + rᵢⱼ) exp(−rᵢⱼ), unit variance
matern_matrix = (1.0 + scaled_distance) * decay  # (D_x, D_x)
# today (lineax)
matern_cov = lx.MatrixLinearOperator(  # (D_x, D_x) PSD op
    matern_matrix, lx.positive_semidefinite_tag
)


# user
class Lorenz96RK4(eqx.Module):
    """One classical RK4 step of the Lorenz-96 vector field, a ForwardModel."""

    vector_field: vdx.Lorenz96
    dt: float = 0.01

    def step(self, state: Float[Array, "k"], dt: float) -> Float[Array, "k"]:
        # today (vardax)
        # k₁ = f(t, x)
        slope_1 = self.vector_field(0.0, state, None)  # (D_x,)
        # x + ½ Δt k₁
        stage_2 = state + 0.5 * dt * slope_1  # (D_x,)
        # today (vardax)
        # k₂ = f(t, x + ½ Δt k₁)
        slope_2 = self.vector_field(0.0, stage_2, None)  # (D_x,)
        # x + ½ Δt k₂
        stage_3 = state + 0.5 * dt * slope_2  # (D_x,)
        # today (vardax)
        # k₃ = f(t, x + ½ Δt k₂)
        slope_3 = self.vector_field(0.0, stage_3, None)  # (D_x,)
        # x + Δt k₃
        stage_4 = state + dt * slope_3  # (D_x,)
        # today (vardax)
        # k₄ = f(t, x + Δt k₃)
        slope_4 = self.vector_field(0.0, stage_4, None)  # (D_x,)
        # k₁ + 2k₂ + 2k₃ + k₄
        increment = slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4  # (D_x,)
        # f(x) = x + (Δt / 6)(k₁ + 2k₂ + 2k₃ + k₄)
        return state + dt / 6.0 * increment  # (D_x,)


# ξ ~ 𝒩(0, I)
background_noise_l96 = jax.random.normal(key_background_l96, (40,))  # (D_x,)
# μ_x = x_true + ξ
background_l96 = truth_l96[0] + background_noise_l96  # (D_x,)

# today (vardax)
lorenz96 = vdx.Lorenz96(forcing=8.0)
# user
# f: x ↦ one RK4 step of the vector field over Δt = 0.01
rk4_forward = Lorenz96RK4(vector_field=lorenz96)
# σ_o² = 0.25 at every site
obs_variance = jnp.full(40, 0.25)  # (D_y,)
# today (lineax)
# Σ_ε = σ_o² I, σ_o = 0.5
obs_cov_l96 = lx.DiagonalLinearOperator(obs_variance)  # (D_y, D_y) op
# today (vardax)
incremental_config = vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True)

# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖mₜ ⊙ (yₜ − fₜ(x₀))‖²_Σ_ε⁻¹
incremental = vdx.IncrementalFourDVar(
    forward=rk4_forward,
    obs_op=masked_identity,
    prior_mean=background_l96,
    prior_cov_op=matern_cov,
    obs_cov_op=obs_cov_l96,
    config=incremental_config,
)

# today (vardax)
# x₀⋆ = argmin 𝒥(x₀), Gauss-Newton outer loop with CG inside
initial_analysis_l96 = incremental(batch_l96)[0]  # (S, D_x) → (D_x,)
# today (vardax)
# Σ_x|y = (Σ_x⁻¹ + ∑ₜ (Hₜ F_{t←0})ᵀ Σ_ε⁻¹ (Hₜ F_{t←0}))⁻¹
# Hₜ = Mₜ = diag(mₜ), the masked identity
posterior = incremental.posterior(batch_l96)
# μ_x − x_true
background_residual = background_l96 - truth_l96[0]  # (D_x,)
# ‖μ_x − x_true‖
background_error_l96 = jnp.linalg.norm(background_residual)  # → ()
# x₀⋆ − x_true
analysis_residual = initial_analysis_l96 - truth_l96[0]  # (D_x,)
# ‖x₀⋆ − x_true‖
analysis_error_l96 = jnp.linalg.norm(analysis_residual)  # → ()
print(f"background error {background_error_l96:.2f}")
print(f"analysis error {analysis_error_l96:.2f}")
print(posterior.mean.shape)
```

It prints `background error 5.76`, `analysis error 2.98` and `(40,)`, the errors being in $\vect{x}_0$.

This walkthrough exercises the periodic Matérn prior as a `lineax` operator, the CVT [](#eq-cvt) with $\gmat{\Sigma}_x^{1/2}$ from the Matérn factorisation, and the Gauss-Newton and CG loop [](#eq-gn-system), whose Hessian [](#eq-gn-hessian) the posterior reuses for free.

The reconstruction recovers the Lorenz-96 trajectory through gaps,
with calibrated posterior uncertainty in unobserved regions.

## Linear-Gaussian agreement test

Lorenz dynamics aren't linear, but we can construct a static
linear-Gaussian test on the Lorenz-63 attractor by treating the
trajectory mean as the prior mean and the empirical covariance as $\gmat{\Sigma}_x$.

With $f(\vect{x}) = \vect{x}$ over one time, the BLUE mean [](#eq-blue-mean) and the minimisers of [](#eq-3dvar-cost), [](#eq-strong4dvar-cost) and [](#eq-incremental-cost) are the same point.
Only the four classical methods take this batch; `FourDVarNet` is left out, as in `tests/test_linear_gaussian_agreement.py`, because its learned components must be trained before they can match.

```{code-block} python
:caption: Checks that the BLUE mean [](#eq-blue-mean) and the minimisers of [](#eq-3dvar-cost), [](#eq-strong4dvar-cost) and [](#eq-incremental-cost) agree on one Lorenz-63 time.
:label: code-lorenz-linear-gaussian-agreement
import equinox as eqx

# Shapes: S = 1 window; one time; D_x = D_y = 3


# user
class StaticForward(eqx.Module):
    """Trivial dynamics, f(x) = x, so a one-time window is a 3D problem."""

    dt: float = 1.0

    def step(self, state: Float[Array, "n"], dt: float) -> Float[Array, "n"]:
        # f(x) = x
        return state


# today (vardax)
single_time = vdx.Batch1D(
    input=batch.input[:, :1],  # (S, T + 1, D_x) → (S, 1, D_x)
    mask=batch.mask[:, :1],
    target=batch.target[:, :1],
)
identity_matrix = jnp.eye(3)  # (D_x, D_x)
# 0.01 I
ridge = 0.01 * identity_matrix  # (D_x, D_x)
# Σ_x = Σ̂_x + 0.01 I
regularised_cov = climatology_cov + ridge  # (D_x, D_x)
# today (lineax)
state_cov = lx.MatrixLinearOperator(  # (D_x, D_x) PSD op
    regularised_cov, lx.positive_semidefinite_tag
)
single_structure = jax.ShapeDtypeStruct((1, 3), jnp.float32)


# user
def apply_single_cov(state: Float[Array, "w n"]) -> Float[Array, "w n"]:
    # (Σ_x v)ₜ = Σ_x vₜ at the one time
    return einx.dot("w n, n m -> w m", state, regularised_cov)


# today (lineax)
time_cov = lx.FunctionLinearOperator(  # (D_x, D_x) op
    apply_single_cov,
    single_structure,
    tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
)
# μ_x = x̄
time_mean = einx.id("n -> 1 n", climatology_mean)  # (D_x,) → (1, D_x)
# today (lineax)
# Σ_ε = I
time_obs_cov = lx.IdentityLinearOperator(single_structure)  # (D_x, D_x) op
# today (lineax)
# Σ_ε = I
state_obs_cov = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# user
# f(x) = x
static_forward = StaticForward()
# today (vardax)
incremental_single_config = vdx.IncrementalConfig(n_outer=2)

# today (vardax)
oi_single = vdx.OptimalInterpolation(masked_identity, time_mean, time_cov, time_obs_cov)
# today (vardax)
# x⋆ = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x)
oi_single_analysis = oi_single(single_time)[:, 0]  # (S, 1, D_x) → (S, D_x)
# today (vardax)
threedvar = vdx.ThreeDVar(masked_identity, time_mean, time_cov, time_obs_cov)
# today (vardax)
# x⋆ = argmin ½ ‖x − μ_x‖²_Σ_x⁻¹ + ½ ‖y − H x‖²_Σ_ε⁻¹
threedvar_analysis = threedvar(single_time)[:, 0]  # (S, 1, D_x) → (S, D_x)
# today (vardax)
strong_single = vdx.StrongFourDVar(
    static_forward, masked_identity, climatology_mean, state_cov, state_obs_cov
)
# today (vardax)
# x₀⋆ = argmin 𝒥(x₀) with f(x) = x
strong_single_analysis = strong_single(single_time)  # (S, D_x)
# today (vardax)
incremental_single = vdx.IncrementalFourDVar(
    static_forward,
    masked_identity,
    climatology_mean,
    state_cov,
    state_obs_cov,
    config=incremental_single_config,
)
# today (vardax)
# x₀⋆ = μ_x + δx⋆, δx⋆ = argmin δ𝒥(δx)
incremental_single_analysis = incremental_single(single_time)  # (S, D_x)

analyses = {
    "oi": oi_single_analysis,
    "3dvar": threedvar_analysis,
    "strong": strong_single_analysis,
    "incremental": incremental_single_analysis,
}
for name, analysis in analyses.items():
    # |x⋆_method − x⋆_OI| ≤ 0.01 elementwise, up to rtol
    agrees = bool(jnp.allclose(analysis, oi_single_analysis, atol=1e-2))
    print(f"{name} agrees with OI: {agrees}")
```

It prints `agrees with OI: True` for all four methods.

This is the Decision D14 invariant in action.

## Notebook examples

The [Tutorials](notebooks/index.md) section renders the end-to-end Lorenz
notebooks (jupytext percent-format sources under `docs/notebooks/`,
executed at docs build time):

- [01: Model-based 4DVar on L63](notebooks/01_model_based_4dvar_L63.ipynb)
- [02: Unrolling vs fixed-point on L63](notebooks/02_unrolling_vs_fixedpoint_L63.ipynb)
- [03: 4DVarNet end-to-end on L63](notebooks/03_4dvarnet_L63.ipynb)
- [05: End-to-end L63 pipeline](notebooks/05_end_to_end_L63.ipynb), including the data utilities
- [06: End-to-end L96 pipeline](notebooks/06_end_to_end_L96.ipynb)
- [07: Prior pre-training on L63](notebooks/07_prior_pretraining_L63.ipynb), pre-training the bilinear prior
- [08: Classical 4DVar vs 4DVarNet on L63](notebooks/08_classical_4dvar_vs_4dvarnet_L63.ipynb), side by side
- [09: Parameter estimation on L63](notebooks/09_param_estimation_L63.ipynb), learnable ODE parameters via `DynTrajectory`
- [10: Bilevel optimisation on L63](notebooks/10_bilevel_opt_L63.ipynb), learning the cost weights through the inner solve
- [11: Learning the gradient update](notebooks/11_gradient_learning_L63.ipynb), the gradient modulator in isolation

To open one as a notebook locally: `jupytext --to notebook docs/notebooks/<name>.py`.

## See also

- [Chapter 6](06_strong_4dvar.md): `StrongFourDVar`, the L63 4DVar walkthrough method
- [Chapter 8](08_incremental_4dvar.md): `IncrementalFourDVar`, the L96 walkthrough method
- [Chapter 9](09_4dvarnet.md): `FourDVarNet`, the learned-prior comparison
- [Chapter 16](16_ssh_example.md): SSH example, the real-world 2D analogue
- [Chapter 17](17_methane_example.md): methane example, the real-world multi-instrument analogue
