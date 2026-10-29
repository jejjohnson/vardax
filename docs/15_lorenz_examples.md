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

where $(x, y, z)$ is the state and $\sigma$, $\rho$, $\beta$ are its three parameters, here the canonical $(10, 28, 8/3)$.
The state dimension is $n = 3$, and the dynamics live on the famous two-winged attractor.

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

# Shapes: T = 5,001 saved times; N = 3 state components

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 5)
key_l63, key_background, key_net, key_l96, key_background_l96 = keys

# today (vardax)
# ẋ = σ(y − x), ẏ = x(ρ − z) − y, ż = xy − βz, integrated with Tsit5
time, trajectory = vdx.simulate_lorenz63(  # (T,), (T, N)
    key_l63, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, n_steps=5000
)
print(trajectory.shape)
```

It prints `(5001, 3)`; the default burn-in of 1,000 steps is dropped.

### Synthetic observation scenario

Take one window of $W = 101$ steps (one time unit, about one Lyapunov time) and observe it every $k = 10$ steps, with Gaussian noise of standard deviation $\sigma_o = 1$.
The data utilities are private helpers under `vardax._src.utils`, which the tutorial notebooks also use.
`xr_to_batch1d` expects a `patch` dimension, so the window gets one.

```{code-block} python
:caption: Builds the masked, noisy Lorenz-63 window, the observations of [](#eq-obs-model) with the masked identity [](#eq-masked-identity) as $H$.
:label: code-lorenz-l63-observations
import vardax._src.utils.masks as masks
import vardax._src.utils.noise as noise
import vardax._src.utils.patches as patches
import vardax._src.utils.preprocessing as preprocessing

# Shapes: S = 1 window; W = 101 window times; N = 3

# today (vardax._src.utils)
dataset = patches.trajectory_to_xr_dataset(trajectory, time)  # (T, N)
window = dataset.isel(time=slice(500, 601))  # (T, N) → (W, N)
window = window.expand_dims("patch")  # (W, N) → (S, W, N)
# today (vardax._src.utils)
# m_t = 1 for t = 0, k, 2k, …, with k = 10
window = masks.regular_mask(window, variable="state", obs_interval=10)
# today (vardax._src.utils)
# y_t = x_t + σ_o ε_t, σ_o = 1, ε_t ~ 𝒩(0, I)
window = noise.add_gaussian_noise(window, variable="state", sigma=1.0, name="obs")
# today (vardax._src.utils)
batch = preprocessing.xr_to_batch1d(
    window, state_var="state", obs_var="obs", mask_var="mask"
)
truth = batch.target[0]  # (S, W, N) → (W, N)
```

### `OptimalInterpolation` won't work directly

OI ([chapter 4](04_oi_blue.md)) assumes a static Gaussian field, which the nonlinear Lorenz dynamics violate.
So OI with the climatological mean $\bar{x}$ and covariance $C$ of the trajectory as its prior gives a poor reconstruction, but it is a useful baseline.
Each time is treated separately, with $B = C$ and $R = I$.

```{code-block} python
:caption: Computes the BLUE mean [](#eq-blue-mean) at each time of the window from the climatological prior, with the masked identity [](#eq-masked-identity) as $H$.
:label: code-lorenz-l63-oi
# Shapes: T = 5,001; W = 101; N = 3


# user
def rmse(
    estimate: Float[Array, "w n"], reference: Float[Array, "w n"]
) -> Float[Array, ""]:
    # e = x̂ − x
    error = estimate - reference
    # e²
    squared_error = error**2
    # (1/(W N)) Σ e²
    mean_squared_error = einx.mean("w n ->", squared_error)
    # RMSE = √((1/(W N)) Σ e²)
    return jnp.sqrt(mean_squared_error)


# x̄ = (1/T) Σₜ xₜ
climatology_mean = einx.mean("t n -> n", trajectory)  # (T, N) → (N,)
# δxₜ = xₜ − x̄
anomaly = einx.subtract("t n, n -> t n", trajectory, climatology_mean)  # (T, N)
# S = Σₜ δxₜ δxₜᵀ
scatter = einx.dot("t n, t m -> n m", anomaly, anomaly)  # (T, N) → (N, N)
# C = S / (T − 1)
climatology_cov = scatter / (trajectory.shape[0] - 1)  # (N, N)
window_structure = jax.ShapeDtypeStruct((101, 3), jnp.float32)


# user
def apply_static_cov(state: Float[Array, "w n"]) -> Float[Array, "w n"]:
    # (B v)ₜ = C vₜ, the same C at every time and no correlation across times
    return einx.dot("w n, n m -> w m", state, climatology_cov)


# today (lineax)
static_cov = lx.FunctionLinearOperator(  # (W N, W N) op
    apply_static_cov,
    window_structure,
    tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
)
# today (lineax)
# R = I
unit_obs_cov = lx.IdentityLinearOperator(window_structure)  # (W N, W N) op
# x_b = x̄ at every time
static_mean = einx.id("n -> w n", climatology_mean, w=101)  # (N,) → (W, N)
# today (vardax)
# H(x) = m ⊙ x
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
# x* = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b), at each time separately
oi_analysis = oi(batch)[0]  # (S, W, N) → (W, N)
# user
oi_rmse = rmse(oi_analysis, truth)  # → ()
print(f"OI trajectory RMSE {oi_rmse:.2f}")
```

It prints `OI trajectory RMSE 6.06`.

### `StrongFourDVar` is the right tool

The background $x_b$ is the true initial state plus noise of standard deviation $\sigma_b = 2$, and the prior $B = 100 I$ is deliberately weak.
$H_t$ is the masked identity and $R = I$.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) for the Lorenz-63 window and rolls the analysis out with [](#eq-rollout).
:label: code-lorenz-l63-strong4dvar
# Shapes: W = 101; N = 3

# today (vardax)
lorenz63 = vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
# today (vardax)
trajectory_model = vdx.DynTrajectory(model=lorenz63)
# today (vardax)
# M: xₜ ↦ xₜ₊₁, one Δt = 0.01 ODE step
forward = trajectory_model.as_forward_model(dt=0.01)
state_structure = jax.ShapeDtypeStruct((3,), jnp.float32)
# δ ~ 𝒩(0, I)
background_noise = jax.random.normal(key_background, (3,))  # (N,)
# σ_b δ, σ_b = 2
background_offset = 2.0 * background_noise  # (N,)
# x_b = x₀ + σ_b δ
background = truth[0] + background_offset  # (N,)
# today (lineax)
# I
state_identity = lx.IdentityLinearOperator(state_structure)  # (N, N) op
# 100 I
scaled_identity = 100.0 * state_identity  # (N, N) op
# today (lineax)
# B = 100 I, tagged PSD so the default CG solve accepts it
weak_prior_cov = lx.TaggedLinearOperator(  # (N, N) PSD op
    scaled_identity, lx.positive_semidefinite_tag
)
# today (optimistix)
minimiser = optx.NonlinearCG(rtol=1e-6, atol=1e-6)

# today (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖mₜ ⊙ (yₜ − M_t(x₀))‖²_{R⁻¹}, R = I
strong = vdx.StrongFourDVar(
    forward=forward,
    obs_op=masked_identity,
    prior_mean=background,
    prior_cov_op=weak_prior_cov,
    obs_cov_op=state_identity,
    minimiser=minimiser,
)
# today (vardax)
# x₀* = argmin J(x₀)
initial_analysis = strong(batch)[0]  # (S, N) → (N,)
step_index = jnp.arange(101)  # (W,)
# tᵢ = i Δt
window_times = 0.01 * step_index  # (W,)
# today (vardax)
# xₜ* = M_t(x₀*)
strong_analysis = trajectory_model(initial_analysis, window_times)  # (W, N)
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
# Shapes: P = 32 patches; G = 8 patches per batch; L = 20 patch times; N = 3

# today (vardax._src.utils)
train_patches = patches.extract_patches(  # (P, L, N)
    dataset, n_patches=32, n_timesteps=20, seed=0
)
# today (vardax._src.utils)
# m_t = 1 for t = 0, 2, 4, …
train_patches = masks.regular_mask(train_patches, variable="state", obs_interval=2)
# today (vardax._src.utils)
# y_t = x_t + σ_o ε_t, σ_o = 1
train_patches = noise.add_gaussian_noise(
    train_patches, variable="state", sigma=1.0, name="obs"
)
train_batches = []
for start in range(0, 32, 8):
    batch_patches = train_patches.isel(patch=slice(start, start + 8))  # (G, L, N)
    # today (vardax._src.utils)
    train_batch = preprocessing.xr_to_batch1d(batch_patches)  # (G, L, N)
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
# θ* = argmin_θ Σ ‖x̂_θ(y) − x‖², Adam, two epochs
net, train_losses, _ = vdx.examples.fit_demo(
    net, train_batches, n_epochs=2, verbose=False
)
# today (vardax)
# x̂ = Ψ_θ(y), 15 unrolled solver steps
net_analysis = net(train_batches[0])  # (G, L, N)
print(net_analysis.shape)
```

It prints `(8, 20, 3)`.

`FourDVarNet1D` learns the attractor structure from the training data
and reconstructs without explicit access to the dynamics. The
comparison `strong_analysis` vs `net_analysis` is the canonical demonstration that
learned priors recover similar information to model-based ones.

## Lorenz-96

The Lorenz-96 system is a one-dimensional periodic version:

$$\dot{X}_k = (X_{k+1} - X_{k-2}) X_{k-1} - X_k + F, \text{ for } k = 0, \ldots, N-1,$$ (eq-lorenz96)

where $X_k$ is the state at site $k$ of a periodic ring (indices taken modulo $N$) and $F$ is the forcing.
We use $N = 40$ and $F = 8$, which is the chaotic regime.
The state is a periodic 1D field, closer in structure to atmospheric problems than Lorenz-63.

```{code-block} python
:caption: Integrates the Lorenz-96 system [](#eq-lorenz96) on a ring of 40 sites.
:label: code-lorenz-l96-simulate
# Shapes: T = 5,001 saved times; K = 40 sites

# today (vardax)
# Ẋ_k = (X_{k+1} − X_{k−2}) X_{k−1} − X_k + F
time_l96, trajectory_l96 = vdx.simulate_lorenz96(  # (T,), (T, K)
    key_l96, n_sites=40, forcing=8.0, dt=0.01, n_steps=5000
)
print(trajectory_l96.shape)
```

It prints `(5001, 40)`.

### Sparse observation scenario

Observe a random half of the cells and times in one window of $W = 21$ steps, with noise of standard deviation $\sigma_o = 0.5$:

```{code-block} python
:caption: Builds the sparse, noisy Lorenz-96 window, the observations of [](#eq-obs-model) with the masked identity [](#eq-masked-identity) as $H$.
:label: code-lorenz-l96-observations
# Shapes: S = 1 window; W = 21 window times; K = 40 sites

site_names = [f"x{site}" for site in range(40)]
# today (vardax._src.utils)
dataset_l96 = patches.trajectory_to_xr_dataset(  # (T, K)
    trajectory_l96, time_l96, feature_names=site_names
)
window_l96 = dataset_l96.isel(time=slice(500, 521))  # (T, K) → (W, K)
window_l96 = window_l96.expand_dims("patch")  # (W, K) → (S, W, K)
# today (vardax._src.utils)
# m_{t,k} ~ Bernoulli(½)
window_l96 = masks.random_mask(window_l96, variable="state", missing_rate=0.5)
# today (vardax._src.utils)
# y = x + σ_o ε, σ_o = 0.5
window_l96 = noise.add_gaussian_noise(window_l96, variable="state", sigma=0.5)
# today (vardax._src.utils)
batch_l96 = preprocessing.xr_to_batch1d(window_l96)
truth_l96 = batch_l96.target[0]  # (S, W, K) → (W, K)
```

### `IncrementalFourDVar` with structured Matérn prior

This is the realistic operational scenario: structured spatial prior, incremental solver ([chapter 8](08_incremental_4dvar.md)), AK-style or masked observations.

The Matérn-3/2 covariance on the periodic grid depends only on the ring distance $d_{ij} = \min(|i - j|, K - |i - j|)$, so it is circulant; here it is a dense $40 \times 40$ matrix.
`gaussx.Toeplitz` would store it by its first column, but it does not implement `lineax.linearise`, which `IncrementalFourDVar` needs.
The forward model is one RK4 step written on the page, which forward- and reverse-mode autodiff both see through, as the Gauss-Newton loop needs.
The background is the true initial state plus noise drawn from the same scale as $B$.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) for Lorenz-96 through the incremental cost [](#eq-incremental-cost) with a periodic Matérn prior, then builds the Gauss-Newton posterior [](#eq-gn-cov).
:label: code-lorenz-l96-incremental
import equinox as eqx

# Shapes: W = 21; K = 40 sites

site_index = jnp.arange(40.0)  # (K,)
# i − j
site_difference = einx.subtract("i, j -> i j", site_index, site_index)  # (K, K)
# |i − j|
offset = jnp.abs(site_difference)  # (K, K)
# K − |i − j|
wrapped_offset = 40.0 - offset  # (K, K)
# d_ij = min(|i − j|, K − |i − j|)
ring_distance = jnp.minimum(offset, wrapped_offset)  # (K, K)
# r = √3 d / ℓ, ℓ = 4
scaled_distance = jnp.sqrt(3.0) * ring_distance / 4.0  # (K, K)
# exp(−r)
decay = jnp.exp(-scaled_distance)  # (K, K)
# B_ij = (1 + r) exp(−r), unit variance
matern_matrix = (1.0 + scaled_distance) * decay  # (K, K)
# today (lineax)
matern_cov = lx.MatrixLinearOperator(  # (K, K) PSD op
    matern_matrix, lx.positive_semidefinite_tag
)


# user
class Lorenz96RK4(eqx.Module):
    """One classical RK4 step of the Lorenz-96 vector field, a ForwardModel."""

    vector_field: vdx.Lorenz96
    dt: float = 0.01

    def step(self, state: Float[Array, "k"], dt: float) -> Float[Array, "k"]:
        # today (vardax)
        # k₁ = f(x)
        slope_1 = self.vector_field(0.0, state, None)  # (K,)
        # x + ½ Δt k₁
        stage_2 = state + 0.5 * dt * slope_1  # (K,)
        # today (vardax)
        # k₂ = f(x + ½ Δt k₁)
        slope_2 = self.vector_field(0.0, stage_2, None)  # (K,)
        # x + ½ Δt k₂
        stage_3 = state + 0.5 * dt * slope_2  # (K,)
        # today (vardax)
        # k₃ = f(x + ½ Δt k₂)
        slope_3 = self.vector_field(0.0, stage_3, None)  # (K,)
        # x + Δt k₃
        stage_4 = state + dt * slope_3  # (K,)
        # today (vardax)
        # k₄ = f(x + Δt k₃)
        slope_4 = self.vector_field(0.0, stage_4, None)  # (K,)
        # k₁ + 2k₂ + 2k₃ + k₄
        increment = slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4  # (K,)
        # x' = x + (Δt / 6)(k₁ + 2k₂ + 2k₃ + k₄)
        return state + dt / 6.0 * increment  # (K,)


# δ ~ 𝒩(0, I)
background_noise_l96 = jax.random.normal(key_background_l96, (40,))  # (K,)
# x_b = x₀ + δ
background_l96 = truth_l96[0] + background_noise_l96  # (K,)

# today (vardax)
lorenz96 = vdx.Lorenz96(forcing=8.0)
# user
# M: x ↦ one RK4 step of f over Δt = 0.01
rk4_forward = Lorenz96RK4(vector_field=lorenz96)
# σ_o² = 0.25 at every site
obs_variance = jnp.full(40, 0.25)  # (K,)
# today (lineax)
# R = σ_o² I, σ_o = 0.5
obs_cov_l96 = lx.DiagonalLinearOperator(obs_variance)  # (K, K) op
# today (vardax)
incremental_config = vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True)

# today (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖mₜ ⊙ (yₜ − M_t(x₀))‖²_{R⁻¹}
incremental = vdx.IncrementalFourDVar(
    forward=rk4_forward,
    obs_op=masked_identity,
    prior_mean=background_l96,
    prior_cov_op=matern_cov,
    obs_cov_op=obs_cov_l96,
    config=incremental_config,
)

# today (vardax)
# x₀* = argmin J(x₀), Gauss-Newton outer loop with CG inside
initial_analysis_l96 = incremental(batch_l96)[0]  # (S, K) → (K,)
# today (vardax)
# P* = (B⁻¹ + Σₜ Gₜᵀ R⁻¹ Gₜ)⁻¹, Gₜ = mₜ ⊙ M′_t
posterior = incremental.posterior(batch_l96)
# x_b − x₀
background_residual = background_l96 - truth_l96[0]  # (K,)
# ‖x_b − x₀‖
background_error_l96 = jnp.linalg.norm(background_residual)  # → ()
# x₀* − x₀
analysis_residual = initial_analysis_l96 - truth_l96[0]  # (K,)
# ‖x₀* − x₀‖
analysis_error_l96 = jnp.linalg.norm(analysis_residual)  # → ()
print(f"background error {background_error_l96:.2f}")
print(f"analysis error {analysis_error_l96:.2f}")
print(posterior.mean.shape)
```

It prints `background error 5.76`, `analysis error 2.98` and `(40,)`, the errors being in $x_0$.

This walkthrough exercises the periodic Matérn prior as a `lineax` operator, the CVT [](#eq-cvt) with $B^{1/2}$ from the Matérn factorisation, and the Gauss-Newton and CG loop [](#eq-gn-system), whose Hessian [](#eq-gn-hessian) the posterior reuses for free.

The reconstruction recovers the Lorenz-96 trajectory through gaps,
with calibrated posterior uncertainty in unobserved regions.

## Linear-Gaussian agreement test

Lorenz dynamics aren't linear, but we can construct a static
linear-Gaussian test on the Lorenz-63 attractor by treating the
trajectory mean as the prior and the empirical covariance as $B$.

With $M(x) = x$ over one time, the BLUE mean [](#eq-blue-mean) and the minimisers of [](#eq-3dvar-cost), [](#eq-strong4dvar-cost) and [](#eq-incremental-cost) are the same point.
Only the four classical methods take this batch; `FourDVarNet` is left out, as in `tests/test_linear_gaussian_agreement.py`, because its learned components must be trained before they can match.

```{code-block} python
:caption: Checks that the BLUE mean [](#eq-blue-mean) and the minimisers of [](#eq-3dvar-cost), [](#eq-strong4dvar-cost) and [](#eq-incremental-cost) agree on one Lorenz-63 time.
:label: code-lorenz-linear-gaussian-agreement
import equinox as eqx

# Shapes: S = 1 window; N = 3


# user
class StaticForward(eqx.Module):
    """Trivial dynamics, M(x) = x, so a one-time window is a 3D problem."""

    dt: float = 1.0

    def step(self, state: Float[Array, "n"], dt: float) -> Float[Array, "n"]:
        # M(x) = x
        return state


# today (vardax)
single_time = vdx.Batch1D(
    input=batch.input[:, :1],  # (S, W, N) → (S, 1, N)
    mask=batch.mask[:, :1],
    target=batch.target[:, :1],
)
identity_matrix = jnp.eye(3)  # (N, N)
# 0.01 I
ridge = 0.01 * identity_matrix  # (N, N)
# B = C + 0.01 I
regularised_cov = climatology_cov + ridge  # (N, N)
# today (lineax)
state_cov = lx.MatrixLinearOperator(  # (N, N) PSD op
    regularised_cov, lx.positive_semidefinite_tag
)
single_structure = jax.ShapeDtypeStruct((1, 3), jnp.float32)


# user
def apply_single_cov(state: Float[Array, "w n"]) -> Float[Array, "w n"]:
    # (B v)ₜ = B vₜ at the one time
    return einx.dot("w n, n m -> w m", state, regularised_cov)


# today (lineax)
time_cov = lx.FunctionLinearOperator(  # (1 N, 1 N) op
    apply_single_cov,
    single_structure,
    tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
)
# x_b = x̄
time_mean = einx.id("n -> 1 n", climatology_mean)  # (N,) → (1, N)
# today (lineax)
# R = I
time_obs_cov = lx.IdentityLinearOperator(single_structure)  # (1 N, 1 N) op
# today (lineax)
# R = I
state_obs_cov = lx.IdentityLinearOperator(state_structure)  # (N, N) op
# user
# M(x) = x
static_forward = StaticForward()
# today (vardax)
incremental_single_config = vdx.IncrementalConfig(n_outer=2)

# today (vardax)
oi_single = vdx.OptimalInterpolation(masked_identity, time_mean, time_cov, time_obs_cov)
# today (vardax)
# x* = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
oi_single_analysis = oi_single(single_time)[:, 0]  # (S, 1, N) → (S, N)
# today (vardax)
threedvar = vdx.ThreeDVar(masked_identity, time_mean, time_cov, time_obs_cov)
# today (vardax)
# x* = argmin ½ ‖x − x_b‖²_{B⁻¹} + ½ ‖y − H x‖²_{R⁻¹}
threedvar_analysis = threedvar(single_time)[:, 0]  # (S, 1, N) → (S, N)
# today (vardax)
strong_single = vdx.StrongFourDVar(
    static_forward, masked_identity, climatology_mean, state_cov, state_obs_cov
)
# today (vardax)
# x₀* = argmin J(x₀) with M(x) = x
strong_single_analysis = strong_single(single_time)  # (S, N)
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
# x₀* = x_b + δx*, δx* = argmin δJ(δx)
incremental_single_analysis = incremental_single(single_time)  # (S, N)

analyses = {
    "oi": oi_single_analysis,
    "3dvar": threedvar_analysis,
    "strong": strong_single_analysis,
    "incremental": incremental_single_analysis,
}
for name, analysis in analyses.items():
    # |x*_method − x*_OI| ≤ 0.01 elementwise, up to rtol
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
