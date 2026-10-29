---
numbering:
  enumerator: "16.%s"
---

# SSH Reconstruction Example

Sea surface height (SSH) reconstruction from satellite altimetry is the
canonical large-scale 4DVarNet benchmark (Fablet et al. 2021, 2023).
The state is a 2D field; observations are along-track samples from
altimetric satellites (Jason, SARAL/AltiKa, SWOT); the gaps between
tracks need to be filled.

This chapter shows three approaches to the same problem: OI as the
classical baseline, `IncrementalFourDVar` as the operational pattern,
and `FourDVarNet` as the learned alternative. Same data, three
analyses, side-by-side comparison.

## Problem setup

State: SSH on a regular lon-lat grid, $H \times W = 128 \times 128$
typical for a regional patch. Observations: along-track samples from
one or more altimeters, with along-track noise $\sigma \sim 3$ cm and
gaps of $\sim 100$ km perpendicular to track.

The arrays `ssh_observed`, `track_mask`, `ssh_truth`, `lat`, `lon`, `altika_uncertainty` and `climatology_ssh` come from preprocessed altimetry and are not built on this page.
The state is flattened to $N = Y X$ cells, with $Y = H$ latitudes and $X = W$ longitudes, so that one `Batch1D` serves OI and incremental 4DVar.

```{code-block} python
:caption: Builds the gridded and flattened batches of along-track observations, the $y$ and $m$ of [](#eq-obs-model) with the masked identity [](#eq-masked-identity).
:label: code-ssh-batches
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 window; T = 10 times; Y = 128 latitudes; X = 128 longitudes; N = Y X

key = jax.random.PRNGKey(0)
key_net, key_skeleton = jax.random.split(key)

# NaN · 0 is NaN, so the gaps are zero-filled before they reach the cost
# ỹ = y where m > 0, else 0
observed_filled = jnp.where(track_mask > 0, ssh_observed, 0.0)  # (S, T, Y, X)
# today (vardax)
grid_batch = vdx.Batch2D(
    input=observed_filled,
    mask=track_mask,  # 1 along-track, 0 in gaps
    target=ssh_truth,  # ground truth, training only
)
observed_flat = einx.id("s t y x -> s t (y x)", observed_filled)  # (S, T, N)
mask_flat = einx.id("s t y x -> s t (y x)", track_mask)  # (S, T, N)
truth_flat = einx.id("s t y x -> s t (y x)", ssh_truth)  # (S, T, N)
# today (vardax)
batch = vdx.Batch1D(input=observed_flat, mask=mask_flat, target=truth_flat)
```

## Approach 1: `OptimalInterpolation`

The classical baseline is OI ([chapter 4](04_oi_blue.md)) with the masked identity [](#eq-masked-identity) as $H$, a separable Matérn prior and a diagonal $R$.

```{code-block} python
:caption: Computes the BLUE mean [](#eq-blue-mean) and covariance [](#eq-blue-cov) in closed form with a separable Matérn prior and a diagonal $R$.
:label: code-ssh-oi
import gaussx as gx

# Shapes: T = 10; Y = 128; X = 128; N = Y X; D = Y or X grid offsets


# user
def matern32_column(
    offset_km: Float[Array, "d"], length_scale_km: float
) -> Float[Array, "d"]:
    # r = √3 d / ℓ
    scaled = jnp.sqrt(3.0) * offset_km / length_scale_km
    # exp(−r)
    decay = jnp.exp(-scaled)
    # ρ(d) = (1 + r) exp(−r)
    return (1.0 + scaled) * decay


# φ − φ₀
lat_difference = lat - lat[0]  # (Y,)
# d_lat = 111 km per degree of latitude
lat_offset_km = 111.0 * lat_difference  # (Y,)
# φ̄ = (1/Y) Σ φ
mean_latitude = einx.mean("y ->", lat)  # (Y,) → ()
# φ̄ in radians
mid_latitude = jnp.deg2rad(mean_latitude)  # → ()
# 111 cos φ̄ km per degree of longitude
km_per_lon_degree = 111.0 * jnp.cos(mid_latitude)  # → ()
# λ − λ₀
lon_difference = lon - lon[0]  # (X,)
# d_lon = 111 cos φ̄ (λ − λ₀)
lon_offset_km = km_per_lon_degree * lon_difference  # (X,)
# user
# ρ(d_lat), ℓ = 100 km
lat_column = matern32_column(lat_offset_km, 100.0)  # (Y,)
# user
# ρ(d_lon), ℓ = 100 km
lon_column = matern32_column(lon_offset_km, 100.0)  # (X,)
# today (gaussx)
lat_corr = gx.Toeplitz(lat_column)  # (Y, Y) op
# today (gaussx)
lon_corr = gx.Toeplitz(lon_column)  # (X, X) op
# today (gaussx)
# C_lat ⊗ C_lon
separable_corr = gx.Kronecker(lat_corr, lon_corr)  # (N, N) op
# B_s = σ² (C_lat ⊗ C_lon), σ = 0.1 m: a separable Matérn-3/2
spatial_cov = 0.01 * separable_corr  # (N, N) op
window_structure = jax.ShapeDtypeStruct((10, 128 * 128), observed_filled.dtype)


# user
def apply_window_cov(state: Float[Array, "t n"]) -> Float[Array, "t n"]:
    batched_cov = jax.vmap(spatial_cov.mv)
    # (B v)ₜ = B_s vₜ, no correlation across times
    return batched_cov(state)


# today (lineax)
window_cov = lx.FunctionLinearOperator(  # (T N, T N) op
    apply_window_cov,
    window_structure,
    tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
)
# σ²_alt
grid_variance = altika_uncertainty**2  # (T, Y, X)
# R = diag(σ²_alt)
obs_variance = einx.id("t y x -> t (y x)", grid_variance)  # (T, N)


# user
def apply_obs_cov(residual: Float[Array, "t n"]) -> Float[Array, "t n"]:
    # R v = σ²_alt ⊙ v; a 2-D DiagonalLinearOperator would break oi.posterior
    return obs_variance * residual


# today (lineax)
window_obs_cov = lx.FunctionLinearOperator(  # (T N, T N) op
    apply_obs_cov,
    window_structure,
    tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
)
# x_b = climatology
window_background = einx.id("t y x -> t (y x)", climatology_ssh)  # (T, N)
# today (vardax)
# H(x) = m ⊙ x
masked_identity = vdx.MaskedIdentity()

# today (vardax)
oi = vdx.OptimalInterpolation(
    obs_op=masked_identity,
    prior_mean=window_background,
    prior_cov_op=window_cov,
    obs_cov_op=window_obs_cov,
)

# Single forward pass, no iteration; run with jax_enable_x64, since
# lineax 0.1.1 CG returned NaN at ~10⁵ unknowns in float32
# today (vardax)
# x* = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
oi_analysis = oi(batch)  # (S, T, N)
# today (vardax)
# P* = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹
posterior_oi = oi.posterior(batch)
# Reconstruction in gaps comes from the Matérn correlation length.
```

What this gives you: the standard OceanBench DUACS baseline. Smoothed
through the prior length scale. Easy to interpret, easy to validate.
Misses mesoscale features that are shorter than the correlation length.

## Approach 2: `IncrementalFourDVar` with shallow-water dynamics

Add a `somax.ShallowWaterModel` as the dynamics. Multi-time observations now constrain the trajectory through the physics, so gaps get filled by dynamical propagation rather than just smoothing.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) with shallow-water dynamics through incremental 4DVar [](#eq-incremental-cost) and builds the Gauss-Newton posterior [](#eq-gn-cov).
:label: code-ssh-incremental
import somax

# Shapes: S = 1 window; T = 10; N = Y X; the control is x₀ (N,)

# not pinned (somax)
shallow_water = somax.ShallowWaterModel(grid=grid, params=ssh_params)
# today (lineax)
# R = diag(σ²_alt) at one time
time_obs_cov = lx.DiagonalLinearOperator(obs_variance[0])  # (N, N) op
# x_b = climatology at the first time
initial_background = einx.id("y x -> (y x)", climatology_ssh[0])  # (N,)
# today (vardax)
incremental_config = vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True)

# today* (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B_s⁻¹} + ½ Σₜ ‖mₜ ⊙ (yₜ − M_t(x₀))‖²_{R⁻¹}
incremental = vdx.IncrementalFourDVar(
    forward=shallow_water,
    obs_op=masked_identity,
    prior_mean=initial_background,
    prior_cov_op=spatial_cov,
    obs_cov_op=time_obs_cov,
    config=incremental_config,
    checkpoint=True,
)

# today (vardax)
# x₀* = argmin J(x₀)
incremental_analysis = incremental(batch)  # (S, N)
# today (vardax)
# P* = (B_s⁻¹ + Σₜ Gₜᵀ R⁻¹ Gₜ)⁻¹, Gₜ = mₜ ⊙ M′_t
posterior_inc = incremental.posterior(batch)
# Mesoscale eddies now propagate consistently across observation gaps;
# the posterior reflects model + observation uncertainty.
```

What this gives you: physically-consistent reconstruction. The incremental machinery scales to operational grids ($10^6$ to $10^7$ cells) thanks to the CVT-preconditioned CG [](#eq-cvt-system).

## Approach 3: `FourDVarNet` with learned prior

Train a `FourDVarNet` ([chapter 9](09_4dvarnet.md)) on (observed, truth) patches. The learned prior $\varphi_\theta$ of the cost [](#eq-4dvarnet-cost) captures the SSH manifold, and the learned gradient modulator $\Phi_\phi$ of the update [](#eq-4dvarnet-update) accelerates the inner iterations.

```{code-block} python
:caption: Trains the learned-prior solver of [](#eq-4dvarnet-cost) and [](#eq-4dvarnet-update) on the loss [](#eq-4dvarnet-loss) and reconstructs the SSH window.
:label: code-ssh-4dvarnet
import optax

# Shapes: S = 1 window; T = 10; Y = 128; X = 128

# today (vardax)
# builds a BilinAEPrior2D prior and a ConvLSTMGradMod2D modulator inside
model = vdx.FourDVarNet2D(
    n_time=10,
    height=128,
    width=128,
    latent_dim=128,
    hidden_dim=64,
    n_solver_steps=15,
    alpha=0.2,
    # O(1) memory training
    solver_adjoint=vdx.OneStepAdjoint(),
    key=key_net,
)

# today (optax)
optimizer = optax.adam(1e-3)
trainable = eqx.filter(model, eqx.is_array)
# today (optax)
opt_state = optimizer.init(trainable)


# user
@eqx.filter_jit
def step(
    model: vdx.FourDVarNet2D, opt_state: optax.OptState, train_batch: vdx.Batch2D
) -> tuple[vdx.FourDVarNet2D, optax.OptState, Float[Array, ""]]:
    # today (vardax)
    # θ ← θ − η Adam(∇_θ ‖x̂_θ(y) − x‖²)
    model, opt_state, loss = vdx.train_step(model, train_batch, optimizer, opt_state)
    return model, opt_state, loss


# Train on the OceanBench dataset; oceanbench_loader yields Batch2D
for _ in range(100):
    for train_batch in oceanbench_loader:
        # user
        model, opt_state, loss = step(model, opt_state, train_batch)

# today (vardax)
# x̂ = Ψ_θ(y)
net_analysis = model(grid_batch)  # (S, T, Y, X)
```

What this gives you: state-of-the-art reconstruction on the
distribution the network was trained on. Better than OI in
information-rich regions; comparable to incremental 4DVar without
needing somax. Limitations: extrapolates poorly outside the training
distribution; posterior calibration via Laplace is less defensible
than the GN-Hessian-based posterior from incremental 4DVar.

## Comparison

| Method | Setup cost | Per-event cost | Recovers mesoscale? | Posterior quality | When |
|---|---|---|---|---|---|
| `OptimalInterpolation` | trivial | low (single pass) | only down to prior length scale | exact (closed form) | Linear-Gaussian baseline |
| `IncrementalFourDVar` | requires `somax` model | medium (3 outer × 20 inner CG) | yes, via dynamics | calibrated (GN-Hessian) | Production / operational |
| `FourDVarNet` | requires training | low (15 solver steps) | yes, via learned prior | needs validation | Data-rich regime |

The OceanBench benchmark (Le Guillou et al. 2023) provides a standard
dataset for these comparisons. The expected ordering is `OI < IncrementalFourDVar
≈ FourDVarNet`, with `FourDVarNet` matching `IncrementalFourDVar` on
in-distribution data but degrading more gracefully when the input
distribution drifts.

## Operational cycling

For continuous SSH analysis (every overpass, every few hours), wrap
the chosen analysis in a `pipekit_cycle.DACycle`:

```{code-block} python
:caption: Cycles the shallow-water forecast and the incremental analysis of [](#code-ssh-incremental) over successive windows, as in the six-step cycle of [chapter 14](14_six_step_cycle.md).
:label: code-ssh-cycle
import pipekit_cycle as pc

# Shapes: N = Y X

# today (vardax)
# x₀* = argmin J(x₀) as a cycle slot; or oi / net
analysis_step = incremental.as_analysis_step()
# today (pipekit_cycle)
ssh_cycle = pc.DACycle(
    forward_model=shallow_water,
    obs_op=masked_identity,
    analysis_step=analysis_step,
    obs_source=along_track_loader,
    n_steps=n_assimilation_windows,
)

# Run continuously
# today (pipekit_cycle)
initial_da_state = pc.DAState(t=0.0, cycle_count=0)
# today* (pipekit_cycle)
# one forecast and one analysis x₀* per window
final_analysis, final_da_state = ssh_cycle(initial_state, initial_da_state)
```

The `analysis_step` slot accepts any of the three methods, so swapping `OptimalInterpolation` for `IncrementalFourDVar` or `FourDVarNet` changes one slot. Nothing in the cycling, data loading, or output serialisation changes.

## Persistence

For trained `FourDVarNet`, persist via `pipekit-jax.JaxModelOp` +
`pipekit-experiment.ModelRegistry`:

```{code-block} python
:caption: Stores the trained weights θ of [](#eq-4dvarnet-loss) and reloads them into a fresh 4DVarNet skeleton.
:label: code-ssh-persistence
import pipekit_experiment
import pipekit_jax

# not pinned (pipekit_experiment)
registry = pipekit_experiment.LocalModelRegistry(root="./ssh_models")
# not pinned (pipekit_jax)
model_op = pipekit_jax.JaxModelOp(model)
# not pinned (pipekit_jax)
weights = model_op.serialize_weights()
# not pinned (pipekit_experiment)
weights_hash = registry.store(
    model_op,
    weights=weights,
    tags={"task": "ssh_oceanbench", "version": "v1"},
)

# Later, reload into a fresh skeleton with the same architecture
# today (vardax)
skeleton = vdx.FourDVarNet2D(
    n_time=10, height=128, width=128, latent_dim=128, hidden_dim=64, key=key_skeleton
)
# not pinned (pipekit_jax)
template = pipekit_jax.JaxModelOp(skeleton)
# not pinned (pipekit_experiment)
stored_weights = registry.load_weights(weights_hash)
# not pinned (pipekit_jax)
reloaded = template.with_weights(stored_weights)
```

`OptimalInterpolation` and `IncrementalFourDVar` have no learned
parameters; they need no persistence (the `prior_cov_op` and config
are reconstructed from numerical parameters).

## See also

- [Chapter 4](04_oi_blue.md): `OptimalInterpolation` (BLUE / OI)
- [Chapter 8](08_incremental_4dvar.md): `IncrementalFourDVar` (operational fast path)
- [Chapter 9](09_4dvarnet.md): `FourDVarNet` (learned variant)
- [Chapter 14](14_six_step_cycle.md): six-step cycle (the methodology that ties this together)

## References

- Fablet, R., Febvre, Q., & Chapron, B. (2023). *Multimodal 4DVarNets
  for the reconstruction of sea surface dynamics from NADIR and
  wide-swath altimetry.* IEEE TGRS 61.
- Le Guillou, F., et al. (2023). *Mapping altimetry in the forthcoming
  SWOT era by back-and-forth nudging a one-layer quasigeostrophic
  model.* JTECH 40(1). [OceanBench reference.]
- Ubelmann, C., Klein, P., & Fu, L.-L. (2015). *Dynamic
  interpolation of sea surface height and potential applications for
  future high-resolution altimetry mapping.* JTECH 32.
