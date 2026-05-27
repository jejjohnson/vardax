# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 — End-to-End Lorenz-63 Pipeline
#
# This notebook demonstrates the full data-preprocessing and training pipeline
# for 4DVarNet on the Lorenz-63 attractor, using the functional utilities in
# `vardax._src.utils`.
#
# **Pipeline overview:**
# 1. Simulate L63 with Diffrax
# 2. Build an xarray Dataset and extract patches
# 3. Add observation masks and Gaussian noise
# 4. Train/test split
# 5. Standardize
# 6. Convert to `Batch1D`
# 7. Visualize raw data
# 8. Train `FourDVarNet1D`
# 9. Evaluate and visualize reconstruction

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import vardax
from vardax import Batch1D, FourDVarNet1D
from vardax._src.utils.dynamical_systems import simulate_lorenz63
from vardax._src.utils.patches import trajectory_to_xr_dataset, extract_patches
from vardax._src.utils.masks import regular_mask
from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.preprocessing import train_test_split, xr_to_batch1d
from vardax._src.utils.standardize import (
    compute_scaler_params,
    apply_standardization,
    inverse_standardization,
)
from vardax._src.utils.viz import (
    plot_3d_attractor,
    plot_state_grid,
    plot_trajectories,
    plot_reconstruction_comparison,
)

# %% [markdown]
# ## 1. Simulate Lorenz-63

# %%
key = jax.random.PRNGKey(0)
time_coords, states = simulate_lorenz63(
    key,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    dt=0.01,
    n_steps=5000,
    n_burn_in=1000,
)
print(f"states shape: {states.shape}, time range: [{time_coords[0]:.2f}, {time_coords[-1]:.2f}]")

# %% [markdown]
# ## 2. Build xarray Dataset and Extract Patches

# %%
ds = trajectory_to_xr_dataset(states, time_coords, feature_names=["X", "Y", "Z"])
ds = extract_patches(ds, n_patches=200, n_timesteps=20, seed=42)
print(ds)

# %% [markdown]
# ## 3. Add Observation Masks and Gaussian Noise

# %%
ds = regular_mask(ds, variable="state", obs_interval=2)
ds = add_gaussian_noise(ds, variable="state", sigma=0.5, seed=0, name="obs")
print(ds)

# %% [markdown]
# ## 4. Train/Test Split

# %%
ds_train, ds_test = train_test_split(ds, n_train=160, n_test=40, seed=0)
print(f"train patches: {ds_train.sizes['patch']}, test patches: {ds_test.sizes['patch']}")

# %% [markdown]
# ## 5. Standardize

# %%
mean, std = compute_scaler_params(ds_train, variable="state", mask_variable="mask")
print(f"mean={mean:.4f}, std={std:.4f}")

ds_train = apply_standardization(ds_train, variables=["state", "obs"], mean=mean, std=std)
ds_test = apply_standardization(ds_test, variables=["state", "obs"], mean=mean, std=std)

# %% [markdown]
# ## 6. Convert to Batch1D

# %%
batch_train = xr_to_batch1d(ds_train, state_var="state", obs_var="obs", mask_var="mask")
batch_test = xr_to_batch1d(ds_test, state_var="state", obs_var="obs", mask_var="mask")
print(f"train: input={batch_train.input.shape}, mask={batch_train.mask.shape}, target={batch_train.target.shape}")

# %% [markdown]
# ## 7. Visualize Raw Data

# %%
fig, ax = plot_3d_attractor(states)
ax.set_title("Lorenz-63 Attractor")
plt.tight_layout()
plt.show()

# %%
fig, ax = plot_state_grid(states[:200], time_coords[:200])
ax.set_title("State Hovmöller")
plt.tight_layout()
plt.show()

# %%
fig, ax = plot_trajectories(states[:200], time_coords[:200])
ax.set_title("L63 Trajectories")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Train FourDVarNet1D

# %%
import flax.nnx as nnx

N = batch_train.input.shape[-1]   # 3 (X, Y, Z)
T = batch_train.input.shape[1]    # 20

model = FourDVarNet1D(
    state_dim=N,
    n_time=T,
    latent_dim=8,
    hidden_dim=16,
    n_solver_steps=5,
    rngs=nnx.Rngs(jax.random.PRNGKey(1)),
)

optimizer, train_losses, _ = vardax.fit(
    model,
    [batch_train],
    n_epochs=5,
    lr=1e-3,
    verbose=True,
)
print("Final train loss:", train_losses[-1])

# %% [markdown]
# ## 9. Evaluate and Visualize Reconstruction

# %%
recon = model(batch_test)

target_np = jnp.array(batch_test.target)
input_np = jnp.array(batch_test.input)
recon_np = jnp.array(recon)

fig, axes = plot_reconstruction_comparison(target_np, input_np, recon_np, sample_idx=0)
plt.suptitle("4DVarNet L63 Reconstruction")
plt.tight_layout()
plt.show()
