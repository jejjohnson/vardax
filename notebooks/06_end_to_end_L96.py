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
# # 06 — End-to-End Lorenz-96 Pipeline
#
# This notebook demonstrates the full data-preprocessing and training pipeline
# for 4DVarNet on the Lorenz-96 attractor, using the functional utilities in
# `vardax._src.utils`.
#
# **Pipeline overview:**
# 1. Simulate L96 with Diffrax (N=40, F=8)
# 2. Build an xarray Dataset and extract patches
# 3. Add observation masks and Gaussian noise
# 4. Train/test split and standardize
# 5. Visualize the L96 attractor (`plot_l96_grid`)
# 6. Train `FourDVarNet1D` with `L96Prior`
# 7. Evaluate and visualize reconstruction

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import vardax
from vardax import (
    Batch1D,
    FourDVarNet1D,
    simulate_lorenz96,
    plot_l96_grid,
    plot_l96_trajectories,
    plot_reconstruction_comparison,
)
from vardax._src.utils.patches import trajectory_to_xr_dataset, extract_patches
from vardax._src.utils.masks import regular_mask
from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.preprocessing import train_test_split, xr_to_batch1d
from vardax._src.utils.standardize import (
    compute_scaler_params,
    apply_standardization,
    inverse_standardization,
)

# %% [markdown]
# ## 1. Simulate Lorenz-96

# %%
key = jax.random.PRNGKey(0)
N = 40
time_coords, states = simulate_lorenz96(
    key,
    N=N,
    F=8.0,
    dt=0.01,
    n_steps=5000,
    n_burn_in=1000,
)
print(f"states shape: {states.shape}, time range: [{time_coords[0]:.2f}, {time_coords[-1]:.2f}]")

# %% [markdown]
# ## 2. Build xarray Dataset and Extract Patches

# %%
ds = trajectory_to_xr_dataset(states, time_coords)
ds = extract_patches(ds, n_patches=200, n_timesteps=20, seed=42)
print(ds)

# %% [markdown]
# ## 3. Add Observation Masks and Gaussian Noise

# %%
ds = regular_mask(ds, variable="state", obs_interval=2)
ds = add_gaussian_noise(ds, variable="state", sigma=0.5, seed=0, name="obs")
print(ds)

# %% [markdown]
# ## 4. Train/Test Split and Standardize

# %%
ds_train, ds_test = train_test_split(ds, n_train=160, n_test=40, seed=0)
print(f"train patches: {ds_train.sizes['patch']}, test patches: {ds_test.sizes['patch']}")

mean, std = compute_scaler_params(ds_train, variable="state", mask_variable="mask")
print(f"mean={mean:.4f}, std={std:.4f}")

ds_train = apply_standardization(ds_train, variables=["state", "obs"], mean=mean, std=std)
ds_test = apply_standardization(ds_test, variables=["state", "obs"], mean=mean, std=std)

# %% [markdown]
# ## 5. Visualize the L96 Attractor

# %%
fig, ax = plot_l96_grid(states[:500], time_coords[:500])
ax.set_title("Lorenz-96 Space-Time (Hovmöller)")
plt.tight_layout()
plt.show()

# %%
fig, ax = plot_l96_trajectories(states[:500], time_coords[:500], n_vars=5)
ax.set_title("L96 Trajectories (5 variables)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Train FourDVarNet1D with L96Prior

# %%
import flax.nnx as nnx

batch_train = xr_to_batch1d(ds_train, state_var="state", obs_var="obs", mask_var="mask")
batch_test = xr_to_batch1d(ds_test, state_var="state", obs_var="obs", mask_var="mask")
print(f"train: input={batch_train.input.shape}, mask={batch_train.mask.shape}, target={batch_train.target.shape}")

# %%
_, T_dim, N_dim = batch_train.input.shape
model = FourDVarNet1D(
    state_dim=N_dim,
    n_time=T_dim,
    latent_dim=16,
    hidden_dim=32,
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
# ## 7. Evaluate and Visualize Reconstruction

# %%
recon = model(batch_test)

target_np = jnp.array(batch_test.target)
input_np = jnp.array(batch_test.input)
recon_np = jnp.array(recon)

fig, axes = plot_reconstruction_comparison(target_np, input_np, recon_np, sample_idx=0)
plt.suptitle("4DVarNet L96 Reconstruction")
plt.tight_layout()
plt.show()
