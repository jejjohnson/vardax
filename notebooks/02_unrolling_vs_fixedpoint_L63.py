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
# # 02 — Unrolling vs Fixed-Point on Lorenz-63
#
# This notebook compares the **unrolled** solver (standard backprop through
# $K$ gradient steps, guided by a ConvLSTM gradient modulator) to the
# **fixed-point** projection solver (repeated prior-projection with
# observation re-insertion) on reconstructing partially-observed Lorenz-63
# trajectories.
#
# **Pipeline overview:**
# 1. Simulate L63 data with Diffrax
# 2. Extract patches, add masks and noise
# 3. Warm-start with `obs_interpolation_init` vs. zero initialisation
# 4. Run unrolled solver (`FourDVarNet1D`, K=10 steps)
# 5. Run fixed-point solver (`solve_4dvarnet_1d_fixedpoint`, K=10 steps)
# 6. Compare MSE across conditions with a bar chart

# %%
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import vardax
from vardax import (
    Batch1D,
    BilinAEPrior1D,
    FourDVarNet1D,
    solve_4dvarnet_1d_fixedpoint,
)
from vardax._src.utils.dynamical_systems import simulate_lorenz63
from vardax._src.utils.patches import trajectory_to_xr_dataset, extract_patches
from vardax._src.utils.masks import regular_mask
from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.preprocessing import (
    train_test_split,
    xr_to_batch1d,
    obs_interpolation_init,
)
from vardax._src.utils.standardize import compute_scaler_params, apply_standardization
import xarray as xr

# %% [markdown]
# ## 1. Simulate Lorenz-63 and build patches

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
print(f"states shape: {states.shape}")

ds = trajectory_to_xr_dataset(states, time_coords, feature_names=["X", "Y", "Z"])
ds = extract_patches(ds, n_patches=200, n_timesteps=20, seed=42)
ds = regular_mask(ds, variable="state", obs_interval=2)
ds = add_gaussian_noise(ds, variable="state", sigma=0.5, seed=0, name="obs")

ds_train, ds_test = train_test_split(ds, n_train=160, n_test=40, seed=0)
mean, std = compute_scaler_params(ds_train, variable="state", mask_variable="mask")
ds_train = apply_standardization(ds_train, variables=["state", "obs"], mean=mean, std=std)
ds_test = apply_standardization(ds_test, variables=["state", "obs"], mean=mean, std=std)

batch_train = xr_to_batch1d(ds_train, state_var="state", obs_var="obs", mask_var="mask")
batch_test = xr_to_batch1d(ds_test, state_var="state", obs_var="obs", mask_var="mask")
print(f"train batch: {batch_train.input.shape}, test batch: {batch_test.input.shape}")

# %% [markdown]
# ## 2. Warm-start initialisation via `obs_interpolation_init`
#
# Build a NaN-masked obs dataset so we can use `obs_interpolation_init`.

# %%
state_vals = ds_test["state"].values
mask_vals = ds_test["mask"].values.astype(bool)
obs_nan = np.where(mask_vals, ds_test["obs"].values, np.nan).astype(np.float32)
obs_nan_da = xr.DataArray(obs_nan, dims=ds_test["obs"].dims, coords=ds_test["obs"].coords)
ds_test_nan = ds_test.assign(obs_nan=obs_nan_da)

ds_test_init = obs_interpolation_init(
    ds_test_nan, variable="state", obs_variable="obs_nan"
)
x_init = jnp.array(ds_test_init["state_init"].values)
print(f"Warm-start init shape: {x_init.shape}")

# MSE of warm-start vs zero init
target = batch_test.target
mse_zero_init = float(jnp.mean((batch_test.input * batch_test.mask - target) ** 2))
mse_warm_init = float(jnp.mean((x_init - target) ** 2))
print(f"Zero-init MSE: {mse_zero_init:.4f}")
print(f"Warm-start MSE: {mse_warm_init:.4f}")

# %% [markdown]
# ## 3. Train FourDVarNet1D (unrolled solver)

# %%
B, T, N = batch_train.input.shape
model_unrolled = FourDVarNet1D(
    state_dim=N,
    n_time=T,
    latent_dim=8,
    hidden_dim=16,
    n_solver_steps=10,
    rngs=nnx.Rngs(jax.random.PRNGKey(1)),
)

optimizer, train_losses, _ = vardax.fit(
    model_unrolled,
    [batch_train],
    n_epochs=5,
    lr=1e-3,
    verbose=True,
)
print(f"Final unrolled train loss: {train_losses[-1]:.4f}")

# %% [markdown]
# ## 4. Fixed-point solver (untrained prior)

# %%
prior = BilinAEPrior1D(state_dim=N, latent_dim=8, n_time=T, rngs=nnx.Rngs(jax.random.PRNGKey(3)))

out_fp = solve_4dvarnet_1d_fixedpoint(batch_test, prior, n_fp_steps=10)
print(f"Fixed-point output shape: {out_fp.shape}")

# %% [markdown]
# ## 5. Evaluate unrolled solver on test batch

# %%
out_unrolled = model_unrolled(batch_test)
print(f"Unrolled output shape: {out_unrolled.shape}")

# %% [markdown]
# ## 6. Compare MSE across conditions

# %%
mse_fp = float(jnp.mean((out_fp - target) ** 2))
mse_unrolled = float(jnp.mean((out_unrolled - target) ** 2))

labels = ["Zero init", "Warm start\n(interp)", "Fixed-point\n(K=10)", "Unrolled\n(trained, K=10)"]
values = [mse_zero_init, mse_warm_init, mse_fp, mse_unrolled]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, values, color=["#aec6cf", "#77dd77", "#fdfd96", "#ff9999"])
ax.set_ylabel("MSE")
ax.set_title("Reconstruction MSE: zero init vs warm start vs fixed-point vs unrolled")
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.002,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.tight_layout()
plt.show()
print("MSE summary:")
for label, val in zip(labels, values):
    print(f"  {label.replace(chr(10), ' ')}: {val:.4f}")

