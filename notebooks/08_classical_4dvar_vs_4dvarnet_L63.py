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
# # 08 — Classical 4DVar vs 4DVarNet on Lorenz-63
#
# This notebook compares:
#
# 1. **Classical 4DVar** — minimise the variational cost
#    $U(x) = \alpha_{obs}\|m\odot(x-y)\|^2 + \alpha_{prior}\|x-\varphi(x)\|^2$
#    with respect to $x$ using a gradient-descent optimisation loop in JAX.
#
# 2. **4DVarNet (learned)** — `FourDVarNet1D` trained end-to-end.
#
# We compare reconstruction MSE and convergence behaviour.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import vardax
from vardax import (
    Batch1D,
    BilinAEPrior1D,
    FourDVarNet1D,
    variational_cost,
    variational_cost_grad,
    decomposed_loss,
)
from vardax._src.utils.dynamical_systems import simulate_lorenz63
from vardax._src.utils.patches import trajectory_to_xr_dataset, extract_patches
from vardax._src.utils.masks import regular_mask
from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.preprocessing import train_test_split, xr_to_batch1d
from vardax._src.utils.standardize import compute_scaler_params, apply_standardization

# %% [markdown]
# ## 1. Prepare L63 data

# %%
key = jax.random.PRNGKey(0)
time_coords, states = simulate_lorenz63(
    key, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, n_steps=5000, n_burn_in=1000
)
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
B, T, N = batch_test.input.shape
print(f"Test batch shape: {batch_test.input.shape}")

# %% [markdown]
# ## 2. Classical 4DVar — gradient descent on $x$
#
# We use a fixed (randomly initialised) prior and minimise $U(x)$ with respect
# to $x$ directly, running a manual gradient-descent loop so we can track the
# convergence trajectory.

# %%
import flax.nnx as nnx

prior = BilinAEPrior1D(state_dim=N, latent_dim=8, n_time=T, rngs=nnx.Rngs(jax.random.PRNGKey(5)))


# Initialise x from masked observations
x_classical = batch_test.input * batch_test.mask

classical_losses = []
# Classical gradient descent uses a larger learning rate than 4DVarNet (1e-3)
# because we are optimising directly in state space (low-dimensional), whereas
# 4DVarNet optimises millions of network parameters requiring a much smaller lr.
lr_classical = 0.05
n_classical_steps = 50

grad_fn = jax.jit(jax.value_and_grad(variational_cost))

for step in range(n_classical_steps):
    loss_val, grad = grad_fn(
        x_classical, batch_test, prior, alpha_obs=0.5, alpha_prior=0.5
    )
    x_classical = x_classical - lr_classical * grad
    classical_losses.append(float(loss_val))

mse_classical = float(jnp.mean((x_classical - batch_test.target) ** 2))
print(f"Classical 4DVar MSE after {n_classical_steps} steps: {mse_classical:.4f}")

# Decomposed loss at convergence
dl = decomposed_loss(x_classical, batch_test, prior)
print(
    f"  obs={float(dl['obs']):.4f}, prior={float(dl['prior']):.4f}, total={float(dl['total']):.4f}"
)

# %% [markdown]
# ## 3. 4DVarNet — trained end-to-end

# %%
model = FourDVarNet1D(
    state_dim=N, n_time=T, latent_dim=8, hidden_dim=16, n_solver_steps=10,
    rngs=nnx.Rngs(jax.random.PRNGKey(1)),
)

_, train_losses, _ = vardax.fit(
    model,
    [batch_train],
    n_epochs=10,
    lr=1e-3,
    verbose=False,
)
print(f"4DVarNet final train loss: {train_losses[-1]:.4f}")

out_4dvarnet = model(batch_test)
mse_4dvarnet = float(jnp.mean((out_4dvarnet - batch_test.target) ** 2))
print(f"4DVarNet test MSE: {mse_4dvarnet:.4f}")

# %% [markdown]
# ## 4. Comparison: convergence and reconstruction quality

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Convergence of classical 4DVar
axes[0].semilogy(classical_losses, color="steelblue")
axes[0].set_xlabel("Gradient-descent step")
axes[0].set_ylabel("Variational cost U(x)")
axes[0].set_title("Classical 4DVar convergence")

# 4DVarNet learning curve
axes[1].semilogy(train_losses, color="tomato")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Train loss")
axes[1].set_title("4DVarNet learning curve")

# MSE bar chart
bars = axes[2].bar(
    ["Classical 4DVar\n(fixed prior)", "4DVarNet\n(trained)"],
    [mse_classical, mse_4dvarnet],
    color=["steelblue", "tomato"],
)
axes[2].set_ylabel("Test MSE")
axes[2].set_title("Reconstruction MSE")
for bar, val in zip(bars, [mse_classical, mse_4dvarnet]):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.001,
        f"{val:.4f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Visualise reconstructions side by side

# %%
sample_idx = 0
feature_idx = 0

fig, axes = plt.subplots(1, 3, figsize=(14, 3), sharey=True)
t_axis = range(T)

for ax, recon, label, color in zip(
    axes,
    [
        x_classical[sample_idx, :, feature_idx],
        out_4dvarnet[sample_idx, :, feature_idx],
        batch_test.target[sample_idx, :, feature_idx],
    ],
    ["Classical 4DVar", "4DVarNet", "Ground truth"],
    ["steelblue", "tomato", "black"],
):
    ax.plot(t_axis, recon, color=color, linewidth=1.5, label=label)
    ax.scatter(
        t_axis,
        batch_test.input[sample_idx, :, feature_idx],
        c="gray",
        s=20,
        zorder=5,
        label="Obs",
    )
    ax.set_title(label)
    ax.set_xlabel("Time step")

axes[0].set_ylabel("State (X)")
plt.suptitle(f"Reconstruction comparison — sample {sample_idx}, feature X")
plt.tight_layout()
plt.show()
