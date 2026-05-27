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
# # 01 — Model-Based 4DVar on Lorenz-63
#
# This notebook demonstrates classical model-based 4DVar on the Lorenz-63
# attractor using `vardax`.

# %%
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import vardax
from vardax import Batch1D, FourDVarNet1D

# %% [markdown]
# ## Generate synthetic Lorenz-63 data

# %%
key = jax.random.PRNGKey(0)

# Simulate a simple synthetic trajectory (placeholder)
B, T, N = 4, 20, 3
k1, k2 = jax.random.split(key)
target = jax.random.normal(k1, (B, T, N))
mask = (jax.random.uniform(k2, (B, T, N)) > 0.5).astype(jnp.float32)
inp = target * mask
batch = Batch1D(input=inp, mask=mask, target=target)

print(f"Batch shapes: input={batch.input.shape}, mask={batch.mask.shape}, target={batch.target.shape}")

# %% [markdown]
# ## Create and initialise model

# %%
model = FourDVarNet1D(
    state_dim=N,
    n_time=T,
    latent_dim=8,
    hidden_dim=16,
    n_solver_steps=5,
    rngs=nnx.Rngs(jax.random.PRNGKey(1)),
)

# %% [markdown]
# ## Run inference

# %%
out = model(batch)
print(f"Output shape: {out.shape}")
mse = jnp.mean((out - target) ** 2)
print(f"MSE (untrained model): {mse:.4f}")

# %% [markdown]
# ## Visualise first sample

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for i, (ax, title) in enumerate(zip(axes, ["Target", "Masked Input", "Reconstruction"])):
    data = [target, inp, out][i]
    ax.imshow(data[0].T, aspect="auto", cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Component")
plt.tight_layout()
plt.show()
