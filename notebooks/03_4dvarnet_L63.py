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
# # 03 — 4DVarNet End-to-End on Lorenz-63
#
# Demonstrates end-to-end training of `FourDVarNet1D` on a synthetic
# Lorenz-63 dataset.

# %%
# (NNX removed in Epic 0 — vardax is now equinox-native)
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import vardax
from vardax import Batch1D, FourDVarNet1D

# %% [markdown]
# ## Generate training data

# %%
def make_batches(key, n_batches=20, B=8, T=20, N=3, obs_rate=0.5):
    batches = []
    for i in range(n_batches):
        k1, k2, key = jax.random.split(key, 3)
        target = jax.random.normal(k1, (B, T, N))
        mask = (jax.random.uniform(k2, (B, T, N)) > obs_rate).astype(jnp.float32)
        batches.append(Batch1D(input=target * mask, mask=mask, target=target))
    return batches


key = jax.random.PRNGKey(0)
train_batches = make_batches(key, n_batches=10)
val_batches = make_batches(jax.random.PRNGKey(99), n_batches=3)

# %% [markdown]
# ## Define and train model

# %%
model = FourDVarNet1D(
    state_dim=3,
    n_time=20,
    latent_dim=8,
    hidden_dim=16,
    n_solver_steps=5,
    key=jax.random.PRNGKey(1),
)

optimizer, train_losses, val_losses = fit(
    model,
    train_batches,
    lr=1e-3,
    n_epochs=3,
    val_batches=val_batches,
    verbose=True,
)

# %% [markdown]
# ## Plot learning curves

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses, label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("4DVarNet Learning Curves (L63)")
ax.legend()
plt.tight_layout()
plt.show()
