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
# # 11 — Learning the Gradient Update
#
# Ported from the mfourdvar *Learning how to Learn* chapter.
#
# The core 4DVarNet idea is that the *solver* can be learned: instead of
# the fixed gradient-descent update $x_{k+1} = x_k - \eta \nabla_x U(x_k)$, a
# recurrent network $g_\phi$ maps the gradient (and its own memory) to the
# update,
#
# $$
# x_{k+1} = x_k - g_\phi\big(\nabla_x U(x_k),\; x_k,\; h_k\big),
# $$
#
# and $\phi$ is trained so that $K$ such steps land close to the truth. In
# vardax this is the `ConvLSTMGradMod1D` *gradient modulator* — see the
# *learned inner solver* and *gradient modulator family* sections of
# chapter [9](../09_4dvarnet.md).
#
# This notebook isolates that one ingredient. The prior $\varphi$ is
# pre-trained and **frozen**, and only the modulator is trained, so any
# improvement over vanilla gradient descent is attributable to the learned
# update rule alone. We then ablate the number of solver steps $K$.

# %%
import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from vardax import (
    BilinAEPrior1D,
    ConvLSTMGradMod1D,
    init_solver_state_1d,
    simulate_lorenz63,
    solve_4dvarnet_1d,
    solver_step_1d,
)
from vardax._src.utils.masks import regular_mask
from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.patches import extract_patches, trajectory_to_xr_dataset
from vardax._src.utils.preprocessing import train_test_split, xr_to_batch1d
from vardax._src.utils.standardize import apply_standardization, compute_scaler_params

# %% [markdown]
# ## 1. Lorenz-63 patches and a frozen prior

# %%
key = jax.random.PRNGKey(0)
time_coords, states = simulate_lorenz63(
    key, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, n_steps=5000, n_burn_in=1000
)
ds = trajectory_to_xr_dataset(states, time_coords, feature_names=["X", "Y", "Z"])
ds = extract_patches(ds, n_patches=96, n_timesteps=20, seed=42)
ds = regular_mask(ds, variable="state", obs_interval=2)
ds = add_gaussian_noise(ds, variable="state", sigma=0.5, seed=0, name="obs")

ds_train, ds_test = train_test_split(ds, n_train=64, n_test=32, seed=0)
mean, std = compute_scaler_params(ds_train, variable="state", mask_variable="mask")
ds_train = apply_standardization(ds_train, variables=["state", "obs"], mean=mean, std=std)
ds_test = apply_standardization(ds_test, variables=["state", "obs"], mean=mean, std=std)

batch_train = xr_to_batch1d(ds_train, state_var="state", obs_var="obs", mask_var="mask")
batch_test = xr_to_batch1d(ds_test, state_var="state", obs_var="obs", mask_var="mask")
B, T, N = batch_train.input.shape
print(f"train {batch_train.input.shape}, test {batch_test.input.shape}")

# %%
prior = BilinAEPrior1D(state_dim=N, latent_dim=8, n_time=T, key=jax.random.PRNGKey(10))
pre_opt = optax.adam(1e-2)
pre_state = pre_opt.init(eqx.filter(prior, eqx.is_array))


@eqx.filter_jit
def pretrain_step(prior, opt_state, x):
    loss, grads = eqx.filter_value_and_grad(lambda p: jnp.mean((x - p(x)) ** 2))(prior)
    updates, opt_state = pre_opt.update(grads, opt_state, prior)
    return eqx.apply_updates(prior, updates), opt_state, loss


for _ in range(300):
    prior, pre_state, pre_loss = pretrain_step(prior, pre_state, batch_train.target)
print(f"prior reconstruction MSE after pre-training: {float(pre_loss):.4f}")

# From here on `prior` is a closed-over constant: nothing below differentiates
# with respect to it.

# %% [markdown]
# ## 2. Baseline — vanilla gradient descent
#
# The solver's cost is the one used by `solver_step_1d`:
# $U(x) = \|m \odot (x - y)\|^2 + \lambda \|x - \varphi(x)\|^2$ (sums, with
# $\lambda = 1$). Vanilla gradient descent takes $K$ steps with a fixed step
# size $\eta$ from the masked observations.

# %%
PRIOR_WEIGHT = 1.0
ETA = 0.1


def solver_cost(x, batch):
    j_obs = jnp.sum((batch.mask * (x - batch.input)) ** 2)
    j_prior = PRIOR_WEIGHT * jnp.sum((x - prior(x)) ** 2)
    return j_obs + j_prior


def mse(x, batch):
    return jnp.mean((x - batch.target) ** 2)


@ft.partial(jax.jit, static_argnames=("n_steps", "eta"))
def vanilla_gd_trace(batch, n_steps, eta=ETA):
    """Per-step MSE of K vanilla gradient-descent steps (index 0 = init)."""

    def body(x, _):
        x = x - eta * jax.grad(solver_cost)(x, batch)
        return x, mse(x, batch)

    x0 = batch.input * batch.mask
    x_final, trace = jax.lax.scan(body, x0, None, length=n_steps)
    return x_final, jnp.concatenate([jnp.array([mse(x0, batch)]), trace])


K = 10
_, gd_trace = vanilla_gd_trace(batch_test, K)
print(f"vanilla GD, K={K}: MSE {float(gd_trace[0]):.4f} -> {float(gd_trace[-1]):.4f}")

# %% [markdown]
# ## 3. Train only the gradient modulator
#
# `solve_4dvarnet_1d(batch, prior, grad_mod, n_steps, hidden_dim)` runs the
# learned solver. The modulator is the only argument we differentiate:
# `eqx.filter_value_and_grad` takes gradients with respect to the arrays of
# its first argument, and the frozen prior is captured by closure.

# %%
HIDDEN = 16


def make_grad_mod(seed):
    return ConvLSTMGradMod1D(state_channels=T, hidden_dim=HIDDEN, key=jax.random.PRNGKey(seed))


def train_grad_mod(grad_mod, batch, n_steps, n_iters=300, lr=3e-3):
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(grad_mod, eqx.is_array))

    def loss_fn(gm):
        x = solve_4dvarnet_1d(batch, prior, gm, n_steps=n_steps, hidden_dim=HIDDEN)
        return mse(x, batch)

    @eqx.filter_jit
    def step(gm, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(gm)
        updates, opt_state = opt.update(grads, opt_state, gm)
        return eqx.apply_updates(gm, updates), opt_state, loss

    history = []
    for _ in range(n_iters):
        grad_mod, opt_state, loss = step(grad_mod, opt_state)
        history.append(float(loss))
    return grad_mod, history


grad_mod, train_hist = train_grad_mod(make_grad_mod(1), batch_train, n_steps=K)
print(f"modulator training loss: {train_hist[0]:.4f} -> {train_hist[-1]:.4f}")


@eqx.filter_jit
def learned_trace(grad_mod, batch, n_steps):
    """Per-step MSE of the learned solver (index 0 = init)."""
    state = init_solver_state_1d(batch, HIDDEN)
    trace = [mse(state.x, batch)]
    for _ in range(n_steps):
        state = solver_step_1d(state, batch, prior, grad_mod, prior_weight=PRIOR_WEIGHT)
        trace.append(mse(state.x, batch))
    return state.x, jnp.stack(trace)


_, lm_trace = learned_trace(grad_mod, batch_test, K)
print(f"learned solver, K={K}: MSE {float(lm_trace[0]):.4f} -> {float(lm_trace[-1]):.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))

axes[0].semilogy(train_hist, color="tab:red")
axes[0].set_xlabel("training iteration")
axes[0].set_ylabel("train MSE after K steps")
axes[0].set_title(f"Modulator training (K={K})")

axes[1].plot(gd_trace, marker="o", color="steelblue", label=f"vanilla GD (η={ETA})")
axes[1].plot(lm_trace, marker="o", color="tomato", label="learned modulator")
axes[1].set_xlabel("solver step k")
axes[1].set_ylabel("held-out MSE")
axes[1].set_title("Reconstruction error along the solve")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Ablation — number of solver steps $K$
#
# For each $K$ we train a fresh modulator *at that* $K$ and compare its
# held-out MSE with vanilla gradient descent run for the same number of
# steps. The learned update is most valuable when the iteration budget is
# small; vanilla descent needs many more steps to catch up.

# %%
K_VALUES = [1, 2, 5, 10, 15]
mse_gd, mse_learned = [], []
for k in K_VALUES:
    _, tr = vanilla_gd_trace(batch_test, k)
    mse_gd.append(float(tr[-1]))
    gm_k, _ = train_grad_mod(make_grad_mod(1), batch_train, n_steps=k, n_iters=200)
    _, tr = learned_trace(gm_k, batch_test, k)
    mse_learned.append(float(tr[-1]))
    print(f"K={k:2d}: vanilla GD={mse_gd[-1]:.4f}  learned={mse_learned[-1]:.4f}")

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(K_VALUES, mse_gd, marker="o", color="steelblue", label="vanilla GD")
ax.plot(K_VALUES, mse_learned, marker="o", color="tomato", label="learned modulator")
ax.axhline(float(gd_trace[0]), color="gray", linestyle=":", label="masked obs (no solve)")
ax.set_xlabel("solver steps K")
ax.set_ylabel("held-out MSE")
ax.set_title("Solver-steps ablation")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - With the prior frozen, replacing the fixed gradient step by a trained
#   `ConvLSTMGradMod1D` is what buys the fast convergence of 4DVarNet: the
#   modulator learns step sizes, momentum, and preconditioning from data.
# - `solve_4dvarnet_1d` / `solver_step_1d` expose the solver as plain
#   functions, so partial training is a matter of which argument you
#   differentiate.
# - `FourDVarNet1D` (notebook [03](03_4dvarnet_L63.py)) trains the prior and
#   the modulator jointly; chapter [9](../09_4dvarnet.md) gives the full
#   picture.
