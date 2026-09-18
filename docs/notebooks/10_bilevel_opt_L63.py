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
# # 10 — Bilevel Optimisation: Learning the Cost Weights
#
# Ported from the mfourdvar *Bi-Level Estimation* chapter.
#
# A variational analysis $x^*(\theta)$ is itself a function of the
# hyper-parameters $\theta$ of its cost — here the weights of the observation
# and prior terms. Bilevel optimisation treats the inner minimisation as a
# differentiable layer and tunes $\theta$ with an *outer* objective evaluated
# on ground truth:
#
# $$
# \begin{aligned}
# \theta^* &= \underset{\theta}{\arg\min}\; L\big(x^*(\theta)\big) \\
# x^*(\theta) &= \underset{x}{\arg\min}\; U(x; \theta),
# \qquad
# U(x; \theta) = \alpha_{obs} \|m \odot (x - y)\|^2 + \alpha_{prior} \|x - \varphi(x)\|^2 .
# \end{aligned}
# $$
#
# The inner problem is solved by $K$ steps of gradient descent on
# [`variational_cost`](../api/costs_priors.md), written as a `lax.scan` so
# that `jax.grad` unrolls through it (chapter [12](../12_adjoint_methods.md)
# covers the implicit alternative).

# %%
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from vardax import (
    BilinAEPrior1D,
    simulate_lorenz63,
    variational_cost,
    variational_cost_grad,
)
from vardax._src.utils.masks import regular_mask
from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.patches import extract_patches, trajectory_to_xr_dataset
from vardax._src.utils.preprocessing import xr_to_batch1d
from vardax._src.utils.standardize import apply_standardization, compute_scaler_params

# %% [markdown]
# ## 1. Lorenz-63 patches
#
# The same preprocessing pipeline as notebook
# [08](08_classical_4dvar_vs_4dvarnet_L63.py): simulate, cut into windows,
# mask every other step, add noise, standardise, and split into a training
# set (used by the outer objective) and a held-out set (used only for the
# final comparison).

# %%
key = jax.random.PRNGKey(0)
time_coords, states = simulate_lorenz63(
    key, sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01, n_steps=5000, n_burn_in=1000
)
ds = trajectory_to_xr_dataset(states, time_coords, feature_names=["X", "Y", "Z"])
# Split the trajectory in time *before* cutting windows, so no window
# straddles the boundary: windows drawn at random from one trajectory and
# split afterwards would leak test timesteps into the training set.
n_time = ds.sizes["time"]
ds_train = extract_patches(ds.isel(time=slice(0, int(0.8 * n_time))), n_patches=64, n_timesteps=20, seed=42)
ds_test = extract_patches(ds.isel(time=slice(int(0.8 * n_time), None)), n_patches=32, n_timesteps=20, seed=43)
ds_train = regular_mask(ds_train, variable="state", obs_interval=2)
ds_test = regular_mask(ds_test, variable="state", obs_interval=2)
ds_train = add_gaussian_noise(ds_train, variable="state", sigma=0.5, seed=0, name="obs")
ds_test = add_gaussian_noise(ds_test, variable="state", sigma=0.5, seed=1, name="obs")
mean, std = compute_scaler_params(ds_train, variable="state", mask_variable="mask")
ds_train = apply_standardization(ds_train, variables=["state", "obs"], mean=mean, std=std)
ds_test = apply_standardization(ds_test, variables=["state", "obs"], mean=mean, std=std)

batch_train = xr_to_batch1d(ds_train, state_var="state", obs_var="obs", mask_var="mask")
batch_test = xr_to_batch1d(ds_test, state_var="state", obs_var="obs", mask_var="mask")
B, T, N = batch_train.input.shape
print(f"train {batch_train.input.shape}, test {batch_test.input.shape}")

# %% [markdown]
# ## 2. A fixed prior
#
# The prior $\varphi$ is a bilinear autoencoder pre-trained on clean
# trajectories (as in notebook [07](07_prior_pretraining_L63.py)) and then
# frozen: the only things learned below are the two cost weights.

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

# %% [markdown]
# ## 3. The inner solver as a differentiable function
#
# The hyper-parameters are stored as logs so the weights stay positive.
# `variational_cost` is a *mean* over all elements, so its gradient is tiny
# per element; scaling the step by the number of elements gives a
# per-element step size $\eta$ that is easy to reason about (it is gradient
# descent on the equivalent *sum* cost). With $\eta$ fixed, the overall scale
# of $(\alpha_{obs}, \alpha_{prior})$ acts as an effective step size and their
# ratio sets the obs/prior trade-off — both are learnable.

# %%
N_INNER = 20
ETA = 0.2


def weights(hyper):
    return jnp.exp(hyper["log_alpha_obs"]), jnp.exp(hyper["log_alpha_prior"])


def inner_solve(hyper, batch, prior, n_steps=N_INNER, eta=ETA):
    """K steps of gradient descent on U(x; theta), starting from masked obs."""
    alpha_obs, alpha_prior = weights(hyper)
    n_elem = batch.input.size

    def body(x, _):
        g = variational_cost_grad(x, batch, prior, alpha_obs, alpha_prior)
        x = x - eta * n_elem * g
        return x, variational_cost(x, batch, prior, alpha_obs, alpha_prior)

    x0 = batch.input * batch.mask
    x_star, cost_trace = jax.lax.scan(body, x0, None, length=n_steps)
    return x_star, cost_trace


def outer_loss(hyper, batch, prior):
    x_star, _ = inner_solve(hyper, batch, prior)
    return jnp.mean((x_star - batch.target) ** 2)


hyper_fixed = {"log_alpha_obs": jnp.log(0.5), "log_alpha_prior": jnp.log(0.5)}
mse_fixed_train = float(outer_loss(hyper_fixed, batch_train, prior))
print(f"outer loss with fixed weights (0.5, 0.5): {mse_fixed_train:.4f}")

# %% [markdown]
# ## 4. Outer loop — learn $(\alpha_{obs}, \alpha_{prior})$
#
# `jax.grad(outer_loss)` differentiates through the unrolled inner solver.
# The outer optimiser is plain Adam on the two log-weights.

# %%
outer_opt = optax.adam(learning_rate=0.05)


@jax.jit
def outer_step(hyper, opt_state):
    loss, grads = jax.value_and_grad(outer_loss)(hyper, batch_train, prior)
    updates, opt_state = outer_opt.update(grads, opt_state, hyper)
    return optax.apply_updates(hyper, updates), opt_state, loss


hyper = dict(hyper_fixed)
opt_state = outer_opt.init(hyper)
hist_alpha, hist_outer = [weights(hyper)], []
for _ in range(150):
    hyper, opt_state, loss = outer_step(hyper, opt_state)
    hist_alpha.append(weights(hyper))
    hist_outer.append(float(loss))
hist_alpha = jnp.array(hist_alpha)

a_obs, a_prior = weights(hyper)
print(f"learned weights: alpha_obs={float(a_obs):.3f}, alpha_prior={float(a_prior):.3f}")
print(f"outer loss: {mse_fixed_train:.4f} -> {hist_outer[-1]:.4f}")

# %% [markdown]
# ## 5. Fixed vs learned weights on held-out windows

# %%
x_fixed, trace_fixed = inner_solve(hyper_fixed, batch_test, prior)
x_learned, trace_learned = inner_solve(hyper, batch_test, prior)
mse_fixed = float(jnp.mean((x_fixed - batch_test.target) ** 2))
mse_learned = float(jnp.mean((x_learned - batch_test.target) ** 2))
mse_obs_only = float(jnp.mean((batch_test.input * batch_test.mask - batch_test.target) ** 2))
print(f"held-out MSE: masked obs={mse_obs_only:.4f}, fixed={mse_fixed:.4f}, learned={mse_learned:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

axes[0].plot(hist_alpha[:, 0], label=r"$\alpha_{obs}$", color="tab:blue")
axes[0].plot(hist_alpha[:, 1], label=r"$\alpha_{prior}$", color="tab:orange")
axes[0].set_xlabel("outer iteration")
axes[0].set_title("Learned cost weights")
axes[0].legend()

axes[1].plot(hist_outer, color="tab:red")
axes[1].set_xlabel("outer iteration")
axes[1].set_ylabel("outer loss (train MSE)")
axes[1].set_title("Outer objective")

bars = axes[2].bar(
    ["fixed\n(0.5, 0.5)", "learned"],
    [mse_fixed, mse_learned],
    color=["steelblue", "tomato"],
)
for bar, val in zip(bars, [mse_fixed, mse_learned]):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f"{val:.4f}",
        ha="center",
        va="bottom",
    )
axes[2].set_ylabel("held-out MSE")
axes[2].set_title("Reconstruction quality")

plt.tight_layout()
plt.show()

# %% [markdown]
# The inner cost traces show why: the learned weights change the geometry of
# $U$ so that $K$ steps land closer to the truth, even though the cost value
# itself is not what the outer loop optimises.

# %%
sample, feature = 0, 0
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))

axes[0].semilogy(trace_fixed, label="fixed weights", color="steelblue")
axes[0].semilogy(trace_learned, label="learned weights", color="tomato")
axes[0].set_xlabel("inner step")
axes[0].set_ylabel("U(x)")
axes[0].set_title("Inner cost on held-out batch")
axes[0].legend()

axes[1].plot(batch_test.target[sample, :, feature], color="black", label="truth")
axes[1].plot(x_fixed[sample, :, feature], color="steelblue", label="fixed")
axes[1].plot(x_learned[sample, :, feature], color="tomato", label="learned")
axes[1].scatter(
    range(T),
    jnp.where(batch_test.mask[sample, :, feature] > 0, batch_test.input[sample, :, feature], jnp.nan),
    color="gray",
    s=20,
    label="obs",
    zorder=5,
)
axes[1].set_xlabel("time step")
axes[1].set_title("Held-out window, component X")
axes[1].legend(ncol=2)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - Any analysis produced by a differentiable inner solver is a function of
#   the cost's hyper-parameters; an outer objective on ground truth turns
#   hand-tuning into optimisation.
# - Here the inner loop is unrolled gradient descent on `variational_cost`
#   and `jax.grad` back-propagates through all $K$ steps. This costs
#   $O(K)$ memory; for large states use `optimistix` with
#   `ImplicitAdjoint` (chapter [12](../12_adjoint_methods.md)) or the
#   one-step adjoint of notebook [02](02_unrolling_vs_fixedpoint_L63.py).
# - The same mechanism learns any other cost hyper-parameter — the inner
#   step size, the prior's own weights (that is 4DVarNet, chapter
#   [9](../09_4dvarnet.md)), or the ODE parameters of notebook
#   [09](09_param_estimation_L63.py).
