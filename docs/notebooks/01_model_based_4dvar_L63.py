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
# Classical strong-constraint 4DVar with the *true* Lorenz-63 dynamics as the
# model. Given a background estimate $x_b$ of the initial state and noisy,
# gappy observations $y_t$ along a window, the analysis is the initial state
# whose model trajectory best fits both:
#
# $$
# x_0^* = \underset{x_0}{\arg\min}\;
#   \tfrac{1}{2}\|x_0 - x_b\|^2_{B^{-1}}
#   + \tfrac{1}{2}\sum_{t=0}^{T} \|m_t \odot (y_t - M_t(x_0))\|^2_{R^{-1}} .
# $$
#
# This is [`StrongFourDVar`](../api/models.md) (chapter
# [6](../06_strong_4dvar.md)). The forward model $M_t$ comes from a
# [`DynTrajectory`](../api/costs_priors.md) ODE prior via
# `as_forward_model`, so the same object that serves as a dynamical prior
# elsewhere drives the rollout here. The later notebooks replace pieces of
# this picture with learned components; this one is the reference point.

# %%
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt

from vardax import (
    Batch1D,
    DynTrajectory,
    Lorenz63,
    MaskedIdentity,
    StrongFourDVar,
    simulate_lorenz63,
)

# %% [markdown]
# ## 1. Truth, background, and observations
#
# We simulate Lorenz-63 on the attractor, take one assimilation window of
# $T + 1$ states at spacing $\Delta t = 0.05$, and observe every other step
# with Gaussian noise. The background $x_b$ is the true initial state
# perturbed with a larger error than the observations carry.

# %%
DT_SIM, SUBSAMPLE = 0.01, 5
DT = DT_SIM * SUBSAMPLE  # observation / model step
T = 20  # rollout steps; the window holds T + 1 states
SIGMA_OBS, SIGMA_BG = 0.5, 2.0

key = jax.random.PRNGKey(0)
time_coords, states = simulate_lorenz63(key, dt=DT_SIM, n_steps=2000, n_burn_in=1000)
start = 500
x_true = states[start : start + SUBSAMPLE * (T + 1) : SUBSAMPLE]  # (T+1, 3)
ts = jnp.arange(T + 1) * DT

k_obs, k_bg = jax.random.split(jax.random.PRNGKey(1))
mask = jnp.zeros((T + 1, 3)).at[::2, :].set(1.0)
y_obs = x_true + SIGMA_OBS * jax.random.normal(k_obs, x_true.shape)
x_b = x_true[0] + SIGMA_BG * jax.random.normal(k_bg, (3,))

# StrongFourDVar consumes a Batch1D of shape (B, T+1, N); one window here.
batch = Batch1D(input=(y_obs * mask)[None], mask=mask[None], target=x_true[None])
print(f"window: {T + 1} states at dt={DT}, observed fraction {float(mask.mean()):.2f}")
print(f"background error |x_b - x_0|: {float(jnp.linalg.norm(x_b - x_true[0])):.3f}")

# %% [markdown]
# ## 2. The model and the analysis step
#
# `Lorenz63` is the vector field; wrapping it in `DynTrajectory` gives an
# ODE solve (Tsit5 with adaptive steps by default), and `as_forward_model`
# exposes the one-step `step(state, dt)` interface `StrongFourDVar` rolls
# out. $B$ and $R$ are diagonal with the true error variances; the tag tells
# lineax they are positive definite so the cost can solve with CG.

# %%
prior = DynTrajectory(model=Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0))
forward = prior.as_forward_model(dt=DT)


def diag_cov(variance: float) -> lx.AbstractLinearOperator:
    op = lx.DiagonalLinearOperator(jnp.full(3, variance))
    return lx.TaggedLinearOperator(op, lx.positive_semidefinite_tag)


model = StrongFourDVar(
    forward=forward,
    obs_op=MaskedIdentity(),
    prior_mean=x_b,
    prior_cov_op=diag_cov(SIGMA_BG**2),
    obs_cov_op=diag_cov(SIGMA_OBS**2),
    max_steps=200,
)

x0_analysis = model(batch)[0]
print(f"analysis error   |x_0* - x_0|: {float(jnp.linalg.norm(x0_analysis - x_true[0])):.3f}")

# %% [markdown]
# ## 3. First guess vs analysis along the window
#
# Rolling the background and the analysis through the same dynamics shows
# what the observations bought: the analysed trajectory tracks the truth
# across the whole window, including the unobserved steps.

# %%
traj_first_guess = prior(x_b, ts)
traj_analysis = prior(x0_analysis, ts)

rmse = lambda traj: float(jnp.sqrt(jnp.mean((traj - x_true) ** 2)))  # noqa: E731
obs_rmse = float(jnp.sqrt(jnp.sum((mask * (y_obs - x_true)) ** 2) / mask.sum()))
print(f"trajectory RMSE: first guess={rmse(traj_first_guess):.3f}, analysis={rmse(traj_analysis):.3f}")
print(f"observation RMSE (at observed points): {obs_rmse:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
    ax.plot(ts, x_true[:, i], color="black", label="truth")
    ax.plot(ts, traj_first_guess[:, i], color="gray", linestyle=":", label="first guess (from $x_b$)")
    ax.plot(ts, traj_analysis[:, i], color="tab:blue", label="analysis (from $x_0^*$)")
    ax.scatter(
        ts,
        jnp.where(mask[:, i] > 0, y_obs[:, i], jnp.nan),
        color="tab:red",
        s=18,
        zorder=5,
        label="observations",
    )
    ax.set_ylabel(name)
axes[0].legend(loc="upper right", ncol=2)
axes[-1].set_xlabel("time")
fig.suptitle("Strong-constraint 4DVar on Lorenz-63")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Error budget

# %%
labels = ["background\n$x_b$", "analysis\n$x_0^*$"]
errors = [
    float(jnp.linalg.norm(x_b - x_true[0])),
    float(jnp.linalg.norm(x0_analysis - x_true[0])),
]
traj_errors = [rmse(traj_first_guess), rmse(traj_analysis)]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, vals, title in zip(
    axes, [errors, traj_errors], ["initial-state error", "trajectory RMSE over window"]
):
    bars = ax.bar(labels, vals, color=["gray", "tab:blue"])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom")
    ax.set_title(title)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - `StrongFourDVar` minimises the classical 4DVar cost over the initial
#   state with `optimistix` (BFGS by default); the dynamics are enforced
#   exactly through the rollout.
# - Any object with `step(state, dt)` can be the model. Here it is the
#   Lorenz-63 ODE via `DynTrajectory.as_forward_model`; chapter
#   [19](../19_physical_models.md) covers the ODE priors and notebook
#   [09](09_param_estimation_L63.py) makes their parameters learnable.
# - Notebook [08](08_classical_4dvar_vs_4dvarnet_L63.py) contrasts this
#   model-based analysis with the learned 4DVarNet solver.
