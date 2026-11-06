---
numbering:
  enumerator: "9.%s"
---

# 4DVarNet (learned 4DVar)

4DVarNet replaces two pieces of classical 4DVar with learned
counterparts: the Gaussian prior $\Vert \vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}}$ becomes a
learned reconstruction $\Vert \vect{x} - g_\text{ae}(\vect{x}; \gvect{\theta}) \Vert_2^2$, and the inner
gradient-descent solver becomes a learned iteration with a ConvLSTM
gradient modulator. The result is trainable end-to-end on
(input, target) pairs.

The "Net" naming follows Fablet et al. (2021). vardax keeps it as the name of the *method*, not the family: 4DVarNet is one of seven peer analysis methods (Decision D14), a learned variant of strong-constraint 4DVar distinguished by its learned components, and not the parent of classical 4DVar.

## Cost function

The variational cost is structurally familiar:

$$\mathcal{J}(\vect{x}) = \alpha_o \frac{1}{2} \Vert h(\vect{x}) - \vect{y} \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}} + \alpha_p \Vert \vect{x} - g_\text{ae}(\vect{x}; \gvect{\theta}) \Vert_2^2$$ (eq-4dvarnet-cost)

where the first term is the observation cost [](#eq-obs-cost), $g_\text{ae}$ is a learned autoencoder with parameters $\gvect{\theta}$, and $\alpha_o$ and $\alpha_p$ weight the two terms.
Here $\vect{x} \in \mathbb{R}^{D_x}$ is the state of a window and $\vect{y} \in \mathbb{R}^{D_y}$ its observations, $h: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$, $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$, $g_\text{ae}: \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ and $\gvect{\theta} \in \mathbb{R}^{D_\theta}$.
The Gaussian background
term $\Vert \vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}}$ is **replaced** (not added). The learned
prior is expressive enough to encode the same information that $\gmat{\Sigma}_x$ +
$\gvect{\mu}_x$ would, plus structure that classical Gaussian priors can't
represent (multimodality, nonlinear manifolds, conditional
dependencies).

## The learned inner solver

Classical 4DVar minimises $\mathcal{J}$ via gradient descent or a quasi-Newton
method. 4DVarNet replaces the inner step with a learned modulator:

$$\vect{x}^{(i+1)} = \vect{x}^{(i)} - \alpha \Delta\vect{x}^{(i)}, \text{ with } (\Delta\vect{x}^{(i)}, \vect{c}^{(i+1)}) = g_\text{mod}(\nabla_{\vect{x}} \mathcal{J}(\vect{x}^{(i)}), \vect{x}^{(i)}, \vect{c}^{(i)}), \text{ for } i = 0, \ldots, N_\text{iter} - 1$$ (eq-4dvarnet-update)

where $\mathcal{J}$ is the cost [](#eq-4dvarnet-cost), $g_\text{mod}$ is a small neural network (ConvLSTM, MLP, or attention) with parameters $\gvect{\theta}_\text{mod}$, $\vect{c}^{(i)}$ is its recurrent carry, $\alpha > 0$ is the step size on the state, and $N_\text{iter}$ is the number of solver steps. The network sees the current gradient and decides how to step, so it can learn momentum, adaptive step sizes, or problem-specific preconditioning that hand-coded solvers don't capture.
Here $\Delta\vect{x}^{(i)} \in \mathbb{R}^{D_x}$, and the carry $\vect{c}^{(i)}$ holds a hidden and a cell tensor with `hidden_dim` channels per cell.

After $N_\text{iter}$ iterations the analysis is $\vect{x}^\star = \vect{x}^{(N_\text{iter})}$. The whole pipeline (forward, gradient, modulator, update) is differentiable end-to-end.

## Training

The training objective minimises reconstruction error against
ground-truth states:

$$\mathcal{L}(\gvect{\theta}, \gvect{\theta}_\text{mod}) = \mathbb{E}_{(\vect{y}, \vect{x}_\text{true})} \Vert \vect{x}^\star(\gvect{\theta}, \gvect{\theta}_\text{mod}; \vect{y}) - \vect{x}_\text{true} \Vert_2^2$$ (eq-4dvarnet-loss)

where $\vect{x}^\star(\gvect{\theta}, \gvect{\theta}_\text{mod}; \vect{y})$ is the analysis after $N_\text{iter}$ steps of [](#eq-4dvarnet-update) from the observations $\vect{y}$, and $\vect{x}_\text{true} \in \mathbb{R}^{D_x}$ is the ground-truth state. The training gradient $\nabla_{\gvect{\theta}, \gvect{\theta}_\text{mod}} \mathcal{L}$ flows through
the inner solver. This is where the adjoint choice (Decision D15)
matters.

### Adjoint choices for the inner solver

Vardax exposes `solver_adjoint: optimistix.AbstractAdjoint` as a
constructor slot. Three viable choices:

| Adjoint | Memory | Convergence requirement | Notes |
|---|---|---|---|
| `optimistix.RecursiveCheckpointAdjoint()` | $O(N_\text{iter})$ checkpoints | None | Standard backprop with recursive checkpointing |
| `vardax.adjoints.OneStepAdjoint()` | $O(1)$ | None | Bolte et al. 2023; only the last step differentiable |
| `optimistix.ImplicitAdjoint()` | $O(1)$ | Yes | IFT-based at the fixed point; exact at convergence |

**`RecursiveCheckpointAdjoint`** is the default. It backpropagates through all $N_\text{iter}$ unrolled steps, with recursive checkpointing to bound memory, so it always works and costs $O(N_\text{iter})$ checkpoints.

**`OneStepAdjoint`** runs $N_\text{iter} - 1$ steps with `jax.lax.stop_gradient` applied to the trajectory, then a single differentiable step, so the training gradient picks up only the last iteration's contribution. Bolte et al. (2023) prove this is exact at convergence; in practice it works well for converged or near-converged inner solvers. Its memory cost is $O(1)$, which makes it the right choice for $N_\text{iter} \ge 20$ when memory is constrained.

vardax ships `OneStepAdjoint` in `vardax._src.adjoints.one_step` with
the goal of upstreaming to `optimistix` once stable (Decision D6).

**`ImplicitAdjoint`** uses the implicit function theorem at the fixed point $\vect{x}^\star = g_\text{step}(\vect{x}^\star; \gvect{\theta})$ of the inner iteration:

$$\frac{\mathrm{d}\vect{x}^\star}{\mathrm{d}\gvect{\theta}} = \left(\mat{I} - \frac{\partial g_\text{step}}{\partial \vect{x}}\right)^{-1} \frac{\partial g_\text{step}}{\partial \gvect{\theta}}$$ (eq-fixed-point-grad)

where $g_\text{step}: \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ is the inner-step map [](#eq-4dvarnet-update), $\gvect{\theta}$ stands for either set of weights, and both partial derivatives are taken at $\vect{x}^\star$. It is the implicit gradient [](#eq-implicit-grad) of chapter 12 with the stationarity condition replaced by the fixed-point residual $\vect{x} - g_\text{step}(\vect{x}; \gvect{\theta}) = \mathbf{0}$. Memory $O(1)$, gradient exact at
convergence. Requires the inner solver to actually converge; if $N_\text{iter}$ is
too small, the IFT gradient is wrong. Use when the inner iteration is
known to be a contraction.

## Implementation in vardax

The example below trains a small `FourDVarNet2D` on $N_\text{batch} = 4$ windows of 3 times ($T = 2$ steps) on an $H \times W = 8 \times 8$ grid, for a few optimiser steps only.
The truth is a travelling wave $x_\text{true}(n, t, i) = \sin(0.8 i + 0.3 t + \xi_n)$, constant along the columns, with a random phase $\xi_n$ per window $n$ and $i$ the row, about half the cells are observed with mask $\vect{m}$, and the observations are $\vect{m} \odot \vect{x}_\text{true}$.
The model's cost is $\mathcal{J}(\vect{x}) = \mathcal{J}_o(\vect{x}) + \beta_p \mathcal{J}_p(\vect{x})$ with $\mathcal{J}_o = \Vert \vect{m} \odot (\vect{x} - \vect{y}) \Vert_2^2$ and $\mathcal{J}_p = \Vert \vect{x} - g_\text{ae}(\vect{x}; \gvect{\theta}) \Vert_2^2$, which is [](#eq-4dvarnet-cost) with $h(\vect{x}) = \vect{m} \odot \vect{x}$, $\gmat{\Sigma}_\varepsilon = \mat{I}$ and no factor ½, and the solver runs [](#eq-4dvarnet-update) with the modulator output scaled by a step size $\alpha$.
In the solver $\mathcal{J}_o$ and $\mathcal{J}_p$ are sums of squares over all entries; `obs_cost_1d`, `obs_cost_2d`, `prior_cost` and `variational_cost` return the corresponding means instead.
Here $\vect{m}, \vect{y}, \vect{x} \in \mathbb{R}^{D_x}$ with $D_x = (T + 1) H W$, so $D_y = D_x$, and $\beta_p \geq 0$ is the prior weight.

```{code-block} python
:caption: Trains a FourDVarNet2D, whose solver iterates [](#eq-4dvarnet-update) on the cost [](#eq-4dvarnet-cost), by minimising the reconstruction loss [](#eq-4dvarnet-loss) on travelling-wave windows.
:label: code-4dvarnet-training
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import vardax as vdx

# Shapes: N_batch = 4 windows; T + 1 = 3 times; H = 8 rows; W = 8 columns

key_model, key_phase, key_mask = jax.random.split(jax.random.PRNGKey(0), 3)

rows = jnp.arange(8.0)  # (H,)
times = jnp.arange(3.0)  # (T + 1,)
# ξ_n ~ U(0, 2π), one phase per window
phase = jax.random.uniform(key_phase, (4,), maxval=2 * jnp.pi)  # (N_batch,)
# 0.8 i
row_phase = 0.8 * rows  # (H,)
# 0.3 t
time_phase = 0.3 * times  # (T + 1,)
# 0.8 i + 0.3 t + ξ_n
wave_phase = einx.add(  # (N_batch, T + 1, H)
    "y, t, s -> s t y", row_phase, time_phase, phase
)
# x_true(n, t, i) = sin(0.8 i + 0.3 t + ξ_n)
wave = jnp.sin(wave_phase)  # (N_batch, T + 1, H)
# the wave is constant along the columns
target = einx.id(  # (N_batch, T + 1, H) → (N_batch, T + 1, H, W)
    "s t y -> s t y x", wave, x=8
)
# m ~ Bernoulli(0.5)
observed = jax.random.bernoulli(  # (N_batch, T + 1, H, W)
    key_mask, 0.5, (4, 3, 8, 8)
)
mask = observed.astype(jnp.float32)  # (N_batch, T + 1, H, W)
# y = m ⊙ x_true
obs = mask * target  # (N_batch, T + 1, H, W)

# today (vardax)
batch = vdx.Batch2D(input=obs, mask=mask, target=target)
# today (vardax)
# x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾, (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)
# N_iter = 3, with a bilinear autoencoder g_ae
model = vdx.FourDVarNet2D(
    n_time=3,
    height=8,
    width=8,
    latent_dim=4,
    hidden_dim=4,
    n_solver_steps=3,
    alpha=0.2,
    prior_weight=1.0,
    solver_adjoint=vdx.OneStepAdjoint(),
    key=key_model,
)

# today (optax)
optimizer = optax.adam(1e-3)
# the trainable leaves (θ, θ_mod)
params = eqx.filter(model, eqx.is_array)
# today (optax)
opt_state = optimizer.init(params)
losses = []
for _ in range(20):
    # today (vardax)
    # (θ, θ_mod) ← adam step on ℒ(θ, θ_mod) = mean((x⋆ − x_true)²)
    model, opt_state, loss = vdx.train_step(model, batch, optimizer, opt_state)
    losses.append(float(loss))
print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
```

It prints `loss 0.263 -> 0.217`.
`train_step` is already wrapped in `eqx.filter_jit`, so the loop calls it directly.

After training, `model.as_analysis_step()` exposes the operational
interface for `pipekit_cycle.DACycle`.

## The prior family

vardax ships these priors:

| Prior | Form | When |
|---|---|---|
| `BilinAEPrior1D/2D/2DMultivar` | $g_\text{ae}(\vect{x}) = g_\text{dec}(\text{ReLU}(\mat{W}_a \vect{x}) \odot \tanh(\mat{W}_b \vect{x}))$ | Standard 4DVarNet starting point |
| `ConvAEPrior1D` | Periodic 1D convolutional AE | Periodic 1D fields (Lorenz-96) |
| `MLPAEPrior1D` | Dense MLP AE | Low-dimensional states (Lorenz-63) |
| `IdentityPrior` | $g_\text{ae}(\vect{x}) = \vect{x}$ | Pure obs-driven baseline; recovers classical 4DVar with a fixed-step modulator |
| `DynamicalPrior` | $g_\text{ae}(\vect{x}) = f^n(\vect{x})$ via somax / plumax forward | Physics-informed 4DVarNet |

`DynamicalPrior` is interesting: it wraps any
`pipekit_cycle.ForwardModel` as the variational prior, so the regulariser
becomes "consistency with the physical dynamics" rather than
"reconstruction by an autoencoder". The gradient back through the
dynamics uses the `forward_adjoint` slot (chapter 3).

## The gradient modulator family

The modulators, shipped and planned:

| Modulator | Form |
|---|---|
| `ConvLSTMGradMod1D/2D` | ConvLSTM over spatial dims, recurrent over solver iterations |
| `MLPGradMod` | Dense MLP, dimension-agnostic via flatten (planned) |
| `AttentionGradMod` | Self-attention over spatial axis (planned) |
| `IdentityGradMod` | $\Delta\vect{x} = \nabla_{\vect{x}} \mathcal{J}$, classical fixed-step descent with step $\alpha$ (planned) |

Notebook [11](notebooks/11_gradient_learning_L63.ipynb) trains a `ConvLSTMGradMod1D` on its own, with the prior frozen, and compares it with the fixed-step baseline of the next section across solver-step budgets.

## Linear-Gaussian agreement

`IdentityPrior` with a fixed-step modulator and many solver steps is the classical baseline, useful for checking agreement with the linear-Gaussian limit (Decision D14 invariant).
With `IdentityPrior` the prior cost $\Vert \vect{x} - g_\text{ae}(\vect{x}) \Vert_2^2$ is zero, and a modulator that passes the gradient through unchanged turns the solver into fixed-step gradient descent on $\mathcal{J}(\vect{x}) = \Vert \vect{m} \odot (\vect{x} - \vect{y}) \Vert_2^2$.
No such modulator ships, so the block defines one and hands it to `solve_4dvarnet_1d`; with step $\alpha = 0.05$ and $N_\text{iter} = 200$ steps it converges to $\vect{m} \odot \vect{y}$.
`OptimalInterpolation` with $\gvect{\mu}_x = \mathbf{0}$, $\gmat{\Sigma}_x = \mat{I}$ and $\gmat{\Sigma}_\varepsilon = \sigma^2 \mat{I}$ gives the same field in the limit $\sigma \to 0$; here $\sigma = 0.01$ on $N_{\Omega_x} = 8$ cells, about 60% of them observed.
All of these are in $\mathbb{R}^{D_x}$ with $D_x = N_{\Omega_x}$, since the window holds a single time ($T = 0$).
The block continues the one above.

```{code-block} python
:caption: Runs [](#eq-4dvarnet-update) with g_mod(∇𝒥, x, c) = (∇𝒥, c) and an identity prior, and checks it against the BLUE mean [](#eq-blue-mean).
:label: code-4dvarnet-oi-agreement
from typing import Any

import lineax as lx
from jaxtyping import Array, Float

# Shapes: N_batch = 1 window; T + 1 = 1 time; N_Ωx = 8 grid cells; D_x = N_Ωx

key_truth, key_cells = jax.random.split(jax.random.PRNGKey(1))
state_structure = jax.ShapeDtypeStruct((1, 8), jnp.float32)

# Σ_x = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# Σ_ε = σ² I, σ = 0.01; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (D_x, D_x) PSD op
    jnp.float32(1e-4) * prior_cov, lx.positive_semidefinite_tag
)
# x_true ~ 𝒩(0, I)
truth = jax.random.normal(key_truth, (1, 1, 8))  # (N_batch, T + 1, N_Ωx)
# m ~ Bernoulli(0.6)
observed_cells = jax.random.bernoulli(  # (N_batch, T + 1, N_Ωx)
    key_cells, 0.6, (1, 1, 8)
)
cell_mask = observed_cells.astype(jnp.float32)  # (N_batch, T + 1, N_Ωx)
# y = m ⊙ x_true
line_obs = cell_mask * truth  # (N_batch, T + 1, N_Ωx)
# today (vardax)
line_batch = vdx.Batch1D(input=line_obs, mask=cell_mask)


# user
class FixedStepGradMod(eqx.Module):
    """Pass the gradient through unchanged, so the solver takes fixed steps."""

    def __call__(
        self, grad: Float[Array, "s t n"], state: Float[Array, "s t n"], carry: Any
    ) -> tuple[Float[Array, "s t n"], Any]:
        # (Δx, c′) = (∇ₓ𝒥, c)
        return grad, carry


# today (vardax)
# g_ae(x) = x, so 𝒥_p = ‖x − g_ae(x)‖² = 0
identity_prior = vdx.IdentityPrior()
# user
grad_mod = FixedStepGradMod()
# today (vardax)
# x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α ∇ₓ𝒥(x⁽ⁱ⁾), with 𝒥(x) = ‖m ⊙ (x − y)‖²
descent = vdx.solve_4dvarnet_1d(  # → (N_batch, T + 1, N_Ωx)
    line_batch,
    identity_prior,
    grad_mod,
    n_steps=200,
    hidden_dim=1,
    alpha=0.05,
)
# today (vardax)
# p(x) = 𝒩(μ_x, Σ_x), p(y | x) = 𝒩(m ⊙ x, Σ_ε)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 8)),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# today (vardax)
# μ_x|y = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x)
closed_form = optimal_interpolation(line_batch)  # → (N_batch, T + 1, N_Ωx)
print(bool(jnp.allclose(descent, closed_form, atol=1e-2)))
```

It prints `True`.

This is the floor. A trained `FourDVarNet` with a non-trivial prior and grad modulator typically does better in nonlinear regimes, but the floor confirms the implementation is correct.

## When to use `FourDVarNet`

Use `FourDVarNet` when:

- You have a training set of (observation, ground-truth) pairs
- The classical $\gmat{\Sigma}_x$ is hard to specify or known to be inadequate
- The regime is data-rich and learning-friendly (smooth manifolds, no
  catastrophic distribution shift between train and test)

Don't use `FourDVarNet` when:

- You have no training pairs; go with classical methods (chapters 4 to 8)
- The training distribution doesn't match deployment, since `FourDVarNet` extrapolates poorly outside its training manifold
- You need defensible posterior covariance, since the Laplace approximation [](#eq-laplace-cov) around a learned-prior MAP can be miscalibrated; `IncrementalFourDVar` with a classical Matérn prior gives a more honest posterior

## Posterior

`LaplaceCovariance` builds its Hessian from the model's `obs_op`, $\gmat{\Sigma}_x$ and $\gmat{\Sigma}_\varepsilon$, which `FourDVarNet2D` does not have, so it raises `AttributeError` on this model.
The Laplace approximation of the learned cost is assembled by hand instead: with the stacked residual $\vect{r}(\vect{x}) = (\vect{m} \odot (\vect{x} - \vect{y}), \sqrt{\beta_p} (\vect{x} - g_\text{ae}(\vect{x}; \gvect{\theta}))) \in \mathbb{R}^{2 D_x}$ and its Jacobian $\mat{J}_r \in \mathbb{R}^{2 D_x \times D_x}$ at the analysis $\vect{x}^\star$, the Gauss-Newton Hessian is $\nabla^2_{\vect{x}} \mathcal{J} \approx 2 \mat{J}_r^\top \mat{J}_r$ and the Laplace covariance [](#eq-laplace-cov) is $\gmat{\Sigma}_{x \mid y} = (\nabla^2_{\vect{x}} \mathcal{J})^{-1}$, applied by a CG solve.
The block continues the first one on this page, for its first window, with $\beta_p = 1$ and $D_x = (T + 1) H W$.

```{code-block} python
:caption: Applies the Laplace covariance [](#eq-laplace-cov) of the learned cost [](#eq-4dvarnet-cost), with a Gauss-Newton Hessian built by hand, to one field.
:label: code-4dvarnet-laplace
# Shapes: N_batch = 1 window; T + 1 = 3 times; H = 8 rows; W = 8 columns

first_obs = obs[:1]  # (N_batch, T + 1, H, W)
first_mask = mask[:1]  # (N_batch, T + 1, H, W)
# today (vardax)
one_window = vdx.Batch2D(input=first_obs, mask=first_mask)
# today (vardax)
# x⋆ = x^(N_iter)
analysis = model(one_window)  # → (N_batch, T + 1, H, W)


# user
def residuals(
    state: Float[Array, "s t y x"],
) -> tuple[Float[Array, "s t y x"], Float[Array, "s t y x"]]:
    """The residuals r(x) = (m ⊙ (x − y), √β_p (x − g_ae(x; θ))), β_p = 1."""
    # x − y
    misfit = state - one_window.input  # (N_batch, T + 1, H, W)
    # r_o = m ⊙ (x − y)
    obs_residual = one_window.mask * misfit  # (N_batch, T + 1, H, W)
    # today (vardax)
    # x̂ = g_ae(x; θ)
    reconstruction = model.prior(state)  # (N_batch, T + 1, H, W)
    # r_p = √β_p (x − x̂), β_p = 1
    prior_residual = state - reconstruction  # (N_batch, T + 1, H, W)
    return obs_residual, prior_residual


# δx ↦ J_r δx at x⋆
_, residual_jvp = jax.linearize(residuals, analysis)
# δr ↦ J_rᵀ δr
residual_vjp = jax.linear_transpose(residual_jvp, analysis)


# user
def precision_mv(direction: Float[Array, "s t y x"]) -> Float[Array, "s t y x"]:
    """The Gauss-Newton Hessian product ∇²ₓ𝒥 δx ≈ 2 J_rᵀ J_r δx."""
    # J_r δx
    residual_direction = residual_jvp(direction)
    # J_rᵀ J_r δx
    (pulled_back,) = residual_vjp(residual_direction)  # (N_batch, T + 1, H, W)
    # ∇²ₓ𝒥 δx ≈ 2 J_rᵀ J_r δx
    return 2.0 * pulled_back  # (N_batch, T + 1, H, W)


analysis_structure = jax.ShapeDtypeStruct(analysis.shape, analysis.dtype)
# Λ_x|y = ∇²ₓ𝒥 ≈ 2 J_rᵀ J_r
posterior_precision = lx.FunctionLinearOperator(  # (D_x, D_x) PSD op
    precision_mv, analysis_structure, lx.positive_semidefinite_tag
)
# 1
ones = jnp.ones_like(analysis)  # (N_batch, T + 1, H, W)
precision_solver = lx.CG(rtol=1e-5, atol=1e-5, max_steps=500)
# today (lineax)
# Σ_x|y 1 = (∇²ₓ𝒥)⁻¹ 1, one matrix-free solve
spread = lx.linear_solve(  # (N_batch, T + 1, H, W) → (N_batch, T + 1, H, W)
    posterior_precision, ones, solver=precision_solver
)
print(spread.value.shape, int(spread.stats["num_steps"]))
```

It prints `(1, 3, 8, 8) 7`.

The Hessian uses the learned prior's curvature, which may be poorly conditioned away from the training manifold. Validate the posterior via the six-step cycle gates of chapter 14 (the posterior gate [](#eq-gate-posterior)): does the posterior diagonal agree with the spread of an ensemble of analyses from perturbed inputs?

## A note on history

4DVarNet originated as a research line (Fablet et al. 2021, 2023) that
demonstrated end-to-end learning of variational DA on satellite SSH.
The original codebase (CIA-Oceanix/4dvarnet-starter) is PyTorch and
ocean-specific. Vardax brings the same algorithm into JAX, peers it
with the classical DA hierarchy, and connects it to the broader
ecosystem (somax / plumax for physics, gaussx for structured priors,
filterax for hybrid EnVar, pipekit-cycle for orchestration).

In the v0.4 design the classical methods are the foundation, not legacy, and 4DVarNet earns its place beside them as a useful learned variant.

## See also

- Chapter 6: `StrongFourDVar` (classical counterpart)
- Chapter 8: `IncrementalFourDVar` (operational baseline)
- Chapter 12: adjoint composition (the inner-solver adjoint choice)
- Chapter 14: six-step cycle (validation methodology for learned methods)
- Design doc: [`design/decisions.md#d14`](design/decisions.md#d14-da-hierarchy-as-horizontal-peer-classes)

## References

- Fablet, R., Chapron, B., Drumetz, L., Mémin, E., Pannekoucke, O., &
  Rousseau, F. (2021). *Learning variational data assimilation models
  and solvers.* JAMES 13(10).
- Fablet, R., Febvre, Q., & Chapron, B. (2023). *Multimodal 4DVarNets
  for the reconstruction of sea surface dynamics from NADIR and
  wide-swath altimetry.* IEEE TGRS 61.
- Bolte, J., Pauwels, E., & Vaiter, S. (2023). *One-step differentiation
  of iterative algorithms.* NeurIPS.
