---
numbering:
  enumerator: "9.%s"
---

# 4DVarNet (learned 4DVar)

4DVarNet replaces two pieces of classical 4DVar with learned
counterparts: the Gaussian prior $\|x - x_b\|^2_{B^{-1}}$ becomes a
learned reconstruction $\|x - \varphi_\theta(x)\|^2$, and the inner
gradient-descent solver becomes a learned iteration with a ConvLSTM
gradient modulator. The result is trainable end-to-end on
(input, target) pairs.

The "Net" naming follows Fablet et al. (2021). vardax keeps it as the name of the *method*, not the family: 4DVarNet is one of seven peer analysis methods (Decision D14), a learned variant of strong-constraint 4DVar distinguished by its learned components, and not the parent of classical 4DVar.

## Cost function

The variational cost is structurally familiar:

$$J(x) = \alpha_\text{obs} \frac{1}{2} \|H(x) - y\|^2_{R^{-1}} + \alpha_\text{prior} \|x - \varphi_\theta(x)\|^2$$ (eq-4dvarnet-cost)

where the first term is the observation cost [](#eq-obs-cost), $\varphi_\theta$ is a learned autoencoder with parameters $\theta$, and $\alpha_\text{obs}$ and $\alpha_\text{prior}$ weight the two terms. The Gaussian background
term $\|x - x_b\|^2_{B^{-1}}$ is **replaced** (not added). The learned
prior is expressive enough to encode the same information that $B$ +
$x_b$ would, plus structure that classical Gaussian priors can't
represent (multimodality, nonlinear manifolds, conditional
dependencies).

## The learned inner solver

Classical 4DVar minimises $J$ via gradient descent or a quasi-Newton
method. 4DVarNet replaces the inner step with a learned modulator:

$$x_{k+1} = x_k - \Phi_\phi(\nabla_x J(x_k), h_k), \text{ for } k = 0, \ldots, K-1$$ (eq-4dvarnet-update)

where $J$ is the cost [](#eq-4dvarnet-cost), $\Phi_\phi$ is a small neural network (ConvLSTM, MLP, or attention) with parameters $\phi$, and $h_k$ is its recurrent hidden state. The network sees the current gradient and decides how to step, so it can learn momentum, adaptive step sizes, or problem-specific preconditioning that hand-coded solvers don't capture.

After $K$ iterations the analysis is $x^* = x_K$. The whole pipeline (forward, gradient, modulator, update) is differentiable end-to-end.

## Training

The training objective minimises reconstruction error against
ground-truth states:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{(y, x_\text{true})} \|x^*(\theta, \phi; y) - x_\text{true}\|^2$$ (eq-4dvarnet-loss)

where $x^*(\theta, \phi; y)$ is the analysis after $K$ steps of [](#eq-4dvarnet-update) from the observations $y$, and $x_\text{true}$ is the ground-truth state. The training gradient $\nabla_{\theta, \phi} \mathcal{L}$ flows through
the inner solver. This is where the adjoint choice (Decision D15)
matters.

### Adjoint choices for the inner solver

Vardax exposes `solver_adjoint: optimistix.AbstractAdjoint` as a
constructor slot. Three viable choices:

| Adjoint | Memory | Convergence requirement | Notes |
|---|---|---|---|
| `optimistix.RecursiveCheckpointAdjoint()` | $O(K)$ checkpoints | None | Standard backprop with recursive checkpointing |
| `vardax.adjoints.OneStepAdjoint()` | $O(1)$ | None | Bolte et al. 2023; only the last step differentiable |
| `optimistix.ImplicitAdjoint()` | $O(1)$ | Yes | IFT-based at the fixed point; exact at convergence |

**`RecursiveCheckpointAdjoint`** is the default. It backpropagates through all $K$ unrolled steps, with recursive checkpointing to bound memory, so it always works and costs $O(K)$ checkpoints.

**`OneStepAdjoint`** runs $K-1$ steps with `jax.lax.stop_gradient` applied to the trajectory, then a single differentiable step, so the training gradient picks up only the last iteration's contribution. Bolte et al. (2023) prove this is exact at convergence; in practice it works well for converged or near-converged inner solvers. Its memory cost is $O(1)$, which makes it the right choice for $K \ge 20$ when memory is constrained.

vardax ships `OneStepAdjoint` in `vardax._src.adjoints.one_step` with
the goal of upstreaming to `optimistix` once stable (Decision D6).

**`ImplicitAdjoint`** uses the implicit function theorem at the fixed point $x^* = f(x^*; \theta)$ of the inner iteration:

$$\frac{d x^*}{d \theta} = \left(I - \frac{\partial f}{\partial x}\right)^{-1} \frac{\partial f}{\partial \theta}$$ (eq-fixed-point-grad)

where $f$ is the inner-step map [](#eq-4dvarnet-update) and both partial derivatives are taken at $x^*$. It is the implicit gradient [](#eq-implicit-grad) of chapter 12 with the stationarity condition replaced by the fixed-point residual $x - f(x; \theta) = 0$. Memory $O(1)$, gradient exact at
convergence. Requires the inner solver to actually converge; if $K$ is
too small, the IFT gradient is wrong. Use when the inner iteration is
known to be a contraction.

## Implementation in vardax

The example below trains a small `FourDVarNet2D` on $S = 4$ windows of $T = 3$ frames on an $8 \times 8$ grid, for a few optimiser steps only.
The truth is a travelling wave $x_\text{true}(s, t, y, x) = \sin(0.8 y + 0.3 t + \varphi_s)$ with a random phase $\varphi_s$ per window, about half the cells are observed with mask $m$, and the observations are $m \odot x_\text{true}$.
The model's cost is $J(x) = \|m \odot (x - y)\|^2 + \lambda \|x - \varphi_\theta(x)\|^2$, which is [](#eq-4dvarnet-cost) with $H(x) = m \odot x$, $R = I$ and no factor ½, and the solver runs [](#eq-4dvarnet-update) with the modulator output scaled by a step size $\alpha$.

```{code-block} python
:caption: Trains a FourDVarNet2D, whose solver iterates [](#eq-4dvarnet-update) on the cost [](#eq-4dvarnet-cost), by minimising the reconstruction loss [](#eq-4dvarnet-loss) on travelling-wave windows.
:label: code-4dvarnet-training
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import vardax as vdx

# Shapes: S = 4 samples; T = 3 times; Y = 8 rows; X = 8 columns

key_model, key_phase, key_mask = jax.random.split(jax.random.PRNGKey(0), 3)

rows = jnp.arange(8.0)  # (Y,)
times = jnp.arange(3.0)  # (T,)
# φₛ ~ U(0, 2π), one phase per window
phase = jax.random.uniform(key_phase, (4,), maxval=2 * jnp.pi)  # (S,)
# 0.8 y
row_phase = 0.8 * rows  # (Y,)
# 0.3 t
time_phase = 0.3 * times  # (T,)
# 0.8 y + 0.3 t + φₛ
wave_phase = einx.add("y, t, s -> s t y", row_phase, time_phase, phase)  # (S, T, Y)
# x_true(s, t, y) = sin(0.8 y + 0.3 t + φₛ)
wave = jnp.sin(wave_phase)  # (S, T, Y)
# the wave is constant along x
target = einx.id("s t y -> s t y x", wave, x=8)  # (S, T, Y) → (S, T, Y, X)
# m ~ Bernoulli(0.5)
observed = jax.random.bernoulli(key_mask, 0.5, (4, 3, 8, 8))  # (S, T, Y, X)
mask = observed.astype(jnp.float32)  # (S, T, Y, X)
# y = m ⊙ x_true
obs = mask * target  # (S, T, Y, X)

# today (vardax)
batch = vdx.Batch2D(input=obs, mask=mask, target=target)
# today (vardax)
# x_{k+1} = x_k − α Φ_ϕ(∇J(x_k), h_k), K = 3, with a bilinear autoencoder φ_θ
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
# the trainable leaves (θ, ϕ)
params = eqx.filter(model, eqx.is_array)
# today (optax)
opt_state = optimizer.init(params)
losses = []
for _ in range(20):
    # today (vardax)
    # (θ, ϕ) ← adam step on ℒ(θ, ϕ) = mean (x*(θ, ϕ; y) − x_true)²
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
| `BilinAEPrior1D/2D/2DMultivar` | $\varphi(x) = \text{decode}(\text{ReLU}(Ax) \odot \tanh(Bx))$ | Standard 4DVarNet starting point |
| `ConvAEPrior1D` | Periodic 1D convolutional AE | Periodic 1D fields (Lorenz-96) |
| `MLPAEPrior1D` | Dense MLP AE | Low-dimensional states (Lorenz-63) |
| `IdentityPrior` | $\varphi(x) = x$ | Pure obs-driven baseline; recovers classical 4DVar with a fixed-step modulator |
| `DynamicalPrior` | $\varphi(x) = M^n(x)$ via somax / plumax forward | Physics-informed 4DVarNet |

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
| `IdentityGradMod` | $\text{update} = -\alpha \cdot \text{grad}$, classical fixed-step descent (planned) |

Notebook [11](notebooks/11_gradient_learning_L63.ipynb) trains a `ConvLSTMGradMod1D` on its own, with the prior frozen, and compares it with the fixed-step baseline of the next section across solver-step budgets.

## Linear-Gaussian agreement

`IdentityPrior` with a fixed-step modulator and many solver steps is the classical baseline, useful for checking agreement with the linear-Gaussian limit (Decision D14 invariant).
With `IdentityPrior` the prior cost $\|x - \varphi(x)\|^2$ is zero, and a modulator that passes the gradient through unchanged turns the solver into fixed-step gradient descent on $J(x) = \|m \odot (x - y)\|^2$.
No such modulator ships, so the block defines one and hands it to `solve_4dvarnet_1d`; with step $\alpha = 0.05$ and $K = 200$ steps it converges to $m \odot y$.
`OptimalInterpolation` with $x_b = 0$, $B = I$ and $R = \sigma^2 I$ gives the same field in the limit $\sigma \to 0$; here $\sigma = 0.01$ on $N = 8$ cells, about 60% of them observed.
The block continues the one above.

```{code-block} python
:caption: Runs [](#eq-4dvarnet-update) with Φ(∇J, h) = ∇J and an identity prior, and checks it against the BLUE mean [](#eq-blue-mean).
:label: code-4dvarnet-oi-agreement
from typing import Any

import lineax as lx
from jaxtyping import Array, Float

# Shapes: S = 1 sample; T = 1 time; N = 8 grid cells

key_truth, key_cells = jax.random.split(jax.random.PRNGKey(1))
state_structure = jax.ShapeDtypeStruct((1, 8), jnp.float32)

# B = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (T N, T N) op
# R = σ² I, σ = 0.01; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (T N, T N) PSD op
    jnp.float32(1e-4) * prior_cov, lx.positive_semidefinite_tag
)
# x_true ~ 𝒩(0, I)
truth = jax.random.normal(key_truth, (1, 1, 8))  # (S, T, N)
# m ~ Bernoulli(0.6)
observed_cells = jax.random.bernoulli(key_cells, 0.6, (1, 1, 8))  # (S, T, N)
cell_mask = observed_cells.astype(jnp.float32)  # (S, T, N)
# y = m ⊙ x_true
line_obs = cell_mask * truth  # (S, T, N)
# today (vardax)
line_batch = vdx.Batch1D(input=line_obs, mask=cell_mask)


# user
class FixedStepGradMod(eqx.Module):
    """Pass the gradient through unchanged, so the solver takes fixed steps."""

    def __call__(
        self, grad: Float[Array, "s t n"], state: Float[Array, "s t n"], carry: Any
    ) -> tuple[Float[Array, "s t n"], Any]:
        # Φ(∇J, h) = ∇J
        return grad, carry


# today (vardax)
# φ(x) = x, so ‖x − φ(x)‖² = 0
identity_prior = vdx.IdentityPrior()
# user
grad_mod = FixedStepGradMod()
# today (vardax)
# x_{k+1} = x_k − α ∇J(x_k), with J(x) = ‖m ⊙ (x − y)‖²
descent = vdx.solve_4dvarnet_1d(  # → (S, T, N)
    line_batch,
    identity_prior,
    grad_mod,
    n_steps=200,
    hidden_dim=1,
    alpha=0.05,
)
# today (vardax)
# p(x) = 𝒩(0, B), p(y | x) = 𝒩(m ⊙ x, R)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 8)),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# today (vardax)
# x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
closed_form = optimal_interpolation(line_batch)  # → (S, T, N)
print(bool(jnp.allclose(descent, closed_form, atol=1e-2)))
```

It prints `True`.

This is the floor. A trained `FourDVarNet` with a non-trivial prior and grad modulator typically does better in nonlinear regimes, but the floor confirms the implementation is correct.

## When to use `FourDVarNet`

Use `FourDVarNet` when:

- You have a training set of (observation, ground-truth) pairs
- The classical $B$ is hard to specify or known to be inadequate
- The regime is data-rich and learning-friendly (smooth manifolds, no
  catastrophic distribution shift between train and test)

Don't use `FourDVarNet` when:

- You have no training pairs; go with classical methods (chapters 4 to 8)
- The training distribution doesn't match deployment, since `FourDVarNet` extrapolates poorly outside its training manifold
- You need defensible posterior covariance, since the Laplace approximation [](#eq-laplace-cov) around a learned-prior MAP can be miscalibrated; `IncrementalFourDVar` with a classical Matérn prior gives a more honest posterior

## Posterior

`LaplaceCovariance` builds its Hessian from the model's `obs_op`, $B$ and $R$, which `FourDVarNet2D` does not have, so it raises `AttributeError` on this model.
The Laplace approximation of the learned cost is assembled by hand instead: with residuals $r(x) = (m \odot (x - y), \sqrt{\lambda} (x - \varphi_\theta(x)))$ and Jacobian $J_r$ at the analysis $x^*$, the Gauss-Newton Hessian is $\nabla^2 J \approx 2 J_r^\top J_r$ and the Laplace covariance [](#eq-laplace-cov) is $P^* = (\nabla^2 J)^{-1}$, applied by a CG solve.
The block continues the first one on this page, for its first window, with $\lambda = 1$.

```{code-block} python
:caption: Applies the Laplace covariance [](#eq-laplace-cov) of the learned cost [](#eq-4dvarnet-cost), with a Gauss-Newton Hessian built by hand, to one field.
:label: code-4dvarnet-laplace
# Shapes: S = 1 sample; T = 3 times; Y = 8 rows; X = 8 columns

first_obs = obs[:1]  # (S, T, Y, X)
first_mask = mask[:1]  # (S, T, Y, X)
# today (vardax)
one_window = vdx.Batch2D(input=first_obs, mask=first_mask)
# today (vardax)
# x* = x_K
analysis = model(one_window)  # → (S, T, Y, X)


# user
def residuals(
    state: Float[Array, "s t y x"],
) -> tuple[Float[Array, "s t y x"], Float[Array, "s t y x"]]:
    """The residuals r(x) = (m ⊙ (x − y), √λ (x − φ_θ(x))), λ = 1."""
    # x − y
    misfit = state - one_window.input  # (S, T, Y, X)
    # r_o = m ⊙ (x − y)
    obs_residual = one_window.mask * misfit  # (S, T, Y, X)
    # today (vardax)
    # φ_θ(x)
    reconstruction = model.prior(state)  # (S, T, Y, X)
    # r_p = √λ (x − φ_θ(x)), λ = 1
    prior_residual = state - reconstruction  # (S, T, Y, X)
    return obs_residual, prior_residual


# δx ↦ J_r δx at x*
_, residual_jvp = jax.linearize(residuals, analysis)
# δr ↦ J_rᵀ δr
residual_vjp = jax.linear_transpose(residual_jvp, analysis)


# user
def precision_mv(direction: Float[Array, "s t y x"]) -> Float[Array, "s t y x"]:
    """The Gauss-Newton Hessian product ∇²J δx ≈ 2 J_rᵀ J_r δx."""
    # J_r δx
    residual_direction = residual_jvp(direction)
    # J_rᵀ J_r δx
    (pulled_back,) = residual_vjp(residual_direction)  # (S, T, Y, X)
    # ∇²J δx ≈ 2 J_rᵀ J_r δx
    return 2.0 * pulled_back  # (S, T, Y, X)


analysis_structure = jax.ShapeDtypeStruct(analysis.shape, analysis.dtype)
# ∇²J ≈ 2 J_rᵀ J_r
posterior_precision = lx.FunctionLinearOperator(  # (S T Y X, S T Y X) PSD op
    precision_mv, analysis_structure, lx.positive_semidefinite_tag
)
# 1
ones = jnp.ones_like(analysis)  # (S, T, Y, X)
precision_solver = lx.CG(rtol=1e-5, atol=1e-5, max_steps=500)
# today (lineax)
# P* 1 = (∇²J)⁻¹ 1, one matrix-free solve
spread = lx.linear_solve(  # (S, T, Y, X) → (S, T, Y, X)
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
