---
numbering:
  enumerator: "13.%s"
---

# Posterior Covariance

A MAP estimate $\vect{x}^\star$ is the start of the answer, not the end. Real DA needs a posterior (mean, covariance and provenance) for downstream consumers (population models, decision pipelines, alerting systems) to propagate uncertainty correctly. This chapter covers the four posterior-covariance computations vardax supports and when to use which; the method chapters point here for that choice.

## The `Posterior` container

Every analysis emits a `Posterior(mean, cov, samples, provenance)`, plus the pieces of a linear control when it has one:

```{code-block} python
:caption: Lists the fields of `vdx.Posterior`, which carries the mean μ_x|y and covariance Σ_x|y of [](#eq-oi-posterior).
:label: code-postcov-container
from typing import Any

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: N_ℰ ensemble members; "..." is the state's own shape


# today (vardax)
# the fields of vdx.Posterior
class Posterior(eqx.Module):
    mean: Float[Array, "..."]
    # Σ_x|y as a lazy gaussx / lineax operator; None for sampling-only heads
    cov: lx.AbstractLinearOperator | None = None
    samples: Float[Array, "member ..."] | None = None
    # forward_model_id, n_iter, J_star, ...
    provenance: dict[str, Any] = eqx.field(default_factory=dict)
    # Φ and Σ_z|y of a linear control x = x⋆ + Φ z, so that Σ_x|y = Φ Σ_z|y Φᵀ
    basis: lx.AbstractLinearOperator | None = None
    coefficient_cov: lx.AbstractLinearOperator | None = None

    # x⁽ᵉ⁾ = μ_x|y + Σ_x|y^{1/2} ξ⁽ᵉ⁾, ξ⁽ᵉ⁾ ~ 𝒩(0, I)
    def sample(
        self, key: PRNGKeyArray, n_samples: int
    ) -> Float[Array, "n_samples ..."]: ...
```

The shapes are $\gvect{\mu}_{x \mid y} \in \mathbb{R}^{D_x}$ (the state's own shape, flattened), $\gmat{\Sigma}_{x \mid y} \in \mathbb{R}^{D_x \times D_x}$, $\vect{x}^{(e)} \in \mathbb{R}^{D_x}$ for the $N_\mathcal{E}$ members $e \in \mathcal{E}$, and, for a linear control with $D_z$ coefficients, $\gmat{\Phi} \in \mathbb{R}^{D_x \times D_z}$ and $\gmat{\Sigma}_{z \mid y} \in \mathbb{R}^{D_z \times D_z}$.
The covariance is an `AbstractLinearOperator` and is never materialised. The mat-vec $\gmat{\Sigma}_{x \mid y} \vect{v}$ and the diagonal $\op{diag}(\gmat{\Sigma}_{x \mid y})$ are computed on demand without building the dense $\gmat{\Sigma}_{x \mid y}$.

`posterior.sample(key, n_samples)` draws $\vect{x}^{(e)} = \gvect{\mu}_{x \mid y} + \gmat{\Sigma}_{x \mid y}^{1/2} \gvect{\xi}^{(e)}$ with $\gvect{\xi}^{(e)} \sim \mathcal{N}(\mathbf{0}, \mat{I})$. The draws come from `gaussx.sample_mvn`, which picks the square root by the structure of $\gmat{\Sigma}_{x \mid y}$ (an FFT circulant embedding for `Toeplitz`, a block-banded Cholesky for `BlockTriDiag`, closed forms for other structured operators) and falls back to the symmetric square root, so a singular $\gmat{\Sigma}_{x \mid y}$ still has finite gradients. When the posterior came from a linear control (a `LinearParameterization`, `ReducedBasisFourDVar(basis=...)` or `IncrementalFourDVar`'s whitened control), it draws in coefficient space instead, $\vect{x}^{(e)} = \gvect{\mu}_{x \mid y} + \gmat{\Phi} \gmat{\Sigma}_{z \mid y}^{1/2} \gvect{\xi}^{(e)}$, which only factors the small $\gmat{\Sigma}_{z \mid y}$. A posterior that already holds `samples` returns a random subset of them.

`OptimalInterpolation` and `IncrementalFourDVar` produce posteriors as
part of the analysis call (closed-form / GN-Hessian reuse). The other
five methods pair with an explicit `PosteriorAdapter`.

## Four posterior families

### 1. Closed form (BLUE / OI)

For `OptimalInterpolation`, the posterior is exact: $\gmat{\Sigma}_{x \mid y}$ is the BLUE covariance [](#eq-blue-cov).

The examples on this page share one toy problem. The state $\vect{x}_t \in \mathbb{R}^{D_x}$ has $D_x = 6$ cells on a ring, and the forward model averages each cell with its right neighbour, $\vect{x}_t = \mat{F} \vect{x}_{t-1}$ with $\mat{F} = \frac{1}{2}(\mat{I} + \mat{P})$ and $\mat{P}$ the cyclic shift. The background is $\gvect{\mu}_x = \mathbf{0}$ with $\gmat{\Sigma}_x = \mat{I}$, a window has $T = 2$ steps, so $T + 1 = 3$ times, only cells 0 to 2 are observed (mask $\vect{m}_t \in \{0, 1\}^{D_y}$ with $D_y = D_x$), and $\vect{y}_t = \vect{x}_t + \sigma \gvect{\varepsilon}_t$ with $\gvect{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mat{I})$, $\gmat{\Sigma}_\varepsilon = \sigma^2 \mat{I}$ and $\sigma = 0.5$. OI analyses the first time of the window on its own, with $\mat{H} = \mat{I}$ on the observed cells. The block continues the one above.

```{code-block} python
:caption: Builds the toy ring problem and reads the diagonal of the OI posterior covariance [](#eq-blue-cov).
:label: code-postcov-oi
import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import vardax as vdx

# Shapes: S = 1 window; T = 2 steps, so T + 1 = 3 times; D_x = D_y = 6 cells;
# N_ℰ = 8 members

key = jax.random.PRNGKey(0)
key_truth, key_noise, key_members = jax.random.split(key, 3)

# P, the cyclic shift (P x)ᵢ = xᵢ₊₁
cyclic_shift = jnp.eye(6, k=1) + jnp.eye(6, k=-5)  # (D_x, D_x)
# F = ½ (I + P)
coupling = 0.5 * (jnp.eye(6) + cyclic_shift)  # (D_x, D_x)


# user
class RingAverage(eqx.Module):
    """xₜ = F xₜ₋₁: each cell averaged with its right neighbour."""

    coupling: Float[Array, "n n"]
    dt: float = eqx.field(static=True, default=1.0)

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # xₜ = F xₜ₋₁
        return einx.dot("i j, j -> i", self.coupling, state)


forward = RingAverage(coupling)
# x₀ ~ 𝒩(0, I)
truth = jax.random.normal(key_truth, (6,))  # (D_x,)
# x₁ = F x₀
truth_1 = forward.step(truth, 1.0)  # (D_x,)
# x₂ = F x₁
truth_2 = forward.step(truth_1, 1.0)  # (D_x,)
truth_trajectory = jnp.stack([truth, truth_1, truth_2])  # (T+1, D_x)
cell_index = jnp.arange(6)  # (D_x,)
# cells 0, 1 and 2 observed
is_observed = cell_index < 3  # (D_x,) bool
observed_cells = is_observed.astype(jnp.float32)  # (D_x,)
# mₜ: the same cells at every time
mask = einx.id("n -> t n", observed_cells, t=3)  # (D_x,) → (T+1, D_y)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (3, 6))  # (T+1, D_y)
# yₜ = xₜ + σ εₜ
window_obs = truth_trajectory + 0.5 * obs_noise  # (T+1, D_y)
window_input = einx.id(  # (T+1, D_y) → (S, T+1, D_y)
    "t n -> 1 t n", window_obs
)
window_mask = einx.id("t n -> 1 t n", mask)  # (T+1, D_y) → (S, T+1, D_y)
# today (vardax)
window = vdx.Batch1D(input=window_input, mask=window_mask)

state_structure = jax.ShapeDtypeStruct((6,), jnp.float32)
# Σ_x = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# Σ_ε = σ² I
scaled_identity = jnp.float32(0.25) * prior_cov  # (D_y, D_y) op
# tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (D_y, D_y) op
    scaled_identity, lx.positive_semidefinite_tag
)

# the same Σ_x and Σ_ε on a single time, shape (1, D_x)
snapshot_structure = jax.ShapeDtypeStruct((1, 6), jnp.float32)
# Σ_x = I
snapshot_prior_cov = lx.IdentityLinearOperator(snapshot_structure)  # (D_x, D_x) op
# Σ_ε = σ² I
snapshot_scaled_identity = jnp.float32(0.25) * snapshot_prior_cov  # (D_y, D_y) op
snapshot_obs_cov = lx.TaggedLinearOperator(  # (D_y, D_y) op
    snapshot_scaled_identity, lx.positive_semidefinite_tag
)
# today (vardax)
snapshot = vdx.Batch1D(
    input=window.input[:, :1],  # (S, T+1, D_y) → (S, 1, D_y)
    mask=window.mask[:, :1],
)
# today (vardax)
oi = vdx.OptimalInterpolation(
    vdx.MaskedIdentity(),
    jnp.zeros((1, 6)),
    snapshot_prior_cov,
    snapshot_obs_cov,
)
# today (vardax)
# Σ_x|y = Σ_x − Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ H Σ_x
oi_posterior = oi.posterior(snapshot)  # cov (D_x, D_x) PSD op
# today (gaussx)
# diag(Σ_x|y) = 1 / (1 + σ⁻²) where observed, 1 elsewhere
oi_variance = gx.diag(oi_posterior.cov)  # (D_x, D_x) op → (D_x,)
print(" ".join(f"{variance:.2f}" for variance in oi_variance))
```

It prints `0.20 0.20 0.20 1.00 1.00 1.00`.

`posterior(batch)` takes one window and returns the exact BLUE covariance in its observation-space form, $\gmat{\Sigma}_{x \mid y} = \gmat{\Sigma}_x - \gmat{\Sigma}_x \mat{H}^\top (\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1} \mat{H} \gmat{\Sigma}_x$. The `cov` operator is lazy: each mat-vec solves the innovation system the analysis solved, with the same solver, and never applies $\gmat{\Sigma}_x^{-1}$, so a smooth $\gmat{\Sigma}_x$ that CG cannot invert is no obstacle. Pass `adapter=vdx.LaplaceCovariance()` for the precision form $(\gmat{\Sigma}_x^{-1} + \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H})^{-1}$ instead.

This is the cheapest and most accurate posterior, when it applies.

### 2. Laplace approximation

For `ThreeDVar` and the 4DVar models, we approximate the posterior at the MAP as a Gaussian whose covariance is the inverse Hessian of the cost, and drop the second-derivative term of the observation misfit (the Gauss-Newton approximation). What the observations depend on decides the precision.

A single analysis time (`ThreeDVar`) gives

$$\gmat{\Sigma}_{x \mid y} = \left(\nabla^2_{\vect{x}} \mathcal{J}(\vect{x}^\star)\right)^{-1} \approx \left(\gmat{\Sigma}_x^{-1} + \mat{H}(\vect{x}^\star)^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}(\vect{x}^\star)\right)^{-1}$$ (eq-laplace-cov)

where $\mathcal{J}$ is the 3DVar cost [](#eq-3dvar-cost) and $\mat{H}(\vect{x}^\star) \in \mathbb{R}^{D_y \times D_x}$ the Jacobian of the observation operator at the MAP, so the bracket is the Gauss-Newton Hessian [](#eq-3dvar-gn-hessian) evaluated at $\vect{x}^\star$.

In a 4DVar window (`StrongFourDVar`, `IncrementalFourDVar`, `ReducedBasisFourDVar` without a basis), the observations at time $t$ depend on $\vect{x}_0$ through the model, so the covariance is the inverse of the Gauss-Newton Hessian [](#eq-gn-hessian):

$$\gmat{\Sigma}_{x \mid y} = \gmat{\Lambda}_{x \mid y}^{-1}$$ (eq-gn-cov)

where $\gmat{\Lambda}_{x \mid y} \in \mathbb{R}^{D_x \times D_x}$ is that Hessian evaluated along the analysis trajectory, $\mat{F}_{t \leftarrow 0}$ is the tangent-linear model from $\vect{x}_0$ to time $t$ and $\mat{H}_t$ includes the mask $\mat{M}_t = \op{diag}(\vect{m}_t)$. Never-observed state components that the dynamics couple to observed ones are constrained through $\mat{F}_{t \leftarrow 0}$, so their posterior variance drops below the prior even though no observation touches them.

For `WeakFourDVar`, the control is $\vect{w} = (\vect{x}_0, \gvect{\eta}_1, \ldots, \gvect{\eta}_T) \in \mathbb{R}^{(T + 1) D_x}$ of the weak-constraint cost [](#eq-weak4dvar-cost), and the precision $\gmat{\Lambda}_{w \mid y}$ gains $\gmat{\Sigma}_{\eta,t}^{-1}$ on each $\gvect{\eta}_t$. The adapter returns the $\vect{x}_0$ block of $\gmat{\Lambda}_{w \mid y}^{-1}$, which is the Kalman-smoother covariance of $\vect{x}_0$ in the linear-Gaussian case. Pass `analysis=(x0, eta)`.

`gaussx.diag` reads the posterior variances off the lazy operator. The block continues the one above.

```{code-block} python
:caption: Runs the strong 4DVar of the toy window and reads the variances of its Laplace posterior [](#eq-gn-cov).
:label: code-postcov-laplace
# Shapes: S = 1 window; T = 2 steps, so T + 1 = 3 times; D_x = D_y = 6 cells

# today (vardax)
strong = vdx.StrongFourDVar(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(6),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# x₀⋆ = argmin 𝒥(x₀), one window
window_analysis = strong(window)  # (S, T+1, D_y) → (S, D_x)
initial_analysis = window_analysis[0]  # (S, D_x) → (D_x,)
# today (vardax)
laplace = vdx.LaplaceCovariance(prior_cov_op=prior_cov, obs_cov_op=obs_cov)
# Σ_x|y = (Σ_x⁻¹ + ∑ₜ (Hₜ F_t←0)ᵀ Σ_ε,t⁻¹ (Hₜ F_t←0))⁻¹
laplace_posterior = laplace(initial_analysis, strong, window)  # cov (D_x, D_x) op
# today (gaussx)
# diag(Σ_x|y)
posterior_variance = gx.diag(laplace_posterior.cov)  # (D_x, D_x) op → (D_x,)
print(" ".join(f"{variance:.2f}" for variance in posterior_variance))
```

It prints `0.17 0.13 0.15 0.38 0.85 1.00`. Cells 3 and 4 are never observed, but the ring average carries them into observed cells within the window, so their variance drops below the prior's 1; cell 5 would need a third step to reach an observed cell and keeps the prior variance.

The adapter reads the model's `obs_op` and rollout (an
`.as_analysis_step()` wrapper works too) and the window's length and mask
from `batch`, which must hold a single window. It returns `cov` as an
`AbstractLinearOperator`: each mat-vec is a Krylov solve (`lineax.CG`
by default, `solver=` to change it), and each Krylov iteration is one
tangent-linear plus one adjoint pass through the rollout. Nothing is
materialised.

A linear control, `parameterization=LinearParameterization(basis, background)` or `ReducedBasisFourDVar(basis=...)`, has $\vect{x} = \gvect{\mu}_x + \gmat{\Phi} \vect{z}$ and the penalty $\frac{1}{2} \vect{z}^\top \gmat{\Sigma}_z^{-1} \vect{z}$ in place of the background term (the reduced-basis cost [](#eq-reduced-basis-cost) of chapter [23](23_reduced_order.md)). The adapters then build the precision in coefficient space and map it to the state:

$$\gmat{\Sigma}_{z \mid y} = \left(\gmat{\Sigma}_z^{-1} + \gmat{\Phi}^\top \mat{H}_{0:T}^\top \gmat{\Sigma}_{\varepsilon,0:T}^{-1} \mat{H}_{0:T} \gmat{\Phi}\right)^{-1}, \text{ and } \gmat{\Sigma}_{x \mid y} = \gmat{\Phi} \gmat{\Sigma}_{z \mid y} \gmat{\Phi}^\top$$ (eq-reduced-basis-cov)

where $\gmat{\Phi}$ is the basis, $\gmat{\Sigma}_z$ the prior covariance of the coefficients $\vect{z}$, $\mat{H}_{0:T}$ the single-time $\mat{H}$ or the strong-constraint window map $\mat{H}_t \mat{F}_{t \leftarrow 0}$ stacked over $t$, and $\gmat{\Sigma}_{\varepsilon,0:T}$ the block-diagonal covariance of the stacked observations. The whitened control $\gmat{\Phi} = \mat{L}_x$, $\gmat{\Sigma}_z = \mat{I}$ with $\gmat{\Sigma}_x = \mat{L}_x \mat{L}_x^\top$ gives exactly the state-space posterior. Any other parameterization (a `NeuralFieldParameterization`, say) and a parameterized `WeakFourDVar` raise `ValueError`: their prior is not $\gmat{\Sigma}_x$, and their decode is not affine.

**Gaussian-likelihood only.** For non-Gaussian likelihoods (heavy
tails, censored, count data), the Laplace approximation is wrong; use
ensemble or amortized posteriors.

**Unimodal-around-MAP only.** If the posterior is multimodal, the
Laplace approximation captures only the local mode. Diagnose by
running multiple MAPs from perturbed starting points.

### 3. Gauss-Newton Hessian

`GaussNewtonHessian` builds the same [](#eq-gn-cov) as `LaplaceCovariance` and exposes the Krylov budget (`n_krylov`, `tol`). The block continues the one above.

```{code-block} python
:caption: Rebuilds the Gauss-Newton posterior [](#eq-gn-cov) with a capped Krylov budget and checks it against the Laplace variances.
:label: code-postcov-gauss-newton
# Shapes: S = 1 window; T = 2 steps, so T + 1 = 3 times; D_x = D_y = 6 cells

# today (vardax)
gauss_newton = vdx.GaussNewtonHessian(
    prior_cov_op=prior_cov, obs_cov_op=obs_cov, n_krylov=50
)
# Σ_x|y, with CG capped at 50 iterations per mat-vec
gauss_newton_posterior = gauss_newton(  # cov (D_x, D_x) op
    initial_analysis, strong, window
)
# today (gaussx)
# diag(Σ_x|y)
gauss_newton_variance = gx.diag(gauss_newton_posterior.cov)  # (D_x, D_x) op → (D_x,)
print(bool(jnp.allclose(gauss_newton_variance, posterior_variance, atol=1e-4)))
```

It prints `True`.

`IncrementalFourDVar` assembles this operator at its last outer
iterate during the analysis; the adapter rebuilds it at the analysis
it is given, at the cost of one linearisation of the rollout.

With `n_krylov=50`, a mat-vec costs up to 50 tangent-linear plus adjoint passes, which is still much cheaper than materialising $\gmat{\Sigma}_{x \mid y}$.

### 4. Ensemble (`filterax` bridge)

For multimodal posteriors, non-Gaussian likelihoods, or hybrid EnVar (Epic 9), build the posterior from an ensemble of analyses:

$$\gmat{\Sigma}_{x \mid y} \approx \frac{1}{N_\mathcal{E} - 1} \sum_{e \in \mathcal{E}} \tilde{\vect{x}}^{(e)} \tilde{\vect{x}}^{(e)\top}, \text{ with } \tilde{\vect{x}}^{(e)} = \vect{x}^{(e)} - \bar{\vect{x}}$$ (eq-ensemble-cov)

where each member $\vect{x}^{(e)} \in \mathbb{R}^{D_x}$ is a separate vardax analysis with a perturbed initial condition or observation realisation, $N_\mathcal{E}$ is the number of members and $\bar{\vect{x}}$ their mean.

Here the $N_\mathcal{E} = 8$ members re-run the strong 4DVar above on perturbed observations $\vect{y}^{(e)} = \vect{y} + \sigma \gvect{\varepsilon}^{(e)}$, and $\zeta = 1.1$ inflates the sample covariance [](#eq-ensemble-cov). `StrongFourDVar` already maps over the leading axis of a batch, so the members run in parallel without a `vmap`. The block continues the one above.

```{code-block} python
:caption: Re-runs the strong 4DVar on perturbed observations and forms the inflated ensemble covariance [](#eq-ensemble-cov).
:label: code-postcov-ensemble
# Shapes: N_ℰ = 8 members; T = 2 steps, so T + 1 = 3 times; D_x = D_y = 6 cells

# ε⁽ᵉ⁾ ~ 𝒩(0, I)
member_eps = jax.random.normal(key_members, (8, 3, 6))  # (N_ℰ, T+1, D_y)
# σ ε⁽ᵉ⁾
member_noise = 0.5 * member_eps  # (N_ℰ, T+1, D_y)
# y⁽ᵉ⁾ = y + σ ε⁽ᵉ⁾
perturbed_obs = einx.add(  # (T+1, D_y) → (N_ℰ, T+1, D_y)
    "t n, m t n -> m t n", window_obs, member_noise
)
member_mask = einx.id(  # (T+1, D_y) → (N_ℰ, T+1, D_y)
    "t n -> m t n", mask, m=8
)
# today (vardax)
member_batch = vdx.Batch1D(input=perturbed_obs, mask=member_mask)
# x⁽ᵉ⁾ = argmin 𝒥(x₀; y⁽ᵉ⁾)
member_analyses = strong(member_batch)  # (N_ℰ, T+1, D_y) → (N_ℰ, D_x)
# today (vardax)
ensemble = vdx.EnsembleCovariance(n_members=8, inflation=1.1)
# today (vardax)
analysis_step = strong.as_analysis_step()
# Σ_x|y = ζ (1 / (N_ℰ − 1)) ∑ₑ x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ
ensemble_posterior = ensemble(member_analyses, analysis_step, window)
print(ensemble_posterior.mean.shape, ensemble_posterior.samples.shape)
```

It prints `(6,) (8, 6)`.

Inflation is built in. Localisation is not: without it, the sample
covariance from $N_\mathcal{E} \ll D_x$ members is rank-deficient and noisy, so taper
it (for example with `filterax`'s localisation) before using it for
anything spatial.

## Picking by inference method

| Method | Default posterior path |
|---|---|
| `OptimalInterpolation` | `.posterior(batch)`, exact |
| `ThreeDVar` | `.posterior(batch)` / `LaplaceCovariance()` |
| `StrongFourDVar` | `.posterior(batch)` / `LaplaceCovariance()` or `GaussNewtonHessian()` |
| `WeakFourDVar` | `.posterior(batch)`, the $\vect{x}_0$ block of the augmented-control inverse |
| `IncrementalFourDVar` | `.posterior(batch)`, the inner loop's Gauss-Newton operator, relinearised at the analysis |
| `FourDVarNet1D` / `FourDVarNet2D` | `EnsembleCovariance()` over perturbed inputs (no `obs_op`, so no Laplace adapter); verify calibration via SBC |
| `AmortizedPosterior` | direct `.sample(batch, key, n)`; a flow or score head gives samples natively |
| Hybrid `EnVarFourDVar` (Epic 9) | `EnsembleCovariance()` |

## Export with `GaussianMarkLikelihood`

Per-event posteriors feed downstream population models (TMTPP,
hierarchical Bayesian) via a serialisable mark-likelihood:

```{code-block} python
:caption: Wraps the Laplace posterior [](#eq-gn-cov) of the toy window in a `GaussianMarkLikelihood` and serialises it to JSON.
:label: code-postcov-mark-export
import json

# Shapes: D_x = 6 cells

# today (vardax)
mark = vdx.GaussianMarkLikelihood(
    posterior=laplace_posterior,
    event_metadata={"event_id": "toy-0001", "time": "2024-01-15T12:00Z"},
)
# today (vardax)
# mean, diag(Σ_x|y), samples, provenance and metadata as plain lists and dicts
record = mark.to_dict()
serialised = json.dumps(record)
print(sorted(record))
```

It prints `['cov_diag', 'event_metadata', 'mean', 'provenance', 'samples']`. vardax ships no catalog writer; the JSON text goes to whatever store the population layer reads.

The serialised form is consumed by Tier V population models without re-running vardax. Decision D10 makes this contract explicit, since it is how the inference layer hands off to the population layer.

## Provenance for auditing the analysis

`Posterior.provenance` is meant to carry at minimum the keys below. Today the adapters record only their name and a few settings (`adapter` and `structure` for `LaplaceCovariance`, plus `n_krylov` for `GaussNewtonHessian`, and `n_members` and `inflation` for `EnsembleCovariance`), so the table is the target contract.

| Key | Type | Description |
|---|---|---|
| `forward_model_id` | str | e.g. `"plumax.tier1.gaussian"` |
| `forward_model_hash` | str | Content hash from `pipekit-experiment.ModelRegistry` |
| `obs_ops_used` | list[str] | Instrument IDs whose obs operators contributed |
| `n_iter` | int | Total inner iterations (or outer × inner for incremental) |
| `J_star` | float | Final cost value |
| `converged` | bool | True if convergence criterion met |
| `model_hash` | str \| None | For learned heads, content hash of trained weights |
| `gaussx_op_hash` | str \| None | Hash of $\gmat{\Sigma}_x$ / $\gmat{\Sigma}_\varepsilon$ operators used |
| `met_source` | str \| None | Met forcing identifier (e.g. `"era5_2024-01-15T12Z"`) |
| `vardax_version` | str | Package version |

This is the contract between the inference layer (vardax) and the
audit layer (catalog / database). Don't break it without coordinating
with downstream consumers.

## Validation gates

Posteriors are only as good as they agree with reality, so the six-step cycle checks a fast posterior against its oracle with the posterior-agreement gate [](#eq-gate-posterior), the Jacobian gate [](#eq-gate-jacobian) and simulation-based calibration; [chapter 14](14_six_step_cycle.md) states the gates and the functions that implement them.

## See also

- Chapter 4: `OptimalInterpolation` (closed-form posterior)
- Chapter 8: `IncrementalFourDVar` (GN-Hessian reuse)
- Chapter 10: `AmortizedPosterior` (direct sampling)
- Chapter 14: six-step cycle (validation methodology)
- Design doc: [`design/posterior.md`](design/posterior.md)
- Design doc: [`design/decisions.md#d10`](design/decisions.md#d10-posterior-export-adapter-pattern)

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning*. Ch. 2.7.
- Cressie, N., & Wikle, C. K. (2011). *Statistics for Spatio-Temporal
  Data*. Wiley.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A.
  (2018). *Validating Bayesian inference algorithms with
  simulation-based calibration.* arXiv:1804.06788.
- Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
  formulation and practical implementation.* Ocean Dynamics 53.
