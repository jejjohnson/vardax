# Posterior Covariance

A MAP estimate $x^*$ is the start of the answer, not the end. Real DA
needs a posterior — mean, covariance, and provenance — for downstream
consumers (population models, decision pipelines, alerting systems)
to propagate uncertainty correctly. This chapter covers the four
posterior-covariance computations vardax supports and when to use
which.

## The `Posterior` container

Every analysis emits a `Posterior(mean, cov, samples, provenance)`:

```python
from typing import Any

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

# Shapes: M ensemble members; "..." is the state's own shape


# today (vardax)
# the fields of vdx.Posterior
class Posterior(eqx.Module):
    mean: Float[Array, "..."]
    # P* as a lazy gaussx / lineax operator; None for sampling-only heads
    cov: lx.AbstractLinearOperator | None = None
    samples: Float[Array, "member ..."] | None = None
    # forward_model_id, n_iter, J_star, ...
    provenance: dict[str, Any] = eqx.field(default_factory=dict)
```

The covariance is an `AbstractLinearOperator` — not materialised. Mat-vec
$P^* v$, the diagonal $\mathrm{diag}(P^*)$, and posterior samples
$x^* + (P^*)^{1/2} \xi$ are computed on demand without ever building
the dense $P^*$.

`OptimalInterpolation` and `IncrementalFourDVar` produce posteriors as
part of the analysis call (closed-form / GN-Hessian reuse). The other
five methods pair with an explicit `PosteriorAdapter`.

## Four posterior families

### 1. Closed form (BLUE / OI)

For `OptimalInterpolation`, the posterior is exact:

$$P^* = (B^{-1} + H^\top R^{-1} H)^{-1}.$$

The examples on this page share one toy problem. The state has $N = 6$ cells on a ring, and the forward model averages each cell with its right neighbour, $x_{t+1} = A x_t$ with $A = \frac{1}{2}(I + P)$ and $P$ the cyclic shift. The background is $x_b = 0$ with $B = I$, a window has $T = 3$ times, only cells 0 to 2 are observed (mask $m_t$), and $y_t = x_t + \sigma \varepsilon_t$ with $R = \sigma^2 I$, $\sigma = 0.5$. OI analyses the first time of the window on its own, with $H = I$ on the observed cells. The block continues the one above.

```python
import einx
import gaussx
import jax
import jax.numpy as jnp
import vardax as vdx

# Shapes: S = 1 window; T = 3 times; N = 6 cells; M = 8 members

key = jax.random.PRNGKey(0)
key_truth, key_noise, key_members = jax.random.split(key, 3)

# P, the cyclic shift (P x)ᵢ = xᵢ₊₁
cyclic_shift = jnp.eye(6, k=1) + jnp.eye(6, k=-5)  # (N, N)
# A = ½ (I + P)
coupling = 0.5 * (jnp.eye(6) + cyclic_shift)  # (N, N)


# user
class RingAverage(eqx.Module):
    """xₜ₊₁ = A xₜ: each cell averaged with its right neighbour."""

    coupling: Float[Array, "n n"]
    dt: float = eqx.field(static=True, default=1.0)

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # xₜ₊₁ = A xₜ
        return einx.dot("i j, j -> i", self.coupling, state)


forward = RingAverage(coupling)
# x₀ ~ 𝒩(0, I)
truth = jax.random.normal(key_truth, (6,))  # (N,)
# x₁ = A x₀
truth_1 = forward.step(truth, 1.0)  # (N,)
# x₂ = A x₁
truth_2 = forward.step(truth_1, 1.0)  # (N,)
truth_trajectory = jnp.stack([truth, truth_1, truth_2])  # (T, N)
# mₜ: cells 0, 1 and 2 observed at every time
observed_cells = (jnp.arange(6) < 3).astype(jnp.float32)  # (N,)
mask = einx.id("n -> t n", observed_cells, t=3)  # (N,) → (T, N)
# yₜ = xₜ + σ εₜ
window_obs = truth_trajectory + 0.5 * jax.random.normal(key_noise, (3, 6))
# today (vardax)
window = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", window_obs),  # (T, N) → (S, T, N)
    mask=einx.id("t n -> 1 t n", mask),
)

state_structure = jax.ShapeDtypeStruct((6,), jnp.float32)
# B = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (N, N) op
# R = σ² I; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (N, N) op
    jnp.float32(0.25) * prior_cov, lx.positive_semidefinite_tag
)

# the same B and R on a single time, shape (1, N)
snapshot_structure = jax.ShapeDtypeStruct((1, 6), jnp.float32)
snapshot_prior_cov = lx.IdentityLinearOperator(snapshot_structure)  # (N, N) op
snapshot_obs_cov = lx.TaggedLinearOperator(  # (N, N) op
    jnp.float32(0.25) * snapshot_prior_cov, lx.positive_semidefinite_tag
)
# today (vardax)
snapshot = vdx.Batch1D(
    input=window.input[:, :1],  # (S, T, N) → (S, 1, N)
    mask=window.mask[:, :1],
)
# today (vardax)
oi = vdx.OptimalInterpolation(
    vdx.MaskedIdentity(),
    jnp.zeros((1, 6)),
    snapshot_prior_cov,
    snapshot_obs_cov,
)
# P* = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹
oi_posterior = oi.posterior(snapshot)
# today (gaussx)
# diag(P*) = 1 / (1 + σ⁻²) where observed, 1 elsewhere
oi_variance = gaussx.diag(oi_posterior.cov)  # (N,)
print(" ".join(f"{variance:.2f}" for variance in oi_variance))
```

It prints `0.20 0.20 0.20 1.00 1.00 1.00`.

`posterior(batch)` takes one window and returns the Laplace posterior
with the model's own $B$ and $R$, which for OI is exact. The `cov`
operator is lazy: each mat-vec is a CG solve against
$B^{-1} + H^\top R^{-1} H$ (closed form for diagonal and other
structured operators), never a dense matrix.

This is the cheapest and most accurate posterior — when applicable.

### 2. Laplace approximation

For `ThreeDVar` and the 4DVar models: at the MAP, approximate the
posterior as Gaussian with covariance the inverse Hessian of the cost,
and drop the second-derivative term of the observation misfit (the
Gauss–Newton approximation). What the observations depend on decides
the precision.

A single analysis time (`ThreeDVar`):

$$P^*_\text{Laplace} = \left(\nabla^2 J(x^*)\right)^{-1} \approx \left(B^{-1} + H'(x^*)^\top R^{-1} H'(x^*)\right)^{-1}.$$

A 4DVar window (`StrongFourDVar`, `IncrementalFourDVar`,
`ReducedBasisFourDVar` without a basis): the observations at time $t$
depend on $x_0$ through the model, so with the tangent-linear model
$M'_{t \leftarrow 0}$ along the analysis trajectory and the mask $m_t$,

$$P^*_\text{GN} = \left(B^{-1} + \sum_t (H'_t M'_{t\leftarrow 0})^\top R_t^{-1} (H'_t M'_{t\leftarrow 0})\right)^{-1}.$$

Never-observed state components that the dynamics couple to observed
ones are constrained through $M'$: their posterior variance drops below
the prior even though no observation touches them.

`WeakFourDVar`: the control is $(x_0, \eta_1, \ldots, \eta_T)$ and the
precision gains $Q^{-1}$ on each $\eta_t$; the adapter returns the
$x_0$ block of the inverse, which is the Kalman-smoother covariance of
$x_0$ in the linear-Gaussian case. Pass `analysis=(x0, eta)`.

The strong 4DVar of the toy window and its Laplace posterior follow; `gaussx.diag` reads the posterior variances off the lazy operator. The block continues the one above.

```python
# Shapes: S = 1 window; T = 3 times; N = 6 cells

# today (vardax)
strong = vdx.StrongFourDVar(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(6),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# x₀* = argmin J(x₀), one window
initial_analysis = strong(window)[0]  # (S, N) → (N,)
# today (vardax)
laplace = vdx.LaplaceCovariance(prior_cov_op=prior_cov, obs_cov_op=obs_cov)
# P*_GN = (B⁻¹ + Σₜ (H′ₜ M′ₜ←₀)ᵀ Rₜ⁻¹ (H′ₜ M′ₜ←₀))⁻¹
laplace_posterior = laplace(initial_analysis, strong, window)
# today (gaussx)
# diag(P*_GN)
posterior_variance = gaussx.diag(laplace_posterior.cov)  # (N,)
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

A linear control, `parameterization=LinearParameterization(basis, x_b)`
or `ReducedBasisFourDVar(basis=...)`, has $x = x_b + \Phi\theta$ and the
penalty $\frac{1}{2} \theta^\top Q^{-1} \theta$ in place of $B$. The adapters
then build the precision in coefficient space and map it to the state:

$$P_\theta = \left(Q^{-1} + \Phi^\top G'^\top R^{-1} G' \Phi\right)^{-1}, \text{ and } P_x = \Phi P_\theta \Phi^\top,$$

with $G'$ the single-time $H'$ or the strong-constraint window map. The
whitened control $\Phi = L$, $Q = I$ with $B = LL^\top$ gives exactly the
state-space posterior. Any other parameterization (a
`NeuralFieldParameterization`, say) and a parameterized `WeakFourDVar`
raise `ValueError`: their prior is not $B$, and their decode is not affine.

**Gaussian-likelihood only.** For non-Gaussian likelihoods (heavy
tails, censored, count data), the Laplace approximation is wrong; use
ensemble or amortized posteriors.

**Unimodal-around-MAP only.** If the posterior is multimodal, the
Laplace approximation captures only the local mode. Diagnose by
running multiple MAPs from perturbed starting points.

### 3. Gauss-Newton Hessian

`GaussNewtonHessian` builds the same $P^*_\text{GN}$ as
`LaplaceCovariance` and exposes the Krylov budget (`n_krylov`, `tol`):

The block continues the one above.

```python
# today (vardax)
gauss_newton = vdx.GaussNewtonHessian(
    prior_cov_op=prior_cov, obs_cov_op=obs_cov, n_krylov=50
)
# P*_GN, with CG capped at 50 iterations per mat-vec
gauss_newton_posterior = gauss_newton(initial_analysis, strong, window)
# today (gaussx)
gauss_newton_variance = gaussx.diag(gauss_newton_posterior.cov)  # (N,)
print(bool(jnp.allclose(gauss_newton_variance, posterior_variance, atol=1e-4)))
```

It prints `True`.

`IncrementalFourDVar` assembles this operator at its last outer
iterate during the analysis; the adapter rebuilds it at the analysis
it is given, at the cost of one linearisation of the rollout.

Now you pay the Krylov cost — typically 50 tangent-linear+adjoint
passes. Still much cheaper than full materialisation.

### 4. Ensemble (`filterax` bridge)

For multimodal posteriors, non-Gaussian likelihoods, or hybrid EnVar
(Epic 9), build the posterior from an ensemble of analyses:

$$P^*_\text{ens} = \frac{1}{M - 1} \sum_{m=1}^{M} (x^{(m)} - \bar{x})(x^{(m)} - \bar{x})^\top.$$

Each $x^{(m)}$ is a separate vardax analysis with a perturbed initial
condition or observation realisation.

Here the $M = 8$ members re-run the strong 4DVar above on perturbed observations $y^{(m)} = y + \sigma \varepsilon^{(m)}$, and $\lambda = 1.1$ inflates the sample covariance. `StrongFourDVar` already maps over the leading axis of a batch, so the members run in parallel without a `vmap`. The block continues the one above.

```python
# Shapes: M = 8 members; T = 3 times; N = 6 cells

# y⁽ᵐ⁾ = y + σ ε⁽ᵐ⁾
member_noise = 0.5 * jax.random.normal(key_members, (8, 3, 6))  # (M, T, N)
perturbed_obs = window.input + member_noise  # (S, T, N) → (M, T, N)
# today (vardax)
member_batch = vdx.Batch1D(
    input=perturbed_obs,
    mask=einx.id("t n -> m t n", mask, m=8),  # (T, N) → (M, T, N)
)
# x⁽ᵐ⁾ = argmin J(x₀; y⁽ᵐ⁾)
member_analyses = strong(member_batch)  # (M, T, N) → (M, N)
# today (vardax)
ensemble = vdx.EnsembleCovariance(n_members=8, inflation=1.1)
# P*_ens = λ (1/(M − 1)) Σₘ (x⁽ᵐ⁾ − x̄)(x⁽ᵐ⁾ − x̄)ᵀ
ensemble_posterior = ensemble(member_analyses, strong.as_analysis_step(), window)
print(ensemble_posterior.mean.shape, ensemble_posterior.samples.shape)
```

It prints `(6,) (8, 6)`.

Inflation is built in. Localisation is not: without it, the sample
covariance from $M \ll N$ members is rank-deficient and noisy, so taper
it (for example with `filterax`'s localisation) before using it for
anything spatial.

## Picking by inference method

| Method | Default posterior path |
|---|---|
| `OptimalInterpolation` | `.posterior(batch)` — exact |
| `ThreeDVar` | `.posterior(batch)` / `LaplaceCovariance()` |
| `StrongFourDVar` | `.posterior(batch)` / `LaplaceCovariance()` or `GaussNewtonHessian()` |
| `WeakFourDVar` | `.posterior(batch)` — $x_0$ block of the augmented-control inverse |
| `IncrementalFourDVar` | `.posterior(batch)` — the inner loop's Gauss–Newton operator, relinearised at the analysis |
| `FourDVarNet1D` / `FourDVarNet2D` | `EnsembleCovariance()` over perturbed inputs (no `obs_op`, so no Laplace adapter); verify calibration via SBC |
| `AmortizedPosterior` | direct `.sample(batch, key, n)` — flow / score gives samples natively |
| Hybrid `EnVarFourDVar` (Epic 9) | `EnsembleCovariance()` |

## Export — `GaussianMarkLikelihood`

Per-event posteriors feed downstream population models (TMTPP,
hierarchical Bayesian) via a serialisable mark-likelihood:

```python
import json

# today (vardax)
mark = vdx.GaussianMarkLikelihood(
    posterior=laplace_posterior,
    event_metadata={"event_id": "toy-0001", "time": "2024-01-15T12:00Z"},
)
# mean, diag(P*), samples, provenance and metadata as plain lists and dicts
record = mark.to_dict()
serialised = json.dumps(record)
print(sorted(record))
```

It prints `['cov_diag', 'event_metadata', 'mean', 'provenance', 'samples']`. vardax ships no catalog writer; the JSON text goes to whatever store the population layer reads.

The serialised form is consumed by Tier V population models without
re-running vardax. Decision D10 makes this contract explicit — it's
how the inference layer hands off to the population layer.

## Provenance — auditing the analysis

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
| `gaussx_op_hash` | str \| None | Hash of $B$ / $R$ operators used |
| `met_source` | str \| None | Met forcing identifier (e.g. `"era5_2024-01-15T12Z"`) |
| `vardax_version` | str | Package version |

This is the contract between the inference layer (vardax) and the
audit layer (catalog / database). Don't break it without coordinating
with downstream consumers.

## Validation gates

Posteriors are only as good as they agree with reality. The six-step
cycle (chapter 14) enforces three gates:

1. **Posterior agreement** — fast (amortized) MAP within
   $1\sigma_\text{post}$ of slow (oracle) MAP.
2. **Adjoint calibration** — fast gradient matches physics gradient to
   $\le 5\%$ operator norm.
3. **Simulation-based calibration** — posterior coverage matches
   nominal coverage (Talts et al. 2018).

The gates ship as `vardax.assert_posterior_agreement`, `vardax.assert_adjoint_calibrated` and `vardax.simulation_based_calibration` (in `vardax._src.utils.validation`), and `tests/test_six_step_validation.py` tests them; failing gates would block promotion of an amortized head to operational use.

## See also

- Chapter 4 — `OptimalInterpolation` (closed-form posterior)
- Chapter 8 — `IncrementalFourDVar` (GN-Hessian reuse)
- Chapter 10 — `AmortizedPosterior` (direct sampling)
- Chapter 14 — six-step cycle (validation methodology)
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
