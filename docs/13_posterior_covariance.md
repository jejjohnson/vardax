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
class Posterior(eqx.Module):
    mean: Array
    cov: AbstractLinearOperator | None       # gaussx / lineax operator
    samples: Array | None                     # (B, M, ...)
    provenance: dict                          # forward_model_id, n_iter, J_star, ...
```

The covariance is an `AbstractLinearOperator` — not materialised. Mat-vec
$P^* v$, the diagonal $\mathrm{diag}(P^*)$, and posterior samples
$x^* + P^{*\,1/2} \xi$ are computed on demand without ever building
the dense $P^*$.

`OptimalInterpolation` and `IncrementalFourDVar` produce posteriors as
part of the analysis call (closed-form / GN-Hessian reuse). The other
five methods pair with an explicit `PosteriorAdapter`.

## Four posterior families

### 1. Closed form (BLUE / OI)

For `OptimalInterpolation`, the posterior is exact:

$$
P^* = (B^{-1} + H^\top R^{-1} H)^{-1}.
$$

```python
oi = OptimalInterpolation(linear_H, x_b, B_op, R_op)
posterior = oi.posterior(batch)
# Posterior(mean=..., cov=AbstractLinearOperator, samples=None, provenance={...})
```

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

$$
P^*_\text{Laplace} = \big(\nabla^2 J(x^*)\big)^{-1}
                   \approx \big(B^{-1} + H'(x^*)^\top R^{-1} H'(x^*)\big)^{-1}.
$$

A 4DVar window (`StrongFourDVar`, `IncrementalFourDVar`,
`ReducedBasisFourDVar` without a basis): the observations at time $t$
depend on $x_0$ through the model, so with the tangent-linear model
$M'_{t \leftarrow 0}$ along the analysis trajectory and the mask $m_t$,

$$
P^*_\text{GN} = \Big(B^{-1} + \sum_t (H'_t M'_{t\leftarrow 0})^\top R_t^{-1}
                    (H'_t M'_{t\leftarrow 0})\Big)^{-1}.
$$

Never-observed state components that the dynamics couple to observed
ones are constrained through $M'$: their posterior variance drops below
the prior even though no observation touches them.

`WeakFourDVar`: the control is $(x_0, \eta_1, \ldots, \eta_T)$ and the
precision gains $Q^{-1}$ on each $\eta_t$; the adapter returns the
$x_0$ block of the inverse, which is the Kalman-smoother covariance of
$x_0$ in the linear-Gaussian case. Pass `analysis=(x0, eta)`.

```python
import vardax

x0_star = strong(batch)[0]                     # one window
posterior = vardax.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)(
    x0_star, strong, batch                    # batch: that one window
)
```

The adapter reads the model's `obs_op` and rollout (an
`.as_analysis_step()` wrapper works too) and the window's length and mask
from `batch`, which must hold a single window. It returns `cov` as an
`AbstractLinearOperator`: each mat-vec is a Krylov solve (`lineax.CG`
by default, `solver=` to change it), and each Krylov iteration is one
tangent-linear plus one adjoint pass through the rollout. Nothing is
materialised.

A linear control, `parameterization=LinearParameterization(basis, x_b)`
or `ReducedBasisFourDVar(basis=...)`, has $x = x_b + \Phi\theta$ and the
penalty $\tfrac12 \theta^\top Q^{-1} \theta$ in place of $B$. The adapters
then build the precision in coefficient space and map it to the state:

```math
P_\theta = \left(Q^{-1} + \Phi^\top G'^\top R^{-1} G' \Phi\right)^{-1},
\qquad
P_x = \Phi P_\theta \Phi^\top,
```

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

```python
posterior = vardax.GaussNewtonHessian(
    prior_cov_op=B_op, obs_cov_op=R_op, n_krylov=50
)(x0_star, strong, batch)
```

`IncrementalFourDVar` assembles this operator at its last outer
iterate during the analysis; the adapter rebuilds it at the analysis
it is given, at the cost of one linearisation of the rollout.

Now you pay the Krylov cost — typically 50 tangent-linear+adjoint
passes. Still much cheaper than full materialisation.

### 4. Ensemble (`filterax` bridge)

For multimodal posteriors, non-Gaussian likelihoods, or hybrid EnVar
(Epic 9), build the posterior from an ensemble of analyses:

$$
P^*_\text{ens} = \frac{1}{M - 1} \sum_{m=1}^{M} (x^{(m)} - \bar{x})(x^{(m)} - \bar{x})^\top.
$$

Each $x^{(m)}$ is a separate vardax analysis with a perturbed initial
condition or observation realisation.

```python
import equinox as eqx
import filterax as fx
from vardax import EnsembleCovariance

# Run M analyses in parallel
analyses = eqx.filter_vmap(strong_4dvar)(perturbed_batches)

posterior = EnsembleCovariance(
    n_members=M, inflation=1.1,
)(analyses, strong_4dvar.as_analysis_step(), batch)
```

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
from vardax import GaussianMarkLikelihood

mark = GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={"event_id": event.id, "time": event.time, "geometry": ...},
)
catalog.write_posterior(event.id, mark.to_dict())
```

The serialised form is consumed by Tier V population models without
re-running vardax. Decision D10 makes this contract explicit — it's
how the inference layer hands off to the population layer.

## Provenance — auditing the analysis

`Posterior.provenance` carries at minimum:

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

These are **planned** as `vardax._src.utils.validation.assert_*`
functions running in `tests/test_six_step_validation.py` (v0.4 design
target, Epic 8). Neither the module nor the test file exists in the
v0.1.x codebase yet; failing gates would block promotion
of an amortized head to operational use.

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
