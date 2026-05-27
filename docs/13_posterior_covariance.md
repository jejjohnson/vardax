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

The `cov` operator is the Sherman-Morrison-Woodbury form chosen by
vardax based on dimensions. Diagonals, samples, and mat-vecs all
compose `gaussx` mat-vecs without materialisation.

This is the cheapest and most accurate posterior — when applicable.

### 2. Laplace approximation

For `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`, `FourDVarNet`: at
the MAP $x^*$, approximate the posterior as Gaussian with covariance

$$
P^*_\text{Laplace} = \big(\nabla^2 J(x^*)\big)^{-1}
                   \approx \big(B^{-1} + H'(x^*)^\top R^{-1} H'(x^*)\big)^{-1}
$$

(the Gauss-Newton approximation drops the second-derivative term).

```python
from vardax.posterior import LaplaceCovariance

x_star = three_dvar(batch)
posterior = LaplaceCovariance()(x_star, three_dvar.as_analysis_step(), batch)
```

`LaplaceCovariance` returns `cov` as an `AbstractLinearOperator` —
mat-vec via Krylov in `lineax.CG`, no materialisation. The cost per
mat-vec is one tangent-linear-plus-adjoint pass through $H$.

**Gaussian-likelihood only.** For non-Gaussian likelihoods (heavy
tails, censored, count data), the Laplace approximation is wrong; use
ensemble or amortized posteriors.

**Unimodal-around-MAP only.** If the posterior is multimodal, the
Laplace approximation captures only the local mode. Diagnose by
running multiple MAPs from perturbed starting points.

### 3. Gauss-Newton Hessian (`IncrementalFourDVar`)

For `IncrementalFourDVar`, the Gauss-Newton Hessian is assembled during
the last outer iteration as part of the algorithm. Reuse it:

$$
P^*_\text{GN} = \big(B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t)\big)^{-1}.
$$

```python
incremental = IncrementalFourDVar(...)
x_star = incremental(batch)
posterior = incremental.posterior(batch)        # GN Hessian reused, free
```

The Hessian is an `AbstractLinearOperator`; inversion via `lineax.CG`
for diagonals or samples. The cost is essentially zero beyond the
analysis cost itself — the Hessian is already there.

For `StrongFourDVar` (same problem, general inner solver), compute
the GN Hessian on demand via

```python
from vardax.posterior import GaussNewtonHessian

posterior = GaussNewtonHessian(n_krylov=50)(x_star, strong.as_analysis_step(), batch)
```

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
from vardax.posterior import EnsembleCovariance

# Run M analyses in parallel
analyses = eqx.filter_vmap(strong_4dvar)(perturbed_batches)

posterior = EnsembleCovariance(
    n_members=M, inflation=1.1, localization_radius=50.0,
)(analyses, strong_4dvar.as_analysis_step(), batch)
```

Localisation and inflation are standard ensemble fixes — delegated to
`filterax`. Without localisation, sample covariance from $M \ll N$
members is rank-deficient and noisy; with it, the covariance is
spatially smoothed and stabilised.

## Picking by inference method

| Method | Default posterior path |
|---|---|
| `OptimalInterpolation` | `.posterior(batch)` — closed form, free |
| `ThreeDVar` | `LaplaceCovariance()` |
| `StrongFourDVar` | `LaplaceCovariance()` or `GaussNewtonHessian()` |
| `WeakFourDVar` | `LaplaceCovariance()` on augmented control |
| `IncrementalFourDVar` | `.posterior(batch)` — GN Hessian reused, free |
| `FourDVarNet` | `LaplaceCovariance()` (verify calibration via SBC) |
| `AmortizedPosterior` | direct `.sample(batch, key, n)` — flow / score gives samples natively |
| Hybrid `EnVarFourDVar` (Epic 9) | `EnsembleCovariance()` |

## Export — `GaussianMarkLikelihood`

Per-event posteriors feed downstream population models (TMTPP,
hierarchical Bayesian) via a serialisable mark-likelihood:

```python
from vardax.posterior import GaussianMarkLikelihood

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
v0.1.6 `fourdvarjax` codebase yet; failing gates would block promotion
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
