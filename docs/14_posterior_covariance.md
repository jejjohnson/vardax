# Posterior Covariance

A point estimate $\mathbf{x}^*$ from minimising $J(\mathbf{x})$ is not
enough — downstream consumers (population models, decision pipelines, alert
systems) need posterior uncertainty. vardax exposes three families of
`PosteriorAdapter` (Decision D10), each appropriate for a different
inference regime.

## Bayesian context

If the prior is Gaussian $\mathbf{x} \sim \mathcal{N}(\mathbf{x}_b,
\mathbf{B})$ and the observation likelihood Gaussian $\mathbf{y} \mid
\mathbf{x} \sim \mathcal{N}(H(\mathbf{x}), \mathbf{R})$, the posterior is

$$p(\mathbf{x} \mid \mathbf{y}) \propto \exp\!\left(-\tfrac{1}{2}\|\mathbf{x} - \mathbf{x}_b\|^2_{\mathbf{B}^{-1}} - \tfrac{1}{2}\|\mathbf{y} - H(\mathbf{x})\|^2_{\mathbf{R}^{-1}}\right)$$

The MAP is $\mathbf{x}^* = \arg\min J(\mathbf{x})$. Around the MAP, the
posterior is approximated by a Gaussian (Laplace) with covariance
$\mathbf{P}^* = (J''(\mathbf{x}^*))^{-1}$.

## Laplace covariance

$$\mathbf{P}^*_\text{Laplace} = \Big((\mathbf{H}')^\top \mathbf{R}^{-1} \mathbf{H}' + \mathbf{B}^{-1}\Big)^{-1}$$

evaluated at $\mathbf{x}^*$. Returned as an `AbstractLinearOperator` —
mat-vec via `lineax.CG`, no full materialisation unless requested.

**When to use.** Gaussian likelihood + Gaussian prior + single posterior
mode (verified by SBC). Default for `VarDANet*` and the cheapest option for
single-event posteriors.

```python
from vardax.posterior import LaplaceCovariance

posterior = LaplaceCovariance()(x_star, model.as_analysis_step(), batch)
# Posterior(mean=x_star, cov=AbstractLinearOperator, samples=None, provenance={...})
```

## Gauss-Newton Hessian

For `IncrementalVarDA*` the Gauss-Newton Hessian is already assembled
during the last outer iteration. Reuse it:

$$\mathbf{P}^*_\text{GN} = \big(\mathbf{J}''_\text{GN}\big)^{-1} = \Big(\mathbf{B}^{-1} + \sum_t (\mathbf{H}'_t \mathbf{M}'_t)^\top \mathbf{R}_t^{-1} (\mathbf{H}'_t \mathbf{M}'_t)\Big)^{-1}$$

Krylov / Lanczos via `lineax.CG` for required marginals (often just
diagonal for facility-scale attribution).

**When to use.** Operational incremental 4DVar. Cost: $n_\text{krylov}$
mat-vec products beyond the inversion.

```python
from vardax.posterior import GaussNewtonHessian

posterior = GaussNewtonHessian(n_krylov=50)(x_star, model.as_analysis_step(), batch)
```

## Ensemble covariance

For multimodal posteriors, non-Gaussian likelihoods, or hybrid EnVar
(Epic 9), build the posterior from an ensemble of analyses via `filterax`:

$$\mathbf{P}^*_\text{ens} = \frac{1}{M-1} \sum_{m=1}^{M} (\mathbf{x}^{(m)} - \bar{\mathbf{x}})(\mathbf{x}^{(m)} - \bar{\mathbf{x}})^\top$$

Each $\mathbf{x}^{(m)}$ is a separate vardax analysis with a perturbed
initial condition / observation realisation.

**When to use.** Suspected multimodal posterior; non-Gaussian regime;
ensemble already available from `filterax`.

```python
import filterax as fx
from vardax.posterior import EnsembleCovariance

# Run M parallel analyses (eqx.filter_vmap)
analyses = eqx.filter_vmap(model)(perturbed_batches)

posterior = EnsembleCovariance(n_members=M, inflation=1.1, localization_radius=50.0)(
    analyses, model.as_analysis_step(), batch,
)
```

Localisation and inflation are standard ensemble fixes — delegated to
`filterax`.

## Selection heuristic

| Inference family | Default posterior adapter |
|---|---|
| `VarDANet*` (learned 4DVarNet) | `LaplaceCovariance` |
| `IncrementalVarDA*` (operational) | `GaussNewtonHessian` (reuses Hessian) |
| `AmortizedVarDA*` (flow / score) | direct sampling (`Posterior.samples`) |
| Hybrid EnVar (`EnVarDA`) | `EnsembleCovariance` |

## Mark-likelihood export

Per-event posteriors feed population models via `GaussianMarkLikelihood`:

```python
from vardax.posterior import GaussianMarkLikelihood

mark = GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={"event_id": event.id, "time": event.time, ...},
)
catalog.write_posterior(event.id, mark.to_dict())
```

The serialised form is consumed by Tier V population models (TMTPP,
hierarchical Bayesian) without re-running vardax inference.

## Validation gates (Six-step cycle, Decision D12)

A posterior is only as good as it agrees with a slower oracle.
`tests/test_six_step_validation.py` enforces:

1. **Agreement.** Step 4 (emulator MAP) within $1\sigma_\text{post}$ of
   Step 2 (physics MAP) on held-out events.
2. **Adjoint calibration.** $\|\partial H_\text{em}/\partial x -
   \partial H_\text{phys}/\partial x\|_\text{op} < 5\%$ via
   random-vector probing.
3. **SBC.** Simulation-based calibration: rank histograms uniform across
   parameters (χ² test of uniformity).

```python
from vardax.utils.validation import assert_posterior_agreement, simulation_based_calibration

assert_posterior_agreement(p_amortized, p_physics, tolerance_sigma=1.0)
simulation_based_calibration(model, prior, forward, n_runs=200)
```

## Provenance

Every `Posterior` carries provenance metadata:

```python
posterior.provenance = {
    "forward_model_id": "plumax.tier1.gaussian",
    "forward_model_hash": "...",            # from pipekit-experiment
    "obs_ops_used": ["TROPOMI", "EMIT", "GHGSat"],
    "n_iter": 60,                            # 3 outer × 20 inner
    "J_star": 12.4,
    "converged": True,
    "model_hash": "...",                     # for learned heads
    "gaussx_op_hash": "...",                 # B / R operator hashes
    "met_source": "era5_2024-01-15T12Z",
    "vardax_version": "0.2.0",
}
```

This is the contract between inference and audit — don't break it lightly.

## See also

- Chapter 13: Incremental 4DVar — GN Hessian assembly
- Chapter 16: Six-step cycle — validation gates
- Decision D10 in design docs — posterior export adapter pattern
- `vardax.posterior` module reference
