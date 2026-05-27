---
status: draft
version: 0.4.0
---

# Posterior Adapter

Per Decision D10, every vardax analysis emits a `Posterior` container —
not just a point estimate. The posterior carries enough information to feed
downstream population models (Tier V TMTPP, hierarchical Bayesian
inversions) without re-running the inner inference.

Two of the seven Layer 2 classes have closed-form fast paths that emit
the posterior as part of the analysis call:

- **`OptimalInterpolation`** — closed-form $P^* = (B^{-1} + H^\top R^{-1} H)^{-1}$
  is returned directly by `.posterior(batch)`.
- **`IncrementalFourDVar`** — the Gauss-Newton Hessian assembled during
  the last outer iteration is reused for $P^*$.

For the other five (`ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`,
`FourDVarNet`, `AmortizedPosterior`), pair the analysis with an
explicit `PosteriorAdapter`.

## The `Posterior` container

```python
class Posterior(eqx.Module):
    mean: Array                                # MAP / posterior mean
    cov: AbstractLinearOperator | None         # gaussx / lineax operator (may be None)
    samples: Array | None                      # (B, M, ...) for ensembles / flow samples
    provenance: dict                           # forward_model_id, n_iter, J_star, …
```

Conventions:

- **`mean`** — always populated.
- **`cov`** — `None` for amortized samples / score-based heads. Otherwise an
  `AbstractLinearOperator` supporting mat-vec (not necessarily
  materialised).
- **`samples`** — `None` for Laplace / GN-Hessian (Gaussian-only). Populated
  for ensemble / flow heads.
- **`provenance`** — dict with at minimum `{forward_model_id, obs_ops_used,
  n_iter, J_star, converged, gaussx_op_hash, model_hash}`.

## Three posterior adapter families

### `LaplaceCovariance` — cheap, Gaussian-only

At the MAP $x^*$:

$$P^* = \big(H'^\top R^{-1} H' + B^{-1}\big)^{-1}$$

where $H' = \partial H / \partial x$ at $x^*$. Returned as an
`AbstractLinearOperator` so mat-vec via `lineax.CG` is cheap; full
materialisation only on request.

**When to use.** Gaussian likelihood, single mode, MAP near posterior mean
(confirmed by SBC). Default for `IncrementalFourDVar`.

**Cost.** One Hessian-vector product family per query. Posterior samples by
$x^* + (H'^\top R^{-1} H' + B^{-1})^{-1/2} \xi$ where $\xi \sim \mathcal{N}(0,I)$ —
requires square-root which may be expensive for unstructured $B$.

```python
class LaplaceCovariance(eqx.Module):
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...
```

### `GaussNewtonHessian` — mid-cost, exact at MAP

Krylov / Lanczos inversion of $J''(x^*)$ via `lineax.CG`. For incremental
4DVar this is the natural posterior — the inner Hessian is already assembled
during the last outer iteration.

**When to use.** Operational incremental 4DVar (`IncrementalFourDVar`). The
Hessian operator from the last GN outer iteration is reused.

**Cost.** `n_krylov` mat-vec products. Materialise only the diagonal /
required marginals.

```python
class GaussNewtonHessian(eqx.Module):
    n_krylov: int = eqx.field(static=True, default=50)
    def __call__(self, analysis: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...
```

### `EnsembleCovariance` — non-Gaussian-aware

Delegates to `filterax`. Build $M$ analyses from $M$ perturbed initial
conditions / observation realisations, return the sample covariance.

**When to use.** Multimodal posterior, non-Gaussian likelihood, hybrid
EnVar inversion.

**Cost.** $M$ vardax analyses (parallel via `eqx.filter_vmap`). Sample
covariance materialisation O($M N^2$) for full; localised / shrunk
estimators available via `filterax`.

```python
class EnsembleCovariance(eqx.Module):
    n_members: int = eqx.field(static=True)
    inflation: float = 1.0
    localization_radius: float | None = None
    def __call__(self, analyses: Array, model: AnalysisStep, batch: Batch) -> Posterior: ...
```

## Posterior selection heuristic

| Inference family | Default posterior adapter |
|---|---|
| `OptimalInterpolation` | closed-form via `.posterior(batch)` (no adapter) |
| `ThreeDVar` | `LaplaceCovariance` at MAP |
| `StrongFourDVar`, `WeakFourDVar` | `LaplaceCovariance` at MAP |
| `IncrementalFourDVar` | `GaussNewtonHessian` via `.posterior(batch)` (Hessian reused) |
| `FourDVarNet` | `LaplaceCovariance` at converged solver state |
| `AmortizedPosterior` | direct sampling (`Posterior.samples`, `cov=None`) |
| Hybrid `EnVarFourDVar` (Epic 9) | `EnsembleCovariance` |
| Hybrid EnVar (`EnVarDA*`, Epic 9) | `EnsembleCovariance` |

## Export to population models — `GaussianMarkLikelihood`

Per-event posteriors feed Tier V (TMTPP, hierarchical Bayesian) via a
mark-likelihood serialisation:

```python
class GaussianMarkLikelihood(eqx.Module):
    """Posterior → mark-likelihood for population models (Tier V, Decision D10)."""

    posterior: Posterior
    event_metadata: dict       # event_id, time, location, instruments_used, …

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_metadata["event_id"],
            "time": self.event_metadata["time"],
            "geometry": self.event_metadata["geometry"],
            "mean": self.posterior.mean.tolist(),
            "cov": self._serialise_cov(),       # diagonal / Cholesky / LowRank repr
            "samples": (self.posterior.samples.tolist()
                        if self.posterior.samples is not None else None),
            "provenance": self.posterior.provenance,
        }

    def _serialise_cov(self) -> dict:
        """Serialise gaussx AbstractLinearOperator to JSON-compatible form."""
        # ...
```

The serialised form is consumed by population models without
re-instantiating the vardax inference — Tier V improvements automatically
absorb upgrades to Tier I-IV forwards.

## Posterior validation gates (Decision D12)

Six-step cycle requires:

1. **Posterior agreement.** Step 4 (emulator MAP) within $1\sigma_\text{post}$
   of Step 2 (physics MAP). Failed when CV(amortized) / CV(physics) > 2 or
   when |mean_amortized - mean_physics| / σ_physics > 1.

2. **Adjoint calibration.** $\|\partial H_\text{em} / \partial x - \partial H_\text{phys} / \partial x\|_\text{op} < 5\%$
   measured by random-vector probing.

3. **Simulation-based calibration (SBC).** Rank histograms uniform across
   parameters, stratified by met regime. Failed when the χ² test of
   uniformity rejects at p < 0.01.

These gates run on a held-out validation set in CI:

```python
# tests/test_six_step_validation.py

def test_posterior_agreement(physics_model, emulator_model, val_batches):
    for batch in val_batches:
        p_phys = LaplaceCovariance()(physics_model(batch), physics_model, batch)
        p_em   = LaplaceCovariance()(emulator_model(batch), emulator_model, batch)
        assert_posterior_agreement(p_em, p_phys, tolerance_sigma=1.0)


def test_adjoint_calibration(physics_obs_op, emulator_obs_op, val_states):
    for x in val_states:
        H_phys = physics_obs_op.linearize(x)
        H_em = emulator_obs_op.linearize(x)
        op_norm = estimate_operator_norm_diff(H_phys, H_em, n_probe=20)
        assert op_norm < 0.05
```

## Provenance schema

`Posterior.provenance` carries at minimum:

| Key | Type | Description |
|---|---|---|
| `forward_model_id` | `str` | Identifier for the `ForwardModel` used (e.g. `"plumax.tier1.gaussian"`) |
| `forward_model_hash` | `str` | Content hash from `pipekit-experiment.ModelRegistry` |
| `obs_ops_used` | `list[str]` | Instrument IDs whose obs operators contributed |
| `n_iter` | `int` | Total inner iterations (or outer × inner for incremental) |
| `J_star` | `float` | Final cost value |
| `converged` | `bool` | True if convergence criterion met |
| `model_hash` | `str \| None` | For learned heads, content hash of trained weights |
| `gaussx_op_hash` | `str \| None` | Hash of $B$ / $R$ operators used |
| `met_source` | `str \| None` | Met forcing identifier (e.g. `"era5_2024-01-15T12Z"`) |
| `vardax_version` | `str` | Package version |

This schema is the contract between the inference layer (vardax) and the
audit layer (catalog / database). Don't break it lightly.
