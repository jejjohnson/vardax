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

```{code-block} python
:caption: Declares the `Posterior` container that every analysis returns, carrying the mean x* and covariance P* of [](#eq-oi-posterior) (Decision D10).
:label: code-design-posterior-container
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size; M = ensemble members or flow samples;
#   *window = optional leading batch of windows


# today* (vardax)
class Posterior(eqx.Module):
    # x*, the MAP or posterior mean
    mean: Float[Array, "*window N"]
    # P*, a gaussx / lineax operator; None for sampling-only heads
    cov: lx.AbstractLinearOperator | None
    # x⁽ᵐ⁾, m = 1, …, M, for ensemble and flow heads
    samples: Float[Array, "*window M N"] | None
    # forward_model_id, n_iter, J_star, …
    provenance: dict
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

```{code-block} python
:caption: Declares `LaplaceCovariance`, which returns the Laplace covariance [](#eq-laplace-cov) at the MAP as a lazy operator.
:label: code-design-posterior-laplace
import equinox as eqx
from jaxtyping import Array, Float

# Shapes: N = state size
# today (pipekit_cycle): AnalysisStep
# today (vardax): Posterior
# proposed (D10): Batch, the union of the vardax batch types


# today* (vardax)
class LaplaceCovariance(eqx.Module):
    # P* = (H′ᵀ R⁻¹ H′ + B⁻¹)⁻¹, with H′ = ∂H/∂x at x*
    def __call__(
        self,
        analysis: Float[Array, " N"],
        model: AnalysisStep,
        batch: Batch,
    ) -> Posterior: ...
```

### `GaussNewtonHessian` — mid-cost, exact at MAP

Krylov / Lanczos inversion of $J''(x^*)$ via `lineax.CG`. For incremental
4DVar this is the natural posterior — the inner Hessian is already assembled
during the last outer iteration.

**When to use.** Operational incremental 4DVar (`IncrementalFourDVar`). The
Hessian operator from the last GN outer iteration is reused.

**Cost.** `n_krylov` mat-vec products. Materialise only the diagonal /
required marginals.

```{code-block} python
:caption: Declares `GaussNewtonHessian`, which inverts the Gauss-Newton Hessian [](#eq-gn-hessian) with at most `n_krylov` CG steps to give [](#eq-gn-cov).
:label: code-design-posterior-gauss-newton
import equinox as eqx
from jaxtyping import Array, Float

# Shapes: N = state size
# today (pipekit_cycle): AnalysisStep
# today (vardax): Posterior
# proposed (D10): Batch, the union of the vardax batch types


# today* (vardax)
class GaussNewtonHessian(eqx.Module):
    n_krylov: int = eqx.field(static=True, default=50)

    # P*_GN = 𝐇_GN⁻¹, applied by n_krylov CG mat-vecs
    def __call__(
        self,
        analysis: Float[Array, " N"],
        model: AnalysisStep,
        batch: Batch,
    ) -> Posterior: ...
```

### `EnsembleCovariance` — non-Gaussian-aware

Delegates to `filterax`. Build $M$ analyses from $M$ perturbed initial
conditions / observation realisations, return the sample covariance.

**When to use.** Multimodal posterior, non-Gaussian likelihood, hybrid
EnVar inversion.

**Cost.** $M$ vardax analyses (parallel via `eqx.filter_vmap`). Sample
covariance materialisation O($M N^2$) for full; localised / shrunk
estimators available via `filterax`.

```{code-block} python
:caption: Declares `EnsembleCovariance`, which turns M member analyses into the inflated sample covariance [](#eq-ensemble-cov).
:label: code-design-posterior-ensemble
import equinox as eqx
from jaxtyping import Array, Float

# Shapes: M = ensemble members; N = state size
# today (pipekit_cycle): AnalysisStep
# today (vardax): Posterior
# proposed (D10): Batch, the union of the vardax batch types


# today* (vardax)
class EnsembleCovariance(eqx.Module):
    n_members: int = eqx.field(static=True)
    inflation: float = 1.0
    # proposed (D10)
    localization_radius: float | None = None

    # P*_ens = λ / (M − 1) Σₘ (x⁽ᵐ⁾ − x̄)(x⁽ᵐ⁾ − x̄)ᵀ
    def __call__(
        self,
        analyses: Float[Array, "M N"],
        model: AnalysisStep,
        batch: Batch,
    ) -> Posterior: ...
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

```{code-block} python
:caption: Serialises a `Posterior`, the Gaussian of [](#eq-oi-posterior), into a JSON mark-likelihood for Tier V population models (Decision D10).
:label: code-design-posterior-mark-likelihood
from typing import Any

import equinox as eqx

# Shapes: N = state size; M = ensemble members or flow samples
# today (vardax): Posterior


# today* (vardax)
# ships to_dict with keys mean, cov_diag, samples, provenance, event_metadata
class GaussianMarkLikelihood(eqx.Module):
    """Posterior → mark-likelihood for population models (Tier V, Decision D10)."""

    posterior: Posterior
    # event_id, time, location, instruments_used, …
    event_metadata: dict

    def to_dict(self) -> dict[str, Any]:
        metadata = self.event_metadata
        samples = self.posterior.samples
        # x* as nested lists
        mean = self.posterior.mean.tolist()  # (N,) → list
        # proposed (D10)
        # P* as a diagonal, Cholesky or low-rank record
        cov = self._serialise_cov()  # (N, N) op → dict
        # x⁽ᵐ⁾ as nested lists, or None
        sample_list = None if samples is None else samples.tolist()  # (M, N) → list
        return {
            "event_id": metadata["event_id"],
            "time": metadata["time"],
            "geometry": metadata["geometry"],
            "mean": mean,
            "cov": cov,
            "samples": sample_list,
            "provenance": self.posterior.provenance,
        }

    # proposed (D10)
    def _serialise_cov(self) -> dict[str, Any]:
        """Serialise gaussx AbstractLinearOperator to JSON-compatible form."""
        ...
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

```{code-block} python
:caption: CI tests for the posterior-agreement gate [](#eq-gate-posterior) and the Jacobian gate [](#eq-gate-jacobian) on held-out windows (Decision D12).
:label: code-design-posterior-gate-tests
# tests/test_six_step_validation.py
import equinox as eqx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size; D = observations per window; V = held-out states


def test_posterior_agreement(
    physics_model: eqx.Module,
    emulator_model: eqx.Module,
    val_batches: list[vdx.Batch1D],
) -> None:
    # today (vardax)
    laplace = vdx.LaplaceCovariance()
    for batch in val_batches:
        # today* (vardax)
        # x*_phys = argmin J_phys(x)
        physics_analysis = physics_model(batch)  # → (N,)
        # today* (vardax)
        # x*_em = argmin J_em(x)
        emulator_analysis = emulator_model(batch)  # → (N,)
        # today (vardax)
        # P*_phys = (H′ᵀ R⁻¹ H′ + B⁻¹)⁻¹ at x*_phys
        physics_posterior = laplace(physics_analysis, physics_model, batch)
        # today (vardax)
        # P*_em at x*_em
        emulator_posterior = laplace(emulator_analysis, emulator_model, batch)
        # today (vardax)
        # |x*_em − x*_phys| / σ_post,phys ≤ 1
        vdx.assert_posterior_agreement(
            emulator_posterior, physics_posterior, tolerance_sigma=1.0
        )


def test_adjoint_calibration(
    physics_obs_op: vdx.ObservationOperator,
    emulator_obs_op: vdx.ObservationOperator,
    val_states: Float[Array, "V N"],
) -> None:
    for state in val_states:
        # today (vardax)
        # H′_phys = ∂H_phys/∂x at x
        physics_tangent = physics_obs_op.linearize(state)  # (N,) → (D, N) op
        # today (vardax)
        # H′_em = ∂H_em/∂x at x
        emulator_tangent = emulator_obs_op.linearize(state)  # (N,) → (D, N) op
        # proposed (D12); vdx.assert_adjoint_calibrated ships the probing gate
        # ‖H′_em − H′_phys‖_op, by 20 random-vector probes
        op_norm = estimate_operator_norm_diff(  # → ()
            physics_tangent, emulator_tangent, n_probe=20
        )
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
