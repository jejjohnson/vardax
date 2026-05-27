---
status: draft
version: 0.3.0
---

# vardax — API Overview

Complete inventory of the vardax public surface, organised by layer and by
protocol family.

## Layer map

- **[primitives.md](primitives.md)** — Layer 0: pure JAX cost functions, solver
  steps (unrolled / one-step / implicit / incremental), CVT transform,
  Laplace covariance, training primitives.
- **[components.md](components.md)** — Layer 1: protocols (`Prior`,
  `GradModulator`, `CostFunction`, `PosteriorAdapter`) + concrete impls.
  Pipekit-cycle protocol re-exports.
- **[observation_operators.md](observation_operators.md)** — Layer 1
  observation operator family: `MaskedIdentity`, `AveragingKernel`,
  `MultiInstrumentFusion`, `InstrumentRegistry`. `MaskedIdentity` and
  `AveragingKernel` satisfy `pipekit_cycle.ObservationOperator` directly;
  `MultiInstrumentFusion` returns per-instrument `dict` outputs and
  satisfies the protocol via its `.to_observation_operator()` adapter
  (flattening / block-diagonal representation).
- **[models.md](models.md)** — Layer 2: `VarDANet*`, `IncrementalVarDA*`,
  `AmortizedVarDA*`. All expose `.as_analysis_step()`.

## Data types

| Export | Shape | Description |
|---|---|---|
| `Batch1D` | `(B, T, N)` | 1D spatiotemporal (input, mask, target, instrument, obs_err) |
| `Batch2D` | `(B, T, H, W)` | 2D spatiotemporal |
| `Batch2DMultivar` | `(B, T, C, H, W)` | Multivariate 2D |
| `LSTMState1D` / `2D` | — | ConvLSTM hidden/cell state |
| `SolverState1D` / `2D` | — | Inner solver state (x, carry, step) |
| `SolverConfig` | — | n_steps, alpha, prior_weight, grad_mode |
| `IncrementalConfig` | — | n_outer, n_inner, cg_atol, cg_rtol, cvt |
| `AmortizedConfig` | — | head_type, n_samples, temperature |
| `GradMode` | — | Literal `"unrolled" \| "one_step" \| "implicit"` |
| `Posterior` | — | mean, cov (gaussx op), samples, provenance |
| `InstrumentSpec` | — | (A, x_a, h, mask, R) tuple per instrument |
| `InstrumentRegistry` | — | dict[instrument_id, InstrumentSpec] |

All containers are `eqx.Module` (not `NamedTuple`) — proper pytrees with
method support, compatible with `pipekit-jax.JaxModelOp` serialisation.

## Protocols

Re-exported from `pipekit_cycle`:

| Protocol | Method signature |
|---|---|
| `ForwardModel` | `step(state, dt) → state`, `dt` property, `state_signature` property |
| `ObservationOperator` | `__call__(state) → obs`, `linearize(state) → AbstractLinearOperator` |
| `AnalysisStep` | `__call__(forecast, obs, *, obs_op, obs_err_cov) → analysis` |

Vardax-specific:

| Protocol | Method signature |
|---|---|
| `Prior` | `__call__(x) → x_prior` |
| `GradModulator` | `__call__(grad, carry) → (update, new_carry)` |
| `CostFunction` | `__call__(x, batch, **kwargs) → scalar` |
| `PosteriorAdapter` | `__call__(analysis, model, batch) → Posterior` |

## Training utilities

| Export | Scope | Library code? |
|---|---|---|
| `train_step` | Single gradient update through model + correct inner-solver differentiation | **Yes** |
| `eval_step` | Forward pass evaluation (no grad) | **Yes** |
| `reconstruction_loss` | MSE vs. target | **Yes** |
| `train_loss_fn` | Wires model to reconstruction loss with correct propagation | **Yes** |
| `fit` | Full training loop with history | **No — example only** |

`train_step` plugs into `pipekit_train.TrainingLoop` via the `pipekit-train`
`Loss` protocol (optional `[train]` extra).

## Posterior utilities (Layer 1)

| Export | Cost | UQ quality |
|---|---|---|
| `LaplaceCovariance` | Cheap — one Hessian-vector product family at MAP | Gaussian-likelihood-only, exact-at-MAP |
| `GaussNewtonHessian` | Mid — Krylov / Lanczos via `lineax.CG` | Exact-at-MAP, structured |
| `EnsembleCovariance` | Expensive — delegates to `filterax` | Non-Gaussian-aware, flow-dependent |
| `GaussianMarkLikelihood` | Free — serialiser only | Export to population models (Tier V) |

## Demo utilities (`vardax._src.utils`, not library API)

| Category | Exports |
|---|---|
| Dynamical systems | `simulate_lorenz63`, `simulate_lorenz96`, `Lorenz63`, `Lorenz96` |
| Visualisation | `plot_3d_attractor`, `plot_state_grid`, `plot_reconstruction_comparison`, `plot_l96_*` |
| Data pipeline | `trajectory_to_xr_dataset`, `extract_patches`, `xr_to_batch1d` |
| Masks | `random_mask`, `regular_mask`, `feature_mask` |
| Noise | `add_gaussian_noise` |
| Standardisation | `compute_scaler_params`, `apply_standardization`, `inverse_standardization` |
| Validation | `assert_posterior_agreement`, `assert_adjoint_calibrated`, `simulation_based_calibration` |

These support tutorial notebooks and validation gates (Decision D12). The
dynamical system simulators (L63, L96) are demos — production forwards come
from `somax` / `plumax`.

## Import conventions

```python
# Protocols (re-exports from pipekit-cycle + vardax-specific additions)
from vardax.protocols import (
    ForwardModel, ObservationOperator, AnalysisStep,  # from pipekit-cycle
    Prior, GradModulator, CostFunction, PosteriorAdapter,  # vardax
)

# Layer 1 — Components
from vardax.priors import BilinAEPrior, ConvAEPrior, MLPAEPrior, IdentityPrior, DynamicalPrior
from vardax.obs_operators import (
    MaskedIdentity, AveragingKernel, MultiInstrumentFusion, InstrumentRegistry,
)
from vardax.costs import variational_cost, obs_cost, prior_cost, incremental_cost
from vardax.grad_mod import ConvLSTMGradMod1D, ConvLSTMGradMod2D, MLPGradMod, IdentityGradMod
from vardax.posterior import (
    LaplaceCovariance, GaussNewtonHessian, EnsembleCovariance, GaussianMarkLikelihood,
)

# Layer 2 — Models
from vardax.models import VarDANet1D, VarDANet2D, IncrementalVarDA2D, AmortizedVarDA
from vardax import SolverConfig, IncrementalConfig, AmortizedConfig, Batch2D, Posterior

# Training
from vardax.training import train_step, eval_step, reconstruction_loss

# Pipekit composition (optional extras)
import pipekit as pk
import pipekit_cycle as pc

# Persistence (optional [persist] extra)
from vardax.persist import save, load
```
