---
status: draft
version: 0.4.0
---

# vardax — API Overview

Complete inventory of the vardax public surface, organised by layer and
by protocol family. v0.4 changes the Layer 2 surface to seven peer
classes (Decision D14) and replaces the `grad_mode` enum with adjoint
slots from `optimistix` / `diffrax` (Decision D15).

## Layer map

- **[primitives.md](primitives.md)** — Layer 0: pure JAX cost functions,
  closed-form BLUE, CVT, Laplace covariance, adjoint passthrough.
- **[components.md](components.md)** — Layer 1: protocols (`Prior`,
  `GradModulator`, `CostFunction`, `PosteriorAdapter`, `Minimiser`) +
  concrete implementations. Pipekit-cycle protocol re-exports.
- **[observation_operators.md](observation_operators.md)** — Layer 1
  observation operator family: `MaskedIdentity`, `LinearObs`,
  `AveragingKernel`, `MultiInstrumentFusion`, `InstrumentRegistry`.
- **[models.md](models.md)** — Layer 2: the seven peer analysis classes.

## Layer 2 — Models

| Class | Method | Learnable? |
|---|---|---|
| `OptimalInterpolation` | BLUE / OI (closed-form) | No |
| `ThreeDVar` | 3D variational | No |
| `StrongFourDVar` | 4DVar, control = $x_0$ | No |
| `WeakFourDVar` | 4DVar, control = $(x_0, \boldsymbol{\eta})$ | No |
| `IncrementalFourDVar` | GN + CG + CVT (operational) | No |
| `FourDVarNet` | Learned 4DVar | Yes |
| `AmortizedPosterior` | Direct $q_\phi(x \mid y)$ head | Yes |

All seven implement `.as_analysis_step()` returning a
`pipekit_cycle.AnalysisStep`-compliant callable.

## Data types

| Export | Shape | Description |
|---|---|---|
| `Batch1D` | `(B, T, N)` | 1D spatiotemporal (input, mask, target, instrument, obs_err) |
| `Batch2D` | `(B, T, H, W)` | 2D spatiotemporal |
| `Batch2DMultivar` | `(B, T, C, H, W)` | Multivariate 2D |
| `Batch3D` | `(B, T, D, H, W)` | Volumetric (planned) |
| `LSTMState1D` / `2D` | — | ConvLSTM hidden / cell state (FourDVarNet only) |
| `SolverConfig` | — | n_steps, alpha, prior_weight (FourDVarNet) |
| `IncrementalConfig` | — | n_outer, n_inner, cg_atol, cg_rtol, cvt |
| `AmortizedConfig` | — | head_type, n_samples, temperature |
| `Posterior` | — | mean, cov (gaussx op), samples, provenance |
| `InstrumentSpec` | — | (obs_op, mask, obs_cov_op, instrument_id) |
| `InstrumentRegistry` | — | dict[instrument_id, InstrumentSpec] |

Removed in v0.4: `GradMode` (replaced by adjoint constructor slots).

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
| `GradModulator` | `__call__(grad, carry) → (update, new_carry)` (FourDVarNet only) |
| `CostFunction` | `__call__(x, batch, **kwargs) → scalar` |
| `PosteriorAdapter` | `__call__(analysis, model, batch) → Posterior` |
| `Minimiser` | wrapper around `optimistix.AbstractMinimiser` |

## Adjoint slots (v0.4 — Decision D15)

Models that involve dynamics or inner minimisation carry adjoint
constructor slots passed straight through to the upstream library:

| Slot | Type | Used by | Default |
|---|---|---|---|
| `checkpoint` | `bool \| int` | `StrongFourDVar`, `WeakFourDVar`, `IncrementalFourDVar`, `ReducedBasisFourDVar` | `False` |
| `forward_adjoint` | `diffrax.AbstractAdjoint`, applied via `forward.with_adjoint` | the same four | `None` (forward's own) |
| `minimiser_adjoint` | `optimistix.AbstractAdjoint` | `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar` | `ImplicitAdjoint()` |
| `solver_adjoint` | `optimistix.AbstractAdjoint` | `FourDVarNet` (through the learned inner solver) | `RecursiveCheckpointAdjoint()` |

The Bolte 2023 one-step method appears as
`vardax.adjoints.OneStepAdjoint`, an
`optimistix.AbstractAdjoint` subclass targeting upstream contribution.

## Training utilities

| Export | Scope | Library code? |
|---|---|---|
| `train_step` | Single gradient update through model + correct adjoint flow | **Yes** |
| `eval_step` | Forward pass evaluation (no grad) | **Yes** |
| `reconstruction_loss` | MSE vs. target | **Yes** |
| `train_loss_fn` | Wires model to reconstruction loss | **Yes** |
| `fit` | Full training loop with history | **No — example only** |

Only `FourDVarNet` and `AmortizedPosterior` use these (the classical
methods are non-learnable).

## Posterior utilities (Layer 1)

| Export | Cost | UQ quality |
|---|---|---|
| `LaplaceCovariance` | Cheap — one Hessian-vector product family at MAP | Gaussian-likelihood-only, exact-at-MAP |
| `GaussNewtonHessian` | Mid — Krylov / Lanczos via `lineax.CG` | Exact-at-MAP, structured |
| `EnsembleCovariance` | Expensive — delegates to `filterax` | Non-Gaussian-aware, flow-dependent |
| `GaussianMarkLikelihood` | Free — serialiser only | Export to population models |

`OptimalInterpolation.posterior(batch)` and
`IncrementalFourDVar.posterior(batch)` are closed-form / reused-Hessian
fast paths that don't need an adapter call.

## Demo utilities (`vardax._src.utils`, not library API)

| Category | Exports |
|---|---|
| Dynamical systems | `simulate_lorenz63`, `simulate_lorenz96`, `Lorenz63`, `Lorenz96` |
| Visualisation | `plot_3d_attractor`, `plot_state_grid`, `plot_reconstruction_comparison`, … |
| Data pipeline | `trajectory_to_xr_dataset`, `extract_patches`, `xr_to_batch1d` |
| Masks | `random_mask`, `regular_mask`, `feature_mask` |
| Noise | `add_gaussian_noise` |
| Standardisation | `compute_scaler_params`, `apply_standardization`, `inverse_standardization` |
| Validation | `assert_posterior_agreement`, `assert_adjoint_calibrated`, `simulation_based_calibration` |

## Import conventions

```{code-block} python
:caption: Lists the target import path of every public name, layer by layer, for the layout in [the target package structure](../architecture.md#package-structure-target).
:label: code-design-api-imports
# Protocols
# today (pipekit_cycle); also re-exported today as vdx.<name>
from pipekit_cycle import AnalysisStep, ForwardModel, ObservationOperator

# proposed (architecture.md); every name ships today as vdx.<name>
from vardax.protocols import (
    CostFunction,
    GradModulator,
    Minimiser,
    PosteriorAdapter,
    Prior,
)

# Layer 1: Components
# proposed (architecture.md); both names ship today as vdx.<name>
from vardax.priors import DynamicalPrior, IdentityPrior

# proposed (architecture.md); ship today only as vdx.BilinAEPrior1D/2D, …
from vardax.priors import BilinAEPrior, ConvAEPrior, MLPAEPrior

# proposed (architecture.md); every name ships today as vdx.<name>
from vardax.obs_operators import (
    AveragingKernel,
    InstrumentRegistry,
    LinearObs,
    MaskedIdentity,
    MultiInstrumentFusion,
)

# proposed (architecture.md); both names ship today as vdx.<name>
from vardax.costs import prior_cost, variational_cost

# proposed (architecture.md); obs_cost ships today only as vdx.obs_cost_1d / _2d
# blue_analysis is the closed form for OptimalInterpolation
from vardax.costs import (
    blue_analysis,
    incremental_cost,
    model_error_cost,
    obs_cost,
    threedvar_cost,
)

# proposed (architecture.md); both names ship today as vdx.<name>
from vardax.grad_mod import ConvLSTMGradMod1D, ConvLSTMGradMod2D

# proposed (architecture.md); not shipped
from vardax.grad_mod import IdentityGradMod, MLPGradMod

# proposed (architecture.md); every name ships today as vdx.<name>
from vardax.posterior import (
    EnsembleCovariance,
    GaussianMarkLikelihood,
    GaussNewtonHessian,
    LaplaceCovariance,
)

# Layer 2: Models (seven peer classes)
# proposed (architecture.md); all but FourDVarNet ship today as vdx.<name>,
# FourDVarNet ships as vdx.FourDVarNet1D / FourDVarNet2D
from vardax.models import (
    AmortizedPosterior,
    FourDVarNet,
    IncrementalFourDVar,
    OptimalInterpolation,
    StrongFourDVar,
    ThreeDVar,
    WeakFourDVar,
)

# Configs + containers
# today (vardax)
from vardax import (
    AmortizedConfig,
    Batch1D,
    Batch2D,
    Batch2DMultivar,
    IncrementalConfig,
    Posterior,
)

# proposed (architecture.md); not shipped
from vardax import SolverConfig

# Training
# proposed (architecture.md); every name ships today as vdx.<name>
from vardax.training import eval_step, reconstruction_loss, train_step

# Adjoints: vardax-owned, otherwise import from optimistix / diffrax
# today (vardax)
from vardax.adjoints import OneStepAdjoint

# today (optimistix)
import optimistix as optx

# today (diffrax)
import diffrax as dfx

# Pipekit composition (required)
# today (pipekit)
import pipekit as pk

# today (pipekit_cycle)
import pipekit_cycle as pc

# Persistence (optional [persist] extra)
# proposed (D13); vardax.persist is not shipped
from vardax.persist import load, save
```
