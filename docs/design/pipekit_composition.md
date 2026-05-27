---
status: draft
version: 0.4.0
---

# pipekit Composition

Per Decision D8, vardax satisfies `pipekit-cycle` protocols **directly**
— no adapter shim module, no `Abstract*` parallel hierarchy. This doc
shows the satisfaction patterns and the orchestration recipes.

## Protocol satisfaction map

| Pipekit-cycle protocol | Satisfied by |
|---|---|
| `ForwardModel` | `vardax.priors.DynamicalPrior` (wraps any forward); somax / plumax forwards directly |
| `ObservationOperator` | `MaskedIdentity`, `LinearObs`, `AveragingKernel` directly; `MultiInstrumentFusion` via `.to_observation_operator()` (returns per-instrument dicts natively) |
| `AnalysisStep` | All seven Layer 2 model classes via `.as_analysis_step()`: `OptimalInterpolation`, `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`, `IncrementalFourDVar`, `FourDVarNet`, `AmortizedPosterior` |

## `ForwardModel` satisfaction

```python
# somax / plumax forwards satisfy pipekit_cycle.ForwardModel natively.
# vardax.priors.DynamicalPrior wraps any ForwardModel as a Prior for the
# variational cost.

import somax

swm = somax.ShallowWaterModel(grid=grid, params=params)
# swm.step(state, dt) → state ✓
# swm.dt → float ✓
# swm.state_signature → Signature ✓
assert isinstance(swm, ForwardModel)
```

`DynamicalPrior` composes multiple `step()` calls into the variational
$\varphi$:

```python
class DynamicalPrior(eqx.Module):
    forward: ForwardModel
    n_steps: int = eqx.field(static=True)
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, x: Array) -> Array:
        for _ in range(self.n_steps):
            x = self.forward.step(x, self.forward.dt)
        return x
```

The `forward_adjoint` choice controls how gradients flow back through
the rollout when this prior is used inside `FourDVarNet` or as the
dynamics of `StrongFourDVar` / `WeakFourDVar` / `IncrementalFourDVar`.

## `ObservationOperator` satisfaction

Every vardax obs operator implements `__call__` + `linearize`:

```python
class MaskedIdentity(eqx.Module):
    def __call__(self, x: Array, mask: Array | None = None) -> Array: ...
    def linearize(self, x: Array) -> AbstractLinearOperator: ...

assert isinstance(MaskedIdentity(), ObservationOperator)
```

The `linearize` default uses `lineax.JacobianLinearOperator` (autodiff
Jacobian). Operators with structure (averaging kernel, spectral) override
with a structured `gaussx` / `lineax` operator for efficient
tangent-linear application during incremental 4DVar.

## `AnalysisStep` satisfaction

All seven Layer 2 model classes expose `.as_analysis_step()`:

```python
class StrongFourDVar(eqx.Module):
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Array:
        """Training / analysis interface."""
        ...

    def as_analysis_step(self) -> AnalysisStep:
        """Operational interface matching pipekit_cycle.AnalysisStep."""
        return _StrongFourDVarAnalysisStep(self)


class _StrongFourDVarAnalysisStep:
    def __init__(self, model: StrongFourDVar):
        self.model = model

    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        batch = build_batch_from_pipekit_args(forecast, obs, obs_op, obs_err_cov)
        return self.model(batch)
```

The adapter shells the model's `__call__` to match the pipekit-cycle
analysis signature. The training interface stays on the model class.

## Orchestration patterns

### Cycling any model through `pipekit_cycle.DACycle`

```python
import pipekit_cycle as pc
import vardax as vdx

# Pick any of the seven classes — orchestration code is identical:
model = vdx.models.IncrementalFourDVar(
    forward=somax_model,
    obs_op=vdx.obs_operators.AveragingKernel(...),
    prior_mean=x_climatology,
    prior_cov_op=B_op, obs_cov_op=R_op,
    config=vdx.IncrementalConfig(),
)
# or vdx.models.OptimalInterpolation(...) — same orchestration
# or vdx.models.FourDVarNet(...) — same orchestration

da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=model.obs_op,
    analysis_step=model.as_analysis_step(),
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)
result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

### Smoother window for retrospective analysis

```python
smoother = pc.SmootherCycle(
    forward_model=somax_model,
    obs_op=vdx.obs_operators.AveragingKernel(...),
    analysis_step=model.as_analysis_step(),
    window_size=72,           # hours
    window_overlap=12,        # hours
)
trajectory = smoother(initial_state, ...)
```

### Composing operators in a pipekit pipeline

vardax operators are `eqx.Module`, not `pipekit.Operator`. To put them
in a `Sequential` pipeline, wrap them with `pipekit.Lambda` (or
`JaxModelOp` for persisted heads):

```python
import pipekit as pk

pipeline = pk.Sequential([
    georeader_step,                                       # IO
    pk.Lambda(lambda data: build_batch(data)),
    pk.Lambda(lambda batch: model(batch)),
    pk.Lambda(posterior_adapter),
    pk.Lambda(GaussianMarkLikelihood.from_posterior),
    catalog_write_step,
])
```

For trained models that need persistence, use `JaxModelOp`:

```python
from pipekit_jax import JaxModelOp

model_op = JaxModelOp(model)
hash_ = registry.store(model_op, weights=model_op.serialize_weights(), tags={"task": "ssh"})

template = JaxModelOp(fresh_skeleton)
reloaded = template.with_weights(registry.load_weights(hash_))
```

## What vardax does NOT shim

- **Cycle orchestration** — `DACycle`, `SmootherCycle`,
  `EnsembleDACycle`, `WindowedCycle` come from `pipekit-cycle`. Vardax
  does not reimplement them.
- **Stateful operator base class** — `StatefulOperator` + `CarryState`
  come from `pipekit`.
- **Loss / Callback / MetricWriter protocols** — come from
  `pipekit-train`. Vardax `train_step` plugs in via `pipekit_train.Loss`
  adapters.
- **ModelRegistry / ExperimentTracker** — come from `pipekit-experiment`.

## Dependency policy

After the equinox migration (Epic 0), `pipekit` and `pipekit-cycle`
become **required** deps. They have zero third-party dependencies
themselves, so the cost is minimal. The current `fourdvarjax` v0.1.6
package does not yet declare them in `pyproject.toml`; the dependency
policy described here is a v0.4 design target.

`pipekit-jax`, `pipekit-experiment`, `pipekit-train` are planned as
**optional extras** (`vardax[persist]`, `vardax[train]`).

## Testing protocol conformance

The test module below is part of the planned Epic 1 conformance suite
— not yet present in the v0.1.6 codebase.

```python
# tests/test_pipekit_protocols.py    (planned, Epic 1)

import pytest
from pipekit_cycle import ObservationOperator, ForwardModel, AnalysisStep
from vardax.obs_operators import (
    MaskedIdentity, LinearObs, AveragingKernel, MultiInstrumentFusion,
)
from vardax.models import (
    OptimalInterpolation, ThreeDVar, StrongFourDVar, WeakFourDVar,
    IncrementalFourDVar, FourDVarNet, AmortizedPosterior,
)


@pytest.mark.parametrize("obs_op", [
    MaskedIdentity(),
    LinearObs(...),
    AveragingKernel(...),
    MultiInstrumentFusion(...).to_observation_operator(),
])
def test_obs_op_satisfies_protocol(obs_op):
    assert isinstance(obs_op, ObservationOperator)


@pytest.mark.parametrize("model_factory", [
    make_oi, make_3dvar, make_strong_4dvar, make_weak_4dvar,
    make_incremental_4dvar, make_fourdvarnet, make_amortized,
])
def test_model_yields_analysis_step(model_factory):
    model = model_factory()
    step = model.as_analysis_step()
    assert isinstance(step, AnalysisStep)
```
