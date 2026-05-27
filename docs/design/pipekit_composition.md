---
status: draft
version: 0.3.0
---

# pipekit Composition

Per Decision D8, vardax satisfies `pipekit-cycle` protocols **directly** — no
adapter shim module, no `Abstract*` parallel hierarchy. This doc shows the
satisfaction patterns and the orchestration recipes.

## Protocol satisfaction map

| Pipekit-cycle protocol | Satisfied by |
|---|---|
| `ForwardModel` | `vardax.priors.DynamicalPrior` (wraps any forward); somax / plumax forwards directly |
| `ObservationOperator` | Every class in `vardax.obs_operators.*` |
| `AnalysisStep` | `VarDANet*.as_analysis_step()`, `IncrementalVarDA*.as_analysis_step()`, `AmortizedVarDA*.as_analysis_step()` |

## `ForwardModel` satisfaction

```python
# vardax.priors.DynamicalPrior wraps any pipekit_cycle.ForwardModel as a Prior.
# Conversely, somax / plumax forwards satisfy ForwardModel natively:

import somax
swm = somax.ShallowWaterModel(grid=grid, params=params)
# swm.step(state, dt) → state ✓
# swm.dt → float ✓
# swm.state_signature → Signature ✓
assert isinstance(swm, ForwardModel)  # passes
```

vardax does not own forward models. The `DynamicalPrior` wrapper just composes
multiple `step()` calls into the variational $\varphi$:

```python
class DynamicalPrior(eqx.Module):
    forward: ForwardModel
    n_steps: int = eqx.field(static=True)

    def __call__(self, x: Array) -> Array:
        for _ in range(self.n_steps):
            x = self.forward.step(x, self.forward.dt)
        return x
```

## `ObservationOperator` satisfaction

Every vardax obs operator (Layer 1) implements both methods:

```python
class MaskedIdentity(eqx.Module):
    def __call__(self, x: Array, mask: Array | None = None) -> Array: ...
    def linearize(self, x: Array) -> AbstractLinearOperator: ...

# At construction:
assert isinstance(MaskedIdentity(), ObservationOperator)
```

The `linearize` default uses `lineax.JacobianLinearOperator` (autodiff
Jacobian). Operators with structure (AK, spectral) override with a
structured `gaussx` / `lineax` operator for efficient tangent-linear
application during incremental 4DVar.

## `AnalysisStep` satisfaction

vardax models satisfy `AnalysisStep` via an explicit adapter method:

```python
class VarDANet2D(eqx.Module):
    prior: Prior
    obs_op: ObservationOperator
    grad_mod: GradModulator
    config: SolverConfig

    def __call__(self, batch: Batch2D) -> Array:
        """Training interface (with target available)."""
        ...

    def as_analysis_step(self) -> AnalysisStep:
        """Operational interface (no target needed)."""
        return _VarDANetAnalysisStep(self)


class _VarDANetAnalysisStep:
    def __init__(self, model: VarDANet2D):
        self.model = model

    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        batch = Batch2D(
            input=obs,
            mask=jnp.where(jnp.isfinite(obs), 1.0, 0.0),
            target=None,
            obs_err=jnp.sqrt(jnp.diag(obs_err_cov)) if obs_err_cov.ndim == 2 else obs_err_cov,
        )
        return self.model(batch)
```

The model's `__call__(batch)` interface is kept for training. The
`as_analysis_step()` adapter exposes the same algorithm to
`pipekit_cycle.DACycle` and friends.

## Orchestration patterns

### Cycling a 4DVarNet over many assimilation windows

```python
import pipekit_cycle as pc
import vardax as vdx

# Build the model
model = vdx.models.VarDANet2D(
    prior=vdx.priors.BilinAEPrior2D(latent_dim=64, n_time=10, height=128, width=128),
    obs_op=vdx.obs_operators.MaskedIdentity(),
    grad_mod=vdx.grad_mod.ConvLSTMGradMod2D(hidden_dim=64),
    config=vdx.SolverConfig(n_steps=15, grad_mode="one_step"),
)

# Train it (separate)
# ...

# Operational cycling:
da_cycle = pc.DACycle(
    forward_model=somax_ssh_model,
    obs_op=vdx.obs_operators.MaskedIdentity(),
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

vardax operators are `eqx.Module` not `pipekit.Operator`, so to put them in
a `Sequential` pipeline they're wrapped with `pipekit.Lambda` (or `JaxModelOp`
for persisted heads):

```python
import pipekit as pk

pipeline = pk.Sequential([
    georeader_step,                    # IO: load satellite + met
    pk.Lambda(lambda data: build_batch(data)),
    pk.Lambda(lambda batch: model(batch)),
    pk.Lambda(posterior_adapter),
    pk.Lambda(GaussianMarkLikelihood.from_posterior),
    catalog_write_step,                # write posterior to GeoCatalog
])
```

For trained models that need persistence, use `JaxModelOp`:

```python
from pipekit_jax import JaxModelOp

# Wrap for registry:
model_op = JaxModelOp(model)
hash_ = registry.store(model_op, weights=model_op.serialize_weights(), tags={"task": "ssh"})

# Reload:
template = JaxModelOp(fresh_skeleton)
reloaded = template.with_weights(registry.load_weights(hash_))
```

## What vardax does NOT shim

- **Cycle orchestration.** `DACycle`, `SmootherCycle`, `EnsembleDACycle`,
  `WindowedCycle` come from `pipekit-cycle`. Vardax does not reimplement
  them.
- **Stateful operator base class.** `StatefulOperator` + `CarryState` come
  from `pipekit`. If vardax needs custom carry state (rare — the existing
  `LSTMState*` is internal to ConvLSTM), subclass `CarryState`.
- **Loss / Callback / MetricWriter protocols.** Come from `pipekit-train`.
  Vardax `train_step` plugs in via `pipekit_train.Loss` adapters.
- **ModelRegistry / ExperimentTracker.** Come from `pipekit-experiment`.

## Dependency policy

`pipekit` and `pipekit-cycle` are **required** deps of vardax. They have
zero third-party dependencies themselves, so the cost is minimal.

`pipekit-jax`, `pipekit-experiment`, `pipekit-train` are **optional extras**
(`vardax[persist]`, `vardax[train]`).

## Testing protocol conformance

```python
# tests/test_pipekit_protocols.py

import pytest
from pipekit_cycle import ObservationOperator, ForwardModel, AnalysisStep
from vardax.obs_operators import MaskedIdentity, AveragingKernel, MultiInstrumentFusion
from vardax.models import VarDANet2D, IncrementalVarDA2D, AmortizedVarDA


@pytest.mark.parametrize("obs_op", [
    MaskedIdentity(),
    AveragingKernel(...),
    MultiInstrumentFusion(...).to_observation_operator(),
])
def test_obs_op_satisfies_protocol(obs_op):
    assert isinstance(obs_op, ObservationOperator)


@pytest.mark.parametrize("model_factory", [
    make_vardanet_2d,
    make_incremental_2d,
    make_amortized,
])
def test_model_yields_analysis_step(model_factory):
    model = model_factory()
    step = model.as_analysis_step()
    assert isinstance(step, AnalysisStep)
```
