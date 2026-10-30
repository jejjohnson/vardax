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

```{code-block} python
:caption: Checks that a somax forward, the M_t of the rollout [](#eq-rollout), satisfies `pipekit_cycle.ForwardModel` natively.
:label: code-design-pipekit-forward-model
# somax / plumax forwards satisfy pipekit_cycle.ForwardModel natively.
# vardax.priors.DynamicalPrior wraps any ForwardModel as a Prior for the
# variational cost.
import pipekit_cycle as pc
import somax

# not pinned (somax)
# xₜ₊₁ = M(xₜ), one step of the shallow-water model
swm = somax.ShallowWaterModel(grid=grid, params=params)
# swm.step(state, dt) → state ✓
# swm.dt → float ✓
# swm.state_signature → Signature ✓
# today (pipekit_cycle)
assert isinstance(swm, pc.ForwardModel)
```

`DynamicalPrior` composes multiple `step()` calls into the variational
$\varphi$:

```{code-block} python
:caption: Declares the target `DynamicalPrior`, which composes `n_steps` forward steps of [](#eq-rollout) into the prior map φ of [](#eq-4dvarnet-cost).
:label: code-design-pipekit-dynamical-prior
import diffrax as dfx
import equinox as eqx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N = state size


# today* (vardax); ships wrapping an ODE right-hand side as
# DynamicalPrior(model, params, solver, stepsize, adjoint), and
# .as_forward_model(dt) adapts it to pc.ForwardModel (D18)
class DynamicalPrior(eqx.Module):
    # today (pipekit_cycle)
    forward: pc.ForwardModel
    n_steps: int = eqx.field(static=True)
    # today (diffrax)
    forward_adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        for _ in range(self.n_steps):
            # today (pipekit_cycle)
            # x ← M(x), over one step of length dt
            x = self.forward.step(x, self.forward.dt)  # (N,) → (N,)
        return x
```

The `forward_adjoint` choice controls how gradients flow back through
the rollout when this prior is used inside `FourDVarNet` or as the
dynamics of `StrongFourDVar` / `WeakFourDVar` / `IncrementalFourDVar`.

## `ObservationOperator` satisfaction

Every vardax obs operator implements `__call__` + `linearize`:

```{code-block} python
:caption: Declares the observation-operator interface for the masked identity [](#eq-masked-identity) and its tangent-linear [](#eq-tlm-obs).
:label: code-design-pipekit-obs-operator
import equinox as eqx
import lineax as lx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N = state size


# today (vardax)
class MaskedIdentity(eqx.Module):
    # H(x) = m ⊙ x
    def __call__(
        self, x: Float[Array, " N"], mask: Float[Array, " N"] | None = None
    ) -> Float[Array, " N"]: ...

    # H′ = ∂H/∂x at x, as a lineax operator
    def linearize(self, x: Float[Array, " N"]) -> lx.AbstractLinearOperator: ...


# today (pipekit_cycle)
assert isinstance(MaskedIdentity(), pc.ObservationOperator)
```

The `linearize` default uses `lineax.JacobianLinearOperator` (autodiff
Jacobian). Operators with structure (averaging kernel, spectral) override
with a structured `gaussx` / `lineax` operator for efficient
tangent-linear application during incremental 4DVar.

## `AnalysisStep` satisfaction

All seven Layer 2 model classes expose `.as_analysis_step()`:

```{code-block} python
:caption: Declares how `StrongFourDVar`, which minimises [](#eq-strong4dvar-cost), keeps its training `__call__` and exposes the pipekit-cycle analysis interface through `.as_analysis_step()`.
:label: code-design-pipekit-analysis-step
import diffrax as dfx
import equinox as eqx
import lineax as lx
import optimistix as optx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N = state size; D = observations per time; T = times in the window
# proposed (D8): Batch, the union of the vardax batch types


# today* (vardax); vardax ships forward_adjoint default None and minimiser
# default None (resolved to optx.BFGS(rtol=1e-6, atol=1e-6))
class StrongFourDVar(eqx.Module):
    # today (pipekit_cycle)
    forward: pc.ForwardModel
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    # x_b
    prior_mean: Float[Array, " N"]
    # B
    prior_cov_op: lx.AbstractLinearOperator
    # R
    obs_cov_op: lx.AbstractLinearOperator
    minimiser: optx.AbstractMinimiser
    # today (optimistix)
    minimiser_adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()
    # today (diffrax)
    forward_adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()

    # x₀* = argmin J(x₀)
    def __call__(self, batch: Batch) -> Float[Array, " N"]:
        """Training / analysis interface."""
        ...

    # today (vardax)
    def as_analysis_step(self) -> pc.AnalysisStep:
        """Operational interface matching pipekit_cycle.AnalysisStep."""
        return _StrongFourDVarAnalysisStep(self)


# today* (vardax); ships as an eqx.Module
class _StrongFourDVarAnalysisStep:
    def __init__(self, model: StrongFourDVar) -> None:
        self.model = model

    def __call__(
        self,
        forecast: Float[Array, " N"],
        obs: Float[Array, "T D"],
        *,
        obs_op: pc.ObservationOperator,
        obs_err_cov: lx.AbstractLinearOperator,
    ) -> Float[Array, " N"]:
        # proposed (D8)
        batch = build_batch_from_pipekit_args(forecast, obs, obs_op, obs_err_cov)
        # today* (vardax)
        # x₀* = argmin J(x₀)
        return self.model(batch)
```

The adapter shells the model's `__call__` to match the pipekit-cycle
analysis signature. The training interface stays on the model class.

## Orchestration patterns

### Cycling any model through `pipekit_cycle.DACycle`

```{code-block} python
:caption: Cycles an `IncrementalFourDVar`, which solves [](#eq-incremental-cost) in each window, through `pipekit_cycle.DACycle`; any of the seven classes drops in the same way.
:label: code-design-pipekit-da-cycle
import pipekit_cycle as pc
import vardax as vdx

# Shapes: N = state size; D = observations per time

# proposed (architecture.md); ships today as vdx.AveragingKernel
# ŷ = A(h ⊙ x + (1 − h) ⊙ x_a)
obs_op = vdx.obs_operators.AveragingKernel(...)  # (N,) → (D,)
# today (vardax)
config = vdx.IncrementalConfig()
# Pick any of the seven classes; orchestration code is identical.
# proposed (architecture.md); ships today as vdx.IncrementalFourDVar
# Gauss-Newton outer loop, CG inner solve of δJ(δx)
model = vdx.models.IncrementalFourDVar(
    forward=somax_model,
    obs_op=obs_op,
    prior_mean=climatology,
    prior_cov_op=background_cov,
    obs_cov_op=obs_cov,
    config=config,
)
# or vdx.models.OptimalInterpolation(...), same orchestration
# or vdx.models.FourDVarNet(...), same orchestration

# today (vardax)
analysis_step = model.as_analysis_step()
# today (pipekit_cycle)
da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=model.obs_op,
    analysis_step=analysis_step,
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)
# today (pipekit_cycle)
cycle_state = pc.DAState(t=0.0, cycle_count=0)
# today (pipekit_cycle)
# forecast xₜᶠ = M(xₜ₋₁ᵃ), then analysis xₜᵃ, for every window
result, final_state = da_cycle(initial_state, cycle_state)
```

### Smoother window for retrospective analysis

```{code-block} python
:caption: Runs the analysis step over overlapping windows with `pipekit_cycle.SmootherCycle` for a retrospective reanalysis.
:label: code-design-pipekit-smoother
import pipekit_cycle as pc
import vardax as vdx

# Shapes: N = state size; D = observations per time

# proposed (architecture.md); ships today as vdx.AveragingKernel
# ŷ = A(h ⊙ x + (1 − h) ⊙ x_a)
obs_op = vdx.obs_operators.AveragingKernel(...)  # (N,) → (D,)
# today (vardax)
analysis_step = model.as_analysis_step()
# today (pipekit_cycle)
smoother = pc.SmootherCycle(
    forward_model=somax_model,
    obs_op=obs_op,
    analysis_step=analysis_step,
    # forecast steps per window
    window=72,
    # step between window starts (12-step overlap)
    stride=60,
)
# today* (pipekit_cycle)
trajectory = smoother(initial_state, ...)
```

### Composing operators in a pipekit pipeline

vardax operators are `eqx.Module`, not `pipekit.Operator`. To put them
in a `Sequential` pipeline, wrap them with `pipekit.Lambda` (or
`JaxModelOp` for persisted heads):

```{code-block} python
:caption: Wraps vardax operators in `pipekit.Lambda` to chain IO, the analysis, the posterior [](#eq-laplace-cov) and its export in one `Sequential` pipeline.
:label: code-design-pipekit-pipeline
import pipekit as pk
import vardax as vdx

# today (pipekit); georeader_step and catalog_write_step are user operators
pipeline = pk.Sequential(
    [
        # IO
        georeader_step,
        # today (pipekit); build_batch is user
        pk.Lambda(lambda data: build_batch(data)),
        # today (pipekit)
        # x* = argmin J(x)
        pk.Lambda(lambda batch: model(batch)),
        # today (pipekit)
        # P* at x*
        pk.Lambda(posterior_adapter),
        # proposed (D10); GaussianMarkLikelihood ships without from_posterior
        pk.Lambda(vdx.GaussianMarkLikelihood.from_posterior),
        catalog_write_step,
    ]
)
```

For trained models that need persistence, use `JaxModelOp`:

```{code-block} python
:caption: Persists a trained model with `JaxModelOp` and reloads its weights from the `ModelRegistry` (Decision D13).
:label: code-design-pipekit-persistence
import pipekit_jax

# not pinned (pipekit_jax)
model_op = pipekit_jax.JaxModelOp(model)
# not pinned (pipekit_jax)
weights = model_op.serialize_weights()
# not pinned (pipekit_experiment)
model_hash = registry.store(model_op, weights=weights, tags={"task": "ssh"})

# not pinned (pipekit_jax)
template = pipekit_jax.JaxModelOp(fresh_skeleton)
# not pinned (pipekit_experiment)
stored_weights = registry.load_weights(model_hash)
# not pinned (pipekit_jax)
reloaded = template.with_weights(stored_weights)
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
themselves, so the cost is minimal. The current v0.1.x package
(previously published as `fourdvarjax`) does not yet declare them in
`pyproject.toml`; the dependency policy described here is a v0.4
design target.

`pipekit-jax`, `pipekit-experiment`, `pipekit-train` are planned as
**optional extras** (`vardax[persist]`, `vardax[train]`).

## Testing protocol conformance

The test module below is part of the planned Epic 1 conformance suite
— not yet present in the v0.1.6 codebase.

```{code-block} python
:caption: Planned conformance tests that every vardax observation operator and every Layer 2 model satisfy the pipekit-cycle protocols (Decision D8).
:label: code-design-pipekit-conformance-tests
# tests/test_pipekit_protocols.py    (planned, Epic 1)
from collections.abc import Callable

import pipekit_cycle as pc
import pytest
import vardax as vdx

# proposed (architecture.md): vdx.obs_operators and vdx.models paths;
#   every name below ships today as vdx.<name> except FourDVarNet
#   (vdx.FourDVarNet1D / FourDVarNet2D)
# user: make_oi, make_3dvar, …, make_amortized model factories


@pytest.mark.parametrize(
    "obs_op",
    [
        # proposed (architecture.md)
        vdx.obs_operators.MaskedIdentity(),
        # proposed (architecture.md)
        vdx.obs_operators.LinearObs(...),
        # proposed (architecture.md)
        vdx.obs_operators.AveragingKernel(...),
        # proposed (architecture.md); .to_observation_operator() ships today
        vdx.obs_operators.MultiInstrumentFusion(...).to_observation_operator(),
    ],
)
def test_obs_op_satisfies_protocol(obs_op: pc.ObservationOperator) -> None:
    # today (pipekit_cycle)
    assert isinstance(obs_op, pc.ObservationOperator)


@pytest.mark.parametrize(
    "model_factory",
    [
        make_oi,
        make_3dvar,
        make_strong_4dvar,
        make_weak_4dvar,
        make_incremental_4dvar,
        make_fourdvarnet,
        make_amortized,
    ],
)
def test_model_yields_analysis_step(model_factory: Callable) -> None:
    # user
    model = model_factory()
    # today (vardax)
    step = model.as_analysis_step()
    # today (pipekit_cycle)
    assert isinstance(step, pc.AnalysisStep)
```
