---
status: draft
version: 0.4.0
---

# Ecosystem Integration

Composition with somax, plumax, gaussx, filterax, pipekit-cycle,
pipekit-experiment, georeader, coordax, GeoCatalog.

---

## somax — Geophysical forward as dynamical prior

```python
import somax
from vardax.priors import DynamicalPrior

# somax forwards satisfy pipekit_cycle.ForwardModel natively
swm = somax.ShallowWaterModel(grid=grid, params=params)

# Wrap as a Prior for the variational cost
prior = DynamicalPrior(forward=swm, n_steps=10)

# Plug into FourDVarNet
model = FourDVarNet(
    prior=prior,
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod2D(hidden_dim=64),
    config=SolverConfig(n_steps=15),
)

# Or into IncrementalFourDVar (no prior wrap — forward is used directly)
incremental = IncrementalFourDVar(
    forward=swm,                  # used as the dynamics M_t
    obs_op=MaskedIdentity(),
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    config=IncrementalConfig(),
)
```

---

## plumax — Methane forward chain (Tier I)

Decision D7: vardax does not own methane forwards. `plumax` provides the
forward chain (Tier I Gaussian plume → Tier IV coupled RTM). vardax does
the inversion.

```python
import plumax
from vardax.priors import DynamicalPrior
from vardax.obs_operators import AveragingKernel, MultiInstrumentFusion
from vardax.models import IncrementalFourDVar

# Forward: parameters Q, x_0, u, θ → predicted XCH4 enhancement
plume_fwd = plumax.tier1.GaussianPlume(met=met_field, dispersion="MO")

# Observation: averaging kernel from TROPOMI L2 metadata
tropomi_ak = AveragingKernel.from_l2(tropomi_l2_ds)

# Multi-instrument fusion (per-instrument bias as joint state — Epic 9)
fusion = MultiInstrumentFusion(
    registry=InstrumentRegistry.from_l2_dict({
        "TROPOMI": tropomi_l2_ds,
        "EMIT": emit_l2_ds,
        "GHGSat": ghgsat_l2_ds,
    }),
)

# Invert
inversion = IncrementalFourDVar(
    forward=plume_fwd,
    obs_op=fusion,
    prior_mean=Q_prior_mean,
    prior_cov_op=gx.LogNormalLinearOperator(Q_prior_mu, Q_prior_sigma_log),
    obs_cov_op=lx.BlockDiagonalLinearOperator([
        spec.R_op for spec in fusion.registry.entries.values()
    ]),
    config=IncrementalConfig(n_outer=3, n_inner=20),
)

x_star = inversion(batch)
posterior = GaussNewtonHessian()(x_star, inversion.as_analysis_step(), batch)
```

---

## gaussx — Structured prior covariance

Decision D11: incremental 4DVar uses the control-variable transform with a
`gaussx.MaternLinearOperator` factorisation of $B$.

```python
import gaussx as gx

# Matérn-3/2 on a regular grid — supports Kronecker structure
B_op = gx.MaternLinearOperator(
    grid_coords=coords,
    length_scale=10.0,         # km — basin-dependent
    nu=1.5,                     # Matérn-3/2
    sigma=1.0,
)

# Returns B^{1/2} for CVT
B_half = B_op.half()

# Or a Kronecker product (separable in lon/lat):
B_kron = gx.KroneckerLinearOperator([
    gx.Matern1DLinearOperator(coords=lon, length_scale=5.0),
    gx.Matern1DLinearOperator(coords=lat, length_scale=3.0),
])
```

For non-separable, structured priors, build from `LowRankUpdate`,
`BlockDiag`, or factorise via Lanczos.

---

## filterax — Hybrid ensemble-variational (Epic 9)

```python
import filterax as fx
from vardax.posterior import EnsembleCovariance
from vardax.models import IncrementalFourDVar

# Ensemble forecast (filterax owns this)
ensemble_forecast = fx.EnsembleForecast(
    forward=somax_model,
    n_members=32,
    initial_perturbations=jax.random.normal(key, (32, *state_shape)),
)
ensemble_states = ensemble_forecast(x_b)

# Hybrid B: blend climatological + ensemble
B_hybrid = 0.5 * B_climatological + 0.5 * fx.ensemble_covariance(ensemble_states)

# Run vardax inversion with hybrid B
model = IncrementalFourDVar(
    forward=somax_model,
    obs_op=fusion,
    prior_mean=ensemble_states.mean(0),
    prior_cov_op=B_hybrid,
    obs_cov_op=R_op,
    config=IncrementalConfig(),
)
x_star = model(batch)

# Posterior from the ensemble itself
posterior = EnsembleCovariance(n_members=32)(ensemble_states, model.as_analysis_step(), batch)
```

---

## pipekit-cycle — Operational DA cycling

```python
import pipekit_cycle as pc
import vardax as vdx

# Build the model once
model = vdx.models.IncrementalFourDVar(
    forward=somax_model,
    obs_op=vdx.obs_operators.AveragingKernel(...),
    prior_mean=x_climatology,
    prior_cov_op=B_op, obs_cov_op=R_op,
    config=vdx.IncrementalConfig(),
)

# Cycle it
da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=model.obs_op,
    analysis_step=model.as_analysis_step(),
    obs_source=load_satellite_op,
    n_steps=n_assimilation_windows,
)
result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

---

## pipekit-experiment + pipekit-jax — Trained model persistence

```python
from pipekit_jax import JaxModelOp
from pipekit_experiment import LocalModelRegistry

# Wrap trained FourDVarNet for serialisation
model_op = JaxModelOp(trained_vardanet)

# Store with content-addressed hash
registry = LocalModelRegistry(root="./models")
hash_ = registry.store(
    model_op,
    weights=model_op.serialize_weights(),
    tags={"task": "ssh_oceanbench", "version": "v1"},
)

# Later — reload with a fresh skeleton
template = JaxModelOp(make_fresh_vardanet_skeleton())
reloaded = template.with_weights(registry.load_weights(hash_))

# Or use vardax shortcuts (D13)
from vardax.persist import save, load

hash_ = save(trained_vardanet, registry, tags={"task": "ssh"})
reloaded = load(registry, hash_, skeleton_factory=make_fresh_vardanet_skeleton)
```

---

## pipekit-train — Training callbacks + metric writers

```python
from pipekit_train import (
    MSE, EarlyStopping, Checkpoint, LogToExperiment, JSONLWriter, TrainingLoop,
)
from pipekit_experiment import WandbTracker

tracker = WandbTracker(project="vardax-ssh", run_name="vardanet-v1")
metric_writer = JSONLWriter("./logs/run.jsonl")

loop = TrainingLoop(
    dataset=oceanbench_ssh_dataset,
    model_op=JaxModelOp(model),
    loss=MSE(),
    callbacks=[
        EarlyStopping(metric="val_mse", patience=10),
        Checkpoint(registry=registry, every_n_epochs=1, metric="val_mse"),
        LogToExperiment(tracker),
    ],
)
trained_op, final_state = loop(JaxModelOp(model), TrainerCarryState(...))
```

`pipekit-train` is an optional `[train]` extra; vardax's `train_step` is
the inner primitive plugged into `pipekit_train.MSE`.

---

## georeader — Sensor data loading

```python
import georeader as gr

# Per-instrument readers know how to handle each L1/L2 format
tropomi_l2 = gr.TROPOMI_L2.from_url("s3://tropomi/2024/01/15/...")
emit_l2    = gr.EMIT_L2.from_url("s3://emit/2024/01/15/...")
ghgsat_l2  = gr.GHGSat_L2.from_url("s3://ghgsat/2024/01/15/...")

# Each returns a GeoTensor with attached AK metadata
# Build vardax Batch from the multi-instrument set
batch = build_batch_from_readers([tropomi_l2, emit_l2, ghgsat_l2], met=met_field)
```

---

## coordax — Coordinate-aware fields

Open question 1 from `boundaries.md`. Current design: vardax `Batch*` use
raw `Array`. Optional `[coords]` extra exposes `coordax` adapters:

```python
from vardax.adapters.coordax import batch_from_coordax_dataset

batch = batch_from_coordax_dataset(coordax_ds, state_var="ssh",
                                    obs_var="ssh_obs", mask_var="mask")
# Coordinates preserved in Batch.metadata for posterior provenance.
```

---

## GeoCatalog — Event-driven batch assembly

```python
from geotoolz import GeoCatalog

# Query overpasses across instruments for a methane event
catalog = GeoCatalog.from_geoparquet("s3://catalog/overpasses.parquet")
overpasses = catalog.query(
    geometry=event.bbox(buffer_km=50),
    interval=event.window(hours=2),
    instruments=["TROPOMI", "EMIT", "GHGSat"],
    quality_min="usable",
)

# Assemble multi-instrument batch
batch = build_event_batch(overpasses, met=met_field)

# Invert
x_star = inversion(batch)
posterior = GaussNewtonHessian()(x_star, inversion.as_analysis_step(), batch)

# Write back to catalog
catalog.write_posterior(event.id, GaussianMarkLikelihood(posterior, event.metadata))
```

---

## Composition patterns

| Pattern | Components | Use case |
|---|---|---|
| **Learned 4DVarNet** | AE prior + masked obs + ConvLSTM | Standard 4DVarNet SSH mapping |
| **Physics-informed** | somax `DynamicalPrior` + learned grad mod | Hybrid learned + physics DA |
| **Operational SSH** | `IncrementalFourDVar` + somax SWM + altimetry + gaussx B | Production ocean DA |
| **Methane Tier I** | plumax Gaussian plume + AK + multi-instrument fusion + Incremental | Single-overpass facility attribution |
| **Methane Tier II** | plumax Lagrangian footprint + linear inversion + gaussx Matérn $B$ | Basin-scale regional inversion |
| **Methane Tier IV** | plumax Eulerian + neural RTM + multi-instrument + amortized | Operational alerts |
| **Hybrid EnVar** | filterax ensemble cov + Incremental + AveragingKernel | Non-Gaussian regimes |
| **Catalog-driven** | GeoCatalog query → georeader → vardax → catalog write | End-to-end pipeline |
| **Research → operations** | Same vardax model: notebook (research), CI batch (regression), API (production) | The whole arc |
