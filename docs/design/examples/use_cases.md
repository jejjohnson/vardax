---
status: draft
version: 0.3.0
---

# End-to-end Use Cases

Two case studies that ground the abstractions on real workflows:

1. **Methane single-overpass MAP** — plumax Tier I + vardax MAP + averaging
   kernel + multi-instrument fusion.
2. **SSH 4DVarNet** — somax shallow-water + learned `VarDANet2D` +
   altimetry-style masking.

---

## Use Case 1: Methane single-overpass attribution

**Goal.** Given multi-instrument satellite observations of an XCH₄
enhancement, estimate the source rate $Q$, location $x_0$, and effective
stack height $H_\text{eff}$ for a known facility. Single overpass.

**Tier.** Plumax Tier I (Gaussian plume / puff).

### Data flow

```
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │  TROPOMI L2  │  │   EMIT L2    │  │  GHGSat L2   │
        │ (XCH4 + AK)  │  │ (XCH4 + AK)  │  │ (XCH4 + AK)  │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
               │                  │                  │
               └─────── georeader ────────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │      Event metadata           │
                  │  (facility, time, geometry)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
            ┌──────────────────────────────────────┐
            │  ERA5 / WRF wind, PBL, stability     │
            └──────────────┬───────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────────────┐
   │ plumax.tier1.GaussianPlume(met=met_field, ...)       │
   │ (satisfies pipekit_cycle.ForwardModel)               │
   └──────────────────────────────┬───────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────┐
   │ MultiInstrumentFusion(InstrumentRegistry({           │
   │   "TROPOMI": InstrumentSpec(AveragingKernel(...)),   │
   │   "EMIT":    InstrumentSpec(AveragingKernel(...)),   │
   │   "GHGSat":  InstrumentSpec(AveragingKernel(...)),   │
   │ }))                                                   │
   └──────────────────────────────┬───────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────┐
   │ IncrementalVarDA(forward=plume, obs_op=fusion,       │
   │   prior_mean=Q_inventory, prior_cov_op=lognormal,    │
   │   obs_cov_op=block_diag_per_instrument, ...)         │
   └──────────────────────────────┬───────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────┐
   │ x_star = model(batch)                                │
   │ posterior = GaussNewtonHessian()(x_star, model, ...) │
   └──────────────────────────────┬───────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────┐
   │ GaussianMarkLikelihood(posterior, event_metadata)    │
   │   .to_dict() → GeoCatalog write                      │
   └──────────────────────────────────────────────────────┘
```

### Sketch

```python
import jax.numpy as jnp
import gaussx as gx
import lineax as lx
import plumax
import vardax as vdx

# (1) Load satellite + met
catalog = GeoCatalog.from_geoparquet("s3://catalog/overpasses.parquet")
overpasses = catalog.query(
    geometry=event.bbox(buffer_km=50),
    interval=event.window(hours=2),
    instruments=["TROPOMI", "EMIT", "GHGSat"],
    quality_min="usable",
)
met = load_era5_at(event.time, event.bbox)

# (2) Forward model — Tier I Gaussian plume
plume = plumax.tier1.GaussianPlume(met=met, dispersion="MO")

# (3) Multi-instrument observation operator
fusion = vdx.obs_operators.MultiInstrumentFusion(
    registry=vdx.obs_operators.InstrumentRegistry.from_l2_dict({
        op.instrument: op.l2_ds for op in overpasses
    }),
)

# (4) Prior: lognormal on Q from EDGAR inventory
Q_prior_mu = jnp.log(edgar_lookup(event.facility_id))
Q_prior_sigma = 1.0   # CV ~ 100%
B_op = gx.LogNormalLinearOperator(mu=Q_prior_mu, sigma=Q_prior_sigma)

# (5) Block-diag R across instruments
R_op = lx.BlockDiagonalLinearOperator([
    spec.R_op for spec in fusion.registry.entries.values()
])

# (6) Configure inversion
inversion = vdx.models.IncrementalVarDA(
    forward=plume, obs_op=fusion,
    prior_mean=jnp.array([Q_prior_mu, 0.0, 0.0, 50.0]),  # Q, x0, y0, H
    prior_cov_op=B_op, obs_cov_op=R_op,
    config=vdx.IncrementalConfig(n_outer=5, n_inner=20, cvt=True),
)

# (7) Build batch from overpasses
batch = vdx.utils.build_event_batch(overpasses, met=met)

# (8) Invert
x_star = inversion(batch)

# (9) Posterior via Gauss-Newton Hessian (reuses last GN outer iteration)
posterior = vdx.posterior.GaussNewtonHessian(n_krylov=50)(
    x_star, inversion.as_analysis_step(), batch,
)

# (10) Export to population layer (Tier V)
mark = vdx.posterior.GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={
        "event_id": event.id,
        "time": event.time,
        "geometry": event.geometry,
        "instruments_used": list(fusion.registry.entries),
        "met_source": "era5_2024-01-15T12Z",
    },
)
catalog.write_posterior(event.id, mark.to_dict())
```

### What this exercises in vardax

- `IncrementalVarDA` (Decision D11)
- `AveragingKernel` + `MultiInstrumentFusion` (Decision D9)
- `GaussNewtonHessian` posterior + `GaussianMarkLikelihood` export (D10)
- `gaussx` structured prior + `lineax` block-diag obs cov
- `plumax.tier1.GaussianPlume` as `pipekit_cycle.ForwardModel` (D8)
- GeoCatalog query → inference → write loop

### Six-step cycle hooks (Decision D12)

| Step | Component |
|---|---|
| 1 (physics) | plumax Gaussian plume |
| 2 (MAP / MCMC) | vardax `IncrementalVarDA` above |
| 3 (emulator) | train plumax neural emulator + adjoint-calibrate |
| 4 (emulator MAP) | swap `forward=plume` → `forward=plume_nn`; vardax code unchanged |
| 5 (amortized) | `AmortizedVarDA(encoder, head=ConditionalFlowHead)` trained on simulated (Q, y) pairs |
| 6 (improve) | swap any block; previous step is the oracle for validation tests |

---

## Use Case 2: Ocean SSH 4DVarNet

**Goal.** Reconstruct sea surface height $\eta(t, x, y)$ from along-track
altimeter observations with mesoscale gaps. End-to-end learned solver.

**Tier.** somax shallow-water as physics oracle, learned `VarDANet2D` as
inference.

### Data flow

```
       ┌──────────────────┐
       │ AltiKa / SARAL   │
       │ along-track SSH  │
       └────────┬─────────┘
                │  georeader
                ▼
   ┌──────────────────────────────────────┐
   │ coordax.Dataset(lat, lon, time, ssh) │
   └────────┬─────────────────────────────┘
            │  build_batch
            ▼
   ┌──────────────────────────────────────┐
   │ Batch2D(input, mask, target?)        │
   └────────┬─────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────────────┐
   │ VarDANet2D(prior=BilinAE2D,              │
   │            obs_op=MaskedIdentity,        │
   │            grad_mod=ConvLSTMGradMod2D,   │
   │            config=SolverConfig("one_step"))│
   └────────┬─────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────────────┐
   │ x_reconstructed = model(batch)           │
   │ posterior = LaplaceCovariance()(...)     │
   └──────────────────────────────────────────┘
```

### Sketch

```python
import equinox as eqx
import optax
import vardax as vdx
from pipekit_jax import JaxModelOp
from pipekit_experiment import LocalModelRegistry

# (1) Build model
model = vdx.models.VarDANet2D(
    prior=vdx.priors.BilinAEPrior2D(latent_dim=128, n_time=10, height=128, width=128),
    obs_op=vdx.obs_operators.MaskedIdentity(),
    grad_mod=vdx.grad_mod.ConvLSTMGradMod2D(state_channels=10, hidden_dim=64),
    config=vdx.SolverConfig(n_steps=15, alpha=0.2, grad_mode="one_step"),
)

# (2) Training (OceanBench SSH benchmark)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

for epoch in range(100):
    for batch in oceanbench_loader:
        model, opt_state, loss = vdx.training.train_step(
            model, batch, optimizer, opt_state,
        )

# (3) Persist via pipekit (Decision D13)
registry = LocalModelRegistry(root="./ssh_models")
hash_ = vdx.persist.save(model, registry, tags={"task": "ssh_oceanbench"})

# (4) Operational cycling on streaming altimetry
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_shallow_water,        # physics oracle for comparison
    obs_op=model.obs_op,
    analysis_step=model.as_analysis_step(),
    obs_source=load_altika_stream,
    n_steps=n_windows,
)

# (5) Posterior for downstream uncertainty propagation
posterior_adapter = vdx.posterior.LaplaceCovariance()
result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
posterior = posterior_adapter(result, model.as_analysis_step(), batch)
```

### What this exercises in vardax

- `VarDANet2D` learned 4DVarNet (D3) with `grad_mode="one_step"` (Bolte 2023)
- `BilinAEPrior2D` learned prior
- `MaskedIdentity` observation operator (simplest case)
- `LaplaceCovariance` posterior (D10)
- `JaxModelOp` + `ModelRegistry` persistence (D13)
- `DACycle` operational cycling (D8)

### Comparison to incremental 4DVar

The same problem can be tackled with `IncrementalVarDA2D` for a physics-based
baseline:

```python
incremental = vdx.models.IncrementalVarDA2D(
    forward=somax_shallow_water,
    obs_op=vdx.obs_operators.MaskedIdentity(),
    prior_mean=climatology_ssh,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=100.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(altika_uncertainty),
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)
x_incremental = incremental(batch)

# Compare:
assert_posterior_agreement(
    LaplaceCovariance()(model(batch), model.as_analysis_step(), batch),
    GaussNewtonHessian()(x_incremental, incremental.as_analysis_step(), batch),
    tolerance_sigma=1.0,
)
```

The same `Batch2D` flows through both inference paths.

---

## Recurring patterns

Both use cases share a structural pattern that vardax codifies:

```
forward (somax / plumax)
   ↓
   ──→  ObservationOperator (Masked / AveragingKernel / MultiInstrument)
            ↓
            ──→  Layer 2 model (VarDANet / IncrementalVarDA / AmortizedVarDA)
                    ↓
                    ──→  PosteriorAdapter (Laplace / GN-Hessian / Ensemble)
                            ↓
                            ──→  GaussianMarkLikelihood → catalog
```

Each block satisfies a pipekit-cycle protocol (Decision D8). The same code
runs in three execution modes:

| Mode | What lives where |
|---|---|
| Research (notebook) | All blocks in a Jupyter cell |
| Reanalysis (batch) | Same blocks in a CI cron pipeline |
| Operations (API) | Same blocks behind a FastAPI handler |

This is the "research → operations arc" promised in `vision.md`.
