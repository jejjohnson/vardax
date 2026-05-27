# Methane Single-Overpass Example

Methane source attribution from satellite XCH₄ observations is the
flagship multi-instrument inversion problem. A single overpass gives
observations from TROPOMI, EMIT, and (sometimes) GHGSat over a
suspected emission event. The question: what was the source rate $Q$
at the facility, and how uncertain is the answer?

This chapter walks an end-to-end methane inversion using
`IncrementalFourDVar` + multi-instrument fusion + averaging-kernel
operators. The forward model comes from `plumax` (Tier I Gaussian
plume); the inference is vardax.

## Problem setup

- **State**: source parameters $(Q, x_0, y_0, H_\text{eff})$ for a
  single point source. Tier I assumes a Gaussian plume forward.
- **Observations**: per-instrument XCH₄ retrievals (TROPOMI L2, EMIT
  L2, GHGSat L2) within a 50 km radius / 2 hour window of the event
  geometry.
- **Met forcing**: ERA5 or WRF wind, PBL height, stability — supplied
  as static metadata (not a control variable).
- **Goal**: posterior on $(Q, x_0, y_0, H_\text{eff})$, exported as a
  mark-likelihood for downstream population models.

## Data flow

```
       ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
       │  TROPOMI L2  │  │   EMIT L2    │  │  GHGSat L2   │
       │ (XCH4 + AK)  │  │ (XCH4 + AK)  │  │ (XCH4 + AK)  │
       └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
              │                  │                  │
              └────── georeader ──────────────────────┘
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
   ┌───────────────────────────────────────────────────────┐
   │ plumax.tier1.GaussianPlume(met=met_field, ...)        │
   │ (satisfies pipekit_cycle.ForwardModel)                │
   └──────────────────────────────┬────────────────────────┘
                                  │
                                  ▼
   ┌───────────────────────────────────────────────────────┐
   │ MultiInstrumentFusion(InstrumentRegistry({            │
   │   "TROPOMI": InstrumentSpec(AveragingKernel(...)),    │
   │   "EMIT":    InstrumentSpec(AveragingKernel(...)),    │
   │   "GHGSat":  InstrumentSpec(AveragingKernel(...)),    │
   │ }))                                                    │
   └──────────────────────────────┬────────────────────────┘
                                  │
                                  ▼
   ┌───────────────────────────────────────────────────────┐
   │ IncrementalFourDVar(forward=plume, obs_op=fusion,     │
   │   prior_mean=Q_inventory, prior_cov_op=lognormal,     │
   │   obs_cov_op=block_diag_per_instrument, ...)          │
   └──────────────────────────────┬────────────────────────┘
                                  │
                                  ▼
   ┌───────────────────────────────────────────────────────┐
   │ x_star = model(batch)                                 │
   │ posterior = model.posterior(batch)                    │
   └──────────────────────────────┬────────────────────────┘
                                  │
                                  ▼
   ┌───────────────────────────────────────────────────────┐
   │ GaussianMarkLikelihood(posterior, event_metadata)     │
   │   .to_dict() → GeoCatalog write                       │
   └───────────────────────────────────────────────────────┘
```

## Sketch

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
# plume satisfies pipekit_cycle.ForwardModel

# (3) Multi-instrument observation operator
fusion = vdx.obs_operators.MultiInstrumentFusion(
    registry=vdx.obs_operators.InstrumentRegistry.from_l2_dict({
        op.instrument: op.l2_ds for op in overpasses
    }),
)

# (4) Prior: lognormal on Q from EDGAR inventory
Q_prior_mu = jnp.log(edgar_lookup(event.facility_id))   # log-emission rate
Q_prior_sigma_log = 1.0                                  # CV ~ 100 %
B_op = gx.LogNormalLinearOperator(mu=Q_prior_mu, sigma=Q_prior_sigma_log)

# (5) Block-diag R across instruments
R_op = lx.BlockDiagonalLinearOperator([
    spec.R_op for spec in fusion.registry.entries.values()
])

# (6) Configure incremental 4DVar inversion
inversion = vdx.models.IncrementalFourDVar(
    forward=plume, obs_op=fusion,
    prior_mean=jnp.array([Q_prior_mu, 0.0, 0.0, 50.0]),  # (logQ, x0, y0, H_eff)
    prior_cov_op=B_op, obs_cov_op=R_op,
    config=vdx.IncrementalConfig(n_outer=5, n_inner=20, cvt=True),
)

# (7) Build batch from overpasses
batch = vdx.utils.build_event_batch(overpasses, met=met)

# (8) Invert
x_star = inversion(batch)

# (9) Posterior via Gauss-Newton Hessian (reuses last outer iteration)
posterior = inversion.posterior(batch)
# Posterior(mean=x_star, cov=GN_Hessian_inv_op, samples=None, provenance={...})

# (10) Export to population layer
mark = vdx.posterior.GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={
        "event_id": event.id, "time": event.time, "geometry": event.geometry,
        "instruments_used": list(fusion.registry.entries),
        "met_source": f"era5_{event.time.isoformat()}",
    },
)
catalog.write_posterior(event.id, mark.to_dict())
```

## What this exercises in vardax

- **`IncrementalFourDVar`** (chapter 8) — operational 4DVar with CVT
- **`AveragingKernel`** (chapter 11) — first-class day-one operator (D9)
- **`MultiInstrumentFusion`** — likelihood-level composition without
  pre-regridding
- **`GaussNewtonHessian` posterior** — free from the last outer iteration
- **`GaussianMarkLikelihood` export** — hand-off to Tier V population
  models (D10)
- **`gaussx` structured prior + `lineax` block-diag obs cov** —
  structured linear algebra throughout
- **`plumax.tier1.GaussianPlume`** as `pipekit_cycle.ForwardModel` (D8)
- **GeoCatalog query → inference → write loop** — operational pattern

## Six-step cycle for methane

The example above is Step 2 (classical inference). The full cycle:

| Step | Component |
|---|---|
| 1 (physics) | `plumax.tier1.GaussianPlume(met)` |
| 2 (classical) | `IncrementalFourDVar` above |
| 3 (emulator) | Train `plumax.NeuralPlume(net)` + adjoint calibration |
| 4 (emulator inference) | Swap `forward=plume` → `forward=neural_plume`; vardax code unchanged |
| 5 (amortized) | `AmortizedPosterior(encoder, head=ConditionalFlowHead)` trained on simulated $(Q, y)$ pairs |
| 6 (improve) | Swap any block; previous step is the oracle for validation |

The same `vardax.models.*` classes appear at Steps 2 and 4 — only the
`forward` slot changes. At Step 5, `AmortizedPosterior` replaces them
entirely, validated against Step 2's output via the gates in chapter
14.

## When to upgrade tiers

`plumax` provides four tiers of forward operator:

| Tier | Forward | When |
|---|---|---|
| I | Gaussian plume (analytical) | Single overpass, single source, fast iteration |
| II | Lagrangian particle transport | Multi-source, sub-hour resolution |
| III | Eulerian finite-volume PDE | Long-range transport, regional inversion |
| IV | Coupled transport + RTM | L1 radiance inversion, multi-instrument |

The vardax inference code is **the same across tiers** — only the
`forward` slot changes. The cost of upgrading from Tier I to Tier III
is in `plumax`, not in vardax.

## Multi-event aggregation

For population-level analysis (annual emissions from a basin, persistent
super-emitters), aggregate per-event posteriors:

```python
events = catalog.query_events(basin="permian", year=2024)
posteriors = []
for event in events:
    batch = build_event_batch(event)
    x_star = inversion(batch)
    posteriors.append(inversion.posterior(batch))

# Pass to TMTPP / hierarchical Bayesian model
marks = [GaussianMarkLikelihood(p, e.metadata) for p, e in zip(posteriors, events)]
population_model = TMTPP(marks)
```

Decision D10's mark-likelihood serialisation is the contract that
makes this work. Vardax doesn't own the population model — that's
downstream — but it produces the standardised input.

## See also

- Chapter 8 — `IncrementalFourDVar` (the analysis method)
- Chapter 11 — observation operators (AK + multi-instrument)
- Chapter 13 — posterior covariance (GN-Hessian reuse)
- Chapter 14 — six-step cycle (research-to-operations)
- Design doc: [`design/examples/use_cases.md`](design/examples/use_cases.md)
- Design doc: [`design/decisions.md#d9`](design/decisions.md#d9-averaging-kernel-multi-instrument-as-first-class)
- Design doc: [`design/decisions.md#d10`](design/decisions.md#d10-posterior-export-adapter-pattern)

## References

- Jacob, D. J., et al. (2022). *Quantifying methane emissions from the
  global scale down to point sources using satellite observations of
  atmospheric methane.* ACP 22(14).
- Varon, D. J., et al. (2019). *Quantifying methane point sources from
  fine-scale satellite observations of atmospheric methane plumes.*
  AMT 12(10).
- Cusworth, D. H., et al. (2021). *Multisatellite imaging of a gas
  well blowout enables quantification of total methane emissions.*
  GRL 48(2).
