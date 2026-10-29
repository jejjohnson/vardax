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
import lineax as lx
import plumax
import vardax as vdx

# Shapes: P = 4 source parameters (log Q, x₀, y₀, H_eff); M retrieved pixels

# (1) Load satellite + met
# not pinned (GeoCatalog)
catalog = GeoCatalog.from_geoparquet("s3://catalog/overpasses.parquet")
# not pinned (GeoCatalog)
overpasses = catalog.query(
    geometry=event.bbox(buffer_km=50),
    interval=event.window(hours=2),
    instruments=["TROPOMI", "EMIT", "GHGSat"],
    quality_min="usable",
)
# user
met = load_era5_at(event.time, event.bbox)

# (2) Forward model, Tier I Gaussian plume
# not pinned (plumax)
plume = plumax.tier1.GaussianPlume(met=met, dispersion="MO")

# (3) Multi-instrument observation operator
# today (vardax)
fusion = vdx.MultiInstrumentFusion(
    # today (vardax)
    registry=vdx.InstrumentRegistry(
        entries={
            # today (vardax)
            overpass.instrument: vdx.InstrumentSpec(
                # user; returns a vdx.AveragingKernel
                # H_i(x) = A_i (h_i ⊙ x + (1 − h_i) ⊙ x_a,i)
                obs_op=averaging_kernel_from_l2(overpass.l2_ds),
                mask=overpass.quality_mask,
                # R_i = diag(σ²_i)
                obs_cov_op=lx.DiagonalLinearOperator(overpass.retrieval_variance),
                instrument_id=overpass.instrument,
            )
            for overpass in overpasses
        }
    ),
)
# today (vardax)
# H(x) = [H_i(x)]ᵢ, concatenated in sorted instrument order; the fusion itself
# returns a dict per instrument, which the cost cannot take
fused_obs_op = fusion.to_observation_operator()

# (4) Prior: lognormal on Q from the EDGAR inventory, Gaussian on log Q
# user
# μ = log Q_EDGAR
log_rate_prior_mean = jnp.log(edgar_lookup(event.facility_id))  # → ()
# σ_log = 1, a coefficient of variation of about 100 %
log_rate_prior_sd = 1.0
# illustrative spreads on the source location and effective height
position_prior_sd_m = 500.0
height_prior_sd_m = 25.0
# B = diag(σ²_log, σ²_x, σ²_y, σ²_H)
prior_variance = jnp.array(  # (P,)
    [
        log_rate_prior_sd**2,
        position_prior_sd_m**2,
        position_prior_sd_m**2,
        height_prior_sd_m**2,
    ]
)
# today (lineax)
prior_cov = lx.DiagonalLinearOperator(prior_variance)  # (P, P) op

# (5) Block-diagonal R across instruments
entries = fusion.registry.entries
obs_variance = jnp.concatenate(  # (M,)
    [entries[instrument].obs_cov_op.diagonal for instrument in sorted(entries)]
)
# today (lineax)
# R = diag(R_i), in the sorted order of the fused observations; each R_i is
# diagonal, and gaussx.BlockDiag lacks the lineax.linearise this model needs
obs_cov = lx.DiagonalLinearOperator(obs_variance)  # (M, M) op

# (6) Configure incremental 4DVar inversion
# today* (vardax)
# J(x) = ½ ‖x − x_b‖²_{B⁻¹} + ½ ‖m ⊙ (y − H(M(x)))‖²_{R⁻¹}
inversion = vdx.IncrementalFourDVar(
    forward=plume,
    obs_op=fused_obs_op,
    # x_b = (log Q_EDGAR, 0, 0, 50 m)
    prior_mean=jnp.array([log_rate_prior_mean, 0.0, 0.0, 50.0]),  # (P,)
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=5, n_inner=20, cvt=True),
)

# (7) Build batch from overpasses
# user; overpasses → vdx.Batch1D of shape (1, 1, M)
batch = build_event_batch(overpasses, met=met)

# (8) Invert
# x* = argminₓ J(x)
analysis = inversion(batch)  # (1, P)

# (9) Gauss-Newton posterior at the analysis (model's own B and R)
# today (vardax)
# P* = (B⁻¹ + Gᵀ R⁻¹ G)⁻¹, G = H′ M′ at x*
posterior = inversion.posterior(batch, adapter=vdx.GaussNewtonHessian())
# Posterior(mean=x*, cov=lazy P*, samples=None, provenance={...})

# (10) Export to population layer
# today (vardax)
mark = vdx.GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={
        "event_id": event.id,
        "time": event.time,
        "geometry": event.geometry,
        "instruments_used": list(entries),
        "met_source": f"era5_{event.time.isoformat()}",
    },
)
# today (vardax)
mark_record = mark.to_dict()
# not pinned (GeoCatalog)
catalog.write_posterior(event.id, mark_record)
```

## What this exercises in vardax

- **`IncrementalFourDVar`** (chapter 8) — operational 4DVar with CVT
- **`AveragingKernel`** (chapter 11) — first-class day-one operator (D9)
- **`MultiInstrumentFusion`** — likelihood-level composition without
  pre-regridding
- **`GaussNewtonHessian` posterior** — free from the last outer iteration
- **`GaussianMarkLikelihood` export** — hand-off to Tier V population
  models (D10)
- **`lineax` diagonal prior and block-diagonal obs cov** — the per-instrument blocks are diagonal, so $R$ is one `DiagonalLinearOperator`
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
# Shapes: E events; P = 4 source parameters

# not pinned (GeoCatalog)
events = catalog.query_events(basin="permian", year=2024)
posteriors = []
for event in events:
    # user
    batch = build_event_batch(event)
    # today (vardax)
    # p(x | y) ≈ 𝒩(x*, P*), one per event
    posteriors.append(inversion.posterior(batch, adapter=vdx.GaussNewtonHessian()))

# Pass to TMTPP / hierarchical Bayesian model
marks = [
    # today (vardax)
    vdx.GaussianMarkLikelihood(posterior, event.metadata)
    for posterior, event in zip(posteriors, events)
]
# not pinned (TMTPP)
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
