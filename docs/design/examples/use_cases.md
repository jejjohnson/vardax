---
status: draft
version: 0.4.0
---

# End-to-end Use Cases

Two case studies that ground the abstractions on real workflows:

1. **Methane single-overpass MAP** — plumax Tier I + vardax MAP + averaging
   kernel + multi-instrument fusion.
2. **SSH 4DVarNet** — somax shallow-water + learned `FourDVarNet` +
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
   │ IncrementalFourDVar(forward=plume, obs_op=fusion,       │
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

```{code-block} python
:caption: Inverts multi-instrument XCH4 for the Tier I source parameters with incremental 4DVar [](#eq-incremental-cost) over the fused averaging kernels [](#eq-fusion-cost) and exports the Gauss-Newton posterior [](#eq-gn-cov).
:label: code-design-usecases-methane
import gaussx as gx
import geotoolz
import jax.numpy as jnp
import lineax as lx
import plumax
import vardax as vdx

# Shapes: P = 4 source parameters (log Q, x₀, y₀, H_eff); Mᵢ observations of
#         instrument i
# Given: event (facility, time, geometry), edgar_lookup, load_era5_at

# (1) Load satellite + met
# not pinned (geotoolz)
catalog = geotoolz.GeoCatalog.from_geoparquet("s3://catalog/overpasses.parquet")
# not pinned (geotoolz)
overpasses = catalog.query(
    geometry=event.bbox(buffer_km=50),
    interval=event.window(hours=2),
    instruments=["TROPOMI", "EMIT", "GHGSat"],
    quality_min="usable",
)
# user
met = load_era5_at(event.time, event.bbox)

# (2) Forward model: Tier I Gaussian plume
# not pinned (plumax)
plume = plumax.tier1.GaussianPlume(met=met, dispersion="MO")

# (3) Multi-instrument observation operator
# proposed (D9)
registry = vdx.InstrumentRegistry.from_l2_dict(
    {op.instrument: op.l2_ds for op in overpasses}
)
# today (vardax)
# J_obs(x) = Σᵢ αᵢ ½ ‖mᵢ ⊙ (yᵢ − Hᵢ(x))‖²_{Rᵢ⁻¹}
fusion = vdx.MultiInstrumentFusion(registry=registry)

# (4) Prior: lognormal on Q from EDGAR inventory
# user
inventory_rate = edgar_lookup(event.facility_id)  # → ()
# μ = log Q_EDGAR
log_rate_prior_mean = jnp.log(inventory_rate)  # → ()
# σ = 1, a coefficient of variation of about 100%
log_rate_prior_std = 1.0
# proposed (gaussx)
# log Q ~ 𝒩(μ, σ²)
prior_cov = gx.LogNormalLinearOperator(
    mu=log_rate_prior_mean, sigma=log_rate_prior_std
)  # (P, P) op

# (5) Block-diag R across instruments
# proposed (lineax; gaussx ships gx.BlockDiag(*operators))
# R = blockdiag(R_TROPOMI, R_EMIT, R_GHGSat)
obs_cov = lx.BlockDiagonalLinearOperator(  # (Σ Mᵢ, Σ Mᵢ) op
    [spec.obs_cov_op for spec in fusion.registry.entries.values()]
)

# (6) Configure inversion
# x_b = (log Q_EDGAR, 0, 0, 50 m) for (Q, x₀, y₀, H)
prior_mean = jnp.array([log_rate_prior_mean, 0.0, 0.0, 50.0])  # (P,)
# today* (vardax; a fused obs_op goes in as fusion.to_observation_operator())
# δJ(δx) over the source parameters, in χ with x − x_b = S χ
inversion = vdx.IncrementalFourDVar(
    forward=plume,
    obs_op=fusion,
    prior_mean=prior_mean,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=5, n_inner=20, cvt=True),
)

# (7) Build batch from overpasses
# proposed (no issue yet)
batch = vdx.utils.build_event_batch(overpasses, met=met)

# (8) Invert
# today (vardax)
# x* after the last outer iteration
analysis = inversion(batch)  # → (1, P)

# (9) Posterior via Gauss-Newton Hessian (reuses last GN outer iteration)
# today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
# P*_GN = 𝐇_GN⁻¹ at x*, from 50 Lanczos vectors
posterior = vdx.GaussNewtonHessian(n_krylov=50)(
    analysis, inversion.as_analysis_step(), batch
)

# (10) Export to population layer (Tier V)
# today (vardax)
mark = vdx.GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={
        "event_id": event.id,
        "time": event.time,
        "geometry": event.geometry,
        "instruments_used": list(fusion.registry.entries),
        "met_source": "era5_2024-01-15T12Z",
    },
)
# today (vardax)
mark_record = mark.to_dict()
# not pinned (geotoolz)
catalog.write_posterior(event.id, mark_record)
```

### What this exercises in vardax

- `IncrementalFourDVar` (Decision D11)
- `AveragingKernel` + `MultiInstrumentFusion` (Decision D9)
- `GaussNewtonHessian` posterior + `GaussianMarkLikelihood` export (D10)
- `gaussx` structured prior + `lineax` block-diag obs cov
- `plumax.tier1.GaussianPlume` as `pipekit_cycle.ForwardModel` (D8)
- GeoCatalog query → inference → write loop

### Six-step cycle hooks (Decision D12)

| Step | Component |
|---|---|
| 1 (physics) | plumax Gaussian plume |
| 2 (MAP / MCMC) | vardax `IncrementalFourDVar` above |
| 3 (emulator) | train plumax neural emulator + adjoint-calibrate |
| 4 (emulator MAP) | swap `forward=plume` → `forward=plume_nn`; vardax code unchanged |
| 5 (amortized) | `AmortizedPosterior(encoder, head=ConditionalFlowHead)` trained on simulated (Q, y) pairs |
| 6 (improve) | swap any block; previous step is the oracle for validation tests |

---

## Use Case 2: Ocean SSH 4DVarNet

**Goal.** Reconstruct sea surface height $\eta(t, x, y)$ from along-track
altimeter observations with mesoscale gaps. End-to-end learned solver.

**Tier.** somax shallow-water as physics oracle, learned `FourDVarNet` as
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
   │ FourDVarNet(prior=BilinAE2D,              │
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

```{code-block} python
:caption: Trains a FourDVarNet on the 4DVarNet loss [](#eq-4dvarnet-loss), persists it, cycles it on streaming altimetry and builds a Laplace posterior [](#eq-laplace-cov).
:label: code-design-usecases-ssh
import equinox as eqx
import optax
import pipekit_cycle as pc
import pipekit_experiment
import vardax as vdx

# Shapes: B batch; T = 10 times; H × W = 128 × 128 grid; K = 15 inner steps
# Given: oceanbench_loader, somax_shallow_water, load_altika_stream, n_windows,
#        initial_state, batch

# (1) Build model
# proposed (D14; ships today as vdx.FourDVarNet2D(n_time, height, width, ..., key))
# x_{k+1} = x_k − Φ(∇ₓJ(x_k), h_k), k = 0 … K − 1
model = vdx.FourDVarNet(
    # today* (vardax; also takes key)
    prior=vdx.BilinAEPrior2D(latent_dim=128, n_time=10, height=128, width=128),
    obs_op=vdx.MaskedIdentity(),
    # today* (vardax; also takes key)
    grad_mod=vdx.ConvLSTMGradMod2D(state_channels=10, hidden_dim=64),
    # proposed (D4)
    config=vdx.SolverConfig(n_steps=15, alpha=0.2),
    # O(1) memory training
    solver_adjoint=vdx.adjoints.OneStepAdjoint(),
)

# (2) Training (OceanBench SSH benchmark)
# today (optax)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

for epoch in range(100):
    for batch in oceanbench_loader:
        # today (vardax)
        # ℒ(θ, φ) = E ‖x*(θ, φ; y) − x_true‖²
        model, opt_state, loss = vdx.train_step(model, batch, optimizer, opt_state)

# (3) Persist via pipekit (Decision D13)
# not pinned (pipekit_experiment)
registry = pipekit_experiment.LocalModelRegistry(root="./ssh_models")
# proposed (D13)
hash_ = vdx.persist.save(model, registry, tags={"task": "ssh_oceanbench"})

# (4) Operational cycling on streaming altimetry
# today (pipekit_cycle)
da_cycle = pc.DACycle(
    # physics oracle for comparison
    forward_model=somax_shallow_water,
    obs_op=model.obs_op,
    analysis_step=model.as_analysis_step(),
    obs_source=load_altika_stream,
    n_steps=n_windows,
)

# (5) Posterior for downstream uncertainty propagation
# today (vardax)
posterior_adapter = vdx.LaplaceCovariance()
# today (pipekit_cycle)
result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
# today* (vardax; the adapter takes one window's analysis, not the cycle result)
# P*_Laplace = (B⁻¹ + H′ᵀ R⁻¹ H′)⁻¹
posterior = posterior_adapter(result, model.as_analysis_step(), batch)
```

### What this exercises in vardax

- `FourDVarNet` learned 4DVarNet (D3) with `solver_adjoint=OneStepAdjoint()` (Bolte 2023, D15)
- `BilinAEPrior2D` learned prior
- `MaskedIdentity` observation operator (simplest case)
- `LaplaceCovariance` posterior (D10)
- `JaxModelOp` + `ModelRegistry` persistence (D13)
- `DACycle` operational cycling (D8)

### Comparison to incremental 4DVar

The same problem can be tackled with `IncrementalFourDVar` for a physics-based
baseline:

```{code-block} python
:caption: Solves the same SSH problem with incremental 4DVar [](#eq-incremental-cost) and checks the 4DVarNet against it with the posterior gate [](#eq-gate-posterior).
:label: code-design-usecases-ssh-incremental
import gaussx as gx
import lineax as lx
import vardax as vdx

# Shapes: N = H × W grid cells; M observations; D = 2 coordinates
# Given: model (the FourDVarNet above), somax_shallow_water, climatology_ssh (N,),
#        coords (N, D), altika_uncertainty (M,), batch

# proposed (gaussx)
# Bᵢⱼ = k_Matérn(sᵢ, sⱼ; ℓ = 100, σ = 0.1)
prior_cov = gx.MaternLinearOperator(  # (N, D) → (N, N) PSD op
    coords, length_scale=100.0, sigma=0.1
)
# today (lineax)
# R = diag(σ²)
obs_cov = lx.DiagonalLinearOperator(altika_uncertainty)  # (M,) → (M, M) op

# today (vardax)
# δJ(δx), 3 outer relinearisations, 20 CG steps each, in χ with x₀ − x_b = S χ
incremental = vdx.IncrementalFourDVar(
    forward=somax_shallow_water,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=climatology_ssh,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)
# today (vardax)
# x₀* after the last outer iteration
x_incremental = incremental(batch)  # → (B, N)

# Compare:
# today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
# P*_Laplace at the 4DVarNet reconstruction
p_learned = vdx.LaplaceCovariance()(model(batch), model.as_analysis_step(), batch)
# today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
# P*_GN at the incremental analysis
p_incremental = vdx.GaussNewtonHessian()(
    x_incremental, incremental.as_analysis_step(), batch
)
# today (vardax)
# |x*_learned − x*_incr| / σ_post,incr ≤ 1
vdx.assert_posterior_agreement(p_learned, p_incremental, tolerance_sigma=1.0)
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
            ──→  Layer 2 model (FourDVarNet / IncrementalFourDVar / AmortizedPosterior)
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
