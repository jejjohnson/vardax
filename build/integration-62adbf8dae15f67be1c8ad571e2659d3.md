---
status: draft
version: 0.4.0
---

# Ecosystem Integration

Composition with somax, plumax, gaussx, filterax, pipekit-cycle,
pipekit-experiment, georeader, coordax, GeoCatalog.

---

## somax — Geophysical forward as dynamical prior

```{code-block} python
:caption: Uses a somax shallow-water forward [](#eq-shallow-water) as the prior of a FourDVarNet [](#eq-4dvarnet-cost) or as the dynamics of an incremental 4DVar [](#eq-incremental-cost).
:label: code-design-integration-somax
import somax
import vardax as vdx

# Shapes: N state cells; K = 15 inner steps
# Given: grid, params, background x_b (N,), prior_cov B op, obs_cov R op

# not pinned (somax)
# somax forwards satisfy pipekit_cycle.ForwardModel natively
swm = somax.ShallowWaterModel(grid=grid, params=params)

# Wrap as a Prior for the variational cost
# today* (vardax; ships as DynamicalPrior(model, params, solver, stepsize, adjoint))
# φ(x₀) = M₁₀(x₀)
prior = vdx.DynamicalPrior(forward=swm, n_steps=10)

# Plug into FourDVarNet
# proposed (D14; ships today as vdx.FourDVarNet1D / vdx.FourDVarNet2D)
# x_{k+1} = x_k − Φ(∇ₓJ(x_k), h_k), with J's prior term ‖x − φ(x)‖²
model = vdx.FourDVarNet(
    prior=prior,
    obs_op=vdx.MaskedIdentity(),
    # today* (vardax; also takes state_channels and key)
    grad_mod=vdx.ConvLSTMGradMod2D(hidden_dim=64),
    # proposed (D4)
    config=vdx.SolverConfig(n_steps=15),
)

# Or into IncrementalFourDVar (no prior wrap: forward is used directly)
# today (vardax)
# δJ(δx) with the somax steps as the dynamics Mₜ
incremental = vdx.IncrementalFourDVar(
    forward=swm,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(),
)
```

---

## plumax — Methane forward chain (Tier I)

Decision D7: vardax does not own methane forwards. `plumax` provides the
forward chain (Tier I Gaussian plume → Tier IV coupled RTM). vardax does
the inversion.

```{code-block} python
:caption: Inverts a plumax Gaussian plume through multi-instrument averaging kernels [](#eq-fusion-cost) with incremental 4DVar [](#eq-incremental-cost) and reads the Gauss-Newton posterior [](#eq-gn-cov).
:label: code-design-integration-plumax
import gaussx as gx
import lineax as lx
import plumax
import vardax as vdx

# Shapes: P = 4 source parameters (Q, x₀, y₀, H_eff); Mᵢ observations of instrument i
# Given: met_field, tropomi_l2_ds, emit_l2_ds, ghgsat_l2_ds (L2 datasets),
#        Q_prior_mean (P,), Q_prior_mu, Q_prior_sigma_log, batch

# not pinned (plumax)
# F(Q, x₀, u, θ) = predicted XCH4 enhancement
plume_fwd = plumax.tier1.GaussianPlume(met=met_field, dispersion="MO")

# Observation: averaging kernel from TROPOMI L2 metadata
# proposed (D9)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a), with A, h, x_a read from the L2 file
tropomi_ak = vdx.AveragingKernel.from_l2(tropomi_l2_ds)

# Multi-instrument fusion (per-instrument bias as joint state, Epic 9)
# proposed (D9)
registry = vdx.InstrumentRegistry.from_l2_dict(
    {
        "TROPOMI": tropomi_l2_ds,
        "EMIT": emit_l2_ds,
        "GHGSat": ghgsat_l2_ds,
    }
)
# today (vardax)
# J_obs(x) = Σᵢ αᵢ ½ ‖mᵢ ⊙ (yᵢ − Hᵢ(x))‖²_{Rᵢ⁻¹}
fusion = vdx.MultiInstrumentFusion(registry=registry)

# proposed (gaussx)
# log Q ~ 𝒩(μ, σ²)
prior_cov = gx.LogNormalLinearOperator(Q_prior_mu, Q_prior_sigma_log)  # (P, P) op
# proposed (lineax; gaussx ships gx.BlockDiag(*operators))
# R = blockdiag(R_TROPOMI, R_EMIT, R_GHGSat)
obs_cov = lx.BlockDiagonalLinearOperator(  # (Σ Mᵢ, Σ Mᵢ) op
    [spec.obs_cov_op for spec in fusion.registry.entries.values()]
)

# Invert
# today* (vardax; a fused obs_op goes in as fusion.to_observation_operator())
# δJ(δx) over the source parameters
inversion = vdx.IncrementalFourDVar(
    forward=plume_fwd,
    obs_op=fusion,
    prior_mean=Q_prior_mean,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20),
)

# today (vardax)
# x* after the last outer iteration
analysis = inversion(batch)  # → (1, P)
# today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
# P*_GN = 𝐇_GN⁻¹ at x*
posterior = vdx.GaussNewtonHessian()(analysis, inversion.as_analysis_step(), batch)
```

---

## gaussx — Structured prior covariance

Decision D11: incremental 4DVar uses the control-variable transform with a
`gaussx.MaternLinearOperator` factorisation of $B$.

```{code-block} python
:caption: Builds a Matérn prior covariance B and its square root S for the control-variable transform [](#eq-cvt), or a separable Kronecker form.
:label: code-design-integration-gaussx
import gaussx as gx

# Shapes: N = N_lon × N_lat grid cells; D = 2 coordinates
# Given: coords (N, D), lon (N_lon,), lat (N_lat,)

# Matérn-3/2 on a regular grid, supports Kronecker structure
# proposed (gaussx)
# Bᵢⱼ = k_Matérn-3/2(sᵢ, sⱼ; ℓ = 10 km, σ = 1)
prior_cov = gx.MaternLinearOperator(  # (N, D) → (N, N) PSD op
    grid_coords=coords,
    # km, basin-dependent
    length_scale=10.0,
    # Matérn-3/2
    nu=1.5,
    sigma=1.0,
)

# Returns B^{1/2} for CVT
# proposed (gaussx; gaussx ships gx.sqrt(operator))
# S = B^{1/2}
prior_sqrt = prior_cov.half()  # (N, N) op

# Or a Kronecker product (separable in lon/lat):
# proposed (gaussx)
# B_lon, ℓ = 5
lon_cov = gx.Matern1DLinearOperator(coords=lon, length_scale=5.0)  # (N_lon, N_lon) op
# proposed (gaussx)
# B_lat, ℓ = 3
lat_cov = gx.Matern1DLinearOperator(coords=lat, length_scale=3.0)  # (N_lat, N_lat) op
# proposed (gaussx; gaussx ships gx.Kronecker)
# B = B_lon ⊗ B_lat
kron_cov = gx.KroneckerLinearOperator([lon_cov, lat_cov])  # (N, N) op
```

For non-separable, structured priors, build from `LowRankUpdate`,
`BlockDiag`, or factorise via Lanczos.

---

## filterax — Hybrid ensemble-variational (Epic 9)

```{code-block} python
:caption: Blends a climatological and an ensemble covariance [](#eq-ensemble-cov) into a hybrid B for incremental 4DVar [](#eq-incremental-cost), then reads the ensemble posterior.
:label: code-design-integration-filterax
import einx
import filterax as fx
import jax
import vardax as vdx

# Shapes: E = 32 members; N state cells
# Given: somax_model, background x_b (N,), state_shape = (N,), B_climatological op,
#        fusion, obs_cov R op, batch

key_perturb = jax.random.PRNGKey(0)

# δx⁽ᵉ⁾ ~ 𝒩(0, I)
initial_perturbations = jax.random.normal(key_perturb, (32, *state_shape))  # (E, N)

# Ensemble forecast (filterax owns this)
# not pinned (filterax)
ensemble_forecast = fx.EnsembleForecast(
    forward=somax_model,
    n_members=32,
    initial_perturbations=initial_perturbations,
)
# not pinned (filterax)
# x⁽ᵉ⁾ = M(x_b + δx⁽ᵉ⁾)
ensemble_states = ensemble_forecast(background)  # (N,) → (E, N)

# Hybrid B: blend climatological + ensemble
# not pinned (filterax)
# P_ens = 1/(E − 1) Σₑ (x⁽ᵉ⁾ − x̄)(x⁽ᵉ⁾ − x̄)ᵀ
ensemble_cov = fx.ensemble_covariance(ensemble_states)  # (E, N) → (N, N) op
# B_hybrid = ½ B_clim + ½ P_ens
hybrid_cov = 0.5 * B_climatological + 0.5 * ensemble_cov  # (N, N) op
# x̄ = 1/E Σₑ x⁽ᵉ⁾
ensemble_mean = einx.mean("e n -> n", ensemble_states)  # (E, N) → (N,)

# Run vardax inversion with hybrid B
# today* (vardax; a fused obs_op goes in as fusion.to_observation_operator())
# δJ(δx) with B = B_hybrid
model = vdx.IncrementalFourDVar(
    forward=somax_model,
    obs_op=fusion,
    prior_mean=ensemble_mean,
    prior_cov_op=hybrid_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(),
)
# today (vardax)
# x₀* after the last outer iteration
analysis = model(batch)  # → (S, N)

# Posterior from the ensemble itself
# today (vardax)
# P*_ens from the E members
posterior = vdx.EnsembleCovariance(n_members=32)(
    ensemble_states, model.as_analysis_step(), batch
)
```

---

## pipekit-cycle — Operational DA cycling

```{code-block} python
:caption: Builds an incremental 4DVar [](#eq-incremental-cost) once and cycles it through pipekit_cycle windows, the variational counterpart of [](#eq-kalman-filter).
:label: code-design-integration-cycle
import pipekit_cycle as pc
import vardax as vdx

# Given: somax_model, x_climatology x_b, prior_cov B op, obs_cov R op,
#        load_satellite_op, n_assimilation_windows, initial_state

# Build the model once
# today (vardax)
model = vdx.IncrementalFourDVar(
    forward=somax_model,
    # today* (vardax)
    obs_op=vdx.AveragingKernel(...),
    prior_mean=x_climatology,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(),
)

# Cycle it
# today (pipekit_cycle)
# forecast x^f_t = M_t(x^a_{t−1}), then analysis x^a_t from the model's step
da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=model.obs_op,
    analysis_step=model.as_analysis_step(),
    obs_source=load_satellite_op,
    n_steps=n_assimilation_windows,
)
# today (pipekit_cycle)
result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

---

## pipekit-experiment + pipekit-jax — Trained model persistence

```{code-block} python
:caption: Stores and reloads a trained FourDVarNet with pipekit content-addressed persistence (Decision D13); no equation is involved.
:label: code-design-integration-persistence
import pipekit_experiment
import pipekit_jax
import vardax as vdx

# Given: trained_vardanet, make_fresh_vardanet_skeleton (user factory)

# Wrap trained FourDVarNet for serialisation
# not pinned (pipekit_jax)
model_op = pipekit_jax.JaxModelOp(trained_vardanet)

# Store with content-addressed hash
# not pinned (pipekit_experiment)
registry = pipekit_experiment.LocalModelRegistry(root="./models")
# not pinned (pipekit_experiment)
hash_ = registry.store(
    model_op,
    weights=model_op.serialize_weights(),
    tags={"task": "ssh_oceanbench", "version": "v1"},
)

# Later, reload with a fresh skeleton
# not pinned (pipekit_jax)
template = pipekit_jax.JaxModelOp(make_fresh_vardanet_skeleton())
# not pinned (pipekit_jax)
reloaded = template.with_weights(registry.load_weights(hash_))

# Or use vardax shortcuts (D13)
# proposed (D13)
hash_ = vdx.persist.save(trained_vardanet, registry, tags={"task": "ssh"})
# proposed (D13)
reloaded = vdx.persist.load(
    registry, hash_, skeleton_factory=make_fresh_vardanet_skeleton
)
```

---

## pipekit-train — Training callbacks + metric writers

```{code-block} python
:caption: Trains a FourDVarNet with a pipekit-train loop on an MSE loss, the empirical form of [](#eq-4dvarnet-loss).
:label: code-design-integration-pipekit-train
import pipekit_experiment
import pipekit_jax
import pipekit_train

# Given: oceanbench_ssh_dataset, model, registry (as above)

# not pinned (pipekit_experiment)
tracker = pipekit_experiment.WandbTracker(project="vardax-ssh", run_name="vardanet-v1")
# not pinned (pipekit_train)
metric_writer = pipekit_train.JSONLWriter("./logs/run.jsonl")

# not pinned (pipekit_train)
# ℒ = mean ‖x* − x_true‖²
loop = pipekit_train.TrainingLoop(
    dataset=oceanbench_ssh_dataset,
    model_op=pipekit_jax.JaxModelOp(model),
    loss=pipekit_train.MSE(),
    callbacks=[
        pipekit_train.EarlyStopping(metric="val_mse", patience=10),
        pipekit_train.Checkpoint(registry=registry, every_n_epochs=1, metric="val_mse"),
        pipekit_train.LogToExperiment(tracker),
    ],
)
# not pinned (pipekit_train)
trained_op, final_state = loop(
    pipekit_jax.JaxModelOp(model), pipekit_train.TrainerCarryState(...)
)
```

`pipekit-train` is an optional `[train]` extra; vardax's `train_step` is
the inner primitive plugged into `pipekit_train.MSE`.

---

## georeader — Sensor data loading

```{code-block} python
:caption: Loads per-instrument L2 products with georeader and assembles a vardax batch for the observation model [](#eq-obs-model).
:label: code-design-integration-georeader
import georeader as gr

# Given: met_field

# Per-instrument readers know how to handle each L1/L2 format
# not pinned (georeader)
tropomi_l2 = gr.TROPOMI_L2.from_url("s3://tropomi/2024/01/15/...")
# not pinned (georeader)
emit_l2 = gr.EMIT_L2.from_url("s3://emit/2024/01/15/...")
# not pinned (georeader)
ghgsat_l2 = gr.GHGSat_L2.from_url("s3://ghgsat/2024/01/15/...")

# Each returns a GeoTensor with attached AK metadata
# Build vardax Batch from the multi-instrument set
# user
batch = build_batch_from_readers([tropomi_l2, emit_l2, ghgsat_l2], met=met_field)
```

---

## coordax — Coordinate-aware fields

Open question 1 from `boundaries.md`. Current design: vardax `Batch*` use
raw `Array`. Optional `[coords]` extra exposes `coordax` adapters:

```{code-block} python
:caption: Builds a vardax batch (y, m) of the observation model [](#eq-obs-model) from a coordax dataset.
:label: code-design-integration-coordax
import vardax as vdx

# Given: coordax_ds (a coordax.Dataset with ssh, ssh_obs and mask)

# proposed (boundaries.md, open question 1)
batch = vdx.adapters.coordax.batch_from_coordax_dataset(
    coordax_ds, state_var="ssh", obs_var="ssh_obs", mask_var="mask"
)
# Coordinates preserved in Batch.metadata for posterior provenance
# (shipped Batch1D / Batch2D carry only input, mask and target).
```

---

## GeoCatalog — Event-driven batch assembly

```{code-block} python
:caption: Queries overpasses for a methane event, inverts them with the Tier I model of [](#code-design-integration-plumax), and writes the Gauss-Newton posterior [](#eq-gn-cov) back to the catalog.
:label: code-design-integration-geocatalog
import geotoolz
import vardax as vdx

# Shapes: P source parameters
# Given: event, met_field, inversion (the IncrementalFourDVar above)

# Query overpasses across instruments for a methane event
# not pinned (geotoolz)
catalog = geotoolz.GeoCatalog.from_geoparquet("s3://catalog/overpasses.parquet")
# not pinned (geotoolz)
overpasses = catalog.query(
    geometry=event.bbox(buffer_km=50),
    interval=event.window(hours=2),
    instruments=["TROPOMI", "EMIT", "GHGSat"],
    quality_min="usable",
)

# Assemble multi-instrument batch
# user
batch = build_event_batch(overpasses, met=met_field)

# Invert
# today (vardax)
# x* after the last outer iteration
analysis = inversion(batch)  # → (1, P)
# today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
# P*_GN = 𝐇_GN⁻¹ at x*
posterior = vdx.GaussNewtonHessian()(analysis, inversion.as_analysis_step(), batch)

# Write back to catalog
# today (vardax)
mark_likelihood = vdx.GaussianMarkLikelihood(posterior, event.metadata)
# not pinned (geotoolz)
catalog.write_posterior(event.id, mark_likelihood)
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
