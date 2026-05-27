---
status: draft
version: 0.3.0
---

# Layer 2 — Model Examples

End-to-end workflows for the three model families.

---

## Training a `VarDANet2D` — learned 4DVarNet

```python
import equinox as eqx
import optax
from vardax.training import train_step

model = VarDANet2D(
    prior=ConvAEPrior(encoder=enc_2d, decoder=dec_2d),
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod2D(hidden_dim=64),
    config=SolverConfig(n_steps=15, grad_mode="one_step"),
)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def step(model, opt_state, batch):
    model, opt_state, loss = train_step(model, batch, optimizer, opt_state)
    return model, opt_state, loss

for epoch in range(100):
    for batch in dataloader:
        model, opt_state, loss = step(model, opt_state, batch)
```

After training, the model is **also** an `AnalysisStep` via
`model.as_analysis_step()` for use in operational cycling.

---

## Running an `IncrementalVarDA2D` — operational 4DVar

`IncrementalVarDA*` is not learned — no training loop. Just configure and
run:

```python
import gaussx as gx
import lineax as lx
from vardax.models import IncrementalVarDA2D
from vardax import IncrementalConfig

# Prior covariance: Matérn-3/2 via gaussx (Decision D11)
B_op = gx.MaternLinearOperator(
    grid_coords=coords, length_scale=10.0, nu=1.5, sigma=1.0,
)

R_op = lx.DiagonalLinearOperator(obs_err_variances)

model = IncrementalVarDA2D(
    forward=somax_model,        # pipekit_cycle.ForwardModel
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),
    prior_mean=x_b,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)

# Operational analysis
x_star = model(batch)

# Or as an AnalysisStep for DACycle:
analysis_step = model.as_analysis_step()
```

For posterior covariance, reuse the GN Hessian from the last outer
iteration:

```python
from vardax.posterior import GaussNewtonHessian

posterior = GaussNewtonHessian(n_krylov=50)(x_star, model.as_analysis_step(), batch)
```

---

## Training an `AmortizedVarDA` — direct posterior head

```python
from vardax.models import AmortizedVarDA
from vardax import AmortizedConfig

model = AmortizedVarDA(
    encoder=ConvObsEncoder(...),       # eqx.Module: y, mask → context
    head=ConditionalFlowHead(...),     # eqx.Module: context → q_φ
    config=AmortizedConfig(head_type="flow", n_samples=64),
)

# Training data from simulation (Decision D12, Step 5):
def sample_train_pair(key):
    x = prior_distribution.sample(key)
    y_clean = forward_model(x)
    y = y_clean + obs_noise.sample(key)
    return Batch2D(input=y, mask=quality_mask, target=x)

# Training loop
for batch in simulation_dataloader:
    model, opt_state, loss = train_step(model, batch, optimizer, opt_state)
```

Validation gates (Decision D12) — must agree with physics oracle:

```python
from vardax.utils.validation import assert_posterior_agreement, simulation_based_calibration

for val_batch in val_loader:
    p_amortized = LaplaceCovariance()(model(val_batch), model.as_analysis_step(), val_batch)
    p_physics = LaplaceCovariance()(physics_model(val_batch),
                                     physics_model.as_analysis_step(), val_batch)
    assert_posterior_agreement(p_amortized, p_physics, tolerance_sigma=1.0)

simulation_based_calibration(model, prior_distribution, forward_model, n_runs=200)
```

---

## Cycling any model through `pipekit_cycle.DACycle`

All three model families satisfy `AnalysisStep` via `.as_analysis_step()`,
so the cycling code is identical:

```python
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=AveragingKernel(...),
    analysis_step=model.as_analysis_step(),   # any of VarDANet / Incremental / Amortized
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)

result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

---

## Dimensional variants

Same algorithm, different spatial dims:

```python
from vardax.models import (
    VarDANet1D, VarDANet2D, VarDANet3D,
    IncrementalVarDA2D, IncrementalVarDA3D,
)

# 1D: time series, transects
model_1d = VarDANet1D(prior=ae_1d, obs_op=obs_op_1d, ...)

# 2D: SSH, SST, surface methane column
model_2d = IncrementalVarDA2D(forward=somax_2d, obs_op=ak_2d, ...)

# 3D: volumetric ocean / atmosphere (e.g. plumax Tier III Eulerian)
model_3d = IncrementalVarDA3D(forward=plumax_eulerian, obs_op=mi_fusion, ...)
```
