---
status: draft
version: 0.4.0
---

# Layer 2 — Model Examples

End-to-end workflows for each of the seven peer analysis classes.

---

## `OptimalInterpolation` — closed-form fast path

```python
import gaussx as gx
import lineax as lx
from vardax.models import OptimalInterpolation
from vardax.obs_operators import LinearObs

model = OptimalInterpolation(
    obs_op=LinearObs(H_mat=along_track_op),     # must be linear
    prior_mean=climatology_ssh,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=100.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(altika_variances),
)

# Single forward pass — no iteration
x_star = model(batch)
posterior = model.posterior(batch)
```

Use when $H$ is linear, $B$ and $R$ are Gaussian. The right default for
SSH altimetry with along-track observations and a Matérn prior.

---

## `ThreeDVar` — nonlinear, single time

```python
import optimistix as optx
from vardax.models import ThreeDVar

model = ThreeDVar(
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),   # nonlinear via h ⊙ x
    prior_mean=x_b,
    prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
)
x_star = model(batch)
```

Use when $H$ is nonlinear (averaging kernel, RTM, custom retrieval) but
there are no dynamics — snapshot inversion.

---

## `StrongFourDVar` — multi-time, exact dynamics

```python
import diffrax as dfx
import optimistix as optx
from vardax.models import StrongFourDVar

model = StrongFourDVar(
    forward=somax_model,                      # pipekit_cycle.ForwardModel
    obs_op=MaskedIdentity(),
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)
x_star = model(batch)
```

For long assimilation windows, switch the dynamics adjoint:

```python
forward_adjoint = dfx.BacksolveAdjoint()    # continuous adjoint, O(1) memory
# or
forward_adjoint = dfx.ForwardMode()         # forward sensitivity (small parameter dim)
```

---

## `WeakFourDVar` — model-error-aware

```python
from vardax.models import WeakFourDVar

model = WeakFourDVar(
    forward=somax_model,                      # M_t^free
    obs_op=obs_op,
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    model_err_cov_op=Q_op,                    # Q for the η_t prior
    minimiser=optx.NonlinearCG(rtol=1e-5),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)

# Returns both the analysis IC and the model-error trajectory
x_0_star, eta_star = model(batch)
```

Use when the model has known biases (climatological drift) and you need
to estimate the time-varying $\eta_t$ alongside the initial state.

---

## `IncrementalFourDVar` — operational fast path

```python
from vardax.models import IncrementalFourDVar
from vardax import IncrementalConfig

model = IncrementalFourDVar(
    forward=somax_model,
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),
    prior_mean=x_b,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=10.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(obs_uncertainty),
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)

# Operational analysis
x_star = model(batch)

# Posterior reuses GN Hessian from last outer iteration
posterior = model.posterior(batch)
```

Use for production / operational 4DVar with structured Matérn priors.
Functionally equivalent to `StrongFourDVar` but with the GN+CG+CVT
operational pattern hard-wired.

---

## `FourDVarNet` — learned 4DVar

```python
import equinox as eqx
import optax
from vardax.models import FourDVarNet
from vardax import SolverConfig
from vardax.adjoints import OneStepAdjoint
from vardax.training import train_step

model = FourDVarNet(
    prior=ConvAEPrior(encoder=enc_2d, decoder=dec_2d),
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod2D(hidden_dim=64),
    config=SolverConfig(n_steps=15, alpha=0.2),
    solver_adjoint=OneStepAdjoint(),
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

After training, `model.as_analysis_step()` gives the operational
interface for use in `pipekit_cycle.DACycle`.

---

## `AmortizedPosterior` — direct head

```python
from vardax.models import AmortizedPosterior
from vardax import AmortizedConfig

model = AmortizedPosterior(
    encoder=ConvObsEncoder(...),       # eqx.Module: (y, mask) → context
    head=ConditionalFlowHead(...),     # gauss_flows-based head
    config=AmortizedConfig(head_type="flow", n_samples=64),
)

# Training data from simulation
def sample_train_pair(key):
    x = prior_distribution.sample(key)
    y_clean = forward_model(x)
    y = y_clean + obs_noise.sample(key)
    return Batch2D(input=y, mask=quality_mask, target=x)

for batch in simulation_loader:
    model, opt_state, loss = train_step(model, batch, optimizer, opt_state)

# Inference: sub-second
x_map = model(batch)
samples = model.sample(batch, key, n=200)
```

Validation gates per Decision D12:

```python
from vardax.utils.validation import (
    assert_posterior_agreement, simulation_based_calibration,
)

for val_batch in val_loader:
    p_amort = LaplaceCovariance()(model(val_batch),
                                    model.as_analysis_step(), val_batch)
    p_phys = LaplaceCovariance()(strong_4dvar(val_batch),
                                   strong_4dvar.as_analysis_step(), val_batch)
    assert_posterior_agreement(p_amort, p_phys, tolerance_sigma=1.0)

simulation_based_calibration(model, prior_distribution, forward_model, n_runs=200)
```

---

## Cycling any model through `pipekit_cycle.DACycle`

All seven satisfy `AnalysisStep` via `.as_analysis_step()` — the cycling
code is identical:

```python
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=AveragingKernel(...),
    analysis_step=model.as_analysis_step(),   # any of the seven
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)

result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

The orchestration is decoupled from the inference algorithm — swap
`OptimalInterpolation` for `IncrementalFourDVar` for `FourDVarNet` by
changing the `analysis_step` slot. Nothing else in the pipeline
changes.

---

## Linear-Gaussian agreement test (Decision D14 invariant)

```python
from vardax.utils.validation import assert_methods_agree

# All seven should agree in the linear-Gaussian limit
batch = make_linear_gaussian_batch()

oi = OptimalInterpolation(linear_obs_op, x_b, B_op, R_op)
threedvar = ThreeDVar(linear_obs_op, x_b, B_op, R_op, optx.GaussNewton(rtol=1e-7))
strong = StrongFourDVar(static_forward, linear_obs_op, x_b, B_op, R_op,
                         optx.NonlinearCG(rtol=1e-7))
incremental = IncrementalFourDVar(static_forward, linear_obs_op, x_b, B_op, R_op,
                                    IncrementalConfig(n_outer=5, n_inner=50))
fourdvarnet = FourDVarNet(IdentityPrior(), linear_obs_op,
                           IdentityGradMod(alpha=0.05),
                           SolverConfig(n_steps=200))

assert_methods_agree(
    {"oi": oi, "3dvar": threedvar, "strong": strong,
     "incremental": incremental, "fourdvarnet": fourdvarnet},
    batch, atol=1e-3,
)
```

This is the canonical correctness baseline — every method, classical and
learned, must produce the same posterior mean in the linear-Gaussian
limit.
