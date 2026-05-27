# SSH Reconstruction Example

Sea surface height (SSH) reconstruction from satellite altimetry is the
canonical large-scale 4DVarNet benchmark (Fablet et al. 2021, 2023).
The state is a 2D field; observations are along-track samples from
altimetric satellites (Jason, SARAL/AltiKa, SWOT); the gaps between
tracks need to be filled.

This chapter shows three approaches to the same problem: OI as the
classical baseline, `IncrementalFourDVar` as the operational pattern,
and `FourDVarNet` as the learned alternative. Same data, three
analyses, side-by-side comparison.

## Problem setup

State: SSH on a regular lon-lat grid, $H \times W = 128 \times 128$
typical for a regional patch. Observations: along-track samples from
one or more altimeters, with along-track noise $\sigma \sim 3$ cm and
gaps of $\sim 100$ km perpendicular to track.

```python
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import gaussx as gx
import optimistix as optx
import diffrax as dfx

from vardax import Batch2D
from vardax.obs_operators import MaskedIdentity, LinearObs

# Build a Batch2D from preprocessed altimetric data
batch = Batch2D(
    input=ssh_observed,          # (B, T, H, W) with NaN in gaps
    mask=track_mask,             # 1 along-track, 0 in gaps
    target=ssh_truth,            # ground truth (training only)
    obs_err=altika_uncertainty,
)
```

## Approach 1 — `OptimalInterpolation`

The classical baseline. Linear $H$ (masked identity), Gaussian
Matérn prior, Gaussian diagonal obs error — closed-form fast path.

```python
from vardax.models import OptimalInterpolation

coords = jnp.stack(jnp.meshgrid(lon, lat, indexing="ij"), axis=-1)
B_op = gx.MaternLinearOperator(
    grid_coords=coords.reshape(-1, 2), length_scale=100.0, sigma=0.1, nu=1.5,
)
R_op = lx.DiagonalLinearOperator(altika_uncertainty.flatten() ** 2)

oi = OptimalInterpolation(
    obs_op=MaskedIdentity(),
    prior_mean=climatology_ssh,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
)

# Single forward pass — no iteration
x_oi = oi(batch)
posterior_oi = oi.posterior(batch)
# Reconstruction in gaps comes from the Matérn correlation length.
```

What this gives you: the standard OceanBench DUACS baseline. Smoothed
through the prior length scale. Easy to interpret, easy to validate.
Misses mesoscale features that are shorter than the correlation length.

## Approach 2 — `IncrementalFourDVar` with shallow-water dynamics

Add a `somax.ShallowWaterModel` as the dynamics. Multi-time
observations now constrain the trajectory through the physics — gaps
get filled by dynamical propagation rather than just smoothing.

```python
import somax
from vardax.models import IncrementalFourDVar
from vardax import IncrementalConfig

swm = somax.ShallowWaterModel(grid=grid, params=ssh_params)

incremental = IncrementalFourDVar(
    forward=swm,
    obs_op=MaskedIdentity(),
    prior_mean=climatology_ssh,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)

x_inc = incremental(batch)
posterior_inc = incremental.posterior(batch)
# Mesoscale eddies now propagate consistently across observation gaps;
# the posterior reflects model + observation uncertainty.
```

What this gives you: physically-consistent reconstruction. The
incremental machinery scales to operational grids
(~$10^6$–$10^7$ cells) thanks to the CVT-preconditioned CG.

## Approach 3 — `FourDVarNet` with learned prior

Train a `FourDVarNet` on (observed, truth) patches. The learned prior
$\varphi_\theta$ captures the SSH manifold; the learned grad modulator
$\Phi_\phi$ accelerates inner iterations.

```python
from vardax.models import FourDVarNet
from vardax.priors import BilinAEPrior2D
from vardax.grad_mod import ConvLSTMGradMod2D
from vardax import SolverConfig
from vardax.adjoints import OneStepAdjoint
from vardax.training import train_step

model = FourDVarNet(
    prior=BilinAEPrior2D(latent_dim=128, n_time=10, height=128, width=128),
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod2D(state_channels=10, hidden_dim=64),
    config=SolverConfig(n_steps=15, alpha=0.2),
    solver_adjoint=OneStepAdjoint(),       # O(1) memory training
)

# Train on the OceanBench dataset
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def step(model, opt_state, batch):
    model, opt_state, loss = train_step(model, batch, optimizer, opt_state)
    return model, opt_state, loss

for epoch in range(100):
    for batch in oceanbench_loader:
        model, opt_state, loss = step(model, opt_state, batch)

# Reconstruction
x_net = model(batch)
```

What this gives you: state-of-the-art reconstruction on the
distribution the network was trained on. Better than OI in
information-rich regions; comparable to incremental 4DVar without
needing somax. Limitations: extrapolates poorly outside the training
distribution; posterior calibration via Laplace is less defensible
than the GN-Hessian-based posterior from incremental 4DVar.

## Comparison

| Method | Setup cost | Per-event cost | Recovers mesoscale? | Posterior quality | When |
|---|---|---|---|---|---|
| `OptimalInterpolation` | trivial | low (single pass) | only down to prior length scale | exact (closed form) | Linear-Gaussian baseline |
| `IncrementalFourDVar` | requires `somax` model | medium (3 outer × 20 inner CG) | yes, via dynamics | calibrated (GN-Hessian) | Production / operational |
| `FourDVarNet` | requires training | low (15 solver steps) | yes, via learned prior | needs validation | Data-rich regime |

The OceanBench benchmark (Le Guillou et al. 2023) provides a standard
dataset for these comparisons. The expected ordering is `OI < IncrementalFourDVar
≈ FourDVarNet`, with `FourDVarNet` matching `IncrementalFourDVar` on
in-distribution data but degrading more gracefully when the input
distribution drifts.

## Operational cycling

For continuous SSH analysis (every overpass, every few hours), wrap
the chosen analysis in a `pipekit_cycle.DACycle`:

```python
import pipekit_cycle as pc

ssh_cycle = pc.DACycle(
    forward_model=swm,
    obs_op=MaskedIdentity(),
    analysis_step=incremental.as_analysis_step(),    # or oi / net
    obs_source=along_track_loader,
    n_steps=n_assimilation_windows,
)

# Run continuously
result, final_state = ssh_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

The `analysis_step` slot accepts any of the three methods — swap
`OptimalInterpolation` for `IncrementalFourDVar` for `FourDVarNet` by
changing one slot. Nothing in the cycling, data loading, or output
serialisation changes.

## Persistence

For trained `FourDVarNet`, persist via `pipekit-jax.JaxModelOp` +
`pipekit-experiment.ModelRegistry`:

```python
from pipekit_jax import JaxModelOp
from pipekit_experiment import LocalModelRegistry

registry = LocalModelRegistry(root="./ssh_models")
model_op = JaxModelOp(model)
hash_ = registry.store(
    model_op, weights=model_op.serialize_weights(),
    tags={"task": "ssh_oceanbench", "version": "v1"},
)

# Later — reload with a fresh skeleton
template = JaxModelOp(make_fresh_skeleton())
reloaded = template.with_weights(registry.load_weights(hash_))
```

`OptimalInterpolation` and `IncrementalFourDVar` have no learned
parameters; they need no persistence (the `prior_cov_op` and config
are reconstructed from numerical parameters).

## See also

- Chapter 4 — `OptimalInterpolation` (BLUE / OI)
- Chapter 8 — `IncrementalFourDVar` (operational fast path)
- Chapter 9 — `FourDVarNet` (learned variant)
- Chapter 14 — six-step cycle (the methodology that ties this together)

## References

- Fablet, R., Febvre, Q., & Chapron, B. (2023). *Multimodal 4DVarNets
  for the reconstruction of sea surface dynamics from NADIR and
  wide-swath altimetry.* IEEE TGRS 61.
- Le Guillou, F., et al. (2023). *Mapping altimetry in the forthcoming
  SWOT era by back-and-forth nudging a one-layer quasigeostrophic
  model.* JTECH 40(1). [OceanBench reference.]
- Ubelmann, C., Klein, P., & Fu, L.-L. (2015). *Dynamic
  interpolation of sea surface height and potential applications for
  future high-resolution altimetry mapping.* JTECH 32.
