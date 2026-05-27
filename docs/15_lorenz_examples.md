# Lorenz Examples

The Lorenz systems are the canonical small-scale testbeds for DA.
Their chaotic dynamics expose what each method does well or badly in a
setting where ground truth is available, integration is cheap, and the
state dimension is small enough to visualise.

This chapter walks the seven analysis methods through Lorenz-63 (3D
chaotic system) and Lorenz-96 (40D periodic system). The goal is to
make each method's behaviour concrete.

## Lorenz-63

The Lorenz-63 system:

$$
\dot{x} = \sigma(y - x), \quad \dot{y} = x(\rho - z) - y, \quad \dot{z} = xy - \beta z,
$$

with canonical $(\sigma, \rho, \beta) = (10, 28, 8/3)$. State
dimension $n = 3$; dynamics live on the famous two-winged attractor.

```python
from vardax._src.utils.dynamical_systems import simulate_lorenz63

key = jax.random.PRNGKey(0)
time, trajectory = simulate_lorenz63(
    key, sigma=10, rho=28, beta=8/3, dt=0.01,
    n_steps=5000, n_burn_in=1000,
)
# trajectory: (5001, 3)
```

### Synthetic observation scenario

Observe the state every $k$ steps with Gaussian noise:

```python
from vardax._src.utils.masks import regular_mask
from vardax._src.utils.noise import add_gaussian_noise

# Wrap in xarray for the data utilities
ds = trajectory_to_xr_dataset(trajectory, time)

# Observe every 10 steps, σ = 1.0
ds = regular_mask(ds, variable="state", obs_interval=10)
ds = add_gaussian_noise(ds, variable="state", sigma=1.0, name="obs")

# Convert to Batch1D
batch = xr_to_batch1d(ds, state_var="state", obs_var="obs", mask_var="mask")
```

### `OptimalInterpolation` won't work directly

The Lorenz dynamics are nonlinear; the assumption of static field
(needed for OI) is violated. OI on the marginal density of the state
gives a poor reconstruction — but it's a useful baseline:

```python
from vardax.models import OptimalInterpolation
from vardax.obs_operators import MaskedIdentity

# Treat as static — wrong but instructive
oi = OptimalInterpolation(
    obs_op=MaskedIdentity(),
    prior_mean=trajectory.mean(0),
    prior_cov_op=lx.MatrixLinearOperator(jnp.cov(trajectory.T)),
    obs_cov_op=lx.DiagonalLinearOperator(jnp.ones(3)),
)
x_oi = oi(batch)
# Reconstruction is biased toward the prior mean — illustrates why
# you need dynamics for chaotic systems.
```

### `StrongFourDVar` is the right tool

```python
import diffrax as dfx
import optimistix as optx
from vardax.models import StrongFourDVar
from vardax.priors import DynamicalPrior

# Wrap the Lorenz-63 vector field as a ForwardModel
forward = wrap_l63_as_forward_model(sigma=10, rho=28, beta=8/3, dt=0.01)

# Strong-constraint 4DVar with the true dynamics
strong = StrongFourDVar(
    forward=forward,
    obs_op=MaskedIdentity(),
    prior_mean=trajectory[0],
    prior_cov_op=lx.DiagonalLinearOperator(jnp.ones(3) * 100),  # weak prior
    obs_cov_op=lx.DiagonalLinearOperator(jnp.ones(3)),
    minimiser=optx.NonlinearCG(rtol=1e-6, atol=1e-6),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)
x_strong = strong(batch)

# x_strong recovers the chaotic trajectory; OI does not
```

Diagnostic: plot the attractor reconstruction. `StrongFourDVar`
recovers the butterfly; `OptimalInterpolation` smears across both
wings because the static prior doesn't know about the dynamics.

### `FourDVarNet` for the learned-prior comparison

Train a small AE prior on a long training trajectory:

```python
from vardax.models import FourDVarNet
from vardax.priors import MLPAEPrior1D
from vardax.grad_mod import MLPGradMod
from vardax import SolverConfig
from vardax.adjoints import OneStepAdjoint

net = FourDVarNet(
    prior=MLPAEPrior1D(state_dim=3, latent_dim=8, hidden_dim=32, n_time=20),
    obs_op=MaskedIdentity(),
    grad_mod=MLPGradMod(hidden_dim=64),
    config=SolverConfig(n_steps=15, alpha=0.2),
    solver_adjoint=OneStepAdjoint(),
)

# Train on (masked_input, ground_truth) pairs from extracted patches
# ... train_step loop ...

x_net = net(batch)
```

`FourDVarNet` learns the attractor structure from the training data
and reconstructs without explicit access to the dynamics. The
comparison `x_strong` vs `x_net` is the canonical demonstration that
learned priors recover similar information to model-based ones.

## Lorenz-96

The Lorenz-96 system is a one-dimensional periodic version:

$$
\dot{X}_k = (X_{k+1} - X_{k-2}) X_{k-1} - X_k + F, \quad k = 0, \ldots, N-1,
$$

with $N = 40$ and forcing $F = 8$ (chaotic regime). The state is a
periodic 1D field — closer in structure to atmospheric problems than
Lorenz-63.

```python
from vardax._src.utils.dynamical_systems import simulate_lorenz96

key = jax.random.PRNGKey(0)
time, trajectory = simulate_lorenz96(key, N=40, F=8.0, dt=0.01, n_steps=5000)
# trajectory: (5001, 40)
```

### Sparse observation scenario

Observe every other cell, sparse in time:

```python
from vardax._src.utils.masks import random_mask

ds = trajectory_to_xr_dataset(trajectory, time, feature_names=[f"x{i}" for i in range(40)])
ds = random_mask(ds, variable="state", missing_rate=0.5)
ds = add_gaussian_noise(ds, variable="state", sigma=0.5)
batch = xr_to_batch1d(ds)
```

### `IncrementalFourDVar` with structured Matérn prior

This is the realistic operational scenario: structured spatial prior,
incremental solver, AK-style or masked observations.

```python
import gaussx as gx
from vardax.models import IncrementalFourDVar
from vardax import IncrementalConfig

# Matérn-3/2 prior on the periodic 40-cell grid
B_op = gx.MaternLinearOperator(
    grid_coords=jnp.arange(40), length_scale=4.0, nu=1.5, sigma=1.0,
    periodic=True,
)

incremental = IncrementalFourDVar(
    forward=wrap_l96_as_forward_model(N=40, F=8.0),
    obs_op=MaskedIdentity(),
    prior_mean=jnp.full(40, 8.0),                # F-mean climatology
    prior_cov_op=B_op,
    obs_cov_op=lx.DiagonalLinearOperator(jnp.full(40, 0.25)),
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)

x_star = incremental(batch)
posterior = incremental.posterior(batch)
```

This walkthrough exercises:
- Periodic Matérn prior via `gaussx`
- CVT preconditioning (B^{1/2} from the Matérn factorisation)
- GN+CG inner loop
- Free posterior reuse of the GN Hessian

The reconstruction recovers the Lorenz-96 trajectory through gaps,
with calibrated posterior uncertainty in unobserved regions.

## Linear-Gaussian agreement test

Lorenz dynamics aren't linear, but we can construct a static
linear-Gaussian test on the Lorenz-63 attractor by treating the
trajectory mean as the prior and the empirical covariance as $B$:

```python
def test_lorenz_linear_gaussian_baseline():
    """In a single-time linear-Gaussian regime, all five classical
    methods + IdentityPrior 4DVarNet must agree."""

    # Build a single-time batch
    batch = Batch1D(
        input=noisy_obs, mask=obs_mask, target=truth,
    )
    x_b = trajectory.mean(0)
    B_op = lx.MatrixLinearOperator(jnp.cov(trajectory.T) + 0.01 * jnp.eye(3))
    R_op = lx.DiagonalLinearOperator(jnp.ones(3))

    oi = OptimalInterpolation(MaskedIdentity(), x_b, B_op, R_op)
    three = ThreeDVar(MaskedIdentity(), x_b, B_op, R_op,
                       minimiser=optx.GaussNewton(rtol=1e-8))
    strong = StrongFourDVar(static_forward, MaskedIdentity(),
                              x_b, B_op, R_op,
                              minimiser=optx.NonlinearCG(rtol=1e-8))
    incremental = IncrementalFourDVar(static_forward, MaskedIdentity(),
                                        x_b, B_op, R_op,
                                        config=IncrementalConfig(n_outer=2))
    net = FourDVarNet(IdentityPrior(), MaskedIdentity(),
                       IdentityGradMod(alpha=0.05),
                       SolverConfig(n_steps=200))

    results = {
        "oi": oi(batch), "3dvar": three(batch),
        "strong": strong(batch), "incremental": incremental(batch),
        "4dvarnet": net(batch),
    }
    reference = results["oi"]
    for name, x in results.items():
        assert jnp.allclose(x, reference, atol=1e-2), \
            f"{name} disagrees with OI in linear-Gaussian limit"
```

This is the Decision D14 invariant in action.

## Notebook examples

The notebooks/ directory contains end-to-end Lorenz tutorials:

- `01_model_based_4dvar_L63.py` — Classical 4DVar with known L63 dynamics
- `02_unrolling_vs_fixedpoint_L63.py` — Adjoint method comparison
- `03_4dvarnet_L63.py` — Full 4DVarNet on L63
- `05_end_to_end_L63.py` — Complete pipeline including data utilities
- `06_end_to_end_L96.py` — Lorenz-96 end-to-end
- `07_prior_pretraining_L63.py` — Pre-training the bilinear prior
- `08_classical_4dvar_vs_4dvarnet_L63.py` — Side-by-side comparison

Each follows jupytext percent-format; open as `.ipynb` with
`jupytext --to notebook`.

## See also

- Chapter 6 — `StrongFourDVar` (the L63 4DVar walkthrough method)
- Chapter 8 — `IncrementalFourDVar` (the L96 walkthrough method)
- Chapter 9 — `FourDVarNet` (the learned-prior comparison)
- Chapter 16 — SSH example (real-world 2D analogue)
- Chapter 17 — methane example (real-world multi-instrument analogue)
