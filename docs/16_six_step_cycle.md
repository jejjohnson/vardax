# Six-Step Inference Cycle

The "six-step inference cycle" is the research-to-operations methodology
vardax is engineered around. Across all forward-model tiers, the same
sequence applies:

```
(1) Physics forward            — somax / plumax
       ↓
(2) Model-based inference      — MAP / MCMC / 4DVarNet    (slow, exact)
       ↓
(3) Neural emulator            — trained from Step 1      (fast surrogate)
       ↓
(4) Emulator-based inference   — same vardax loop         (100–1000× faster)
       ↓
(5) Amortized predictor        — y → posterior directly   (sub-second)
       ↓
(6) Improve                    — swap any block; oracle = previous step
```

vardax's job: make Steps 2, 4, and 5 use **the same library code**
parameterised only by which `ForwardModel` and which inference family
(`VarDANet*` / `IncrementalVarDA*` / `AmortizedVarDA*`) is plugged in. The
fixed protocol surface (Decision D8) is what makes this possible.

## Step 1 — Physics forward

Implemented in `somax` / `plumax`. Satisfies `pipekit_cycle.ForwardModel`.

```python
forward = plumax.tier1.GaussianPlume(met=met, dispersion="MO")
# forward.step(state, dt) → state ✓
```

vardax doesn't own this — its job starts when a forward is available.

## Step 2 — Model-based inference

Run vardax `IncrementalVarDA*` (operational), `VarDANet*` (learned), or
NumPyro MCMC. Slow but exact:

```python
inversion = vdx.models.IncrementalVarDA2D(
    forward=forward, obs_op=fusion,
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20),
)
x_star = inversion(batch)
posterior = vdx.posterior.GaussNewtonHessian()(x_star, inversion.as_analysis_step(), batch)
```

This is the **oracle** for Step 4.

## Step 3 — Train a neural emulator

The forward is now the bottleneck. Train an emulator $F_\psi$ that mimics
the physics forward:

$$F_\psi(\mathbf{x}) \approx \text{Forward}(\mathbf{x})$$

Two training-time gates must hold before promotion to Step 4:

1. **Forward agreement.** $\|F_\psi(\mathbf{x}) - \text{Forward}(\mathbf{x})\| / \|\text{Forward}(\mathbf{x})\| < \epsilon_\text{fwd}$
   on held-out states (typically $\epsilon_\text{fwd} = 0.01$).
2. **Adjoint calibration.** $\|\partial F_\psi / \partial \mathbf{x} - \partial \text{Forward} / \partial \mathbf{x}\|_\text{op} < 0.05$
   via random-vector probing. **This is the hard gate** — a fast forward
   with a wrong Jacobian gives wrong posteriors.

The emulator $F_\psi$ implements `pipekit_cycle.ForwardModel` so it's a
drop-in for `forward` in Step 4.

## Step 4 — Emulator-based inference

Swap `forward` → `emulator` in the same vardax code:

```python
emulator = trained_neural_forward  # satisfies ForwardModel

inversion_em = vdx.models.IncrementalVarDA2D(
    forward=emulator,             # ← only change from Step 2
    obs_op=fusion,
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20),
)
x_star_em = inversion_em(batch)
posterior_em = vdx.posterior.GaussNewtonHessian()(x_star_em, inversion_em.as_analysis_step(), batch)
```

**Validation gate (post-promotion):** Step 4 output agrees with Step 2:

$$|\mathbf{x}^*_\text{em} - \mathbf{x}^*_\text{phys}| / \sigma_\text{post,phys} \le 1$$

on a held-out set of events. Fails ⇒ either retrain emulator with broader
distribution or roll back.

## Step 5 — Amortized predictor

Train `AmortizedVarDA` head $q_\phi(\mathbf{x} \mid \mathbf{y})$ on
simulated $(\mathbf{x}, \mathbf{y})$ pairs from the physics forward (or
emulator, if already validated):

```python
amort = vdx.models.AmortizedVarDA(
    encoder=ConvObsEncoder(...),
    head=ConditionalFlowHead(...),
    config=vdx.AmortizedConfig(head_type="flow"),
)

# Train on simulations
for batch in simulation_loader:
    amort, opt_state, loss = vdx.training.train_step(amort, batch, optimizer, opt_state)

# Inference is now sub-second
x_map = amort(batch)
samples = amort.sample(batch, key, n=200)
```

**Validation gates** (Decision D12):
- Posterior agreement vs Step 2 within $1\sigma_\text{post}$.
- Adjoint calibration $< 5\%$.
- SBC rank histograms uniform.

## Step 6 — Improve

The cycle is a loop. Improvements at any step trigger re-validation
downstream:

| Change | Triggers re-validation of |
|---|---|
| New physics in Step 1 | Steps 2, 3, 4, 5 |
| New inference method in Step 2 | Step 4 (Step 2 is oracle) |
| Better emulator (Step 3) | Steps 4, 5 |
| Better amortized head (Step 5) | Step 5 vs Step 2/4 |
| New observation operator (e.g. instrument added) | Steps 2, 4, 5 |

vardax's contribution: the validation gates are **part of the test suite**,
not just documentation. `tests/test_six_step_validation.py` enforces them
in CI.

```python
# Pseudocode for the validation suite

def test_step4_agrees_with_step2(physics_inversion, emulator_inversion, val_events):
    for ev in val_events:
        p_phys = LaplaceCovariance()(physics_inversion(ev.batch),
                                       physics_inversion.as_analysis_step(), ev.batch)
        p_em   = LaplaceCovariance()(emulator_inversion(ev.batch),
                                       emulator_inversion.as_analysis_step(), ev.batch)
        assert_posterior_agreement(p_em, p_phys, tolerance_sigma=1.0)


def test_amortized_adjoint_calibrated(amortized, physics_inversion, val_events):
    for ev in val_events:
        grad_amort = jax.grad(lambda y: amortized.encode_decode_map(y))(ev.y)
        grad_phys = jax.grad(lambda y: physics_inversion(ev.batch_with_y(y)))(ev.y)
        op_norm = relative_operator_norm(grad_amort, grad_phys, n_probe=20)
        assert op_norm < 0.05


def test_sbc_uniform(amortized, prior, forward, n_runs=200):
    ranks = simulation_based_calibration(amortized, prior, forward, n_runs=n_runs)
    chi2_p = chi2_uniformity_test(ranks)
    assert chi2_p > 0.01, f"SBC failed uniformity test: p={chi2_p}"
```

## Why the cycle matters

The Step 6 loop is the key. Without explicit gates, "faster inference"
silently becomes "wrong inference, faster". By codifying:

- Step 2 = oracle for Step 4
- Step 4 = oracle for Step 5 (when Step 4 is itself validated)
- Adjoint calibration as a hard gate before any promotion
- SBC as continuous monitoring

vardax makes the research-to-operations arc auditable, not just fast.

## See also

- Chapter 13: Incremental 4DVar — Step 2 / Step 4 implementation
- Chapter 15: Amortized inference — Step 5 implementation
- Chapter 14: Posterior covariance — gate computations
- Decision [D12](design/decisions.md#d12-six-step-inference-cycle-as-testing-scaffold)
  — methodology rationale
- Design doc: [`design/examples/use_cases.md`](design/examples/use_cases.md)
  — methane single-overpass walkthrough exercises Steps 1–5
- `vardax.utils.validation` module — gate implementations
