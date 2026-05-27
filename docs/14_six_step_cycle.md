# Six-Step Inference Cycle

The "six-step cycle" is the research-to-operations methodology vardax
is engineered around. Across all forward-model tiers — geophysical
fluids, atmospheric transport, methane radiative transfer — the same
sequence applies:

```
(1) Physics forward            — somax / plumax
      ↓
(2) Classical inference        — OI / 3DVar / 4DVar           (slow, exact)
      ↓
(3) Neural emulator            — trained from Step 1          (fast surrogate)
      ↓
(4) Emulator-based inference   — same vardax code             (100–1000× faster)
      ↓
(5) Amortized predictor        — y → posterior in one pass    (sub-second)
      ↓
(6) Improve                    — swap any block; prior step is the oracle
```

The crucial property: **Steps 2, 4, and 5 use the same vardax code**.
The forward model is swapped via the `pipekit_cycle.ForwardModel`
protocol; the analysis class doesn't know whether $M_t$ is physics or
an emulator. The amortized predictor satisfies the same
`AnalysisStep` protocol as the classical method it replaces.

This is what makes the cycle a cycle and not a rewrite.

## Step 1 — Physics forward

Implemented in `somax` (geophysical fluids), `plumax` (atmospheric
transport / methane), or any user code that satisfies
`pipekit_cycle.ForwardModel`:

```python
import somax
swm = somax.ShallowWaterModel(grid=grid, params=params)
# swm.step(state, dt) → state ✓
# swm.dt → float ✓
# swm.state_signature → Signature ✓
assert isinstance(swm, ForwardModel)
```

Vardax does not own this layer.

## Step 2 — Classical inference

Pick one of the five classical methods from chapters 4–8 based on
regime:

| Regime | Method |
|---|---|
| Linear $H$, Gaussian $B$ / $R$ | `OptimalInterpolation` (chapter 4) |
| Nonlinear $H$, single time | `ThreeDVar` (chapter 5) |
| Multi-time, exact dynamics | `StrongFourDVar` (chapter 6) |
| Multi-time, model error | `WeakFourDVar` (chapter 7) |
| Operational 4DVar | `IncrementalFourDVar` (chapter 8) |

The result is the **oracle** for Step 4 and Step 5. Slow but exact (or
as exact as the prior / likelihood allow).

```python
oracle = IncrementalFourDVar(
    forward=somax_model, obs_op=obs_op,
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    config=IncrementalConfig(n_outer=3, n_inner=20),
)
x_star_oracle = oracle(batch)
posterior_oracle = oracle.posterior(batch)
```

## Step 3 — Train a neural emulator

The forward model is now the bottleneck. Train a surrogate $F_\psi$
that mimics the physics forward:

$$
F_\psi(x) \approx \text{Forward}(x).
$$

Two training-time gates must hold before promotion to Step 4:

### Forward agreement

$$
\frac{\|F_\psi(x) - \text{Forward}(x)\|}{\|\text{Forward}(x)\|} < \epsilon_\text{fwd}
$$

on held-out states (typically $\epsilon_\text{fwd} = 0.01$). This is
the obvious gate: does the emulator give the right answer on the
forward pass?

### Adjoint calibration — the hard gate

$$
\frac{\|\partial F_\psi / \partial x - \partial \text{Forward} / \partial x\|_\text{op}}{\|\partial \text{Forward} / \partial x\|_\text{op}} < 0.05
$$

via random-vector probing (estimate the operator norm of the
difference). A fast forward with a wrong Jacobian gives wrong
posteriors — the gradient that drives the inner minimisation will be
mis-directed.

This gate is **hard**: vardax refuses to promote an emulator that
fails adjoint calibration, even if the forward agreement is excellent.
Wrong-direction gradients are silently corrosive.

```python
from vardax.utils.validation import (
    assert_forward_agreement, assert_adjoint_calibrated,
)

assert_forward_agreement(emulator, physics_forward, val_states, eps=0.01)
assert_adjoint_calibrated(emulator, physics_forward, val_states, threshold=0.05)
```

The emulator $F_\psi$ satisfies `pipekit_cycle.ForwardModel`. It's a
drop-in replacement for `forward` in Step 4.

## Step 4 — Emulator-based inference

Swap `forward` → `emulator` in the same vardax code:

```python
emulator = trained_neural_forward                      # satisfies ForwardModel

fast_analysis = IncrementalFourDVar(
    forward=emulator,                                  # ← only change from Step 2
    obs_op=obs_op,
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    config=IncrementalConfig(n_outer=3, n_inner=20),
)
x_star_em = fast_analysis(batch)
posterior_em = fast_analysis.posterior(batch)
```

Validation gate (post-promotion): Step 4 output agrees with Step 2:

$$
\frac{|x^*_\text{em} - x^*_\text{phys}|}{\sigma_\text{post,phys}} \le 1
$$

on a held-out set of events. If this fails, the emulator is worse than
it looked on the per-state agreement test — retrain with broader
distribution or roll back.

The wall-clock speedup is typically 100–1000× over Step 2, depending on
the emulator's runtime vs. the physics forward.

## Step 5 — Amortized predictor

Train `AmortizedPosterior` (chapter 10) on simulated $(x, y)$ pairs
from the physics forward (or the emulator, if Step 3+4 are validated):

```python
amort = AmortizedPosterior(
    encoder=ConvObsEncoder(...),
    head=ConditionalFlowHead(...),
    config=AmortizedConfig(head_type="flow"),
)

# Train on simulations
for batch in simulation_loader:
    amort, opt_state, loss = train_step(amort, batch, optimizer, opt_state)

# Inference — sub-second
x_map = amort(batch)
samples = amort.sample(batch, key, n=200)
```

Validation gates (Decision D12):

- **Posterior agreement** with Step 2 within $1\sigma_\text{post}$
- **Adjoint calibration** $< 5\%$
- **SBC** rank histograms uniform

```python
from vardax.utils.validation import (
    assert_posterior_agreement, simulation_based_calibration,
)

for val_batch in val_loader:
    p_amort = LaplaceCovariance()(amort(val_batch),
                                    amort.as_analysis_step(), val_batch)
    p_phys = oracle.posterior(val_batch)
    assert_posterior_agreement(p_amort, p_phys, tolerance_sigma=1.0)

assert_adjoint_calibrated(amort, oracle, val_batches, threshold=0.05)
simulation_based_calibration(amort, prior, forward_model, n_runs=200)
```

`amort.as_analysis_step()` returns a `pipekit_cycle.AnalysisStep` —
same interface as Step 2 and Step 4. Operational pipelines swap in
the amortized predictor without touching the orchestration.

## Step 6 — Improve, repeat

The cycle is a loop. Improvements at any step trigger re-validation
downstream:

| Change | Re-validates |
|---|---|
| New physics in Step 1 | Steps 2, 3, 4, 5 |
| New inference method in Step 2 | Step 4 (Step 2 is oracle) |
| Better emulator (Step 3) | Steps 4, 5 |
| Better amortized head (Step 5) | Step 5 vs Step 2 / 4 |
| New observation operator | Steps 2, 4, 5 |

The validation gates are the contract — they're how you know an
"improvement" is actually an improvement.

## Why the cycle is the right framing

Three observations from operational DA:

- **Fast inference is dangerous when wrong.** Replacing a 4DVar with a
  flow predictor that runs in milliseconds is appealing — until the
  flow's posterior is miscalibrated and the alerting system fires on
  noise. The gates between steps make "fast" and "correct" both
  requirements, not just the former.
- **The forward model is the slow part, not the inference.** When
  the forward is a high-resolution ocean / atmosphere / plume model,
  iteration count matters less than per-iteration cost. Replacing the
  forward with an emulator (Step 3) is the largest single speedup
  available.
- **Research and operations need the same code.** A research scientist
  validates the methodology in a Jupyter notebook. An operational
  engineer deploys it behind a FastAPI handler. If the code paths
  diverge, drift between research validation and operational
  performance is inevitable. The cycle keeps them aligned.

Vardax codifies the cycle: the protocols are the same across steps,
the gates are part of the test suite, and the same `AnalysisStep`
instance can run in a notebook, a batch pipeline, and a streaming API.

## Implementation

> **Planned (v0.4 design target).** The validation utilities and the
> test module below are part of the equinox migration roadmap (Epic 1
> + Epic 8); they are not yet present in the v0.1.x codebase.

`vardax._src.utils.validation` will ship:

- `assert_forward_agreement(emulator, physics, val_states, eps)`
- `assert_adjoint_calibrated(emulator, physics, val_states, threshold)` —
  random-vector operator norm probing
- `assert_posterior_agreement(p_fast, p_oracle, tolerance_sigma)`
- `simulation_based_calibration(model, prior, forward, n_runs)` —
  returns ranks, callable as a pytest fixture

`tests/test_six_step_validation.py` wires these into CI. The gates
block merges that promote an emulator or amortized head past its
validation level.

## See also

- Chapter 9 — `FourDVarNet` (a hybrid case: classical 4DVar with
  learned components, validated against classical oracle)
- Chapter 10 — `AmortizedPosterior` (Step 5 implementation)
- Chapter 13 — posterior covariance (gate computations)
- Design doc: [`design/decisions.md#d12`](design/decisions.md#d12-six-step-inference-cycle-as-testing-scaffold)

## References

- Cranmer, K., Brehmer, J., & Louppe, G. (2020). *The frontier of
  simulation-based inference.* PNAS 117(48). [Frames the
  simulation-based-inference half of the cycle.]
- Talts, S., et al. (2018). *Validating Bayesian inference algorithms
  with simulation-based calibration.* arXiv:1804.06788.
- Kashinath, K., et al. (2021). *Physics-informed machine learning:
  case studies for weather and climate modelling.* Phil. Trans. R.
  Soc. A 379(2194). [Forward-emulator step in atmospheric context.]
