# Six-Step Inference Cycle

The "six-step cycle" is the research-to-operations methodology vardax
is engineered around. Across all forward-model tiers — geophysical
fluids, atmospheric transport, methane radiative transfer — the same
sequence applies:

1. Physics forward: somax / plumax.
2. Classical inference: OI / 3DVar / 4DVar (slow, exact).
3. Neural emulator: trained from Step 1 (fast surrogate).
4. Emulator-based inference: the same vardax code (100–1000× faster).
5. Amortized predictor: $y$ to posterior in one pass (sub-second).
6. Improve: swap any block; the prior step is the oracle.

The crucial property: **Steps 2, 4, and 5 use the same vardax code**.
The forward model is swapped via the `pipekit_cycle.ForwardModel`
protocol; the analysis class doesn't know whether $M_t$ is physics or
an emulator. The amortized predictor satisfies the same
`AnalysisStep` protocol as the classical method it replaces.

This is what makes the cycle a cycle and not a rewrite.

## Step 1 — Physics forward

Implemented in `somax` (geophysical fluids), `plumax` (atmospheric
transport / methane), or any user code that satisfies
`pipekit_cycle.ForwardModel`.

This chapter runs the cycle on a toy that stands in for somax. The state has $N = 6$ cells on a ring, and the physics forward averages each cell with its right neighbour and damps the result, $M(x) = A x - c (A x)^3$ with $A = \frac{1}{2}(I + P)$, $P$ the cyclic shift and $c = 0.02$. The PRNG keys for the whole chapter are split here.

```python
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = 6 cells

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 10)
key_truth, key_noise, key_val, key_probe_emulator, key_prior = keys[:5]
key_sim, key_head, key_sample, key_probe_amortized, key_sbc = keys[5:]

# P, the cyclic shift (P x)ᵢ = xᵢ₊₁
cyclic_shift = jnp.eye(6, k=1) + jnp.eye(6, k=-5)  # (N, N)
# A = ½ (I + P)
coupling = 0.5 * (jnp.eye(6) + cyclic_shift)  # (N, N)


# user
class CubicRing(eqx.Module):
    """M(x) = A x − c (A x)³ on a ring of cells."""

    coupling: Float[Array, "n n"]
    cubic: float = eqx.field(static=True, default=0.02)
    dt: float = eqx.field(static=True, default=1.0)
    state_signature: Any = eqx.field(static=True, default=None)

    def step(
        self, state: Float[Array, "*batch n"], dt: float
    ) -> Float[Array, "*batch n"]:
        # A x
        averaged = einx.dot("i j, ... j -> ... i", self.coupling, state)
        # M(x) = A x − c (A x)³
        return averaged - self.cubic * averaged**3


physics = CubicRing(coupling)
# today (vardax)
# step(state, dt), dt and state_signature: the pipekit_cycle.ForwardModel contract
print(isinstance(physics, vdx.ForwardModel))
```

It prints `True`.

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

The oracle sees a window of $T = 3$ times, every cell observed, with $y_t = x_t + \sigma \varepsilon_t$, $R = \sigma^2 I$, $\sigma = 0.5$, around a truth $x_0 \sim \mathcal{N}(0, I)$; the background is $x_b = 0$ with $B = I$. The block continues the one above.

```python
import lineax as lx

# Shapes: S = 1 window; T = 3 times; N = 6 cells


# user
def simulate_window(
    initial_state: Float[Array, " n"], key: jax.Array
) -> Float[Array, "t n"]:
    # x₁ = M(x₀)
    state_1 = physics.step(initial_state, 1.0)  # (N,)
    # x₂ = M(x₁)
    state_2 = physics.step(state_1, 1.0)  # (N,)
    trajectory = jnp.stack([initial_state, state_1, state_2])  # (T, N)
    # yₜ = xₜ + σ εₜ
    return trajectory + 0.5 * jax.random.normal(key, (3, 6))  # (T, N)


state_structure = jax.ShapeDtypeStruct((6,), jnp.float32)
# B = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (N, N) op
# R = σ² I; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (N, N) op
    jnp.float32(0.25) * prior_cov, lx.positive_semidefinite_tag
)
# x₀ ~ 𝒩(0, I)
truth = jax.random.normal(key_truth, (6,))  # (N,)
window_obs = simulate_window(truth, key_noise)  # (T, N)
full_mask = jnp.ones((1, 3, 6))  # (S, T, N)
# today (vardax)
window = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", window_obs),  # (T, N) → (S, T, N)
    mask=full_mask,
)

# today (vardax)
incremental_config = vdx.IncrementalConfig(n_outer=3, n_inner=20)
# today (vardax)
oracle = vdx.IncrementalFourDVar(
    forward=physics,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(6),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=incremental_config,
)
# x₀* = argmin J(x₀), Gauss–Newton outer loop with CG inner solves
oracle_analysis = oracle(window)  # (S, T, N) → (S, N)
# P*_GN at x₀*
oracle_posterior = oracle.posterior(window)
print(oracle_analysis.shape, oracle_posterior.mean.shape)
```

It prints `(1, 6) (6,)`.

## Step 3 — Train a neural emulator

The forward model is now the bottleneck. Train a surrogate $F_\psi$
that mimics the physics forward:

$$F_\psi(x) \approx \text{Forward}(x).$$

Two training-time gates must hold before promotion to Step 4:

### Forward agreement

$$\frac{\|F_\psi(x) - \text{Forward}(x)\|}{\|\text{Forward}(x)\|} < \epsilon_\text{fwd}$$

on held-out states (typically $\epsilon_\text{fwd} = 0.01$). This is
the obvious gate: does the emulator give the right answer on the
forward pass?

### Adjoint calibration — the hard gate

$$\frac{\|\partial F_\psi / \partial x - \partial \text{Forward} / \partial x\|_\text{op}}{\|\partial \text{Forward} / \partial x\|_\text{op}} < 0.05$$

via random-vector probing (estimate the operator norm of the
difference). A fast forward with a wrong Jacobian gives wrong
posteriors — the gradient that drives the inner minimisation will be
mis-directed.

This gate is **hard**: vardax refuses to promote an emulator that
fails adjoint calibration, even if the forward agreement is excellent.
Wrong-direction gradients are silently corrosive.

Here a copy of the physics with $c$ off by 2% stands in for a trained network $F_\psi$, and both gates run on $V = 64$ held-out states. The block continues the one above.

```python
# Shapes: V = 64 held-out states; N = 6 cells

# F_ψ: the cubic coefficient 2% off, standing in for a trained network
emulator = CubicRing(coupling, cubic=0.0204)
# x ~ 𝒩(0, I)
val_states = jax.random.normal(key_val, (64, 6))  # (V, N)
emulated = emulator.step(val_states, 1.0)  # (V, N)
simulated = physics.step(val_states, 1.0)  # (V, N)
# ‖F_ψ(x) − M(x)‖ / ‖M(x)‖ < ε_fwd = 0.01
forward_error = jnp.linalg.norm(emulated - simulated) / jnp.linalg.norm(simulated)
assert forward_error < 0.01


# user
def emulator_step(states: Float[Array, "v n"]) -> Float[Array, "v n"]:
    return emulator.step(states, 1.0)


# user
def physics_step(states: Float[Array, "v n"]) -> Float[Array, "v n"]:
    return physics.step(states, 1.0)


# today (vardax)
# ‖∂F_ψ v − ∂M v‖ / ‖∂M v‖ < 0.05 for random probes v
vdx.assert_adjoint_calibrated(
    emulator_step, physics_step, val_states, key=key_probe_emulator, threshold=0.05
)
print(f"forward error {forward_error:.4f}")
```

It prints `forward error 0.0007`.

The emulator $F_\psi$ satisfies `pipekit_cycle.ForwardModel`. It's a
drop-in replacement for `forward` in Step 4.

## Step 4 — Emulator-based inference

Swap `forward` → `emulator` in the same vardax code:

The block continues the one above and ends with the post-promotion gate stated below it.

```python
# Shapes: S = 1 window; T = 3 times; N = 6 cells

# today (vardax)
fast_analysis = vdx.IncrementalFourDVar(
    forward=emulator,  # the only change from Step 2
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(6),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=incremental_config,
)
emulator_analysis = fast_analysis(window)  # (S, T, N) → (S, N)
emulator_posterior = fast_analysis.posterior(window)
# today (vardax)
# |x*_em − x*_phys| / σ_post,phys ≤ 1, cell by cell
vdx.assert_posterior_agreement(
    emulator_posterior, oracle_posterior, tolerance_sigma=1.0
)
```

Validation gate (post-promotion): Step 4 output agrees with Step 2:

$$\frac{|x^*_\text{em} - x^*_\text{phys}|}{\sigma_\text{post,phys}} \le 1$$

on a held-out set of events. If this fails, the emulator is worse than
it looked on the per-state agreement test — retrain with broader
distribution or roll back.

The wall-clock speedup is typically 100–1000× over Step 2, depending on
the emulator's runtime vs. the physics forward.

## Step 5 — Amortized predictor

Train `AmortizedPosterior` (chapter 10) on simulated $(x, y)$ pairs
from the physics forward (or the emulator, if Step 3+4 are validated):

`ConditionalFlowHead` is still a stub that raises until its `gauss_flows` dependency lands, so the example trains the Gaussian `RegressionHead`, $q_\phi(x_0 \mid y) = \mathcal{N}(\mu_\phi(y), \mathrm{diag}(\sigma^2_\phi(y)))$, on $P = 65536$ simulated pairs by maximum likelihood. With `depth=0` and the `IdentityObsEncoder`, $\mu_\phi$ is affine in $y$, which is exact for a linear-Gaussian problem and close for this weakly nonlinear one. The block continues the one above.

```python
import optax

# Shapes: P = 65,536 training pairs; T = 3 times; N = 6 cells; D = 36 context

# x₀ ~ 𝒩(0, I)
train_states = jax.random.normal(key_prior, (65536, 6))  # (P, N)
simulation_keys = jax.random.split(key_sim, 65536)
# y = H(M(x₀)) + σ ε for every pair
train_obs = jax.vmap(simulate_window)(train_states, simulation_keys)  # (P, T, N)
# today (vardax)
train_batch = vdx.Batch1D(
    input=train_obs, mask=jnp.ones((65536, 3, 6)), target=train_states
)
# today (vardax)
amort = vdx.AmortizedPosterior(
    # today (vardax)
    # c = [vec(y); vec(m)]
    encoder=vdx.IdentityObsEncoder(),
    # today (vardax)
    # μ_φ(c), log σ²_φ(c): one linear layer each
    head=vdx.RegressionHead(context_dim=36, state_shape=(6,), depth=0, key=key_head),
    # today (vardax)
    config=vdx.AmortizedConfig(),
)
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(amort, eqx.is_array))
# today (vardax)
train_step = eqx.filter_jit(vdx.amortized_train_step)
for _ in range(500):
    # 𝓛(φ) = −mean log q_φ(x₀ | y)
    amort, opt_state, loss = train_step(amort, train_batch, optimizer, opt_state)

# Inference — sub-second
# x̂₀ = μ_φ(y)
x_map = amort(window)  # (S, T, N) → (S, N)
# x₀⁽ʲ⁾ ~ q_φ(x₀ | y)
samples = amort.sample(window, key_sample, n=200)  # (S, T, N) → (S, 200, N)
print(x_map.shape, samples.shape)
```

It prints `(1, 6) (1, 200, 6)`.

Validation gates (Decision D12):

- **Posterior agreement** with Step 2 within $1\sigma_\text{post}$
- **Adjoint calibration** $< 5\%$
- **SBC** rank histograms uniform

The amortized head has no observation operator, so its posterior for the first gate is its mean alone; the gate reads $\sigma_\text{post}$ from the oracle. The adjoint gate probes $\partial x_0^* / \partial y$ of both predictors, and SBC ranks the truth among 20 posterior samples in each of 20 simulated runs. The block continues the one above.

```python
# Shapes: S = 1 window; T = 3 times; N = 6 cells


# user
@eqx.filter_jit
def amortized_map(obs: Float[Array, "s t n"]) -> Float[Array, "s n"]:
    # today (vardax)
    return amort(vdx.Batch1D(input=obs, mask=full_mask))


# user
@eqx.filter_jit
def oracle_map(obs: Float[Array, "s t n"]) -> Float[Array, "s n"]:
    # today (vardax)
    return oracle(vdx.Batch1D(input=obs, mask=full_mask))


# user
@eqx.filter_jit
def sample_posterior(
    obs: Float[Array, "t n"], key: jax.Array, n_samples: int
) -> Float[Array, "j n"]:
    # today (vardax)
    single = vdx.Batch1D(input=einx.id("t n -> 1 t n", obs), mask=full_mask)
    # today (vardax)
    # x₀⁽ʲ⁾ ~ q_φ(x₀ | y)
    return amort.sample(single, key, n_samples)[0]  # (S, J, N) → (J, N)


# user
def sample_prior(key: jax.Array) -> Float[Array, " n"]:
    # x₀ ~ 𝒩(0, I)
    return jax.random.normal(key, (6,))


# today (vardax)
amortized_posterior = vdx.Posterior(mean=amortized_map(window.input)[0])
# today (vardax)
# |x̂₀ − x₀*| / σ_post ≤ 1, cell by cell
vdx.assert_posterior_agreement(
    amortized_posterior, oracle_posterior, tolerance_sigma=1.0
)
# today (vardax)
# ‖∂x̂₀/∂y v − ∂x₀*/∂y v‖ / ‖∂x₀*/∂y v‖ < 0.05
vdx.assert_adjoint_calibrated(
    amortized_map, oracle_map, window.input, key=key_probe_amortized, threshold=0.05
)
# today (vardax)
# rank of x₀ among J posterior samples, per simulated run
ranks = vdx.simulation_based_calibration(
    sample_posterior,
    sample_prior,
    simulate_window,
    key=key_sbc,
    n_runs=20,
    n_samples=20,
)
print(ranks)
```

All three gates pass, and it prints the ranks `[17 11 18 11 16 19  9 11  0  8 13 18 20 15 12  8  9 11 11 15]`, each in $[0, 20]$.

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

The validation utilities ship in `vardax._src.utils.validation` and are re-exported at the top level:

- `assert_adjoint_calibrated(fn_fast, fn_oracle, y, *, key, threshold=0.05, n_probes=10)`: random-vector operator norm probing
- `assert_posterior_agreement(p_fast, p_oracle, *, tolerance_sigma=1.0)`
- `simulation_based_calibration(sample_posterior, sample_prior, simulate_obs, *, key, n_runs=200, n_samples=200)`: returns the ranks

`assert_forward_agreement` is not shipped yet; Step 3 computes the forward error inline.

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
