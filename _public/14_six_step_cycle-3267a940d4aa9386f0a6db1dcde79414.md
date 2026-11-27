---
numbering:
  enumerator: "14.%s"
---

# Six-Step Inference Cycle

The "six-step cycle" is the research-to-operations methodology vardax is engineered around. Across all forward-model tiers (geophysical fluids, atmospheric transport, methane radiative transfer) the same sequence applies:

1. Physics forward: somax / plumax.
2. Classical inference: OI / 3DVar / 4DVar (slow, exact).
3. Neural emulator: trained from Step 1 (fast surrogate).
4. Emulator-based inference: the same vardax code (100 to 1000× faster).
5. Amortized predictor: $\vect{y}$ to posterior in one pass (sub-second).
6. Improve: swap any block; the prior step is the oracle.

The key property is that **Steps 2, 4, and 5 use the same vardax code**.
The forward model is swapped via the `pipekit_cycle.ForwardModel`
protocol; the analysis class doesn't know whether $f_t$ is physics or
an emulator. The amortized predictor satisfies the same
`AnalysisStep` protocol as the classical method it replaces.

This is what makes the cycle a cycle and not a rewrite.

## Step 1: Physics forward

Implemented in `somax` (geophysical fluids), `plumax` (atmospheric
transport / methane), or any user code that satisfies
`pipekit_cycle.ForwardModel`.

This chapter runs the cycle on a toy that stands in for somax. The state $\vect{x} \in \mathbb{R}^{D_x}$ has $D_x = 6$ cells on a ring, and the physics forward averages each cell with its right neighbour and damps the result, $f(\vect{x}) = \mat{C} \vect{x} - c (\mat{C} \vect{x})^3$ with $\mat{C} = \frac{1}{2}(\mat{I} + \mat{P})$, $\mat{P}$ the cyclic shift and $c = 0.02$. The PRNG keys for the whole chapter are split here.

```{code-block} python
:caption: Defines the toy physics forward f of the rollout [](#eq-rollout) and checks it satisfies the `ForwardModel` protocol.
:label: code-cycle-physics
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import vardax as vdx
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: D_x = 6 cells

key = jax.random.PRNGKey(0)
(
    key_truth,
    key_noise,
    key_val,
    key_probe_emulator,
    key_prior,
    key_sim,
    key_head,
    key_sample,
    key_probe_amortized,
    key_sbc,
) = jax.random.split(key, 10)

# P, the cyclic shift (P x)ᵢ = xᵢ₊₁
cyclic_shift = jnp.eye(6, k=1) + jnp.eye(6, k=-5)  # (D_x, D_x)
# C = ½ (I + P)
coupling = 0.5 * (jnp.eye(6) + cyclic_shift)  # (D_x, D_x)


# user
class CubicRing(eqx.Module):
    """f(x) = C x − c (C x)³ on a ring of cells."""

    coupling: Float[Array, "n n"]
    cubic: float = eqx.field(static=True, default=0.02)
    dt: float = eqx.field(static=True, default=1.0)
    state_signature: Any = eqx.field(static=True, default=None)

    def step(
        self, state: Float[Array, "*batch n"], dt: float
    ) -> Float[Array, "*batch n"]:
        # C x
        averaged = einx.dot("i j, ... j -> ... i", self.coupling, state)
        # f(x) = C x − c (C x)³
        return averaged - self.cubic * averaged**3


# user
physics = CubicRing(coupling)
# today (vardax)
# step(state, dt), dt and state_signature: the pipekit_cycle.ForwardModel contract
print(isinstance(physics, vdx.ForwardModel))
```

It prints `True`.

Vardax does not own this layer.

## Step 2: Classical inference

Pick one of the five classical methods from chapters 4 to 8 based on regime:

| Regime | Method |
|---|---|
| Linear $\mat{H}$, Gaussian $\gmat{\Sigma}_x$ / $\gmat{\Sigma}_\varepsilon$ | `OptimalInterpolation` (chapter 4) |
| Nonlinear $h$, single time | `ThreeDVar` (chapter 5) |
| Multi-time, exact dynamics | `StrongFourDVar` (chapter 6) |
| Multi-time, model error | `WeakFourDVar` (chapter 7) |
| Operational 4DVar | `IncrementalFourDVar` (chapter 8) |

The result is the **oracle** for Step 4 and Step 5. Slow but exact (or
as exact as the prior / likelihood allow).

The oracle sees a window of $T = 2$ steps, so $T + 1 = 3$ times, every cell observed, with $\vect{y}_t = \vect{x}_t + \sigma \gvect{\varepsilon}_t \in \mathbb{R}^{D_y}$, $D_y = D_x$, $\gvect{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mat{I})$, $\gmat{\Sigma}_\varepsilon = \sigma^2 \mat{I}$, $\sigma = 0.5$, around a truth $\vect{x}_0 \sim \mathcal{N}(\mathbf{0}, \mat{I})$; the background is $\gvect{\mu}_x = \mathbf{0}$ with $\gmat{\Sigma}_x = \mat{I}$. The oracle is `IncrementalFourDVar`. The block continues the one above.

```{code-block} python
:caption: Runs the `IncrementalFourDVar` oracle, which minimises [](#eq-strong4dvar-cost) through [](#eq-incremental-cost), and reads its posterior [](#eq-gn-cov).
:label: code-cycle-oracle
import lineax as lx

# Shapes: S = 1 window; T = 2 steps, so T + 1 = 3 times; D_x = D_y = 6 cells


# user
def simulate_window(
    initial_state: Float[Array, " n"], key: PRNGKeyArray
) -> Float[Array, "t n"]:
    # x₁ = f(x₀)
    state_1 = physics.step(initial_state, 1.0)  # (D_x,)
    # x₂ = f(x₁)
    state_2 = physics.step(state_1, 1.0)  # (D_x,)
    trajectory = jnp.stack([initial_state, state_1, state_2])  # (T+1, D_x)
    # εₜ ~ 𝒩(0, I)
    noise = jax.random.normal(key, (3, 6))  # (T+1, D_y)
    # yₜ = xₜ + σ εₜ
    return trajectory + 0.5 * noise  # (T+1, D_y)


state_structure = jax.ShapeDtypeStruct((6,), jnp.float32)
# Σ_x = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# Σ_ε = σ² I
scaled_identity = jnp.float32(0.25) * prior_cov  # (D_y, D_y) op
# tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (D_y, D_y) op
    scaled_identity, lx.positive_semidefinite_tag
)
# x₀ ~ 𝒩(0, I)
truth = jax.random.normal(key_truth, (6,))  # (D_x,)
# user
# yₜ = f_t(x₀) + σ εₜ
window_obs = simulate_window(truth, key_noise)  # (D_x,) → (T+1, D_y)
# m = 1, every cell observed
full_mask = jnp.ones((1, 3, 6))  # (S, T+1, D_y)
window_input = einx.id(  # (T+1, D_y) → (S, T+1, D_y)
    "t n -> 1 t n", window_obs
)
# today (vardax)
window = vdx.Batch1D(input=window_input, mask=full_mask)

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
# x₀⋆ = argmin 𝒥(x₀), Gauss-Newton outer loop with CG inner solves
oracle_analysis = oracle(window)  # (S, T+1, D_y) → (S, D_x)
# today (vardax)
# Σ_x|y at x₀⋆
oracle_posterior = oracle.posterior(window)  # cov (D_x, D_x) PSD op
print(oracle_analysis.shape, oracle_posterior.mean.shape)
```

It prints `(1, 6) (6,)`.

## Step 3: Train a neural emulator

The forward model is now the bottleneck. Train a surrogate $g_\text{emu}$ that mimics the physics forward:

$$g_\text{emu}(\vect{x}; \gvect{\theta}) \approx f(\vect{x})$$ (eq-emulator)

where $g_\text{emu}$ is the emulator with weights $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ and $f$ is the physics forward of Step 1. Two training-time gates must hold before promotion to Step 4.

### Forward agreement

$$\frac{\Vert g_\text{emu}(\vect{x}; \gvect{\theta}) - f(\vect{x}) \Vert_2}{\Vert f(\vect{x}) \Vert_2} < \mathrm{tol}_\text{fwd}$$ (eq-gate-forward)

on held-out states $\vect{x}$, where $\mathrm{tol}_\text{fwd}$ is the tolerance (typically $0.01$). This is the obvious gate: does the emulator give the right answer on the forward pass?

### Adjoint calibration, the hard gate

$$\frac{\Vert \partial g_\text{emu} / \partial \vect{x} - \partial f / \partial \vect{x} \Vert_\text{op}}{\Vert \partial f / \partial \vect{x} \Vert_\text{op}} < 0.05$$ (eq-gate-jacobian)

where $\Vert \cdot \Vert_\text{op}$ is the operator norm of a $D_x \times D_x$ Jacobian, estimated by random-vector probing of the Jacobian difference. A fast forward with a wrong Jacobian gives wrong posteriors, because the gradient that drives the inner minimisation is mis-directed.

This gate is **hard**: vardax refuses to promote an emulator that
fails adjoint calibration, even if the forward agreement is excellent.
Wrong-direction gradients are silently corrosive.

Here a copy of the physics with $c$ off by 2% stands in for a trained network $g_\text{emu}$, tested on $N_\text{val} = 64$ held-out states. The block continues the one above.

```{code-block} python
:caption: Checks a stand-in emulator against the forward-agreement gate [](#eq-gate-forward) and the Jacobian gate [](#eq-gate-jacobian).
:label: code-cycle-emulator-gates
# Shapes: N_val = 64 held-out states; D_x = 6 cells

# user
# g_emu: the cubic coefficient 2% off, standing in for a trained network
emulator = CubicRing(coupling, cubic=0.0204)
# x ~ 𝒩(0, I)
val_states = jax.random.normal(key_val, (64, 6))  # (N_val, D_x)
# g_emu(x)
emulated = emulator.step(val_states, 1.0)  # (N_val, D_x) → (N_val, D_x)
# f(x)
simulated = physics.step(val_states, 1.0)  # (N_val, D_x) → (N_val, D_x)
# g_emu(x) − f(x)
emulation_misfit = emulated - simulated  # (N_val, D_x)
# ‖g_emu(x) − f(x)‖
misfit_norm = jnp.linalg.norm(emulation_misfit)  # → ()
# ‖f(x)‖
simulated_norm = jnp.linalg.norm(simulated)  # → ()
# ‖g_emu(x) − f(x)‖ / ‖f(x)‖ < tol_fwd = 0.01
forward_error = misfit_norm / simulated_norm  # → ()
assert forward_error < 0.01


# user
def emulator_step(states: Float[Array, "v n"]) -> Float[Array, "v n"]:
    # g_emu(x)
    return emulator.step(states, 1.0)


# user
def physics_step(states: Float[Array, "v n"]) -> Float[Array, "v n"]:
    # f(x)
    return physics.step(states, 1.0)


# today (vardax)
# ‖∂g_emu v − ∂f v‖ / ‖∂f v‖ < 0.05 for random probes v
vdx.assert_adjoint_calibrated(
    emulator_step, physics_step, val_states, key=key_probe_emulator, threshold=0.05
)
print(f"forward error {forward_error:.4f}")
```

It prints `forward error 0.0007`.

The emulator $g_\text{emu}$ satisfies `pipekit_cycle.ForwardModel`. It's a
drop-in replacement for `forward` in Step 4.

## Step 4: Emulator-based inference

Swap `forward` for `emulator` in the same vardax code. The post-promotion gate asks that the Step 4 output agrees with Step 2,

$$\frac{\vert x^\star_{\text{em},d} - x^\star_{\text{phys},d} \vert}{\sigma_d} \le 1$$ (eq-gate-posterior)

cell by cell, for $d = 1, \ldots, D_x$, where $\vect{x}^\star_\text{em}$ and $\vect{x}^\star_\text{phys} \in \mathbb{R}^{D_x}$ are the emulator and physics analyses and $\sigma_d$ is the physics posterior standard deviation, $\sigma_d^2 = [\gmat{\Sigma}_{x \mid y}]_{dd}$ (the diagonal of [](#eq-gn-cov)). The gate runs on a held-out set of events. If it fails, the emulator is worse than it looked on the per-state agreement test, so retrain with a broader distribution or roll back.

The block continues the one above.

```{code-block} python
:caption: Reruns the oracle analysis with the emulator and applies the posterior-agreement gate [](#eq-gate-posterior).
:label: code-cycle-emulator-inference
# Shapes: S = 1 window; T = 2 steps, so T + 1 = 3 times; D_x = D_y = 6 cells

# today (vardax)
fast_analysis = vdx.IncrementalFourDVar(
    # the only change from Step 2
    forward=emulator,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(6),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=incremental_config,
)
# x₀⋆_em = argmin 𝒥(x₀) with f replaced by g_emu
emulator_analysis = fast_analysis(window)  # (S, T+1, D_y) → (S, D_x)
# today (vardax)
# Σ_x|y at x₀⋆_em
emulator_posterior = fast_analysis.posterior(window)  # cov (D_x, D_x) PSD op
# today (vardax)
# |x⋆_em,d − x⋆_phys,d| / σ_d ≤ 1, cell by cell
vdx.assert_posterior_agreement(
    emulator_posterior, oracle_posterior, tolerance_sigma=1.0
)
```

The wall-clock speedup is typically 100 to 1000× over Step 2, depending on the emulator's runtime against the physics forward.

## Step 5: Amortized predictor

Train `AmortizedPosterior` (chapter 10) on simulated $(\vect{x}, \vect{y})$ pairs from the physics forward (or the emulator, if Steps 3 and 4 are validated).

`ConditionalFlowHead` is still a stub that raises until its `gauss_flows` dependency lands, so the example trains the Gaussian `RegressionHead` [](#eq-gaussian-head), with mean $\gvect{\mu}_{x \mid y}(\vect{c}; \gvect{\nu})$ and diagonal covariance $\gmat{\Sigma}_{x \mid y}(\vect{c}; \gvect{\nu})$ of the context $\vect{c} \in \mathbb{R}^{D_c}$ the encoder builds from $\vect{y}$, on $N_\text{train} = 65536$ simulated pairs by maximum likelihood [](#eq-mle-loss). With `depth=0` and the `IdentityObsEncoder`, $\gvect{\mu}_{x \mid y}$ is affine in $\vect{y}$, which is exact for a linear-Gaussian problem and close for this weakly nonlinear one. The block continues the one above.

```{code-block} python
:caption: Trains the Gaussian regression head [](#eq-gaussian-head) by maximum likelihood [](#eq-mle-loss) on simulated pairs and samples its posterior.
:label: code-cycle-amortized-training
import optax

# Shapes: N_train = 65,536 training pairs; T = 2 steps, so T + 1 = 3 times;
# D_x = D_y = 6 cells; D_c = 2 (T + 1) D_y = 36 context

# x₀ ~ 𝒩(0, I)
train_states = jax.random.normal(key_prior, (65536, 6))  # (N_train, D_x)
simulation_keys = jax.random.split(key_sim, 65536)  # (N_train,)
simulate_pairs = jax.vmap(simulate_window)
# user
# y = h_t(f_t(x₀)) + σ ε for every pair
train_obs = simulate_pairs(  # (N_train, D_x) → (N_train, T+1, D_y)
    train_states, simulation_keys
)
# today (vardax)
train_batch = vdx.Batch1D(
    input=train_obs, mask=jnp.ones((65536, 3, 6)), target=train_states
)
# today (vardax)
# z = [vec(y); vec(m)]
encoder = vdx.IdentityObsEncoder()
# today (vardax)
# μ_x|y(z; ν), diag Σ_x|y(z; ν): one linear layer each
head = vdx.RegressionHead(context_dim=36, state_shape=(6,), depth=0, key=key_head)
# today (vardax)
amortized_config = vdx.AmortizedConfig()
# today (vardax)
amort = vdx.AmortizedPosterior(encoder=encoder, head=head, config=amortized_config)
optimizer = optax.adam(1e-2)
amortized_params = eqx.filter(amort, eqx.is_array)
opt_state = optimizer.init(amortized_params)
# today (vardax)
train_step = eqx.filter_jit(vdx.amortized_train_step)
for _ in range(500):
    # ℒ_NLL(ν) = −mean log q(x₀ ∣ y; ν)
    amort, opt_state, loss = train_step(amort, train_batch, optimizer, opt_state)

# Inference, sub-second
# x̂₀ = μ_x|y(y; ν)
x_map = amort(window)  # (S, T+1, D_y) → (S, D_x)
# today (vardax)
# x₀⁽ᵉ⁾ ~ q(x₀ ∣ y; ν), N_ℰ = 200 draws
samples = amort.sample(window, key_sample, n=200)  # (S, T+1, D_y) → (S, N_ℰ, D_x)
print(x_map.shape, samples.shape)
```

It prints `(1, 6) (1, 200, 6)`.

Decision D12 holds the amortized head to three gates: posterior agreement with Step 2 [](#eq-gate-posterior), the Jacobian gate [](#eq-gate-jacobian) applied to the maps $\vect{y} \mapsto \vect{x}_0^\star$, and uniform simulation-based calibration (SBC) rank histograms. The amortized head has no observation operator, so its posterior for the first gate is its mean alone, and the gate reads $\sigma_d$ from the oracle. The Jacobian gate probes $\partial \vect{x}_0^\star / \partial \vect{y}$ of both predictors, and SBC ranks the truth among 20 posterior samples in each of 20 simulated runs. The block continues the one above.

```{code-block} python
:caption: Applies the posterior-agreement gate [](#eq-gate-posterior), the Jacobian gate [](#eq-gate-jacobian) and simulation-based calibration to the amortized head.
:label: code-cycle-amortized-gates
# Shapes: S = 1 window; T = 2 steps, so T + 1 = 3 times; D_x = D_y = 6 cells


# user
@eqx.filter_jit
def amortized_map(obs: Float[Array, "s t n"]) -> Float[Array, "s n"]:
    # today (vardax)
    obs_batch = vdx.Batch1D(input=obs, mask=full_mask)
    # x̂₀ = μ_x|y(y; ν)
    return amort(obs_batch)  # (S, T+1, D_y) → (S, D_x)


# user
@eqx.filter_jit
def oracle_map(obs: Float[Array, "s t n"]) -> Float[Array, "s n"]:
    # today (vardax)
    obs_batch = vdx.Batch1D(input=obs, mask=full_mask)
    # x₀⋆ = argmin 𝒥(x₀)
    return oracle(obs_batch)  # (S, T+1, D_y) → (S, D_x)


# user
@eqx.filter_jit
def sample_posterior(
    obs: Float[Array, "t n"], key: PRNGKeyArray, n_samples: int
) -> Float[Array, "j n"]:
    single_input = einx.id("t n -> 1 t n", obs)  # (T+1, D_y) → (S, T+1, D_y)
    # today (vardax)
    single = vdx.Batch1D(input=single_input, mask=full_mask)
    # today (vardax)
    # x₀⁽ᵉ⁾ ~ q(x₀ ∣ y; ν)
    samples = amort.sample(single, key, n_samples)  # (S, T+1, D_y) → (S, N_ℰ, D_x)
    return samples[0]  # (S, N_ℰ, D_x) → (N_ℰ, D_x)


# user
def sample_prior(key: PRNGKeyArray) -> Float[Array, " n"]:
    # x₀ ~ 𝒩(0, I)
    return jax.random.normal(key, (6,))


# user
# x̂₀ = μ_x|y(y; ν)
amortized_window = amortized_map(window.input)  # (S, T+1, D_y) → (S, D_x)
# today (vardax)
amortized_posterior = vdx.Posterior(mean=amortized_window[0])
# today (vardax)
# |x̂₀,d − x₀⋆,d| / σ_d ≤ 1, cell by cell
vdx.assert_posterior_agreement(
    amortized_posterior, oracle_posterior, tolerance_sigma=1.0
)
# today (vardax)
# ‖∂x̂₀/∂y v − ∂x₀⋆/∂y v‖ / ‖∂x₀⋆/∂y v‖ < 0.05
vdx.assert_adjoint_calibrated(
    amortized_map, oracle_map, window.input, key=key_probe_amortized, threshold=0.05
)
# today (vardax)
# rank of x₀ among N_ℰ posterior samples, per simulated run
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

`amort.as_analysis_step()` returns a `pipekit_cycle.AnalysisStep` with the same interface as Step 2 and Step 4, so operational pipelines swap in the amortized predictor without touching the orchestration.

## Step 6: Improve, repeat

The cycle is a loop. Improvements at any step trigger re-validation
downstream:

| Change | Re-validates |
|---|---|
| New physics in Step 1 | Steps 2, 3, 4, 5 |
| New inference method in Step 2 | Step 4 (Step 2 is oracle) |
| Better emulator (Step 3) | Steps 4, 5 |
| Better amortized head (Step 5) | Step 5 vs Step 2 / 4 |
| New observation operator | Steps 2, 4, 5 |

The validation gates are the contract, since they are how you know an "improvement" is actually an improvement.

## Why the cycle is the right framing

Three observations from operational DA:

- **Fast inference is dangerous when wrong.** Replacing a 4DVar with a flow predictor that runs in milliseconds is appealing, until the flow's posterior is miscalibrated and the alerting system fires on noise. The gates between steps make "fast" and "correct" both requirements, not just the former.
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

- Chapter 9: `FourDVarNet` (a hybrid case: classical 4DVar with learned components, validated against a classical oracle)
- Chapter 10: `AmortizedPosterior` (Step 5 implementation)
- Chapter 13: posterior covariance (gate computations)
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
