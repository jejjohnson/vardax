# Amortized Inference

Variational and incremental 4DVar (chapters 4, 13) optimise per-event. Each
new observation triggers a fresh minimisation — fine for retrospective
analysis, too slow for real-time alerts. **Amortized inference** trains a
neural network $q_\phi(\mathbf{x} \mid \mathbf{y})$ that maps observations
directly to a posterior, in a single forward pass.

vardax exposes amortized inference via `AmortizedVarDA*` (Decision D12).

## Formulation

Goal: learn $q_\phi(\mathbf{x} \mid \mathbf{y})$ that approximates the
exact posterior $p(\mathbf{x} \mid \mathbf{y})$:

$$\phi^* = \underset{\phi}{\arg\min}\; \mathbb{E}_{\mathbf{y} \sim p(\mathbf{y})}\; \mathrm{KL}\big(q_\phi(\cdot \mid \mathbf{y})\,\|\,p(\cdot \mid \mathbf{y})\big)$$

When the forward and prior are samplable (which they are when supplied by
`somax` / `plumax`), the training distribution comes from **simulation**:

1. Draw $\mathbf{x} \sim p(\mathbf{x})$ (prior on source / state).
2. Simulate $\mathbf{y} \mid \mathbf{x} = H(\mathrm{Forward}(\mathbf{x})) + \boldsymbol{\varepsilon}$.
3. Train $q_\phi$ to recover $\mathbf{x}$ from $\mathbf{y}$.

This is "simulation-based inference" (Cranmer et al. 2020).

## Three head variants

### Conditional normalising flow (`head_type="flow"`)

Learn an invertible map $f_\phi$ from a base Gaussian to the posterior,
conditioned on the observation context:

$$\mathbf{x} = f_\phi(\mathbf{z};\, c_\psi(\mathbf{y})), \qquad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$

Exact density: $\log q_\phi(\mathbf{x} \mid \mathbf{y}) = \log
p_\mathbf{z}(f_\phi^{-1}(\mathbf{x})) - \log|\det \partial f_\phi /
\partial \mathbf{z}|$. Exact sampling. Use `gauss_flows`.

### Score-based diffusion (`head_type="score"`)

Learn $s_\phi(\mathbf{x}, t \mid \mathbf{y}) \approx \nabla_\mathbf{x} \log
p_t(\mathbf{x} \mid \mathbf{y})$ — the score of a noise-perturbed
posterior at noise scale $t$. Sample via reverse SDE:

$$d\mathbf{x} = [\,f(\mathbf{x}, t) - g(t)^2 s_\phi(\mathbf{x}, t \mid \mathbf{y})\,]\,dt + g(t)\,d\bar{\mathbf{w}}$$

No exact density (sampling-only). High capacity for multimodal posteriors.

### Direct regression (`head_type="regression"`)

Predict posterior parameters directly:

$$q_\phi(\mathbf{x} \mid \mathbf{y}) = \mathcal{N}\big(\boldsymbol{\mu}_\phi(\mathbf{y}),\, \boldsymbol{\Sigma}_\phi(\mathbf{y})\big)$$

Cheapest. Restricted to Gaussian families. Good for Gaussian-prior /
Gaussian-likelihood settings (e.g., a learned Laplace approximation).

## Training objective

For flow / regression heads — maximum likelihood on simulated pairs:

$$\mathcal{L}_\text{MLE}(\phi) = -\mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim p_\text{sim}}\,\log q_\phi(\mathbf{x} \mid \mathbf{y})$$

For score-based heads — denoising score matching:

$$\mathcal{L}_\text{DSM}(\phi) = \mathbb{E}_{t, \mathbf{x}, \mathbf{y}, \boldsymbol{\varepsilon}}\,\big\|s_\phi(\mathbf{x}_t, t \mid \mathbf{y}) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{x})\big\|^2$$

with $\mathbf{x}_t = \mathbf{x} + \sigma(t)\boldsymbol{\varepsilon}$.

## Implementation

```python
from vardax.models import AmortizedVarDA
from vardax import AmortizedConfig

model = AmortizedVarDA(
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

# Standard train loop
for batch in simulation_loader:
    model, opt_state, loss = train_step(model, batch, optimizer, opt_state)

# Inference (single forward pass — sub-second)
x_map = model(batch)                        # MAP / mode of q_φ
samples = model.sample(batch, key, n=200)   # posterior samples

# Pipekit-cycle integration
analysis_step = model.as_analysis_step()
```

## When amortized inference helps

| Regime | Amortized helps? |
|---|---|
| Single retrospective analysis | No — solver-based is fine |
| Real-time alerts | Yes — sub-second inference |
| Many independent events (catalog reprocessing) | Yes — amortise over events |
| Same forward, varying observations | Yes — train once, infer N times |
| Each event has a different forward | No — would need to retrain |

## Validation gates (Decision D12)

Amortized inference is dangerous when wrong. vardax codifies three gates:

1. **Posterior agreement.** Amortized MAP within $1\sigma_\text{post}$ of
   physics MAP on held-out events:
   $$\frac{|\mathbf{x}^*_\text{amort} - \mathbf{x}^*_\text{phys}|}{\sigma_\text{post}} \le 1$$

2. **Adjoint calibration.** Amortized gradient matches physics gradient:
   $$\frac{\|\partial \mathbf{x}^*_\text{amort}/\partial \mathbf{y} - \partial \mathbf{x}^*_\text{phys}/\partial \mathbf{y}\|_\text{op}}{\|\partial \mathbf{x}^*_\text{phys}/\partial \mathbf{y}\|_\text{op}} < 0.05$$

3. **Simulation-based calibration (SBC).** Rank histograms uniform:
   $$\mathrm{rank}(\mathbf{x}_i \mid \text{samples from } q_\phi(\cdot \mid \mathbf{y}_i)) \sim \mathrm{Uniform}$$
   χ² test of uniformity at p < 0.01 fails ⇒ retrain.

```python
from vardax.utils.validation import (
    assert_posterior_agreement, assert_adjoint_calibrated, simulation_based_calibration,
)

for val_batch in val_loader:
    p_amort = LaplaceCovariance()(model(val_batch), model.as_analysis_step(), val_batch)
    p_phys = LaplaceCovariance()(physics_model(val_batch),
                                  physics_model.as_analysis_step(), val_batch)
    assert_posterior_agreement(p_amort, p_phys, tolerance_sigma=1.0)

assert_adjoint_calibrated(model, physics_model, val_batches, threshold=0.05)
simulation_based_calibration(model, prior_distribution, forward_model, n_runs=200)
```

## Trade-offs

| Aspect | Solver-based (`VarDANet*`, `IncrementalVarDA*`) | Amortized (`AmortizedVarDA*`) |
|---|---|---|
| Cost per event | High (iterative solve) | Low (single fwd pass) |
| Training cost | Lower (smaller model) | Higher (head + encoder) |
| Generalisation | Strong (uses physics) | Limited (depends on training distribution) |
| Multi-modal handling | Hard | Easy with score-based heads |
| Adjoint correctness | Exact (autodiff) | Approximate (calibrated) |
| Posterior structure | Gaussian (Laplace / GN) | Flexible (flow / score) |

## See also

- Chapter 16: Six-step inference cycle — methodology framing
- Chapter 14: Posterior covariance — UQ for amortized samples
- Decision D12 in design docs — six-step cycle as testing scaffold
- `gauss_flows` for the conditional flow head
- Cranmer, K., Brehmer, J., Louppe, G. (2020). "The frontier of
  simulation-based inference." *PNAS*.
