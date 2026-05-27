# Amortized Inference

The six methods covered so far (chapters 4–9) all solve a fresh
minimisation per event. A new observation triggers a new analysis —
fine for retrospective work, too slow for real-time. Amortized
inference learns a function $q_\phi(x \mid y)$ that maps observations
directly to posteriors, in a single forward pass.

The cost of training amortizes over many events; the per-event cost
becomes a single neural network evaluation. This is the difference
between research and operations.

## Formulation

Learn a conditional density $q_\phi(x \mid y)$ that approximates the
exact Bayesian posterior $p(x \mid y) \propto p(y \mid x) p(x)$:

$$
\phi^* = \underset{\phi}{\arg\min}\;
         \mathbb{E}_{y \sim p(y)}\;
         \mathrm{KL}\big(q_\phi(\cdot \mid y) \,\|\, p(\cdot \mid y)\big).
$$

When the prior and forward model are samplable (which they are when
supplied by `somax` or `plumax`), the training distribution comes from
**simulation**:

1. Draw $x \sim p(x)$ from the prior.
2. Simulate $y \mid x = H(M(x)) + \varepsilon$ from the forward and
   noise model.
3. Train $q_\phi$ to recover $x$ from $y$ on these synthetic pairs.

This is "simulation-based inference" (Cranmer et al. 2020). The
training distribution is fully controlled — every $(x, y)$ pair has
known ground truth — but the deployment distribution must match for
$q_\phi$ to be calibrated.

## Three head variants

vardax `AmortizedPosterior` exposes three head choices via the
`head_type` config:

### Conditional normalising flow (`head_type="flow"`)

Learn an invertible map $f_\phi$ from a base Gaussian to the
posterior, conditioned on the observation context:

$$
x = f_\phi(z;\, c_\psi(y)), \quad z \sim \mathcal{N}(0, I).
$$

Density:

$$
\log q_\phi(x \mid y) = \log p_z\big(f_\phi^{-1}(x; c_\psi(y))\big) -
                       \log \big| \det \partial f_\phi / \partial z \big|.
$$

Exact density, exact samples. Use `gauss_flows` for the flow
implementation. Good for moderate-dimensional state (up to $\sim 10^4$).
Higher-dimensional flows are open research.

### Score-based diffusion (`head_type="score"`)

Learn the score of a noise-perturbed posterior at noise scale $t$:

$$
s_\phi(x, t \mid y) \approx \nabla_x \log p_t(x \mid y).
$$

Sampling via reverse SDE:

$$
d x = \big[\,f(x, t) - g(t)^2 s_\phi(x, t \mid y)\,\big]\, d t + g(t)\, d \bar{w}.
$$

No exact density (sampling-only). High capacity for multimodal
posteriors — the diffusion process can represent disconnected
components, which flows struggle with. Scales to higher state
dimensions than flows.

### Direct regression (`head_type="regression"`)

Predict Gaussian posterior parameters directly:

$$
q_\phi(x \mid y) = \mathcal{N}\big(x; \mu_\phi(y), \Sigma_\phi(y)\big).
$$

Cheapest by far. Restricted to unimodal Gaussian posteriors. Useful
when the posterior is known to be Gaussian (e.g., as a learned
replacement for the Laplace approximation around a `IncrementalFourDVar`
MAP).

## Training objective

For flow and regression heads — maximum likelihood on simulated pairs:

$$
\mathcal{L}_\text{MLE}(\phi) = -\mathbb{E}_{(x, y) \sim p_\text{sim}}\, \log q_\phi(x \mid y).
$$

For score heads — denoising score matching:

$$
\mathcal{L}_\text{DSM}(\phi) = \mathbb{E}_{t, x, y, \varepsilon}\,
                                \big\| s_\phi(x_t, t \mid y) -
                                       \nabla_{x_t} \log p_t(x_t \mid x) \big\|^2,
$$

with $x_t = x + \sigma(t)\varepsilon$.

Both reduce to a standard `train_step` over simulated minibatches —
the vardax training infrastructure (chapter 9) applies.

## Implementation in vardax

```python
from vardax.models import AmortizedPosterior
from vardax import AmortizedConfig
from vardax.training import train_step

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

# Inference — sub-second
x_map = model(batch)                        # MAP / mode of q_φ(x | y)
samples = model.sample(batch, key, n=200)   # posterior samples
log_p = model.log_prob(x, batch)            # exact for flow, NotImplemented for score
```

## Validation — the hard gates

Amortized inference is dangerous when miscalibrated. A well-trained
flow with a tight $q_\phi$ that's centred wrong gives confident wrong
answers. Decision D12 (the six-step cycle, chapter 14) requires three
gates before promoting an amortized head to operational use:

### 1. Posterior agreement

The amortized MAP must fall within $1\sigma_\text{post}$ of an
oracle MAP from a slower method (`StrongFourDVar` /
`IncrementalFourDVar`):

$$
\frac{|x^*_\text{amort} - x^*_\text{oracle}|}{\sigma_\text{post, oracle}} \le 1.
$$

```python
from vardax.utils.validation import assert_posterior_agreement

for val_batch in val_loader:
    p_amort = LaplaceCovariance()(model(val_batch), model.as_analysis_step(), val_batch)
    p_oracle = oracle.posterior(val_batch)
    assert_posterior_agreement(p_amort, p_oracle, tolerance_sigma=1.0)
```

### 2. Adjoint calibration

The amortized model's gradient with respect to observations must match
the physics-based gradient:

$$
\frac{\|\partial x^*_\text{amort} / \partial y - \partial x^*_\text{oracle} / \partial y\|_\text{op}}{\|\partial x^*_\text{oracle} / \partial y\|_\text{op}} < 0.05.
$$

This is measured by random-vector probing — randomised tangent
linearisation tests for operator agreement. Without this gate, the
amortized model may match the oracle MAP on the training distribution
but extrapolate badly when the observation distribution shifts.

### 3. Simulation-based calibration (SBC)

Per Talts et al. (2018): sample $x^{(j)}$ from the prior, simulate
$y^{(j)}$, draw $q_\phi$ samples conditioned on $y^{(j)}$, compute the
rank of $x^{(j)}$ in the sample. If $q_\phi$ is calibrated, the ranks
are uniform across the sample. Failure indicates over- or
under-confident posteriors.

```python
from vardax.utils.validation import simulation_based_calibration

ranks = simulation_based_calibration(
    model, prior_distribution, forward_model, n_runs=200,
)
# Plot rank histogram; assert χ² uniformity test p > 0.01
```

All three gates are **planned** to ship as `assert_*` functions in
`vardax._src.utils.validation` and to be wired into
`tests/test_six_step_validation.py` as part of the v0.4 design target
(Epic 8). Neither the module nor the test file exists in the v0.1.6
`fourdvarjax` codebase yet.

## When amortized inference helps

| Regime | Amortized helps? |
|---|---|
| Single retrospective analysis | No — solver-based is fine |
| Real-time alerts | Yes — sub-second |
| Many independent events (catalog reprocessing) | Yes — amortise training cost |
| Same forward, varying observations | Yes — train once, infer N times |
| Each event has a different forward | No — would need to retrain |
| Posterior shape is multimodal | Yes — flow / score heads can represent it; Laplace can't |

## Trade-offs vs solver-based methods

| Aspect | Solver-based (chapters 4–9) | Amortized (this chapter) |
|---|---|---|
| Cost per event | High (iterative solve) | Low (single forward pass) |
| Training cost | None / low (only `FourDVarNet`) | High (head + encoder) |
| Generalisation | Strong (uses physics) | Limited (training distribution) |
| Multimodal handling | Hard (Laplace assumes unimodal) | Easy with score-based heads |
| Adjoint correctness | Exact via autodiff | Approximate, calibrated by gates |
| Posterior structure | Gaussian (Laplace / GN-Hessian) | Flexible (flow / score / Gaussian) |
| Failure mode | Slow convergence | Confident wrong answer outside training distribution |

The right pattern is **both**: use solver-based methods to produce
oracle posteriors on a representative sample, train an amortized head
against the oracle, validate via the gates, then deploy the head for
real-time work. The six-step cycle (chapter 14) formalises this.

## See also

- Chapter 9 — `FourDVarNet` (learned solver, same training machinery)
- Chapter 13 — posterior covariance (UQ for amortized samples)
- Chapter 14 — six-step cycle (the methodology that frames amortization)
- Design doc: [`design/decisions.md#d12`](design/decisions.md#d12-six-step-inference-cycle-as-testing-scaffold)

## References

- Cranmer, K., Brehmer, J., & Louppe, G. (2020). *The frontier of
  simulation-based inference.* PNAS 117(48).
- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., &
  Lakshminarayanan, B. (2021). *Normalizing flows for probabilistic
  modeling and inference.* JMLR 22(57).
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., &
  Poole, B. (2021). *Score-based generative modeling through
  stochastic differential equations.* ICLR.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A.
  (2018). *Validating Bayesian inference algorithms with
  simulation-based calibration.* arXiv:1804.06788.
