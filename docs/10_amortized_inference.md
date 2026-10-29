---
numbering:
  enumerator: "10.%s"
---

# Amortized Inference

The six methods covered so far (chapters 4 to 9) all solve a fresh minimisation per event. A new observation triggers a new analysis, which is fine for retrospective work but too slow for real-time. Amortized
inference learns a function $q_\phi(x \mid y)$ that maps observations
directly to posteriors, in a single forward pass.

The cost of training amortizes over many events; the per-event cost
becomes a single neural network evaluation. This is the difference
between research and operations.

## Formulation

Learn a conditional density $q_\phi(x \mid y)$ that approximates the exact Bayesian posterior [](#eq-bayes):

$$\phi^* = \underset{\phi}{\arg\min} \mathbb{E}_{y \sim p(y)} \mathrm{KL}\left(q_\phi(\cdot \mid y) \| p(\cdot \mid y)\right)$$ (eq-amortized-objective)

where $\phi$ are the network parameters and the expectation runs over the observations the model will see.

When the prior and forward model are samplable (which they are when
supplied by `somax` or `plumax`), the training distribution comes from
**simulation**:

1. Draw $x \sim p(x)$ from the prior.
2. Simulate $y = H(M(x)) + \varepsilon$, the observation model [](#eq-obs-model) applied to the forecast $M(x)$.
3. Train $q_\phi$ to recover $x$ from $y$ on these synthetic pairs.

This is "simulation-based inference" (Cranmer et al. 2020). The training distribution is fully controlled (every $(x, y)$ pair has known ground truth), but the deployment distribution must match for $q_\phi$ to be calibrated.

## Three head variants

vardax `AmortizedPosterior` exposes three head choices via the
`head_type` config:

### Conditional normalising flow (`head_type="flow"`)

Learn an invertible map $f_\phi$ from a base Gaussian to the
posterior, conditioned on the observation context:

$$x = f_\phi(z; c_\psi(y)), \text{ with } z \sim \mathcal{N}(0, I)$$ (eq-flow-sample)

where $c_\psi(y)$ is the context vector from an observation encoder with parameters $\psi$. The change of variables gives the density

$$\log q_\phi(x \mid y) = \log p_z\left(f_\phi^{-1}(x; c_\psi(y))\right) - \log \left| \det \partial f_\phi / \partial z \right|$$ (eq-flow-logdensity)

where $p_z$ is the base Gaussian density and the Jacobian is taken at $z = f_\phi^{-1}(x; c_\psi(y))$.

Exact density, exact samples. Use `gauss_flows` for the flow
implementation. Good for moderate-dimensional state (up to $\sim 10^4$).
Higher-dimensional flows are open research.

### Score-based diffusion (`head_type="score"`)

Learn the score of a noise-perturbed posterior at noise scale $t$:

$$s_\phi(x, t \mid y) \approx \nabla_x \log p_t(x \mid y)$$ (eq-score)

where $p_t$ is the posterior perturbed by noise up to scale $t$. Sampling runs the reverse SDE

$$d x = \left[f(x, t) - g(t)^2 s_\phi(x, t \mid y) \right] d t + g(t) d \bar{w}$$ (eq-reverse-sde)

backwards in $t$, where $f$ and $g$ are the drift and diffusion of the forward noising process and $\bar{w}$ is a reverse-time Wiener process.

No exact density (sampling-only). High capacity for multimodal posteriors, since the diffusion process can represent disconnected components, which flows struggle with. Scales to higher state dimensions than flows.

### Direct regression (`head_type="regression"`)

Predict Gaussian posterior parameters directly:

$$q_\phi(x \mid y) = \mathcal{N}\left(x; \mu_\phi(y), \Sigma_\phi(y)\right)$$ (eq-gaussian-head)

where $\mu_\phi$ and $\Sigma_\phi$ are the network's predicted posterior mean and covariance.

Cheapest by far. Restricted to unimodal Gaussian posteriors. Useful when the posterior is known to be Gaussian (e.g., as a learned replacement for the Laplace approximation [](#eq-laplace-cov) around an `IncrementalFourDVar` MAP).

## Training objective

Flow and regression heads use maximum likelihood on simulated pairs:

$$\mathcal{L}_\text{MLE}(\phi) = -\mathbb{E}_{(x, y) \sim p_\text{sim}} \log q_\phi(x \mid y)$$ (eq-mle-loss)

where $p_\text{sim}$ is the joint distribution of the simulated pairs. Score heads use denoising score matching:

$$\mathcal{L}_\text{DSM}(\phi) = \mathbb{E}_{t, x, y, \varepsilon} \left\| s_\phi(x_t, t \mid y) - \nabla_{x_t} \log p_t(x_t \mid x) \right\|^2$$ (eq-dsm-loss)

where $x_t = x + \sigma(t)\varepsilon$ is the noised state, $\varepsilon \sim \mathcal{N}(0, I)$, and $\sigma(t)$ is the noise scale at $t$.

Both reduce to a standard training step over simulated minibatches: `amortized_train_step` is chapter 9's `train_step` with the negative log-likelihood [](#eq-mle-loss) in place of the reconstruction loss [](#eq-4dvarnet-loss).

## Implementation in vardax

Only the regression head ships today: `ConditionalFlowHead` and `ScoreDiffusionHead` are stubs that raise `NotImplementedError` when constructed, so the example uses `RegressionHead`.
The toy problem has a state of $T = 1$ time and $N = 4$ cells with prior $p(x) = \mathcal{N}(0, I)$ and observations $y = x + \sigma \varepsilon$, $\sigma = 0.5$, fully observed.
Its exact posterior is $\mathcal{N}(y / (1 + \sigma^2), \sigma^2 / (1 + \sigma^2) I)$, the BLUE [](#eq-blue-mean) and [](#eq-blue-cov) with $B = I$, $H = I$ and $R = \sigma^2 I$, which the trained head should recover.
Each training step draws $S = 64$ fresh simulated pairs, the head is the Gaussian head [](#eq-gaussian-head) with diagonal covariance, and `amortized_train_step` minimises [](#eq-mle-loss).

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import vardax as vdx
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: S = 64 simulated pairs; T = 1 time; N = 4 grid cells; C = 16 context

key_encoder, key_head, key_train, key_test, key_samples = jax.random.split(
    jax.random.PRNGKey(0), 5
)
obs_std = 0.5  # σ


# user
def sample_train_pair(
    key: PRNGKeyArray,
) -> tuple[Float[Array, "t n"], Float[Array, "t n"]]:
    key_state, key_noise = jax.random.split(key)
    # x ~ p(x) = 𝒩(0, I)
    state = jax.random.normal(key_state, (1, 4))  # (T, N)
    # y = x + σ ε, ε ~ 𝒩(0, I)
    obs = state + obs_std * jax.random.normal(key_noise, (1, 4))  # (T, N)
    return state, obs


# user
@eqx.filter_jit
def simulate_batch(key: PRNGKeyArray) -> vdx.Batch1D:
    pair_keys = jax.random.split(key, 64)
    states, obs = jax.vmap(sample_train_pair)(pair_keys)  # (S, T, N)
    # today (vardax)
    return vdx.Batch1D(input=obs, mask=jnp.ones_like(obs), target=states)


# today (vardax)
# c = MLP([vec(y); vec(m)])
encoder = vdx.MLPObsEncoder(input_size=4, context_dim=16, key=key_encoder)
# today (vardax)
# q_φ(x | y) = 𝒩(μ_φ(c), diag σ²_φ(c))
head = vdx.RegressionHead(context_dim=16, state_shape=(1, 4), key=key_head)
# today (vardax)
model = vdx.AmortizedPosterior(
    encoder=encoder,
    head=head,
    config=vdx.AmortizedConfig(head_type="regression", n_samples=64),
)

# today (optax)
optimizer = optax.adam(3e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
for batch_key in jax.random.split(key_train, 1000):
    batch = simulate_batch(batch_key)
    # today (vardax)
    # ℒ_MLE(φ) = −mean log q_φ(x | y)
    model, opt_state, loss = vdx.amortized_train_step(
        model, batch, optimizer, opt_state
    )
print(f"final NLL {float(loss):.2f}")

# inference: one forward pass per batch
test_batch = simulate_batch(key_test)
# today (vardax)
# x* = μ_φ(y), the mode of q_φ(x | y)
x_map = model(test_batch)  # → (S, T, N)
# today (vardax)
# xⱼ ~ q_φ(x | y), j = 1, …, 200
samples = model.sample(test_batch, key_samples, n=200)  # → (S, 200, T, N)
# today (vardax)
# log q_φ(x | y), exact for the regression head
log_density = model.log_prob(test_batch.target, test_batch)  # → (S,)
# E[x | y] = y / (1 + σ²)
exact_mean = test_batch.input / (1 + obs_std**2)  # (S, T, N)
mean_error = jnp.max(jnp.abs(x_map - exact_mean))  # → ()
print(samples.shape, log_density.shape, f"max mean error {float(mean_error):.2f}")
```

It prints `final NLL 2.48` and `(64, 200, 1, 4) (64,) max mean error 0.37`; the exact posterior's NLL is $2 (\log(2 \pi \cdot 0.2) + 1) \approx 2.46$.

## Validation: the hard gates

Amortized inference is dangerous when miscalibrated. A well-trained
flow with a tight $q_\phi$ that's centred wrong gives confident wrong
answers. Decision D12 (the six-step cycle, chapter 14) requires three
gates before promoting an amortized head to operational use:

### 1. Posterior agreement

The amortized MAP must fall within $1\sigma_\text{post}$ of an
oracle MAP from a slower method (`StrongFourDVar` /
`IncrementalFourDVar`):

$$\frac{|x^*_\text{amort} - x^*_\text{oracle}|}{\sigma_\text{post, oracle}} \le 1$$ (eq-amortized-gate-mean)

where $\sigma_\text{post, oracle}$ is the oracle's posterior standard deviation, cell by cell. This is the posterior gate [](#eq-gate-posterior) of chapter 14 applied to the amortized map.

For the toy problem the oracle is `ThreeDVar` with $B = I$ and $R = \sigma^2 I$, whose Laplace posterior [](#eq-laplace-cov) is exact here because the problem is linear-Gaussian; the block checks [](#eq-amortized-gate-mean) on four windows.
`assert_posterior_agreement` reads only the mean of the fast posterior, so the amortized MAP goes in a `Posterior` without a covariance.
The block continues the one above.

```python
import lineax as lx

# Shapes: S = 1 window; T = 1 time; N = 4 grid cells

key_val, key_sbc = jax.random.split(jax.random.PRNGKey(1))
state_structure = jax.ShapeDtypeStruct((1, 4), jnp.float32)
# B = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (T N, T N) op
# R = σ² I; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (T N, T N) op
    jnp.float32(obs_std**2) * prior_cov, lx.positive_semidefinite_tag
)
# today (vardax)
oracle = vdx.ThreeDVar(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 4)),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)

val_batch = simulate_batch(key_val)
for window_index in range(4):
    # today (vardax)
    window = vdx.Batch1D(
        input=val_batch.input[window_index : window_index + 1],
        mask=val_batch.mask[window_index : window_index + 1],
    )
    # today (vardax)
    # P* = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹ at the 3DVar MAP
    p_oracle = oracle.posterior(window)
    # today (vardax)
    p_amort = vdx.Posterior(mean=model(window)[0])
    # today (vardax)
    # |x*_amort − x*_oracle| / σ_oracle ≤ 1, cell by cell
    vdx.assert_posterior_agreement(p_amort, p_oracle, tolerance_sigma=1.0)
print("posterior agreement passed")
```

It prints `posterior agreement passed`.

### 2. Adjoint calibration

The amortized model's gradient with respect to observations must match
the physics-based gradient:

$$\frac{\|\partial x^*_\text{amort} / \partial y - \partial x^*_\text{oracle} / \partial y\|_\text{op}}{\|\partial x^*_\text{oracle} / \partial y\|_\text{op}} < 0.05$$ (eq-amortized-gate-jacobian)

where $\|\cdot\|_\text{op}$ is the operator norm. This is the Jacobian gate [](#eq-gate-jacobian) of chapter 14 applied to the amortized map, with $y$ as the input. It is measured by random-vector probing (randomised tangent-linear tests for operator agreement). Without this gate, the amortized model may match the oracle MAP on the training distribution but extrapolate badly when the observation distribution shifts.

### 3. Simulation-based calibration (SBC)

Per Talts et al. (2018): sample $x^{(j)}$ from the prior, simulate
$y^{(j)}$, draw $q_\phi$ samples conditioned on $y^{(j)}$, compute the
rank of $x^{(j)}$ in the sample. If $q_\phi$ is calibrated, the ranks
are uniform across the sample. Failure indicates over- or
under-confident posteriors.

`simulation_based_calibration` takes three callables (posterior sampler, prior sampler, simulator) rather than the model, and reduces each state to its $\ell_2$ norm before ranking it among the samples.
With 19 samples per run the rank takes 20 values, grouped here into 5 bins of 4; a calibrated head gives about 20 runs per bin.
The block continues the one above.

```python
import einx
import jax.scipy.stats as jstats

# Shapes: T = 1 time; N = 4 grid cells; J = 19 posterior samples per run


# user
@eqx.filter_jit
def sample_posterior(
    obs: Float[Array, "t n"], key: PRNGKeyArray, n: int
) -> Float[Array, "j t n"]:
    # today (vardax)
    window = vdx.Batch1D(
        input=einx.id("t n -> 1 t n", obs), mask=jnp.ones((1, *obs.shape))
    )
    # xⱼ ~ q_φ(x | y)
    return model.sample(window, key, n)[0]  # (1, J, T, N) → (J, T, N)


# user
def sample_prior(key: PRNGKeyArray) -> Float[Array, "t n"]:
    # x ~ 𝒩(0, I)
    return jax.random.normal(key, (1, 4))


# user
def simulate_obs(state: Float[Array, "t n"], key: PRNGKeyArray) -> Float[Array, "t n"]:
    # y = x + σ ε
    return state + obs_std * jax.random.normal(key, (1, 4))


# today (vardax)
# rank of ‖x‖ among ‖xⱼ‖, xⱼ ~ q_φ(x | y), over 100 runs
ranks = vdx.simulation_based_calibration(  # → (100,)
    sample_posterior,
    sample_prior,
    simulate_obs,
    key=key_sbc,
    n_runs=100,
    n_samples=19,
)
bin_counts = jnp.bincount(ranks // 4, length=5)  # (100,) → (5,)
# χ² = Σ_b (n_b − 20)² / 20
chi_square = jnp.sum((bin_counts - 20) ** 2) / 20  # → ()
p_value = jstats.chi2.sf(chi_square, 4)  # → ()
print(bin_counts.tolist(), f"chi2 p = {float(p_value):.3f}")
```

It prints `[30, 21, 19, 18, 12] chi2 p = 0.075`: the lowest bin is over-full, but the $\chi^2$ test does not reject uniformity at the $p > 0.01$ threshold.

All three gates ship in `vardax._src.utils.validation`, re-exported as `assert_posterior_agreement`, `assert_adjoint_calibrated` and `simulation_based_calibration`, and `tests/test_six_step_validation.py` tests them.

## When amortized inference helps

| Regime | Amortized helps? |
|---|---|
| Single retrospective analysis | No, solver-based is fine |
| Real-time alerts | Yes, sub-second |
| Many independent events (catalog reprocessing) | Yes, amortise training cost |
| Same forward, varying observations | Yes, train once, infer N times |
| Each event has a different forward | No, would need to retrain |
| Posterior shape is multimodal | Yes, flow / score heads can represent it; Laplace can't |

## Trade-offs vs solver-based methods

| Aspect | Solver-based (chapters 4 to 9) | Amortized (this chapter) |
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

- Chapter 9: `FourDVarNet` (learned solver, same training machinery)
- Chapter 13: posterior covariance (UQ for amortized samples)
- Chapter 14: six-step cycle (the methodology that frames amortization)
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
