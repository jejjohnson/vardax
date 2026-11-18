---
numbering:
  enumerator: "10.%s"
---

# Amortized Inference

The six methods covered so far (chapters 4 to 9) all solve a fresh minimisation per event. A new observation triggers a new analysis, which is fine for retrospective work but too slow for real-time. Amortized
inference learns a function $q(\vect{x} \mid \vect{y}; \gvect{\nu})$ that maps observations
directly to posteriors, in a single forward pass.

The cost of training amortizes over many events; the per-event cost
becomes a single neural network evaluation. This is the difference
between research and operations.

## Formulation

Learn a conditional density $q(\vect{x} \mid \vect{y}; \gvect{\nu})$ that approximates the exact Bayesian posterior [](#eq-bayes):

$$\gvect{\nu}^\star = \underset{\gvect{\nu}}{\mathrm{argmin}} \ \mathbb{E}_{\vect{y} \sim p(\vect{y})} \mathrm{KL}\left(q(\cdot \mid \vect{y}; \gvect{\nu}) \Vert p(\cdot \mid \vect{y})\right)$$ (eq-amortized-objective)

where $\gvect{\nu} \in \mathbb{R}^{D_\nu}$ are the network parameters, $\vect{x} \in \mathbb{R}^{D_x}$ is the state, $\vect{y} \in \mathbb{R}^{D_y}$ the observations, and the expectation runs over the observations the model will see.

When the prior and forward model are samplable (which they are when
supplied by `somax` or `plumax`), the training distribution comes from
**simulation**:

1. Draw $\vect{x} \sim p(\vect{x})$ from the prior.
2. Simulate $\vect{y} = h(f(\vect{x})) + \gvect{\varepsilon}$, the observation model [](#eq-obs-model) applied to the forecast $f(\vect{x})$, with $\gvect{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \gmat{\Sigma}_\varepsilon)$ in $\mathbb{R}^{D_y}$.
3. Train $q$ to recover $\vect{x}$ from $\vect{y}$ on these synthetic pairs.

This is "simulation-based inference" (Cranmer et al. 2020). The training distribution is fully controlled (every $(\vect{x}, \vect{y})$ pair has known ground truth), but the deployment distribution must match for $q$ to be calibrated.

## Three head variants

vardax `AmortizedPosterior` exposes three head choices via the
`head_type` config:

### Conditional normalising flow (`head_type="flow"`)

Learn an invertible map $g_\text{flow}$ from a base Gaussian to the
posterior, conditioned on the observation context:

$$\vect{x} = g_\text{flow}(\vect{z}; \vect{c}, \gvect{\nu}), \text{ with } \vect{z} \sim \mathcal{N}(\mathbf{0}, \mat{I}), \vect{c} = g_\text{enc}(\vect{y}; \gvect{\nu})$$ (eq-flow-sample)

where $\vect{c} \in \mathbb{R}^{D_c}$ is the context vector from an observation encoder $g_\text{enc}$, $\vect{z} \in \mathbb{R}^{D_x}$ is the base variable, and $\gvect{\nu}$ collects the weights of the encoder and of the flow. The change of variables gives the density

$$\log q(\vect{x} \mid \vect{y}; \gvect{\nu}) = \log p_z\left(g_\text{flow}^{-1}(\vect{x}; \vect{c}, \gvect{\nu})\right) - \log \vert \det \partial g_\text{flow} / \partial \vect{z} \vert$$ (eq-flow-logdensity)

where $p_z$ is the base Gaussian density and the Jacobian is taken at $\vect{z} = g_\text{flow}^{-1}(\vect{x}; \vect{c}, \gvect{\nu})$.

Exact density, exact samples. Use `gauss_flows` for the flow
implementation. Good for moderate-dimensional state (up to $\sim 10^4$).
Higher-dimensional flows are open research.

### Score-based diffusion (`head_type="score"`)

Learn the score of a noise-perturbed posterior at noise scale $\tau$:

$$\vect{s}(\vect{x}, \tau \mid \vect{y}; \gvect{\nu}) \approx \nabla_{\vect{x}} \log p_\tau(\vect{x} \mid \vect{y})$$ (eq-score)

where $\vect{s} \in \mathbb{R}^{D_x}$ and $p_\tau$ is the posterior perturbed by noise up to scale $\tau$. Sampling runs the reverse SDE

$$\mathrm{d}\vect{x} = \left[f_\text{sde}(\vect{x}, \tau) - g_\text{sde}(\tau)^2 \vect{s}(\vect{x}, \tau \mid \vect{y}; \gvect{\nu}) \right] \mathrm{d}\tau + g_\text{sde}(\tau) \mathrm{d}\vect{w}_\text{rev}$$ (eq-reverse-sde)

backwards in $\tau$, where $f_\text{sde}: \mathbb{R}^{D_x} \times \mathbb{R} \to \mathbb{R}^{D_x}$ and the scalar $g_\text{sde}(\tau)$ are the drift and diffusion of the forward noising process and $\vect{w}_\text{rev}$ is a reverse-time Wiener process in $\mathbb{R}^{D_x}$.

No exact density (sampling-only). High capacity for multimodal posteriors, since the diffusion process can represent disconnected components, which flows struggle with. Scales to higher state dimensions than flows.

### Direct regression (`head_type="regression"`)

Predict Gaussian posterior parameters directly:

$$q(\vect{x} \mid \vect{y}; \gvect{\nu}) = \mathcal{N}\left(\vect{x}; \gvect{\mu}_{x \mid y}(\vect{y}; \gvect{\nu}), \gmat{\Sigma}_{x \mid y}(\vect{y}; \gvect{\nu})\right)$$ (eq-gaussian-head)

where $\gvect{\mu}_{x \mid y} \in \mathbb{R}^{D_x}$ and $\gmat{\Sigma}_{x \mid y} \in \mathbb{R}^{D_x \times D_x}$ are the network's predicted posterior mean and covariance.

Cheapest by far. Restricted to unimodal Gaussian posteriors. Useful when the posterior is known to be Gaussian (e.g., as a learned replacement for the Laplace approximation [](#eq-laplace-cov) around an `IncrementalFourDVar` MAP).

## Training objective

Flow and regression heads use maximum likelihood on simulated pairs:

$$\mathcal{L}_\text{NLL}(\gvect{\nu}) = -\mathbb{E}_{(\vect{x}, \vect{y}) \sim p_\text{sim}} \log q(\vect{x} \mid \vect{y}; \gvect{\nu})$$ (eq-mle-loss)

where $p_\text{sim}$ is the joint distribution of the simulated pairs. Score heads use denoising score matching:

$$\mathcal{L}_\text{DSM}(\gvect{\nu}) = \mathbb{E}_{\tau, \vect{x}, \vect{y}, \gvect{\xi}} \Vert \vect{s}(\vect{x}_\tau, \tau \mid \vect{y}; \gvect{\nu}) - \nabla_{\vect{x}_\tau} \log p_\tau(\vect{x}_\tau \mid \vect{x}) \Vert_2^2$$ (eq-dsm-loss)

where $\vect{x}_\tau = \vect{x} + \sigma_\text{sde}(\tau) \gvect{\xi}$ is the noised state, $\gvect{\xi} \sim \mathcal{N}(\mathbf{0}, \mat{I})$ in $\mathbb{R}^{D_x}$, and $\sigma_\text{sde}(\tau)$ is the noise scale at $\tau$.

Both reduce to a standard training step over simulated minibatches: `amortized_train_step` is chapter 9's `train_step` with the negative log-likelihood [](#eq-mle-loss) in place of the reconstruction loss [](#eq-4dvarnet-loss).

## Implementation in vardax

Only the regression head ships today: `ConditionalFlowHead` and `ScoreDiffusionHead` are stubs that raise `NotImplementedError` when constructed, so the example uses `RegressionHead`.
The toy problem has a state of one time ($T = 0$ steps) and $N_{\Omega_x} = 4$ cells, so $D_x = D_y = 4$, with prior $p(\vect{x}) = \mathcal{N}(\mathbf{0}, \mat{I})$ and observations $\vect{y} = \vect{x} + \sigma \gvect{\varepsilon}$, $\sigma = 0.5$, fully observed.
Its exact posterior is $\mathcal{N}(\vect{y} / (1 + \sigma^2), \sigma^2 / (1 + \sigma^2) \mat{I})$, the BLUE [](#eq-blue-mean) and [](#eq-blue-cov) with $\gmat{\Sigma}_x = \mat{I}$, $\mat{H} = \mat{I}$ and $\gmat{\Sigma}_\varepsilon = \sigma^2 \mat{I}$, which the trained head should recover.
Each training step draws $N_\text{batch} = 64$ fresh simulated pairs, and the head is the Gaussian head [](#eq-gaussian-head) with diagonal covariance and context size $D_c = 16$.

```{code-block} python
:caption: Trains the Gaussian head [](#eq-gaussian-head) by minimising [](#eq-mle-loss) on simulated pairs and compares its mean with the exact posterior mean [](#eq-blue-mean).
:label: code-amortized-regression-head
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import vardax as vdx
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: N_batch = 64 simulated pairs; T + 1 = 1 time; N_Ωx = 4 grid cells;
#         D_x = N_Ωx; D_c = 16 context

key_encoder, key_head, key_train, key_test, key_samples = jax.random.split(
    jax.random.PRNGKey(0), 5
)
obs_std = 0.5  # σ


# user
def sample_train_pair(
    key: PRNGKeyArray,
) -> tuple[Float[Array, "t n"], Float[Array, "t n"]]:
    """Draw one simulated pair (x, y) from the prior and the observation model."""
    key_state, key_noise = jax.random.split(key)
    # x ~ p(x) = 𝒩(0, I)
    state = jax.random.normal(key_state, (1, 4))  # (T + 1, N_Ωx)
    # ε ~ 𝒩(0, I)
    noise = jax.random.normal(key_noise, (1, 4))  # (T + 1, N_Ωx)
    # y = x + σ ε
    obs = state + obs_std * noise  # (T + 1, N_Ωx)
    return state, obs


# user
@eqx.filter_jit
def simulate_batch(key: PRNGKeyArray) -> vdx.Batch1D:
    """Draw N_batch simulated pairs as a fully observed batch."""
    pair_keys = jax.random.split(key, 64)
    states, obs = jax.vmap(sample_train_pair)(pair_keys)  # (N_batch, T + 1, N_Ωx)
    # m = 1 everywhere
    all_observed = jnp.ones_like(obs)  # (N_batch, T + 1, N_Ωx)
    # today (vardax)
    return vdx.Batch1D(input=obs, mask=all_observed, target=states)


# today (vardax)
# c = g_enc([vec(y); vec(m)]; ν), an MLP
encoder = vdx.MLPObsEncoder(input_size=4, context_dim=16, key=key_encoder)
# today (vardax)
# q(x | y; ν) = 𝒩(μ_x|y(c), Σ_x|y(c)), Σ_x|y = diag(σ²_x|y)
head = vdx.RegressionHead(context_dim=16, state_shape=(1, 4), key=key_head)
# today (vardax)
config = vdx.AmortizedConfig(head_type="regression", n_samples=64)
# today (vardax)
# x ↦ q(x | y; ν), one encoder pass and one head pass
model = vdx.AmortizedPosterior(encoder=encoder, head=head, config=config)

# today (optax)
optimizer = optax.adam(3e-3)
# the trainable leaves ν
params = eqx.filter(model, eqx.is_array)
# today (optax)
opt_state = optimizer.init(params)
for batch_key in jax.random.split(key_train, 1000):
    batch = simulate_batch(batch_key)
    # today (vardax)
    # ℒ_NLL(ν) = −mean(log q(x | y; ν))
    model, opt_state, loss = vdx.amortized_train_step(
        model, batch, optimizer, opt_state
    )
print(f"final NLL {float(loss):.2f}")

# inference: one forward pass per batch
test_batch = simulate_batch(key_test)
# today (vardax)
# x⋆ = μ_x|y(y; ν), the mode of q(x | y; ν)
x_map = model(test_batch)  # → (N_batch, T + 1, N_Ωx)
# today (vardax)
# x⁽ᵉ⁾ ~ q(x | y; ν), e = 1, …, 200
samples = model.sample(  # → (N_batch, 200, T + 1, N_Ωx)
    test_batch, key_samples, n=200
)
# today (vardax)
# log q(x | y; ν), exact for the regression head
log_density = model.log_prob(  # → (N_batch,)
    test_batch.target, test_batch
)
# 𝔼[x | y] = y / (1 + σ²)
exact_mean = test_batch.input / (1 + obs_std**2)  # (N_batch, T + 1, N_Ωx)
# x⋆ − 𝔼[x | y]
mean_gap = x_map - exact_mean  # (N_batch, T + 1, N_Ωx)
# |x⋆ − 𝔼[x | y]|
abs_mean_gap = jnp.abs(mean_gap)  # (N_batch, T + 1, N_Ωx)
# max |x⋆ − 𝔼[x | y]|
mean_error = einx.max("s t n ->", abs_mean_gap)  # → ()
print(samples.shape, log_density.shape, f"max mean error {float(mean_error):.2f}")
```

It prints `final NLL 2.48` and `(64, 200, 1, 4) (64,) max mean error 0.37`; the exact posterior's NLL is $2 (\log(2 \pi \cdot 0.2) + 1) \approx 2.46$.

## Validation: the hard gates

Amortized inference is dangerous when miscalibrated. A well-trained
flow with a tight $q$ that's centred wrong gives confident wrong
answers. Decision D12 (the six-step cycle, chapter 14) requires three
gates before promoting an amortized head to operational use:

### 1. Posterior agreement

The amortized MAP must fall within one posterior standard deviation of an
oracle MAP from a slower method (`StrongFourDVar` /
`IncrementalFourDVar`):

$$\frac{\vert \vect{x}^\star_\text{amort} - \vect{x}^\star_\text{oracle} \vert}{\gvect{\sigma}_{x \mid y,\text{oracle}}} \le 1$$ (eq-amortized-gate-mean)

where $\gvect{\sigma}_{x \mid y,\text{oracle}} \in \mathbb{R}^{D_x}$ is the oracle's posterior standard deviation, cell by cell. This is the posterior gate [](#eq-gate-posterior) of chapter 14 applied to the amortized map.

For the toy problem the oracle is `ThreeDVar` with $\gmat{\Sigma}_x = \mat{I}$ and $\gmat{\Sigma}_\varepsilon = \sigma^2 \mat{I}$, whose Laplace posterior [](#eq-laplace-cov) is exact here because the problem is linear-Gaussian.
`assert_posterior_agreement` reads only the mean of the fast posterior, so the amortized MAP goes in a `Posterior` without a covariance.
The block continues the one above.

```{code-block} python
:caption: Checks the posterior-agreement gate [](#eq-amortized-gate-mean) for the amortized MAP against a 3DVar oracle with the Laplace posterior [](#eq-laplace-cov), on four windows.
:label: code-amortized-posterior-gate
import lineax as lx

# Shapes: N_batch = 1 window; T + 1 = 1 time; N_Ωx = 4 grid cells; D_x = N_Ωx

key_val, key_sbc = jax.random.split(jax.random.PRNGKey(1))
state_structure = jax.ShapeDtypeStruct((1, 4), jnp.float32)
# Σ_x = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# Σ_ε = σ² I; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (D_x, D_x) PSD op
    jnp.float32(obs_std**2) * prior_cov, lx.positive_semidefinite_tag
)
# today (vardax)
# 𝒥(x) = ½ ‖x − μ_x‖²_Σ_x⁻¹ + ½ ‖y − h(x)‖²_Σ_ε⁻¹, μ_x = 0
oracle = vdx.ThreeDVar(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 4)),
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)

val_batch = simulate_batch(key_val)
for window_index in range(4):
    # (N_batch, T + 1, N_Ωx)
    window_obs = val_batch.input[window_index : window_index + 1]
    window_mask = val_batch.mask[window_index : window_index + 1]
    # today (vardax)
    window = vdx.Batch1D(input=window_obs, mask=window_mask)
    # today (vardax)
    # Σ_x|y = (Σ_x⁻¹ + Hᵀ Σ_ε⁻¹ H)⁻¹ at the 3DVar MAP
    p_oracle = oracle.posterior(window)
    # today (vardax)
    # x⋆_amort = μ_x|y(y; ν)
    amortized_map = model(window)  # → (N_batch, T + 1, N_Ωx)
    # today (vardax)
    p_amort = vdx.Posterior(mean=amortized_map[0])
    # today (vardax)
    # |x⋆_amort − x⋆_oracle| / σ_x|y,oracle ≤ 1, cell by cell
    vdx.assert_posterior_agreement(p_amort, p_oracle, tolerance_sigma=1.0)
print("posterior agreement passed")
```

It prints `posterior agreement passed`.

### 2. Adjoint calibration

The amortized model's gradient with respect to observations must match
the physics-based gradient:

$$\frac{\Vert \partial \vect{x}^\star_\text{amort} / \partial \vect{y} - \partial \vect{x}^\star_\text{oracle} / \partial \vect{y} \Vert_\text{op}}{\Vert \partial \vect{x}^\star_\text{oracle} / \partial \vect{y} \Vert_\text{op}} < 0.05$$ (eq-amortized-gate-jacobian)

where the Jacobians are $D_x \times D_y$ matrices and $\Vert \cdot \Vert_\text{op}$ is the operator norm. This is the Jacobian gate [](#eq-gate-jacobian) of chapter 14 applied to the amortized map, with $\vect{y}$ as the input. It is measured by random-vector probing (randomised tangent-linear tests for operator agreement). Without this gate, the amortized model may match the oracle MAP on the training distribution but extrapolate badly when the observation distribution shifts.

(sec-amortized-sbc)=
### 3. Simulation-based calibration (SBC)

Per Talts et al. (2018): sample $\vect{x}_n$ from the prior, simulate
$\vect{y}_n$, draw $N_\mathcal{E}$ samples $\vect{x}^{(e)}$ from $q$ conditioned on $\vect{y}_n$, compute the
rank of $\vect{x}_n$ in the sample. If $q$ is calibrated, the ranks
are uniform across the sample. Failure indicates over- or
under-confident posteriors.

`simulation_based_calibration` takes three callables (posterior sampler, prior sampler, simulator) rather than the model, and reduces each state to its $\ell_2$ norm before ranking it among the samples.
With $N_\mathcal{E} = 19$ samples per run the rank takes 20 values, grouped here into 5 bins of 4; a calibrated head gives about 20 runs per bin.
The block continues the one above.

```{code-block} python
:caption: Runs the simulation-based calibration gate of [](#sec-amortized-sbc) on the amortized posterior and tests the rank histogram for uniformity with a χ² test.
:label: code-amortized-sbc
import jax.scipy.stats as jstats

# Shapes: T + 1 = 1 time; N_Ωx = 4 grid cells; N_E = 19 posterior samples per run


# user
@eqx.filter_jit
def sample_posterior(
    obs: Float[Array, "t n"], key: PRNGKeyArray, n: int
) -> Float[Array, "j t n"]:
    """Draw n samples from q(x | y; ν) for one fully observed window."""
    obs_window = einx.id(  # (T + 1, N_Ωx) → (1, T + 1, N_Ωx)
        "t n -> 1 t n", obs
    )
    # m = 1 everywhere
    all_observed = jnp.ones_like(obs_window)  # (1, T + 1, N_Ωx)
    # today (vardax)
    window = vdx.Batch1D(input=obs_window, mask=all_observed)
    # today (vardax)
    # x⁽ᵉ⁾ ~ q(x | y; ν)
    window_samples = model.sample(window, key, n)  # → (1, N_E, T + 1, N_Ωx)
    return window_samples[0]  # (1, N_E, ...) → (N_E, T + 1, N_Ωx)


# user
def sample_prior(key: PRNGKeyArray) -> Float[Array, "t n"]:
    """Draw x from the prior p(x)."""
    # x ~ 𝒩(0, I)
    return jax.random.normal(key, (1, 4))  # (T + 1, N_Ωx)


# user
def simulate_obs(state: Float[Array, "t n"], key: PRNGKeyArray) -> Float[Array, "t n"]:
    """Simulate y from x through the observation model."""
    # ε ~ 𝒩(0, I)
    noise = jax.random.normal(key, (1, 4))  # (T + 1, N_Ωx)
    # y = x + σ ε
    return state + obs_std * noise  # (T + 1, N_Ωx)


# today (vardax)
# rank of ‖x‖ among ‖x⁽ᵉ⁾‖, x⁽ᵉ⁾ ~ q(x | y; ν), over 100 runs
ranks = vdx.simulation_based_calibration(  # → (100,)
    sample_posterior,
    sample_prior,
    simulate_obs,
    key=key_sbc,
    n_runs=100,
    n_samples=19,
)
# b = ⌊rank / 4⌋, five bins of four ranks
rank_bins = ranks // 4  # (100,)
# N_b, the runs in bin b
bin_counts = jnp.bincount(rank_bins, length=5)  # (100,) → (5,)
# (N_b − 20)²
squared_excess = (bin_counts - 20) ** 2  # (5,)
# χ² = ∑_b (N_b − 20)² / 20
chi_square = einx.sum("b ->", squared_excess) / 20  # → ()
# p = ℙ(χ²₄ ≥ χ²)
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
