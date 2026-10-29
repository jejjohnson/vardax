---
status: draft
version: 0.4.0
---

# Layer 2 — Model Examples

End-to-end workflows for each of the seven peer analysis classes.

---

## `OptimalInterpolation` — closed-form fast path

```{code-block} python
:caption: Maps along-track SSH with the closed-form BLUE mean [](#eq-blue-mean) and covariance [](#eq-blue-cov) under a Matérn prior.
:label: code-design-models-oi
import gaussx as gx
import lineax as lx
import vardax as vdx

# Shapes: S windows; N grid cells; M along-track observations; D = 2 coordinates
# Given: along_track_op H (M, N) lineax op, climatology_ssh x_b (N,),
#        coords (N, D), altika_variances (M,), batch

# today (vardax)
# H(x) = H x, must be linear
obs_op = vdx.LinearObs(operator=along_track_op)
# proposed (gaussx)
# Bᵢⱼ = k_Matérn(sᵢ, sⱼ; ℓ = 100, σ = 0.1)
prior_cov = gx.MaternLinearOperator(  # (N, D) → (N, N) PSD op
    coords, length_scale=100.0, sigma=0.1
)
# today (lineax)
# R = diag(σ²)
obs_cov = lx.DiagonalLinearOperator(altika_variances)  # (M,) → (M, M) op

# today (vardax)
model = vdx.OptimalInterpolation(
    obs_op=obs_op,
    prior_mean=climatology_ssh,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)

# Single forward pass, no iteration
# today (vardax)
# x* = x_b + K (y − H x_b)
analysis = model(batch)  # → (S, N)
# today (vardax)
# P* = (I − K H) B
posterior = model.posterior(batch)  # Posterior(mean (N,), cov (N, N) op)
```

Use when $H$ is linear, $B$ and $R$ are Gaussian. The right default for
SSH altimetry with along-track observations and a Matérn prior.

---

## `ThreeDVar` — nonlinear, single time

```{code-block} python
:caption: Minimises the 3DVar cost [](#eq-3dvar-cost) with Gauss-Newton for an averaging-kernel observation operator [](#eq-averaging-kernel).
:label: code-design-models-threedvar
import optimistix as optx
import vardax as vdx

# Shapes: S windows; N levels
# Given: A (N, N) lineax op, xa x_a (N,), h (N,), background x_b (N,),
#        prior_cov B op, obs_cov R op, batch

# today (vardax)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a), nonlinear via h ⊙ x
obs_op = vdx.AveragingKernel(kernel_matrix=A, prior_profile=xa, pressure_weights=h)

# today* (vardax; optx.GaussNewton is a least-squares solver, not an
#   AbstractMinimiser, so the model rejects it: use optx.BFGS or NonlinearCG)
# J(x) = ½ ‖x − x_b‖²_{B⁻¹} + ½ ‖y − H(x)‖²_{R⁻¹}
model = vdx.ThreeDVar(
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
)
# today (vardax)
# x* = argmin J(x)
analysis = model(batch)  # → (S, N)
```

Use when $H$ is nonlinear (averaging kernel, RTM, custom retrieval) but
there are no dynamics — snapshot inversion.

---

## `StrongFourDVar` — multi-time, exact dynamics

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) with nonlinear CG over a somax forward.
:label: code-design-models-strong4dvar
import diffrax as dfx
import optimistix as optx
import vardax as vdx

# Shapes: S windows; T times; N state cells
# Given: somax_model (a pipekit_cycle.ForwardModel), background x_b (N,),
#        prior_cov B op, obs_cov R op, batch

# today (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖mₜ ⊙ (yₜ − Mₜ(x₀))‖²_{R⁻¹}
model = vdx.StrongFourDVar(
    forward=somax_model,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)
# today (vardax)
# x₀* = argmin J(x₀)
analysis = model(batch)  # → (S, N)
```

For long assimilation windows, switch the dynamics adjoint:

```{code-block} python
:caption: Swaps the dynamics adjoint for the continuous adjoint ODE [](#eq-adjoint-ode) or for forward sensitivities.
:label: code-design-models-forward-adjoint
import diffrax as dfx

# today (diffrax)
# dλ/dt = −M′(x(t))ᵀ λ(t) − ∇J, O(1) memory
forward_adjoint = dfx.BacksolveAdjoint()
# or
# today (diffrax)
# forward sensitivity, for a small parameter dimension
forward_adjoint = dfx.ForwardMode()
```

---

## `WeakFourDVar` — model-error-aware

```{code-block} python
:caption: Minimises the weak-constraint cost [](#eq-weak4dvar-cost) jointly over the initial state and the model errors of [](#eq-weak-model).
:label: code-design-models-weak4dvar
import diffrax as dfx
import optimistix as optx
import vardax as vdx

# Shapes: S windows; T times; N state cells
# Given: somax_model M_t^free, obs_op, background x_b (N,), prior_cov B op,
#        obs_cov R op, model_err_cov Q op (prior on ηₜ), batch

# today* (vardax; optx.NonlinearCG also requires atol)
# J(x₀, η) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖yₜ − Hₜ(xₜ)‖²_{R⁻¹} + ½ Σₜ ‖ηₜ‖²_{Q⁻¹}
model = vdx.WeakFourDVar(
    forward=somax_model,
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    model_err_cov_op=model_err_cov,
    minimiser=optx.NonlinearCG(rtol=1e-5),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)

# Returns both the analysis IC and the model-error trajectory
# today (vardax)
# (x₀*, η*) = argmin J(x₀, η)
analysis, model_error = model(batch)  # → (S, N), (S, T, N)
```

Use when the model has known biases (climatological drift) and you need
to estimate the time-varying $\eta_t$ alongside the initial state.

---

## `IncrementalFourDVar` — operational fast path

```{code-block} python
:caption: Minimises the incremental cost [](#eq-incremental-cost) by Gauss-Newton and CG in the control variables [](#eq-cvt), then reads the posterior [](#eq-gn-cov) from the last Hessian.
:label: code-design-models-incremental
import gaussx as gx
import lineax as lx
import vardax as vdx

# Shapes: S windows; T times; N state cells; M observations; D = 2 coordinates
# Given: somax_model, A (N, N) lineax op, xa x_a (N,), h (N,),
#        background x_b (N,), coords (N, D), obs_uncertainty (M,), batch

# today (vardax)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a)
obs_op = vdx.AveragingKernel(kernel_matrix=A, prior_profile=xa, pressure_weights=h)
# proposed (gaussx)
# Bᵢⱼ = k_Matérn(sᵢ, sⱼ; ℓ = 10, σ = 0.1)
prior_cov = gx.MaternLinearOperator(  # (N, D) → (N, N) PSD op
    coords, length_scale=10.0, sigma=0.1
)
# today (lineax)
# R = diag(σ²)
obs_cov = lx.DiagonalLinearOperator(obs_uncertainty)  # (M,) → (M, M) op

# today (vardax)
# 3 outer relinearisations, 20 CG steps each, in χ with x₀ − x_b = S χ
model = vdx.IncrementalFourDVar(
    forward=somax_model,
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)

# Operational analysis
# today (vardax)
# x₀* after the last outer iteration
analysis = model(batch)  # → (S, N)

# Posterior reuses GN Hessian from last outer iteration
# today (vardax)
# P*_GN = 𝐇_GN⁻¹
posterior = model.posterior(batch)  # Posterior(mean (N,), cov (N, N) op)
```

Use for production / operational 4DVar with structured Matérn priors.
Functionally equivalent to `StrongFourDVar` but with the GN+CG+CVT
operational pattern hard-wired.

---

## `FourDVarNet` — learned 4DVar

```{code-block} python
:caption: Trains a composable FourDVarNet on the 4DVarNet loss [](#eq-4dvarnet-loss) through its unrolled update [](#eq-4dvarnet-update).
:label: code-design-models-fourdvarnet
import equinox as eqx
import optax
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B batch; T times; H × W grid; K = 15 inner steps
# Given: enc_2d, dec_2d autoencoder halves, dataloader of vdx.Batch2D

# proposed (D14; ships today as vdx.FourDVarNet2D(n_time, height, width, ..., key))
# x_{k+1} = x_k − Φ(∇ₓJ(x_k), h_k), with J = α_obs ½ ‖H(x) − y‖² + α_prior ‖x − φ(x)‖²
model = vdx.FourDVarNet(
    # proposed (D14; vardax ships ConvAEPrior1D and BilinAEPrior2D)
    prior=vdx.ConvAEPrior(encoder=enc_2d, decoder=dec_2d),
    obs_op=vdx.MaskedIdentity(),
    # today* (vardax; also takes state_channels and key)
    grad_mod=vdx.ConvLSTMGradMod2D(hidden_dim=64),
    # proposed (D4)
    config=vdx.SolverConfig(n_steps=15, alpha=0.2),
    solver_adjoint=vdx.adjoints.OneStepAdjoint(),
)

# today (optax)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


# user
@eqx.filter_jit
def step(
    model: eqx.Module, opt_state: optax.OptState, batch: vdx.Batch2D
) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
    """One optimiser step on the 4DVarNet loss."""
    # today (vardax)
    # ℒ(θ, φ) = E ‖x*(θ, φ; y) − x_true‖²
    model, opt_state, loss = vdx.train_step(model, batch, optimizer, opt_state)
    return model, opt_state, loss


for epoch in range(100):
    for batch in dataloader:
        model, opt_state, loss = step(model, opt_state, batch)
```

After training, `model.as_analysis_step()` gives the operational
interface for use in `pipekit_cycle.DACycle`.

---

## `AmortizedPosterior` — direct head

```{code-block} python
:caption: Trains an amortized flow posterior [](#eq-amortized-objective) on simulated pairs and draws samples through the flow [](#eq-flow-sample).
:label: code-design-models-amortized
import jax
import vardax as vdx
from jaxtyping import PRNGKeyArray

# Shapes: B batch; N state cells; M observations; n = 200 samples
# Given: prior_distribution p(x), forward_model, obs_noise p(ε), quality_mask m,
#        simulation_loader, optimizer, opt_state, batch

key_infer = jax.random.PRNGKey(0)

# today (vardax)
model = vdx.AmortizedPosterior(
    # proposed (D12; vardax ships MLPObsEncoder and IdentityObsEncoder)
    # c = c_ψ(y, m)
    encoder=vdx.ConvObsEncoder(...),
    # proposed (D12; vdx.ConditionalFlowHead is a stub that raises until gauss_flows)
    # x = f_φ(z; c), z ~ 𝒩(0, I)
    head=vdx.ConditionalFlowHead(...),
    config=vdx.AmortizedConfig(head_type="flow", n_samples=64),
)


# user
def sample_train_pair(key: PRNGKeyArray) -> vdx.Batch2D:
    """Draw one simulated (y, x) training pair."""
    key_state, key_noise = jax.random.split(key)
    # x ~ p(x)
    state = prior_distribution.sample(key_state)  # (N,)
    # F(x)
    obs_clean = forward_model(state)  # (N,) → (M,)
    # ε ~ p(ε)
    noise = obs_noise.sample(key_noise)  # (M,)
    # y = F(x) + ε
    obs = obs_clean + noise  # (M,)
    # today (vardax)
    return vdx.Batch2D(input=obs, mask=quality_mask, target=state)


for batch in simulation_loader:
    # today* (vardax; the amortized step ships as vdx.amortized_train_step)
    # φ* = argmin E KL(q_φ(· ∣ y) ‖ p(· ∣ y))
    model, opt_state, loss = vdx.train_step(model, batch, optimizer, opt_state)

# Inference: sub-second
# today (vardax)
# x̂ = mode of q_φ(· ∣ y)
point_estimate = model(batch)  # → (B, N)
# today (vardax)
# xⁱ ~ q_φ(· ∣ y), i = 1 … n
samples = model.sample(batch, key_infer, n=200)  # → (B, n, N)
```

Validation gates per Decision D12:

```{code-block} python
:caption: Checks the amortized posterior against the physics posterior with the posterior gate [](#eq-gate-posterior) on Laplace covariances [](#eq-laplace-cov), then runs simulation-based calibration.
:label: code-design-models-amortized-gates
import vardax as vdx

# Given: model (amortized), strong_4dvar (a vdx.StrongFourDVar), val_loader,
#        prior_distribution, forward_model

for val_batch in val_loader:
    # today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
    # P*_amort = (B⁻¹ + H′ᵀ R⁻¹ H′)⁻¹ at the amortized estimate
    p_amort = vdx.LaplaceCovariance()(
        model(val_batch), model.as_analysis_step(), val_batch
    )
    # today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
    # P*_phys at the physics MAP
    p_phys = vdx.LaplaceCovariance()(
        strong_4dvar(val_batch), strong_4dvar.as_analysis_step(), val_batch
    )
    # today (vardax)
    # |x*_amort − x*_phys| / σ_post,phys ≤ 1
    vdx.assert_posterior_agreement(p_amort, p_phys, tolerance_sigma=1.0)

# today* (vardax; takes sample_posterior, sample_prior, simulate_obs and key)
# rank of x_true among posterior samples, uniform if calibrated
vdx.simulation_based_calibration(model, prior_distribution, forward_model, n_runs=200)
```

---

## Cycling any model through `pipekit_cycle.DACycle`

All seven satisfy `AnalysisStep` via `.as_analysis_step()` — the cycling
code is identical:

```{code-block} python
:caption: Cycles any of the seven analysis steps through forecast and analysis windows, the variational counterpart of the recursion [](#eq-kalman-filter).
:label: code-design-models-cycle
import pipekit_cycle as pc
import vardax as vdx

# Given: somax_model, model (any of the seven), satellite_loader,
#        n_assimilation_windows, initial_state

# today (pipekit_cycle)
# forecast x^f_t = M_t(x^a_{t−1}), then analysis x^a_t from the model's step
da_cycle = pc.DACycle(
    forward_model=somax_model,
    # today* (vardax)
    obs_op=vdx.AveragingKernel(...),
    # any of the seven
    analysis_step=model.as_analysis_step(),
    obs_source=satellite_loader,
    n_steps=n_assimilation_windows,
)

# today (pipekit_cycle)
result, final_state = da_cycle(initial_state, pc.DAState(t=0.0, cycle_count=0))
```

The orchestration is decoupled from the inference algorithm — swap
`OptimalInterpolation` for `IncrementalFourDVar` for `FourDVarNet` by
changing the `analysis_step` slot. Nothing else in the pipeline
changes.

---

## Linear-Gaussian agreement test (Decision D14 invariant)

```{code-block} python
:caption: Checks that every method recovers the BLUE mean [](#eq-blue-mean) in the linear-Gaussian limit.
:label: code-design-models-agreement
import optimistix as optx
import vardax as vdx

# Shapes: N state cells; M observations
# Given: linear_obs_op, background x_b (N,), prior_cov B op, obs_cov R op,
#        static_forward (Mₜ = I)

# All seven should agree in the linear-Gaussian limit
# user
batch = make_linear_gaussian_batch()

# today (vardax)
# x* = x_b + K (y − H x_b)
oi = vdx.OptimalInterpolation(linear_obs_op, background, prior_cov, obs_cov)
# today* (vardax; optx.GaussNewton also requires atol, and is a least-squares
#   solver that ThreeDVar rejects as minimiser)
threedvar = vdx.ThreeDVar(
    linear_obs_op, background, prior_cov, obs_cov, optx.GaussNewton(rtol=1e-7)
)
# today* (vardax; optx.NonlinearCG also requires atol)
strong = vdx.StrongFourDVar(
    static_forward,
    linear_obs_op,
    background,
    prior_cov,
    obs_cov,
    optx.NonlinearCG(rtol=1e-7),
)
# today (vardax)
incremental = vdx.IncrementalFourDVar(
    static_forward,
    linear_obs_op,
    background,
    prior_cov,
    obs_cov,
    vdx.IncrementalConfig(n_outer=5, n_inner=50),
)
# proposed (D14; IdentityGradMod and SolverConfig are planned, IdentityPrior ships)
fourdvarnet = vdx.FourDVarNet(
    vdx.IdentityPrior(),
    linear_obs_op,
    vdx.IdentityGradMod(alpha=0.05),
    vdx.SolverConfig(n_steps=200),
)

# proposed (D14)
# ‖x*_method − x*_OI‖_∞ ≤ 10⁻³ for every method
vdx.utils.validation.assert_methods_agree(
    {
        "oi": oi,
        "3dvar": threedvar,
        "strong": strong,
        "incremental": incremental,
        "fourdvarnet": fourdvarnet,
    },
    batch,
    atol=1e-3,
)
```

This is the canonical correctness baseline — every method, classical and
learned, must produce the same posterior mean in the linear-Gaussian
limit.
