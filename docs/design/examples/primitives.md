---
status: draft
version: 0.4.0
---

# Layer 0 — Primitive Examples

Pure JAX cost terms, closed-form BLUE, control-variable transform,
posterior primitives, adjoint composition.

---

## Closed-form BLUE (`OptimalInterpolation`)

```{code-block} python
:caption: Computes the closed-form BLUE analysis [](#eq-blue-mean) and its lazy covariance [](#eq-blue-cov) with structured B and R.
:label: code-design-primitives-blue
import gaussx as gx
import lineax as lx
import vardax as vdx

# Shapes: N state cells; M observations; D = 2 coordinates
# Given: coords (N, D), obs_variances (M,), background x_b (N,), obs y (M,),
#        obs_op a linear vardax observation operator

# proposed (gaussx)
# Bᵢⱼ = k_Matérn-3/2(sᵢ, sⱼ; ℓ = 10, σ = 1)
prior_cov = gx.MaternLinearOperator(  # (N, D) → (N, N) PSD op
    coords, length_scale=10.0, nu=1.5, sigma=1.0
)
# today (lineax)
# R = diag(σ²)
obs_cov = lx.DiagonalLinearOperator(obs_variances)  # (M,) → (M, M) op
# today (vardax)
# H = H′(x_b), exact since obs_op is linear
obs_jacobian = obs_op.linearize(background)  # (N,) → (M, N) op

# proposed (D16)
# x* = x_b + K (y − H x_b), P* = (I − K H) B
analysis, posterior_cov = vdx.costs.blue_analysis(  # (N,), (N, N) op
    background, obs, prior_cov, obs_cov, obs_jacobian
)
# P* stays an AbstractLinearOperator: mat-vecs via lineax.CG, never materialised
```

The implementation picks between the $B$-space and $R$-space forms
(Sherman-Morrison-Woodbury) based on dimensions.

---

## Variational cost

```{code-block} python
:caption: Evaluates the observation cost [](#eq-obs-cost), the classical and learned prior terms of [](#eq-3dvar-cost) and [](#eq-4dvarnet-cost), and the gradient of the 4DVarNet cost.
:label: code-design-primitives-variational-cost
import jax
import vardax as vdx

# Shapes: S windows; T times; N state cells
# Given: state x (S, T, N), batch (y, m), obs_op H, obs_precision R⁻¹ op,
#        background x_b, prior_precision B⁻¹ op, ae_prior φ

# proposed (D19)
# J_obs(x) = ½ ‖m ⊙ (y − H(x))‖²_{R⁻¹}
obs_term = vdx.obs_cost(  # → ()
    state, batch.input, batch.mask, obs_operator=obs_op, R_inv_op=obs_precision
)

# today* (vardax)
# J_b(x) = ½ ‖x − x_b‖²_{B⁻¹}
prior_term_classical = vdx.prior_cost(  # → ()
    state, prior_mean=background, B_inv_op=prior_precision
)

# today* (vardax)
# J_prior(x) = ‖x − φ(x)‖²
prior_term_learned = vdx.prior_cost(state, prior_fn=ae_prior)  # → ()

# today* (vardax)
# J(x) = α_obs ½ ‖H(x) − y‖²_{R⁻¹} + α_prior ‖x − φ(x)‖²
total_cost = vdx.variational_cost(  # → ()
    state,
    batch,
    prior_fn=ae_prior,
    obs_operator=obs_op,
    alpha_obs=1.0,
    alpha_prior=0.1,
)

# today* (vardax)
# ∇ₓJ(x)
cost_grad = jax.grad(vdx.variational_cost)(  # (S, T, N)
    state, batch, ae_prior, obs_op, 1.0, 0.1
)
```

---

## Incremental 4DVar inner loop

```{code-block} python
:caption: Linearises the model and observation operator [](#eq-tlm-forward), [](#eq-tlm-obs) at the outer iterate and solves the Gauss-Newton system [](#eq-gn-system) of the incremental cost [](#eq-incremental-cost) with CG.
:label: code-design-primitives-incremental-inner
import jax
import jax.numpy as jnp
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N state cells; M observations
# Given: background x_b (N,), batch, forward_model and its step length dt,
#        obs_op, prior_precision B⁻¹ op, obs_precision R⁻¹ op, plus
#        initial_state, prior_cov B, obs_cov R and config for the helper


# user
def step(state: Float[Array, " N"]) -> Float[Array, " N"]:
    """One model step Mₜ(x) of the forward."""
    return forward_model.step(state, dt)


# Mₜ(x_b + δx) ≈ Mₜ(x_b) + M′ₜ δx
forward_lin = jax.linearize(step, background)  # (Mₜ(x_b), δx ↦ M′ₜ δx)
# today (vardax)
# Hₜ(z + δz) ≈ Hₜ(z) + H′ₜ δz
obs_op_lin = obs_op.linearize(background)  # (N,) → (M, N) op
# δx⁽⁰⁾ = 0
increment_start = jnp.zeros_like(background)  # (N,)

# proposed (D11)
# 𝐇_GN δx* = −∇J(x_b), solved with at most 20 CG steps
increment = vdx.solver.gauss_newton_inner(  # (N,)
    dx0=increment_start,
    x_b=background,
    batch=batch,
    forward_lin=forward_lin,
    obs_op_lin=obs_op_lin,
    B_inv_op=prior_precision,
    R_inv_op=obs_precision,
    n_inner=20,
    cg_atol=1e-5,
    cg_rtol=1e-5,
)

# x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + δx*
background = background + increment  # (N,)

# proposed (D11)
# x* after config.n_outer relinearisations
analysis = vdx.solver.incremental_outer(  # (N,)
    initial_state, batch, forward_model, obs_op, prior_cov, obs_cov, config
)
```

---

## Control-variable transform

```{code-block} python
:caption: Maps a state to control variables and back with the Matérn square root S = B^{1/2} of the control-variable transform [](#eq-cvt).
:label: code-design-primitives-cvt
import gaussx as gx
import vardax as vdx

# Shapes: N state cells; D = 2 coordinates
# Given: coords (N, D), state x (N,), background x_b (N,)

# proposed (gaussx)
# Bᵢⱼ = k_Matérn-3/2(sᵢ, sⱼ; ℓ = 10, σ = 1)
prior_cov = gx.MaternLinearOperator(  # (N, D) → (N, N) PSD op
    grid_coords=coords, length_scale=10.0, nu=1.5, sigma=1.0
)
# proposed (gaussx; gaussx ships gx.sqrt(operator))
# S = B^{1/2}
prior_sqrt = prior_cov.half()  # (N, N) op

# proposed (D11)
# χ = B^{−1/2} (x − x_b)
control = vdx.cvt.cvt_transform(state, background, prior_sqrt)  # (N,) → (N,)
# proposed (D11)
# x = x_b + B^{1/2} χ
state = vdx.cvt.cvt_inverse(control, background, prior_sqrt)  # (N,) → (N,)
```

In CVT coordinates the prior cost is $\|\chi\|^2$, preconditioning CG.

---

## Adjoint composition (Decision D15)

Gradients through ODE integration come from `diffrax.AbstractAdjoint`;
gradients through inner minimisation come from
`optimistix.AbstractAdjoint`. Vardax exposes both as constructor slots.

```{code-block} python
:caption: Configures StrongFourDVar for the cost [](#eq-strong4dvar-cost) with an implicit minimiser adjoint [](#eq-implicit-grad) and a continuous dynamics adjoint [](#eq-adjoint-ode), or with checkpointed backprop.
:label: code-design-primitives-adjoint-slots
import diffrax as dfx
import optimistix as optx
import vardax as vdx

# Shapes: N state cells; T times
# Given: somax_model (a diffrax-integrated forward), obs_op, background x_b (N,),
#        prior_cov B op, obs_cov R op

# today* (vardax; optx.GaussNewton is a least-squares solver, not an
#   AbstractMinimiser, so the model rejects it: use optx.BFGS or NonlinearCG)
# memory-efficient operational configuration
# x₀* = argmin J(x₀), dx₀*/dθ by the implicit function theorem, λ(t) by a backsolve
model = vdx.StrongFourDVar(
    forward=somax_model,
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
    # continuous adjoint, O(1) memory in T
    forward_adjoint=dfx.BacksolveAdjoint(),
)

# today* (vardax)
# standard configuration: backprop through both, with checkpoints
model = vdx.StrongFourDVar(
    ...,
    minimiser_adjoint=optx.RecursiveCheckpointAdjoint(),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)

# today (diffrax)
# forward sensitivity, good when the parameter count is far below N
forward_adjoint = dfx.ForwardMode()
```

For `FourDVarNet`, the inner-solver adjoint controls how training
gradients flow through the unrolled learned iteration:

```{code-block} python
:caption: Chooses how training gradients flow back through the unrolled 4DVarNet update [](#eq-4dvarnet-update): full backprop, last step only, or the fixed-point gradient [](#eq-fixed-point-grad).
:label: code-design-primitives-fourdvarnet-adjoint
import optimistix as optx
import vardax as vdx

# Shapes: K = 15 inner steps
# Given: ae_prior φ, obs_op H, conv_lstm Φ

# Three adjoint options for FourDVarNet's inner solver:
# today (optimistix)
# O(K) memory, standard backprop through x₁ … x_K
solver_adjoint = optx.RecursiveCheckpointAdjoint()
# today (vardax)
# O(1) memory, last step only (Bolte et al. 2023)
solver_adjoint = vdx.adjoints.OneStepAdjoint()
# today (optimistix)
# O(1) memory, dx*/dθ = (I − ∂f/∂x)⁻¹ ∂f/∂θ, exact at a fixed point
solver_adjoint = optx.ImplicitAdjoint()

# proposed (D14; ships today as vdx.FourDVarNet1D / vdx.FourDVarNet2D)
# x_{k+1} = x_k − Φ(∇ₓJ(x_k), h_k), k = 0 … K − 1
model = vdx.FourDVarNet(
    prior=ae_prior,
    obs_op=obs_op,
    grad_mod=conv_lstm,
    # proposed (D4)
    config=vdx.SolverConfig(n_steps=15),
    solver_adjoint=solver_adjoint,
)
```

---

## Laplace posterior covariance

```{code-block} python
:caption: Builds the Laplace covariance [](#eq-laplace-cov) at the MAP and the inverse Gauss-Newton Hessian [](#eq-gn-cov) of [](#eq-gn-hessian) as function primitives.
:label: code-design-primitives-laplace
import jax
import vardax as vdx

# Shapes: N state cells
# Given: analysis x* (N,), threedvar_cost J, batch, forward_model, obs_op,
#        prior_precision B⁻¹ op, obs_precision R⁻¹ op, prior_cov B op, obs_cov R op

# ∇J
cost_grad_fn = jax.grad(threedvar_cost)

# proposed (D10; ships today as the vdx.LaplaceCovariance adapter)
# Laplace at MAP, for ThreeDVar / StrongFourDVar / WeakFourDVar / FourDVarNet
# P*_Laplace = (∇²J(x*))⁻¹ ≈ (B⁻¹ + H′ᵀ R⁻¹ H′)⁻¹
posterior_cov = vdx.posterior.laplace_covariance(  # (N, N) op
    analysis,
    cost_grad_fn=cost_grad_fn,
    B_inv_op=prior_precision,
    R_inv_op=obs_precision,
)

# proposed (D10; ships today as the vdx.GaussNewtonHessian adapter)
# reused inside IncrementalFourDVar.posterior(batch)
# P*_GN = 𝐇_GN⁻¹, from 50 Lanczos vectors
gn_posterior_cov = vdx.posterior.gauss_newton_hessian(  # (N, N) op
    analysis, batch, forward_model, obs_op, prior_cov, obs_cov, n_krylov=50
)
```

`OptimalInterpolation` and `IncrementalFourDVar` return posteriors
directly via `.posterior(batch)` — no `laplace_covariance` call needed.
