---
status: draft
version: 0.4.0
---

# Layer 0 — Primitive Examples

Pure JAX cost terms, closed-form BLUE, control-variable transform,
posterior primitives, adjoint composition.

---

## Closed-form BLUE (`OptimalInterpolation`)

```python
import gaussx as gx
import lineax as lx
from vardax.costs import blue_analysis

# Structured B and R via gaussx / lineax
B_op = gx.MaternLinearOperator(coords, length_scale=10.0, nu=1.5, sigma=1.0)
R_op = lx.DiagonalLinearOperator(obs_variances)
H_op = obs_op.linearize(x_b)   # AbstractLinearOperator (since obs_op is linear)

x_star, P_star_op = blue_analysis(x_b, y, B_op, R_op, H_op)
# P_star_op is an AbstractLinearOperator — mat-vec via lineax.CG, no full materialisation
```

The implementation picks between the $B$-space and $R$-space forms
(Sherman-Morrison-Woodbury) based on dimensions.

---

## Variational cost

```python
from vardax.costs import obs_cost, prior_cost, variational_cost

# Observation cost (classical with R^{-1})
j_obs = obs_cost(x, batch.input, batch.mask, obs_operator=obs_op, R_inv_op=R_inv)

# Prior cost — classical with B^{-1}
j_prior_classical = prior_cost(x, prior_mean=x_b, B_inv_op=B_inv)

# Prior cost — learned (FourDVarNet)
j_prior_learned = prior_cost(x, prior_fn=ae_model)

# Total (FourDVarNet form)
j_total = variational_cost(
    x, batch, prior_fn=ae_model, obs_operator=obs_op,
    alpha_obs=1.0, alpha_prior=0.1,
)

grad_J = jax.grad(variational_cost)(x, batch, ae_model, obs_op, 1.0, 0.1)
```

---

## Incremental 4DVar inner loop

```python
import jax, lineax as lx
from vardax.solver import gauss_newton_inner, incremental_outer

# At outer iterate x_b, linearise via jax.linearize
forward_lin = jax.linearize(lambda s: forward_model.step(s, dt), x_b)
obs_op_lin = obs_op.linearize(x_b)

# Inner CG solve
dx_star = gauss_newton_inner(
    dx0=jnp.zeros_like(x_b),
    x_b=x_b, batch=batch,
    forward_lin=forward_lin, obs_op_lin=obs_op_lin,
    B_inv_op=B_inv, R_inv_op=R_inv,
    n_inner=20, cg_atol=1e-5, cg_rtol=1e-5,
)

# Outer update
x_b = x_b + dx_star

# Or use the high-level helper
x_star = incremental_outer(x0, batch, forward, obs_op, B_op, R_op, config)
```

---

## Control-variable transform

```python
import gaussx as gx
from vardax.cvt import cvt_transform, cvt_inverse

B_half = gx.MaternLinearOperator(
    grid_coords=coords, length_scale=10.0, nu=1.5, sigma=1.0,
).half()

chi = cvt_transform(x, x_b, B_half)    # χ = B^{-1/2}(x - x_b)
x = cvt_inverse(chi, x_b, B_half)      # x = x_b + B^{1/2} χ
```

In CVT coordinates the prior cost is $\|\chi\|^2$, preconditioning CG.

---

## Adjoint composition (Decision D15)

Gradients through ODE integration come from `diffrax.AbstractAdjoint`;
gradients through inner minimisation come from
`optimistix.AbstractAdjoint`. Vardax exposes both as constructor slots.

```python
import diffrax as dfx
import optimistix as optx
from vardax.models import StrongFourDVar

# Memory-efficient operational configuration
model = StrongFourDVar(
    forward=somax_model,
    obs_op=obs_op,
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
    forward_adjoint=dfx.BacksolveAdjoint(),    # continuous adjoint, O(1) memory
)

# Standard configuration
model = StrongFourDVar(
    ...,
    minimiser_adjoint=optx.RecursiveCheckpointAdjoint(),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)

# Forward sensitivity (good when params << state)
forward_adjoint = dfx.ForwardMode()
```

For `FourDVarNet`, the inner-solver adjoint controls how training
gradients flow through the unrolled learned iteration:

```python
from vardax.models import FourDVarNet
from vardax.adjoints import OneStepAdjoint   # Bolte et al. 2023

# Three adjoint options for FourDVarNet's inner solver:
solver_adjoint = optx.RecursiveCheckpointAdjoint()   # O(K) memory, standard backprop
solver_adjoint = OneStepAdjoint()                    # O(1) memory, last step only
solver_adjoint = optx.ImplicitAdjoint()              # O(1) memory, exact at fixed point

model = FourDVarNet(
    prior=ae_prior, obs_op=obs_op, grad_mod=conv_lstm,
    config=SolverConfig(n_steps=15),
    solver_adjoint=solver_adjoint,
)
```

---

## Laplace posterior covariance

```python
from vardax.posterior import laplace_covariance, gauss_newton_hessian

# Laplace at MAP — for ThreeDVar / StrongFourDVar / WeakFourDVar / FourDVarNet
P_star = laplace_covariance(
    x_star, cost_grad_fn=jax.grad(threedvar_cost),
    B_inv_op=B_inv, R_inv_op=R_inv,
)

# Gauss-Newton Hessian — reused inside IncrementalFourDVar.posterior(batch)
H_inv = gauss_newton_hessian(
    x_star, batch, forward, obs_op, B_op, R_op, n_krylov=50,
)
```

`OptimalInterpolation` and `IncrementalFourDVar` return posteriors
directly via `.posterior(batch)` — no `laplace_covariance` call needed.
