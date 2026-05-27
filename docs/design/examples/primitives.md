---
status: draft
version: 0.3.0
---

# Layer 0 — Primitive Examples

Pure JAX cost functions, solver steps, control-variable transform, posterior
primitives.

---

## Variational cost

```python
from vardax.costs import obs_cost, prior_cost, variational_cost

# Observation cost: masked MSE
j_obs = obs_cost(x, batch.input, batch.mask, obs_operator=None)

# Prior cost: autoencoder reconstruction
j_prior = prior_cost(x, prior_fn=ae_model)

# Total: weighted sum
j_total = variational_cost(x, batch, prior_fn=ae_model, obs_operator=None,
                            alpha_obs=1.0, alpha_prior=0.1)

# Gradient — input to the inner solver
grad_J = jax.grad(variational_cost)(x, batch, ae_model, None, 1.0, 0.1)
```

---

## Solver step (one inner iteration)

```python
from vardax.solver import solver_step

# x_{k+1} = x_k - Φ(∇J, carry)
x_new, carry_new = solver_step(x, grad_J, grad_modulator_fn, carry)
```

---

## Three gradient modes for `VarDANet*`

```python
from vardax.solver import unrolled_solve, one_step_solve, implicit_solve

# Unrolled — O(K) memory, backprop through all K steps
x_star, final_carry = unrolled_solve(x0, cost_fn, grad_mod_fn, n_steps=15, carry0=carry)

# One-step — O(1) memory, Bolte et al. (2023)
x_star, final_carry = one_step_solve(x0, cost_fn, grad_mod_fn, n_steps=15, carry0=carry)

# Implicit — O(1) memory, optimistix.FixedPointIteration
x_star, final_carry = implicit_solve(x0, cost_fn, grad_mod_fn, n_steps=15, carry0=carry)
```

---

## Incremental 4DVar inner loop (Decision D11)

```python
import jax
import lineax as lx
from vardax.costs import incremental_cost
from vardax.solver import gauss_newton_inner, incremental_outer

# At outer iterate x_b, linearise forward + obs operator
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
# ... repeat for n_outer iterations
```

Or use the high-level helper:

```python
x_star = incremental_outer(x0, batch, forward, obs_op, B_op, R_op, config)
```

---

## Control-variable transform

```python
import gaussx as gx
from vardax.cvt import cvt_transform, cvt_inverse

# Build Matérn B^{1/2} via gaussx
B_half = gx.MaternLinearOperator(
    grid_coords=coords,
    length_scale=10.0,    # km — basin-dependent
    nu=1.5,                # Matérn-3/2
    sigma=1.0,
).half()                   # gives B^{1/2}

# Forward CVT: χ = B^{-1/2}(x - x_b)
chi = cvt_transform(x, x_b, B_half)

# Inverse: x = x_b + B^{1/2}·χ
x = cvt_inverse(chi, x_b, B_half)
```

In CVT coordinates the prior cost is $\|\chi\|^2$ (identity Gaussian),
which preconditions CG.

---

## Laplace posterior covariance

```python
from vardax.posterior import laplace_covariance

# At MAP x_star, return P* as an AbstractLinearOperator
P_star = laplace_covariance(x_star, cost_grad_fn=jax.grad(variational_cost),
                             B_inv_op=B_inv, R_inv_op=R_inv)

# Mat-vec via lineax.CG — no full materialisation
diag = jnp.diag(P_star.as_matrix())  # if needed
```

For incremental 4DVar, reuse the Hessian from the last GN outer iteration:

```python
from vardax.posterior import gauss_newton_hessian

H_inv = gauss_newton_hessian(x_star, batch, forward, obs_op,
                              B_op, R_op, n_krylov=50)
```
