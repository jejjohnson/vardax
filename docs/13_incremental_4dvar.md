# Incremental 4DVar with Control-Variable Transform

Operational 4DVar — used by ECMWF, NCEP, JMA, UKMO — solves the
weak-constraint variational problem by Gauss-Newton outer iterations on the
full nonlinear cost, with CG / Lanczos inner iterations on a linearised
quadratic subproblem. The **control-variable transform** preconditions the
inner solve. vardax implements this in `IncrementalVarDA*` (Decision D11).

## Full nonlinear cost

$$J(\mathbf{x}_0) = \frac{1}{2}\|\mathbf{x}_0 - \mathbf{x}_b\|^2_{\mathbf{B}^{-1}} + \frac{1}{2}\sum_{t=1}^{T} \|\mathbf{y}_t - H_t(M_t(\mathbf{x}_0))\|^2_{\mathbf{R}_t^{-1}}$$

with:

| Symbol | Description |
|---|---|
| $\mathbf{x}_0$ | Initial-time state (control variable) |
| $\mathbf{x}_b$ | Background / prior mean |
| $\mathbf{B}$ | Background error covariance (often Matérn) |
| $\mathbf{R}_t$ | Observation error covariance at time $t$ |
| $M_t$ | Nonlinear forward model from $t_0$ to $t_t$ |
| $H_t$ | Nonlinear observation operator at $t_t$ |
| $\mathbf{y}_t$ | Observations at $t_t$ |

The forward $M_t$ is supplied by `somax` (geophysics) or `plumax`
(atmospheric transport / methane) — vardax does not own forward models.

## Incremental linearisation

At outer iterate $\mathbf{x}_b^{(k)}$, linearise $M_t$ and $H_t$ via
`jax.linearize`:

$$M_t(\mathbf{x}_b + \delta\mathbf{x}) \approx M_t(\mathbf{x}_b) + \mathbf{M}'_t \delta\mathbf{x}$$
$$H_t(\mathbf{z} + \delta\mathbf{z}) \approx H_t(\mathbf{z}) + \mathbf{H}'_t \delta\mathbf{z}$$

The linearised (quadratic) increment cost:

$$\delta J(\delta\mathbf{x}) = \frac{1}{2}\|\delta\mathbf{x}\|^2_{\mathbf{B}^{-1}} + \frac{1}{2}\sum_t \|\mathbf{d}_t - \mathbf{H}'_t \mathbf{M}'_t \delta\mathbf{x}\|^2_{\mathbf{R}_t^{-1}}$$

with innovation $\mathbf{d}_t = \mathbf{y}_t - H_t(M_t(\mathbf{x}_b))$. The
Gauss-Newton Hessian:

$$\mathbf{J}''_\text{GN} = \mathbf{B}^{-1} + \sum_t (\mathbf{H}'_t \mathbf{M}'_t)^\top \mathbf{R}_t^{-1} (\mathbf{H}'_t \mathbf{M}'_t)$$

The inner subproblem $\delta\mathbf{x}^* = \arg\min \delta J$ has the
normal equation

$$\mathbf{J}''_\text{GN} \delta\mathbf{x}^* = \sum_t (\mathbf{H}'_t \mathbf{M}'_t)^\top \mathbf{R}_t^{-1} \mathbf{d}_t$$

solved by `lineax.CG` (Krylov / Lanczos). Outer update:
$\mathbf{x}_b^{(k+1)} = \mathbf{x}_b^{(k)} + \delta\mathbf{x}^*$.

## Control-Variable Transform (CVT)

For Matérn-correlated background error, the inner Hessian
$\mathbf{J}''_\text{GN}$ is ill-conditioned (eigenvalues span the
correlation-length range). The CVT preconditions:

$$\boldsymbol{\chi} = \mathbf{B}^{-1/2}(\delta\mathbf{x})$$

In CVT coordinates the prior cost becomes $\|\boldsymbol{\chi}\|^2$
(identity Gaussian). The transformed Hessian:

$$\tilde{\mathbf{J}}''_\text{GN} = \mathbf{I} + \mathbf{B}^{1/2} \mathbf{J}''_\text{obs} \mathbf{B}^{1/2}$$

has spectrum bunched around 1 — CG converges in ~tens of iterations
regardless of grid resolution.

The factor $\mathbf{B}^{1/2}$ comes from `gaussx.MaternLinearOperator.half()`:

```python
import gaussx as gx
from vardax.cvt import cvt_transform, cvt_inverse

B_half = gx.MaternLinearOperator(
    grid_coords=coords, length_scale=10.0, nu=1.5, sigma=1.0,
).half()

# Forward CVT
chi = cvt_transform(x, x_b, B_half)        # χ = B^{-1/2}(x - x_b)
# Inverse
x = cvt_inverse(chi, x_b, B_half)          # x = x_b + B^{1/2} χ
```

## Algorithm

```
Inputs:
   x_b      — background / prior mean
   B_op     — gaussx prior covariance (Matérn by default)
   R_op     — obs error covariance (per-instrument block-diag possible)
   batch    — Batch* with multi-time observations
   forward  — pipekit_cycle.ForwardModel (from somax / plumax)
   obs_op   — pipekit_cycle.ObservationOperator (e.g. AveragingKernel)
   config   — IncrementalConfig(n_outer, n_inner, cg_atol, cg_rtol, cvt=True)

x ← x_b
for k in range(config.n_outer):

    # 1. Linearise forward + obs operator at current outer iterate
    M_lin = jax.linearize(forward.step, x, ...)
    H_lin = obs_op.linearize(forward_trajectory(x, batch))

    # 2. Innovations
    d = obs - H_lin(forward_trajectory(x, batch))

    # 3. Hessian as a LinearOperator (not materialised)
    J_pp = B_inv_op + (H_lin @ M_lin).T @ R_inv_op @ (H_lin @ M_lin)
    rhs = (H_lin @ M_lin).T @ R_inv_op @ d

    # 4. Inner CG solve (in χ-space if CVT enabled)
    if config.cvt:
        # Preconditioned: solve (I + B^{1/2} J''_obs B^{1/2}) χ = B^{1/2} rhs
        chi = lineax.linear_solve(
            B_half @ J_obs_pp @ B_half + identity, B_half @ rhs,
            solver=lineax.CG(atol=cg_atol, rtol=cg_rtol, max_steps=n_inner),
        )
        dx = B_half @ chi
    else:
        # Identity preconditioner
        dx = lineax.linear_solve(J_pp, rhs, solver=lineax.CG(...))

    # 5. Outer update
    x = x + dx

return x
```

## Implementation

```python
import gaussx as gx
import lineax as lx
from vardax.models import IncrementalVarDA2D
from vardax import IncrementalConfig

model = IncrementalVarDA2D(
    forward=somax_model,
    obs_op=AveragingKernel(...),
    prior_mean=x_b,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=10.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(obs_uncertainty),
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)

x_star = model(batch)
```

## Strong-constraint vs weak-constraint

The formulation above is **strong-constraint** — the forward model is
treated as exact, the only control variable is the initial condition
$\mathbf{x}_0$. For **weak-constraint** 4DVar (model error allowed at each
timestep), augment the control vector with per-step model error
$\boldsymbol{\eta}_t$:

$$J(\mathbf{x}_0, \{\boldsymbol{\eta}_t\}) = J_\text{bg} + \sum_t \|\boldsymbol{\eta}_t\|^2_{\mathbf{Q}^{-1}} + J_\text{obs}$$

Reserved for follow-up; current `IncrementalVarDA*` is strong-constraint.

## See also

- Chapter 12: Observation operators — including `linearize()` contract
- Chapter 14: Posterior covariance — GN Hessian is reusable for UQ
- Decision D11 in design docs — CVT as operational path
- `vardax.cvt` module reference
