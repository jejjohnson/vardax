# Incremental 4DVar with Control-Variable Transform

Operational 4DVar at ECMWF, NCEP, JMA, and UKMO all follow a common
template: Gauss-Newton outer iterations on the full nonlinear cost,
CG (or Lanczos) inner iterations on a linearised quadratic
subproblem, with a control-variable transform that preconditions the
inner solve. `IncrementalFourDVar` packages this pattern.

It is functionally equivalent to `StrongFourDVar` (chapter 6) — same
problem, same answer — but the specialised inner solver is dramatically
faster when the prior is structured (Matérn, Kronecker) and the
window is long.

## The outer / inner decomposition

Strong-constraint 4DVar minimises

$$
J(x_0) = \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}} +
         \tfrac{1}{2} \sum_t \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}.
$$

Linearise around the current outer iterate $x_b^{(k)}$:

$$
M_t(x_b^{(k)} + \delta x) \approx M_t(x_b^{(k)}) + M'_t \delta x,
$$

$$
H_t(z + \delta z) \approx H_t(z) + H'_t \delta z,
$$

where $M'_t = \partial M_t / \partial x|_{x_b^{(k)}}$ from
`jax.linearize` and similarly $H'_t$ from `obs_op.linearize(...)`. The
linearised (quadratic) increment cost is

$$
\delta J(\delta x) = \tfrac{1}{2} \|\delta x\|^2_{B^{-1}} +
                     \tfrac{1}{2} \sum_t \|d_t - H'_t M'_t \delta x\|^2_{R_t^{-1}},
$$

with innovation $d_t = y_t - H_t(M_t(x_b^{(k)}))$. The Gauss-Newton
Hessian:

$$
\mathbf{H}_\text{GN} = B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t).
$$

The inner CG / Lanczos solver finds $\delta x^*$ such that

$$
\mathbf{H}_\text{GN}\, \delta x^* = \sum_t (H'_t M'_t)^\top R_t^{-1} d_t.
$$

Outer update: $x_b^{(k+1)} = x_b^{(k)} + \delta x^*$. Relinearise.

This pattern converges in a handful of outer iterations (typically
3–5) regardless of nonlinearity strength, because each outer step
sees a fresh tangent linearisation.

## Control-variable transform (CVT)

The Hessian $\mathbf{H}_\text{GN}$ has condition number set by the
prior covariance eigenvalues spread over the correlation length scales
of $B$. For a Matérn $B$ on a 2D grid, this can be hundreds of orders
of magnitude — CG converges slowly without preconditioning.

CVT changes variables:

$$
\chi = B^{-1/2}(\delta x),
$$

so $\delta x = B^{1/2} \chi$. In $\chi$-space the prior cost is

$$
\tfrac{1}{2} \|\delta x\|^2_{B^{-1}} = \tfrac{1}{2} \|\chi\|^2,
$$

and the Hessian becomes

$$
\tilde{\mathbf{H}}_\text{GN} = I + B^{1/2} \Big(\sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t)\Big) B^{1/2}.
$$

The spectrum of $\tilde{\mathbf{H}}_\text{GN}$ is bunched around 1 — CG
converges in tens of iterations regardless of grid resolution.

$B^{1/2}$ comes from `gaussx.MaternLinearOperator.half()`:

```python
import gaussx as gx

B_half = gx.MaternLinearOperator(
    grid_coords=coords, length_scale=10.0, nu=1.5, sigma=1.0,
).half()
```

For non-Matérn structured priors, factorise via Cholesky (small grids)
or Lanczos (large grids).

## Implementation in vardax

```python
import gaussx as gx
import lineax as lx
from vardax.models import IncrementalFourDVar
from vardax import IncrementalConfig

model = IncrementalFourDVar(
    forward=somax_model,
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),
    prior_mean=x_b,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=10.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(obs_uncertainty),
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)
x_star = model(batch)

# Posterior is free — reuses the GN Hessian from the last outer iteration
posterior = model.posterior(batch)
```

`IncrementalConfig`:

- `n_outer` — number of Gauss-Newton outer iterations (typical: 3)
- `n_inner` — max CG iterations per outer (typical: 20–50)
- `cg_atol`, `cg_rtol` — CG convergence tolerances
- `cvt` — apply control-variable transform (default `True`)

## Algorithm

```
Inputs:
   x_b           — background mean
   B_op          — prior covariance (gaussx; supports .half())
   R_op          — observation-error covariance
   forward       — pipekit_cycle.ForwardModel
   obs_op        — pipekit_cycle.ObservationOperator
   batch         — observations y_t for t = 0, …, T
   config        — IncrementalConfig

x ← x_b
for k in range(config.n_outer):

    # 1. Linearise M_t and H_t at the current outer iterate
    M_lin = jax.linearize(lambda s: forward.step(s, dt), x)
    H_lin = obs_op.linearize(forward_trajectory(x, batch))

    # 2. Compute innovations
    d = batch.y - H_lin(forward_trajectory(x, batch))

    # 3. Inner solve via lineax.CG (in χ-space if config.cvt)
    if config.cvt:
        B_half = B_op.half()
        # Solve  (I + B^{1/2} (H'M')ᵀ R⁻¹ (H'M') B^{1/2}) χ = B^{1/2} (H'M')ᵀ R⁻¹ d
        preconditioned_hessian = identity_op + B_half @ sum_t_hmrhm @ B_half
        rhs = B_half @ adj_innovation
        chi = lineax.linear_solve(
            preconditioned_hessian, rhs,
            solver=lineax.CG(atol=config.cg_atol, rtol=config.cg_rtol,
                              max_steps=config.n_inner),
        )
        dx = B_half @ chi
    else:
        # Identity preconditioner — slower but works without B^{1/2}
        dx = lineax.linear_solve(...)

    # 4. Outer update
    x = x + dx

return x
```

## Why it's the operational fast path

Three multiplicative speedups over `StrongFourDVar` with `NonlinearCG`:

1. **CVT preconditioning** — CG converges in $O(10)$ iterations
   instead of $O(\sqrt{\kappa(B)})$ where $\kappa(B)$ is the prior
   condition number.
2. **Linearisation reuse** — the tangent linear $M'$ is computed once
   per outer iteration and reused across all $n_\text{inner}$ inner CG
   steps. `NonlinearCG` on the full nonlinear cost recomputes the
   gradient (which requires a full forward + adjoint pass) every step.
3. **Free posterior** — the GN Hessian assembled in the last outer
   iteration is reused for `LaplaceCovariance` / `GaussNewtonHessian`.
   No separate Krylov pass needed for posterior diagonal.

For a typical operational problem ($T = 12$ hours, $n = 10^7$ grid
points, $m = 10^5$ observations), `IncrementalFourDVar` converges in
minutes; `StrongFourDVar` with `NonlinearCG` takes hours.

## Linear-Gaussian agreement

For $T = 0$ and linear $H$, `IncrementalFourDVar` reduces to
`OptimalInterpolation`. The outer iteration converges in one step (no
relinearisation needed), and the inner CG produces the closed-form
Kalman update:

```python
def test_incremental_recovers_oi():
    oi = OptimalInterpolation(linear_H, x_b, B_op, R_op)
    inc = IncrementalFourDVar(
        forward=static_forward, obs_op=linear_H,
        prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
        config=IncrementalConfig(n_outer=2, n_inner=50, cvt=True),
    )
    batch = make_linear_gaussian_batch(T=0)
    assert jnp.allclose(oi(batch), inc(batch), atol=1e-3)
```

Part of the Decision D14 invariant.

## Strong- vs weak-constraint

`IncrementalFourDVar` is strong-constraint by default. For
weak-constraint incremental 4DVar (the augmented control vector with
$\eta_t$), use `WeakFourDVar` directly — the incremental machinery for
the augmented problem is heavier and not always worth the complexity.
The strong / weak choice is structural; the inner-solver choice
(general vs incremental) is performance.

## Posterior covariance — for free

The last outer iteration's GN Hessian is an `AbstractLinearOperator`
that can be inverted via Krylov for posterior diagonal or sampling:

```python
posterior = model.posterior(batch)
# Posterior(mean=x_star, cov=GN_Hessian_inv_op, samples=None, provenance={...})
```

Compare `StrongFourDVar`, where computing the posterior requires a
fresh Krylov pass over the same Hessian. For `IncrementalFourDVar` the
Hessian is already there.

## When to switch from `IncrementalFourDVar` to `StrongFourDVar`

Almost never, for operational work. `IncrementalFourDVar` is faster
and produces the same answer. Use `StrongFourDVar` only when:

- The prior is exotic (not Matérn / Kronecker / structured) and
  `gaussx.MaternLinearOperator.half()` doesn't apply. The CVT
  preconditioner is then unavailable.
- You want to swap in a custom optimistix minimiser (e.g.
  `optx.LevenbergMarquardt` for very nonlinear cases). Incremental
  uses a fixed GN+CG; the rest of the optimistix zoo isn't compatible
  with the incremental machinery.

For most users, `IncrementalFourDVar` is the right default.

## See also

- Chapter 6 — `StrongFourDVar` (equivalent problem, slower solver)
- Chapter 12 — adjoint composition (CVT relies on `gaussx` factorisation)
- Chapter 13 — posterior covariance (reuses GN Hessian)
- Design doc: [`design/decisions.md#d11`](design/decisions.md#d11-incrementalfourdvar-as-the-operational-fast-path-of-strongfourdvar)

## References

- Courtier, P., Thépaut, J.-N., & Hollingsworth, A. (1994). *A strategy
  for operational implementation of 4D-Var, using an incremental
  approach.* QJRMS 120(519).
- Lorenc, A. C. (1997). *Development of an operational variational
  assimilation scheme.* JMSJ 75(1B).
- Bannister, R. N. (2017). *A review of operational methods of
  variational and ensemble-variational data assimilation.* QJRMS
  143(703).
