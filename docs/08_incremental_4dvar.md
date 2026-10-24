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

Write $x^{(k)}$ for the outer iterate, starting from $x^{(0)} = x_b$.
The background $x_b$ stays fixed across outer loops; only $x^{(k)}$
moves. Linearise around $x^{(k)}$:

$$
M_t(x^{(k)} + \delta x) \approx M_t(x^{(k)}) + M'_t \delta x,
$$

$$
H_t(z + \delta z) \approx H_t(z) + H'_t \delta z,
$$

where $M'_t = \partial M_t / \partial x|_{x^{(k)}}$ from
`jax.linearize` and $H'_t$ is the tangent of $H_t$ at
$M_t(x^{(k)})$. With the innovation
$d_t^{(k)} = y_t - H_t(M_t(x^{(k)}))$, the linearised (quadratic)
cost of an increment $\delta x$ is

$$
\delta J(\delta x) = \tfrac{1}{2} \|x^{(k)} + \delta x - x_b\|^2_{B^{-1}} +
                     \tfrac{1}{2} \sum_t \|d_t^{(k)} - H'_t M'_t \delta x\|^2_{R_t^{-1}}.
$$

The background term measures the *total* departure
$x^{(k)} + \delta x - x_b$, not $\delta x$ alone. The Gauss-Newton
Hessian is

$$
\mathbf{H}_\text{GN} = B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t),
$$

and setting $\nabla \delta J = 0$ gives the inner system that the CG /
Lanczos solver solves for $\delta x^*$:

$$
\mathbf{H}_\text{GN}\, \delta x^* =
  -B^{-1}\big(x^{(k)} - x_b\big)
  + \sum_t (H'_t M'_t)^\top R_t^{-1} d_t^{(k)}
  = -\nabla J\big(x^{(k)}\big).
$$

The right-hand side is the negative gradient of the full nonlinear cost
at the current iterate. On the first outer loop $x^{(0)} = x_b$, so the
background term vanishes and only the innovation term is left. On later
loops it does not vanish: dropping it would pull every increment toward
$x^{(k)}$ instead of $x_b$, and the iteration would no longer converge
to the 4DVar minimum.

Outer update: $x^{(k+1)} = x^{(k)} + \delta x^*$. Relinearise.

This pattern converges in a handful of outer iterations (typically
3–5) regardless of nonlinearity strength, because each outer step
sees a fresh tangent linearisation.

## Control-variable transform (CVT)

!!! note
    `IncrementalFourDVar` does not implement the CVT yet: its inner CG
    solve runs in state space with no preconditioner. The section below
    describes the method. To minimise over a transformed control today,
    use `ReducedBasisFourDVar` with `basis=` (a reduced basis) or
    `parameterization=` (any [`StateParameterization`](api/protocols.md)).

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
    config=IncrementalConfig(n_outer=3, n_inner=20),
)
x_star = model(batch)

# Posterior is free — reuses the GN Hessian from the last outer iteration
posterior = model.posterior(batch)
```

`IncrementalConfig`:

- `n_outer` — number of Gauss-Newton outer iterations (typical: 3)
- `n_inner` — max CG iterations per outer (typical: 20–50)
- `cg_atol`, `cg_rtol` — CG convergence tolerances

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

x ← x_b                       # x⁽⁰⁾ = x_b; x_b itself never changes
for k in range(config.n_outer):

    # 1. Tangent of the observed trajectory G(x) = [H_t(M_t(x))]_t at x⁽ᵏ⁾
    #    G'δx = [H'_t M'_t δx]_t   (jax.linearize; Gᵀ via jax.vjp)
    G_x, G_lin = jax.linearize(observed_trajectory, x)

    # 2. Innovations at the current iterate
    #    d_tᵏ = y_t − H_t(M_t(x⁽ᵏ⁾))
    d = batch.y - G_x

    # 3. Right-hand side: the negative gradient of the full cost at x⁽ᵏ⁾
    #    −∇J(x⁽ᵏ⁾) = −B⁻¹(x⁽ᵏ⁾ − x_b) + G'ᵀ R⁻¹ d
    rhs = -B_inv @ (x - x_b) + G_lin.T @ (R_inv @ d)

    # 4. Inner solve via lineax.CG
    if use_cvt:                  # χ-space shown; not implemented yet
        B_half = B_op.half()
        # (I + B^{1/2}ᵀ G'ᵀ R⁻¹ G' B^{1/2}) δχ = B^{1/2}ᵀ (−∇J(x⁽ᵏ⁾))
        preconditioned_hessian = identity_op + B_half.T @ GtRG @ B_half
        chi = lineax.linear_solve(
            preconditioned_hessian, B_half.T @ rhs,
            solver=lineax.CG(atol=config.cg_atol, rtol=config.cg_rtol,
                              max_steps=config.n_inner),
        )
        # δx = B^{1/2} δχ
        dx = B_half @ chi
    else:
        # (B⁻¹ + G'ᵀ R⁻¹ G') δx = −∇J(x⁽ᵏ⁾)
        dx = lineax.linear_solve(B_inv + GtRG, rhs, solver=lineax.CG(...))

    # 5. Outer update: x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + δx*
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
