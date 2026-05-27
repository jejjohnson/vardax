# Strong-constraint 4DVar

When observations are spread across an assimilation window and the
state evolves through known dynamics, 3DVar is no longer enough — we
need to relate observations at later times to the initial state via
the forward model. Strong-constraint 4DVar treats the dynamics as
exact and makes the initial state $x_0$ the control variable.

## Cost function

Given observations $y_t$ at times $t = 0, 1, \ldots, T$, dynamics $M_t$
propagating $x_0$ to time $t$, and obs operator $H_t$ at each time:

$$
J(x_0) =
  \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}} +
  \tfrac{1}{2} \sum_{t=0}^{T} \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}.
$$

**Strong-constraint** means the model is treated as exact: any $x_0$
determines the trajectory $x_t = M_t(x_0)$ exactly. Model error is not
admitted. (If you need model error, use `WeakFourDVar` — chapter 7.)

The gradient combines the background term with the adjoint of every
forward step:

$$
\nabla J(x_0) = B^{-1}(x_0 - x_b) +
                \sum_{t=0}^{T} (M'_t)^\top H'_t{}^\top R_t^{-1} (H_t(M_t(x_0)) - y_t).
$$

In hand-coded operational 4DVar, $(M'_t)^\top$ is the adjoint model —
historically a major engineering effort (Talagrand and Courtier 1987,
Errico 1997). With `diffrax`, it's a constructor argument.

## Implementation in vardax

```python
import diffrax as dfx
import optimistix as optx
from vardax.models import StrongFourDVar

model = StrongFourDVar(
    forward=somax_model,                              # pipekit_cycle.ForwardModel
    obs_op=MaskedIdentity(),                          # or AveragingKernel, etc.
    prior_mean=x_b,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)
x_0_star = model(batch)
```

Two adjoint slots, two purposes:

- **`forward_adjoint`** — controls how gradients flow back through the
  dynamics rollout (chapter 3). For long windows, switch to
  `dfx.BacksolveAdjoint()` for constant memory.
- **`minimiser_adjoint`** — controls how gradients flow back through
  the outer minimiser if the model itself is differentiated through
  (e.g., for training hyperparameters of $B$). `ImplicitAdjoint` is
  exact at the optimum with $O(1)$ memory.

## The adjoint problem

The gradient computation reads "integrate the trajectory forward,
then propagate the observation residuals backward through the adjoint
of the dynamics." In equations, this is the backward sweep

$$
\lambda_T = H'_T{}^\top R_T^{-1} (H_T(x_T) - y_T)
$$

$$
\lambda_{t-1} = (M'_t)^\top \lambda_t + H'_{t-1}{}^\top R_{t-1}^{-1} (H_{t-1}(x_{t-1}) - y_{t-1})
$$

$$
\nabla_{x_0} J = B^{-1}(x_0 - x_b) + \lambda_0.
$$

This is exactly what `diffrax.RecursiveCheckpointAdjoint` or
`BacksolveAdjoint` compute under the hood. Vardax just composes them
into the cost-gradient call.

## Algorithm

```
Inputs:
   x_b, B_op, R_op           — prior and observation-error covariances
   forward                   — pipekit_cycle.ForwardModel (M_t)
   obs_op                    — pipekit_cycle.ObservationOperator (H_t)
   batch                     — observations y_t for t = 0, …, T
   minimiser                 — optimistix.AbstractMinimiser
   minimiser_adjoint         — optimistix.AbstractAdjoint
   forward_adjoint           — diffrax.AbstractAdjoint

def cost_fn(x_0, batch):
    # Forward rollout via diffrax with forward_adjoint
    trajectory = rollout(x_0, forward, n_steps=T, adjoint=forward_adjoint)

    # Observation cost summed over time
    j_obs = 0
    for t in range(T + 1):
        residual = batch.y[t] - obs_op(trajectory[t])
        j_obs = j_obs + 0.5 * residual @ R_inv_op @ residual

    # Background term
    j_b = 0.5 * (x_0 - x_b) @ B_inv_op @ (x_0 - x_b)
    return j_b + j_obs

# Iterate
result = optimistix.minimise(
    fn=cost_fn, solver=minimiser, y0=x_b, args=batch,
    adjoint=minimiser_adjoint,
)
x_0_star = result.value
```

## Linear-Gaussian agreement

When $H_t$ and $M_t$ are linear and $T = 0$, `StrongFourDVar` reduces
to `OptimalInterpolation`. The conformance test enforces this:

```python
def test_strong_4dvar_recovers_oi():
    oi = OptimalInterpolation(linear_H, x_b, B_op, R_op)
    strong = StrongFourDVar(
        forward=static_forward,                # M_t = I for T = 0
        obs_op=linear_H,
        prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
        minimiser=optx.NonlinearCG(rtol=1e-8),
    )
    batch = make_linear_gaussian_batch(T=0)
    assert jnp.allclose(oi(batch), strong(batch), atol=1e-3)
```

Part of the Decision D14 invariant.

## When the window is long

For $T = 100$+ assimilation cycles, the forward trajectory dominates
memory. Three strategies:

1. **`dfx.BacksolveAdjoint()`** — solve the adjoint ODE backwards in
   time. $O(1)$ memory in the trajectory length. Caveat: reverse-time
   stiffness can require an implicit integrator (`dfx.Kvaerno5`,
   `dfx.ImplicitEuler`).
2. **`dfx.RecursiveCheckpointAdjoint(checkpoints=N)`** — store $N$
   checkpoints, recompute segments between them. $O(\sqrt{T})$ memory.
3. **Shorter windows.** Run multiple `StrongFourDVar` analyses with
   shorter windows and cycle them via `pipekit_cycle.DACycle`. This is
   what operational systems do.

For most research-scale problems with $T \le 20$, the default
`RecursiveCheckpointAdjoint` is fine.

## When the dynamics are nonlinear

The 4DVar cost $J$ is nonconvex when $M_t$ is nonlinear — multiple
local minima exist (and matter, especially for chaotic systems like
Lorenz). Strategies:

- **Multi-start.** Run `optimistix.NonlinearCG` from several
  perturbed initial guesses, pick the lowest cost. vardax provides
  `vardax.utils.multi_start(model, batch, n_starts=10, key)` as a
  helper.
- **Quasi-static gradient.** Use a warm-start from a previous cycle
  (the usual operational pattern).
- **Switch to `IncrementalFourDVar`** (chapter 8) — the outer
  Gauss-Newton iterations relinearise around the current iterate,
  which is more robust than NonlinearCG on the full nonlinear cost.

## Posterior

Pair the analysis with a `PosteriorAdapter`:

```python
from vardax.posterior import LaplaceCovariance

posterior = LaplaceCovariance()(x_0_star, model.as_analysis_step(), batch)
```

The Laplace covariance involves $(B^{-1} + \sum_t (M'_t H'_t)^\top R_t^{-1} (M'_t H'_t))^{-1}$,
computed by Krylov / Lanczos via `lineax.CG`. The mat-vec includes a
tangent-linear-plus-adjoint pass through the trajectory, so it's
expensive per query — `n_krylov = 50` typical mat-vec count gives
posterior diagonal at the cost of 50 trajectory tangent + adjoint
passes.

For operational use, `IncrementalFourDVar` is preferred — it produces
the Gauss-Newton Hessian as a side effect of the last outer iteration,
so posterior is essentially free.

## When `StrongFourDVar` is the right answer

Use `StrongFourDVar` when:

- Multi-time observations with dynamics
- Model error is small (within the noise floor of the observations)
- General-purpose 4DVar — when you want to try `optimistix.BFGS` or
  `NonlinearCG` directly on the full nonlinear cost without the
  incremental machinery

For production / operational 4DVar with structured priors and CVT
preconditioning, use `IncrementalFourDVar` (chapter 8) instead — it's
the same problem, solved faster.

If model error matters, use `WeakFourDVar` (chapter 7).

## See also

- Chapter 3 — dynamical model and `forward_adjoint` options
- Chapter 7 — `WeakFourDVar` (admits model error)
- Chapter 8 — `IncrementalFourDVar` (operational fast path)
- Chapter 12 — adjoint composition

## References

- Talagrand, O., & Courtier, P. (1987). *Variational assimilation of
  meteorological observations with the adjoint vorticity equation.*
  QJRMS 113(478).
- Le Dimet, F.-X., & Talagrand, O. (1986). *Variational algorithms for
  analysis and assimilation of meteorological observations.* Tellus A
  38(2).
- Errico, R. M. (1997). *What is an adjoint model?* BAMS 78(11).
