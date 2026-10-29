# 3DVar — 3D Variational

When the observation operator $H$ is nonlinear, the closed-form OI
analysis (chapter 4) no longer applies — the posterior is no longer
Gaussian. 3DVar gives up the closed form, keeps the Gaussian prior,
and computes the MAP estimate iteratively.

## Cost function

Minimise

$$J(x) = \frac{1}{2} \|x - x_b\|^2_{B^{-1}} + \frac{1}{2} \|y - H(x)\|^2_{R^{-1}}.$$

The gradient:

$$\nabla J(x) = B^{-1} (x - x_b) - H'(x)^\top R^{-1} (y - H(x)).$$

The Hessian (for Gauss-Newton):

$$J''(x) \approx B^{-1} + H'(x)^\top R^{-1} H'(x).$$

The Gauss-Newton approximation drops the second-derivative term in
$H''(x)^\top R^{-1} (y - H(x))$, which vanishes at the MAP and is
small near it. This makes the Hessian positive-definite and CG-friendly.

## The single time

"3D" refers to the three spatial dimensions of the state; $T = 0$ in
the 4D notation of chapter 1. No dynamics. One observation set, one
inversion.

This is the right regime for:

- Snapshot inversions (single satellite overpass, single sounding)
- Static fields (long-term mean, climatology)
- Multi-instrument fusion at a single time
- Methane source attribution from a single overpass (chapter 17)

For time series and dynamics, use `StrongFourDVar` (chapter 6).

## Implementation in vardax

The example below has a nonlinear operator $H(x) = \exp(x)$, applied cell by cell, on a grid of $N = 8$ cells at a single time.
The background is $x_b = 0$ with $B = I$, the truth is a draw around it, and the observations are $y = \exp(x_\text{true}) + \varepsilon$ with $R = \sigma^2 I$ and $\sigma = 0.1$.
`ThreeDVar` takes a batch of $S$ samples, each of shape $(T, N)$.

```python
import einx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import vardax as vdx

# Shapes: S = 1 sample; T = 1 time; N = 8 grid cells

key_truth, key_noise = jax.random.split(jax.random.PRNGKey(0))
state_structure = jax.ShapeDtypeStruct((1, 8), jnp.float32)

# B = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (T N, T N) op
# R = σ² I, σ = 0.1; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (T N, T N) op
    jnp.float32(0.01) * prior_cov, lx.positive_semidefinite_tag
)
# x_b = 0
background = jnp.zeros((1, 8))  # (T, N)
# x_true = x_b + 0.3 δ,  δ ~ 𝒩(0, I)
truth = background + 0.3 * jax.random.normal(key_truth, (1, 8))  # (T, N)
# y = exp(x_true) + σ ε,  ε ~ 𝒩(0, I)
obs = jnp.exp(truth) + 0.1 * jax.random.normal(key_noise, (1, 8))  # (T, N)

# today (vardax)
# H(x) = exp(x), elementwise
obs_op = vdx.CallableObs(fn=jnp.exp)
# today (vardax)
# J(x) = ½ ‖x − x_b‖²_{B⁻¹} + ½ ‖y − H(x)‖²_{R⁻¹}
model = vdx.ThreeDVar(
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.BFGS(rtol=1e-6, atol=1e-6),
)
# today (vardax)
batch = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", obs),  # (T, N) → (S, T, N)
    mask=jnp.ones((1, 1, 8)),
)
# x* = argminₓ J(x)
analysis, result = model.analyse(batch)  # → (S, T, N)

background_error = jnp.linalg.norm(background - truth)
analysis_error = jnp.linalg.norm(analysis[0] - truth)
print(f"background error {background_error:.2f}, analysis error {analysis_error:.2f}")
print(f"converged {bool(result.converged[0])} in {int(result.num_steps[0])} steps")
```

It prints `background error 0.76, analysis error 0.40` and `converged True in 63 steps`.

The minimiser is any `optimistix.AbstractMinimiser`, or a vardax `Minimiser` such as `OptaxMinimiser`; the choice depends on the cost function's geometry:

| Minimiser | When |
|---|---|
| `optx.BFGS(...)` | General-purpose; no special structure required (the default) |
| `optx.NonlinearCG(...)` | Memory-constrained; many parameters |
| `optx.NelderMead(...)` | Derivative-free; the cost has kinks |

`optx.GaussNewton` and `optx.LevenbergMarquardt` are least-squares solvers, not minimisers, so `ThreeDVar` rejects them; the Gauss-Newton fast path is `IncrementalFourDVar` (chapter 8).

The `minimiser_adjoint` controls how gradients flow through the
minimiser if the model is itself differentiated through (e.g., for
training hyperparameters of $B$). `ImplicitAdjoint` is the default —
it uses the implicit function theorem at the optimum, giving exact
gradients with $O(1)$ memory.

## Algorithm

Written out, the cost is two whitened quadratic forms, one per term of $J$, each applied with a CG solve rather than an inverse.
`ThreeDVar` builds the same cost, with the residual masked where observations are missing (the mask here is all ones), and hands it to `optimistix.minimise`.
The block continues the one above.

```python
from jaxtyping import Array, Float

# Shapes: T = 1 time; N = 8 grid cells

cov_solver = lx.CG(atol=1e-6, rtol=1e-6)


# user
def threedvar_cost(
    state: Float[Array, "t n"], obs: Float[Array, "t n"]
) -> Float[Array, ""]:
    # δx = x − x_b
    increment = state - background  # (T, N)
    # B⁻¹ δx
    prior_solve = lx.linear_solve(prior_cov, increment, solver=cov_solver)
    # ½ δxᵀ B⁻¹ δx
    background_term = 0.5 * einx.dot("t n, t n ->", increment, prior_solve.value)
    # d = y − H(x)
    innovation = obs - obs_op(state)  # (T, N)
    # R⁻¹ d
    obs_solve = lx.linear_solve(obs_cov, innovation, solver=cov_solver)
    # ½ dᵀ R⁻¹ d
    obs_term = 0.5 * einx.dot("t n, t n ->", innovation, obs_solve.value)
    # J(x) = ½ δxᵀ B⁻¹ δx + ½ dᵀ R⁻¹ d
    return background_term + obs_term  # → ()


# today (optimistix)
# x* = argminₓ J(x), started at x_b
solution = optx.minimise(
    threedvar_cost,
    optx.BFGS(rtol=1e-6, atol=1e-6),
    background,
    args=obs,
    adjoint=optx.ImplicitAdjoint(),
    max_steps=200,
)
# the hand-written cost reaches the same minimum as ThreeDVar
print(bool(jnp.allclose(solution.value, analysis[0], atol=1e-3)))
```

Convergence is typically tens of iterations for well-conditioned
problems. The optimistix machinery handles step size, line search, and
termination criteria internally.

## Linear-Gaussian agreement

When $H$ is linear, `ThreeDVar` reduces to `OptimalInterpolation` —
the iterative minimiser converges to the closed-form Kalman analysis.
With $H = I$ the same background and covariances give the BLUE in closed form; `tests/test_linear_gaussian_agreement.py` enforces the agreement for every method:

```python
# today (vardax)
linear_three = vdx.ThreeDVar(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# today (vardax)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
closed_form = optimal_interpolation(batch)  # → (S, T, N)
iterative = linear_three(batch)  # → (S, T, N)
print(bool(jnp.allclose(closed_form, iterative, atol=1e-3)))
```

This is part of the Decision D14 invariant — all seven methods agree
in the linear-Gaussian limit. If they don't, the implementation is
wrong.

## Posterior

For Gaussian likelihood near the MAP, the Laplace approximation gives

$$p(x \mid y) \approx \mathcal{N}(x; x^*, P^*), \text{ with } P^* = \left(B^{-1} + H'(x^*)^\top R^{-1} H'(x^*)\right)^{-1}.$$

`ThreeDVar.posterior` runs the analysis and applies `LaplaceCovariance` with the model's $B$ and $R$; the covariance stays a lazy operator:

```python
# today (vardax)
# P* = (B⁻¹ + H′(x*)ᵀ R⁻¹ H′(x*))⁻¹
posterior = model.posterior(batch)
# P* 1, one matrix-free product
spread = posterior.cov.mv(jnp.ones((1, 8)))  # (T, N) → (T, N)
print(posterior.mean.shape, spread.shape)
```

For strongly-nonlinear obs operators, the Laplace approximation
underestimates uncertainty. Use `EnsembleCovariance` (via `filterax`)
or `AmortizedPosterior` with a flow head for better calibration —
chapters 10, 13.

## When 3DVar is the right answer

Use `ThreeDVar` when:

- $H$ is nonlinear (averaging kernel, RTM, custom retrieval)
- Single timestep (no dynamics)
- Posterior assumed unimodal Gaussian-around-MAP
- $B$ and $R$ are Gaussian

If any of these fail:

- Linear $H$ → `OptimalInterpolation` (chapter 4) — same answer, free
  posterior
- Dynamics → `StrongFourDVar` (chapter 6)
- Multimodal posterior → ensemble or amortized methods (chapters 10, 13)

## When the minimiser fails to converge

If `optimistix.GaussNewton` fails (the residual stops decreasing,
hessian becomes indefinite, line search rejects every step), the
likely causes:

- Cost function is far from quadratic — switch to `LevenbergMarquardt`
  (damped Gauss-Newton) or `BFGS`
- Prior is too tight (small variance) but bias-misspecified — widen
  $B$ or check $x_b$ against innovation statistics
- Obs operator has discontinuities or near-discontinuities (saturated
  retrievals, mask edges) — smooth them in preprocessing or switch to
  derivative-free `optimistix.NelderMead`
- A linear solve inside the cost returns NaN ("A linear solver returned
  non-finite output") — the cost applies $B^{-1}$ and $R^{-1}$ with
  `lx.CG(atol=1e-6, rtol=1e-6)` by default, which breaks down on an
  ill-conditioned or tiny covariance (an $R$ of $10^{-14}$, say). Pass
  `cov_solver=lx.AutoLinearSolver(well_posed=True)` (or `lx.Cholesky()`)
  when the operators can be materialised, or rescale the problem so the
  covariances are $O(1)$. The 4D-Var models take the same `cov_solver=`.

vardax does not retry or fall back automatically — if your minimiser
fails, that's information. Diagnose, don't hide. `three.analyse(batch)`
returns the analysis together with a `MinimiseResult` per sample: the
cost and gradient norm at the minimum, the number of steps taken, and
`converged`, the minimiser's own verdict (optimistix's
`result == successful`). Read `converged` rather than comparing
`num_steps` with `max_steps`: a line search that gives up stops early
without converging, and a tolerance can be met on the last allowed step.

## See also

- Chapter 4 — `OptimalInterpolation` (linear-$H$ closed form)
- Chapter 6 — `StrongFourDVar` (add dynamics)
- Chapter 8 — `IncrementalFourDVar` (specialised GN+CG inner loop)
- Chapter 13 — posterior covariance (Laplace and beyond)

## References

- Lorenc, A. C. (1986). *Analysis methods for numerical weather
  prediction.* QJRMS 112(474).
- Talagrand, O. (1997). *Assimilation of observations, an
  introduction.* JMSJ 75(1B).
- Bannister, R. N. (2008). *A review of forecast error covariance
  statistics in atmospheric variational data assimilation.* QJRMS
  134(637).
