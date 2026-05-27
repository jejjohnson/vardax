# 3DVar — 3D Variational

When the observation operator $H$ is nonlinear, the closed-form OI
analysis (chapter 4) no longer applies — the posterior is no longer
Gaussian. 3DVar gives up the closed form, keeps the Gaussian prior,
and computes the MAP estimate iteratively.

## Cost function

Minimise

$$
J(x) = \tfrac{1}{2} \|x - x_b\|^2_{B^{-1}}
     + \tfrac{1}{2} \|y - H(x)\|^2_{R^{-1}}.
$$

The gradient:

$$
\nabla J(x) = B^{-1} (x - x_b) - H'(x)^\top R^{-1} (y - H(x)).
$$

The Hessian (for Gauss-Newton):

$$
J''(x) \approx B^{-1} + H'(x)^\top R^{-1} H'(x).
$$

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

```python
import optimistix as optx
from vardax.models import ThreeDVar

model = ThreeDVar(
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),   # nonlinear via h ⊙ x
    prior_mean=x_b,
    prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
)
x_star = model(batch)
```

The minimiser is any `optimistix.AbstractMinimiser` — the choice
depends on the cost function's geometry:

| Minimiser | When |
|---|---|
| `optx.GaussNewton(...)` | Cost is approximately quadratic; obs operator's Jacobian is well-behaved |
| `optx.BFGS(...)` | General-purpose; no special structure required |
| `optx.NonlinearCG(...)` | Memory-constrained; many parameters |
| `optx.LevenbergMarquardt(...)` | Highly nonlinear obs operator; needs damping |

The `minimiser_adjoint` controls how gradients flow through the
minimiser if the model is itself differentiated through (e.g., for
training hyperparameters of $B$). `ImplicitAdjoint` is the default —
it uses the implicit function theorem at the optimum, giving exact
gradients with $O(1)$ memory.

## Algorithm

```
Inputs:
   x_b      — background mean
   B_op     — prior covariance
   R_op     — observation-error covariance
   obs_op   — pipekit_cycle.ObservationOperator (linearize() required)
   y        — observations
   minimiser — optimistix.AbstractMinimiser

# Define the cost (closure over batch)
def cost_fn(x, batch):
    return threedvar_cost(x, batch, obs_op, x_b, B_inv_op, R_inv_op)

# Iterate
result = optimistix.minimise(
    fn=cost_fn,
    solver=minimiser,
    y0=x_b,                                   # warm start at background
    args=batch,
    adjoint=minimiser_adjoint,
)
return result.value                           # MAP estimate
```

Convergence is typically tens of iterations for well-conditioned
problems. The optimistix machinery handles step size, line search, and
termination criteria internally.

## Linear-Gaussian agreement

When $H$ is linear, `ThreeDVar` reduces to `OptimalInterpolation` —
the iterative minimiser converges to the closed-form Kalman analysis.
The conformance test enforces agreement:

```python
def test_threedvar_recovers_oi_in_linear_limit():
    oi = OptimalInterpolation(linear_H_op, x_b, B_op, R_op)
    three = ThreeDVar(
        obs_op=linear_H_op, prior_mean=x_b,
        prior_cov_op=B_op, obs_cov_op=R_op,
        minimiser=optx.GaussNewton(rtol=1e-8, atol=1e-8),
    )
    batch = make_linear_gaussian_batch()
    assert jnp.allclose(oi(batch), three(batch), atol=1e-3)
```

This is part of the Decision D14 invariant — all seven methods agree
in the linear-Gaussian limit. If they don't, the implementation is
wrong.

## Posterior

For Gaussian likelihood near the MAP, the Laplace approximation gives

$$
p(x \mid y) \approx \mathcal{N}(x; x^*, P^*), \quad
P^* = \big(B^{-1} + H'(x^*)^\top R^{-1} H'(x^*)\big)^{-1}.
$$

Compute via `LaplaceCovariance`:

```python
from vardax.posterior import LaplaceCovariance

posterior = LaplaceCovariance()(x_star, model.as_analysis_step(), batch)
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

vardax does not retry or fall back automatically — if your minimiser
fails, that's information. Diagnose, don't hide.

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
