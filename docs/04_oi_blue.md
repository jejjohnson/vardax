# Optimal Interpolation / BLUE

For linear-Gaussian state estimation, the posterior is Gaussian in
closed form and the analysis is a single matrix expression. This is
the Best Linear Unbiased Estimator (BLUE), known in geosciences as
Optimal Interpolation (OI). It is the first method you should reach
for when the regime allows it (Decision D16).

## Setup

Prior:

$$
p(x) = \mathcal{N}(x; x_b, B), \quad x \in \mathbb{R}^n.
$$

Likelihood (linear $H$, Gaussian errors):

$$
p(y \mid x) = \mathcal{N}(y; H x, R), \quad H \in \mathbb{R}^{m \times n}.
$$

By the standard Gaussian conjugacy result, the posterior is

$$
p(x \mid y) = \mathcal{N}(x; x^*, P^*),
$$

with

$$
\boxed{\;
x^* = x_b + K(y - H x_b), \qquad K = B H^\top (H B H^\top + R)^{-1}.
\;}
$$

$K$ is the Kalman gain. The posterior covariance is

$$
P^* = (I - K H) B = (B^{-1} + H^\top R^{-1} H)^{-1}.
$$

The two forms of $P^*$ are equivalent by Sherman-Morrison-Woodbury;
which one to compute depends on which is cheaper.

## Two equivalent expressions, two computations

The Kalman gain $K = B H^\top (H B H^\top + R)^{-1}$ requires inverting
an $m \times m$ matrix (observation-space form). The alternative
posterior precision $(B^{-1} + H^\top R^{-1} H)$ requires inverting an
$n \times n$ matrix (state-space form).

Pick the form whose matrix is smaller:

| Regime | Form | Inversion size |
|---|---|---|
| $m \ll n$ (sparse observations) | $K = B H^\top (H B H^\top + R)^{-1}$ | $m \times m$ |
| $m \gg n$ (dense observations) | $P^{*\,-1} = B^{-1} + H^\top R^{-1} H$ | $n \times n$ |

For SSH altimetry with sparse along-track samples and a 2D grid, $m \ll n$
— observation-space form. For a satellite imager with one observation
per state cell, $m \approx n$ — either works.

vardax picks automatically based on dimensions, unless you override.

## Why the matrices are never materialised

Both forms require solving a linear system, not inverting a matrix
explicitly. `gaussx` supplies structured factorisations:

- $B$ as `gaussx.MaternLinearOperator(coords, length_scale, sigma)` —
  Matérn covariance on a regular grid, with $O(N \log N)$ matrix-vector
  products via FFT (or Kronecker structure on separable grids).
- $R$ as `lineax.DiagonalLinearOperator(variances)` — diagonal,
  trivial.
- $H B H^\top$ never materialised; only $v \mapsto H B H^\top v$ is
  needed, and that's two mat-vecs in $B$ and two in $H^\top$.

The linear system $(H B H^\top + R) v = (y - H x_b)$ is then solved by
`lineax.CG`, returning $v$ in $O(k \cdot m)$ time where $k$ is the CG
iteration count (typically tens). The analysis $x^* = x_b + B H^\top v$
is one more mat-vec.

A full OI on a 2D Matérn prior with 50,000 grid cells and 5,000
observations takes seconds on a single GPU. No materialised matrices,
no $O(n^3)$ factorisations.

## Implementation in vardax

```python
import gaussx as gx
import lineax as lx
from vardax.models import OptimalInterpolation
from vardax.obs_operators import LinearObs

model = OptimalInterpolation(
    obs_op=LinearObs(H_mat=along_track_op),       # must be linear
    prior_mean=climatology_ssh,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=100.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(obs_variances),
)

# Single forward pass — no iteration
x_star = model(batch)

# Posterior is exact and free
posterior = model.posterior(batch)
# Posterior(mean=x_star, cov=AbstractLinearOperator, samples=None, provenance={...})
```

The `posterior.cov` is an `AbstractLinearOperator` representing $P^*$.
Sampling from the posterior is a mat-vec in $P^{*\,1/2}$; the diagonal
of $P^*$ (marginal variances) is one more mat-vec family. Nothing is
ever materialised.

## Validation

`OptimalInterpolation` is the canonical baseline. Every more
sophisticated method must agree with it in the linear-Gaussian limit
(Decision D14 invariant). The test suite enforces this:

```python
def test_three_dvar_recovers_oi():
    oi = OptimalInterpolation(linear_obs, x_b, B_op, R_op)
    three = ThreeDVar(linear_obs, x_b, B_op, R_op,
                       minimiser=optx.GaussNewton(rtol=1e-8))
    batch = make_linear_gaussian_batch()
    assert jnp.allclose(oi(batch), three(batch), atol=1e-3)

def test_strong_4dvar_recovers_oi():
    oi = OptimalInterpolation(linear_obs, x_b, B_op, R_op)
    strong = StrongFourDVar(static_forward, linear_obs, x_b, B_op, R_op,
                              minimiser=optx.NonlinearCG(rtol=1e-8))
    batch = make_linear_gaussian_batch(T=0)
    assert jnp.allclose(oi(batch), strong(batch), atol=1e-3)

# … and similar for IncrementalFourDVar, FourDVarNet (IdentityPrior),
#                  AmortizedPosterior (trained to convergence)
```

If a new method disagrees with `OptimalInterpolation` in this limit,
something is wrong. The agreement is the definition of correctness.

## Connection to the Kalman filter

The Kalman filter analysis step is OI applied at each cycle:

$$
\begin{aligned}
\text{forecast:}  &\quad x^f_t = M_t x^a_{t-1}, \quad P^f_t = M_t P^a_{t-1} M_t^\top + Q \\
\text{analysis:}  &\quad K_t = P^f_t H^\top (H P^f_t H^\top + R)^{-1} \\
                  &\quad x^a_t = x^f_t + K_t (y_t - H x^f_t) \\
                  &\quad P^a_t = (I - K_t H) P^f_t
\end{aligned}
$$

The analysis step is exactly `OptimalInterpolation(prior_mean=x^f_t,
prior_cov_op=P^f_t, obs_cov_op=R, obs_op=H)`. The forecast step lives
in `filterax` (it's about propagating uncertainty through dynamics, not
about inference per se).

This means `filterax`'s Kalman filter can be built compositionally:
forecast via `filterax`, analysis via vardax `OptimalInterpolation`,
update covariance via `filterax`. The interfaces line up because both
libraries respect the same protocols.

## When OI is the right answer

Use `OptimalInterpolation` when **all** of:

- $H$ is linear (or the linearisation around $x_b$ is good enough that
  the residual nonlinearity is dominated by observation noise)
- $B$ and $R$ are Gaussian
- The posterior is unimodal (no symmetry-breaking, no bistable
  regimes)
- A single timestep (no dynamics) or the dynamics are linear

If any fails:

- Nonlinear $H$ → `ThreeDVar` (chapter 5)
- Dynamics → `StrongFourDVar` / `IncrementalFourDVar` (chapters 6, 8)
- Non-Gaussian likelihood → `AmortizedPosterior` with score head (chapter 10)
- Multimodal posterior → ensemble methods via `filterax`

The conformance test ensures that if you mistakenly call
`OptimalInterpolation` with a nonlinear `obs_op`, `__init__` raises a
clear error pointing you to `ThreeDVar` instead.

## See also

- Chapter 2 — observation model (the linear $H$ assumption)
- Chapter 5 — `ThreeDVar` (nonlinear extension)
- Chapter 13 — posterior covariance (the closed-form $P^*$ is the
  simplest case)
- Design doc: [`design/decisions.md#d16`](design/decisions.md#d16-blue-oi-as-a-first-class-method)

## References

- Lorenc, A. (1981). *A global three-dimensional multivariate
  statistical interpolation scheme.* MWR 109(4).
- Daley, R. (1991). *Atmospheric Data Analysis*. CUP. Ch. 4.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning*. Ch. 2.7.
