# Weak-constraint 4DVar

Strong-constraint 4DVar (chapter 6) treats the forward model as exact.
For real systems this is wrong — models have biases, missing physics,
and discretisation error. Over long assimilation windows the trajectory
drifts away from observations not because the analysis is bad but
because the model itself is. Weak-constraint 4DVar acknowledges this
by admitting model error as an additional control variable.

## Cost function

The state evolves under a noisy dynamics:

$$x_t = M_t^\text{free}(x_{t-1}) + \eta_t, \text{ with } \eta_t \sim \mathcal{N}(0, Q_t).$$

The control vector is augmented from $x_0$ to $(x_0, \eta_1, \ldots,
\eta_T)$. The cost adds a model-error term:

$$J(x_0, \boldsymbol{\eta}) = \underbrace{\frac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}}_{\text{background}} + \underbrace{\frac{1}{2} \sum_t \|y_t - H_t(x_t)\|^2_{R_t^{-1}}}_{\text{observations}} + \underbrace{\frac{1}{2} \sum_t \|\eta_t\|^2_{Q_t^{-1}}}_{\text{model error}}.$$

The trajectory is recovered from $(x_0, \boldsymbol{\eta})$ by stepping
$x_t = M_t^\text{free}(x_{t-1}) + \eta_t$ recursively.

## Why the model-error term matters

Two failure modes of strong-constraint 4DVar:

- **Climatological drift.** The free model's climatology differs from
  reality. Long windows produce $x_T$ far from $y_T$ regardless of
  $x_0$ — the analysis tries to compensate by moving $x_0$, which
  damages the early-time fit.
- **Missing fast physics.** The model resolves the slow manifold but
  not the fast (cloud microphysics, sub-grid turbulence). Strong 4DVar
  attributes fast-physics errors to $x_0$, which is the wrong place.

Allowing $\eta_t$ to absorb these errors per-step gives a better fit at
every time without distorting $x_0$.

## Implementation in vardax

The example below has $N = 8$ sites and a window of $T = 6$ times (5 steps).
The free model decays every site, $M^\text{free}(x) = 0.8 x$, but the truth also gains a constant $0.5$ per step, $x_t = 0.8 x_{t-1} + 0.5$, which the free model misses.
Every site is observed at every time with $R = 0.01 I$; the background is $x_b = 0$ with $B = I$, and the model-error covariance is $Q = 0.04 I$.

```python
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 window; T = 6 times; N = 8 sites

key_truth, key_noise = jax.random.split(jax.random.PRNGKey(0))


# user
class Decay(eqx.Module):
    """The free model M^free(x) = 0.8 x, missing the truth's forcing."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        return 0.8 * state


# user
def truth_step(state: Float[Array, " n"], _) -> tuple[Array, Array]:
    # xₜ = 0.8 xₜ₋₁ + 0.5
    next_state = 0.8 * state + 0.5  # (N,)
    return next_state, next_state


psd = lx.positive_semidefinite_tag
# B = I
prior_cov = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((8,), jnp.float32))
# R = 0.01 I, tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(0.01 * prior_cov, psd)  # (N, N) op
# Q = 0.04 I
model_err_cov = lx.TaggedLinearOperator(0.04 * prior_cov, psd)  # (N, N) op
# x_b = 0
background = jnp.zeros(8)  # (N,)
# x₀ ~ 𝒩(0, I)
truth_start = jax.random.normal(key_truth, (8,))  # (N,)
_, truth_later = jax.lax.scan(truth_step, truth_start, length=5)  # (T − 1, N)
# [x₀; x₁ … x₅]
truth = einx.id("n, t n -> (1 + t) n", truth_start, truth_later)  # (T, N)
# yₜ = xₜ + 0.1 εₜ, εₜ ~ 𝒩(0, I)
obs = truth + 0.1 * jax.random.normal(key_noise, (6, 8))  # (T, N)

# today (vardax)
# J(x₀, η) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖yₜ − xₜ‖²_{R⁻¹} + ½ Σₜ ‖ηₜ‖²_{Q⁻¹}
model = vdx.WeakFourDVar(
    forward=Decay(),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    model_err_cov_op=model_err_cov,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
    # chapter 3
    checkpoint=True,
)
# today (vardax)
batch = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", obs),  # (T, N) → (S, T, N)
    mask=jnp.ones((1, 6, 8)),
)
# (x₀*, η*) = argmin J(x₀, η): the analysis IC and the model-error trajectory
initial_star, model_error_star = model(batch)  # → (S, N), (S, T − 1, N)

# today (vardax)
# the same window with no model-error term
strong = vdx.StrongFourDVar(
    forward=Decay(),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
)
strong_initial = strong(batch)  # → (S, N)
# ‖x₀* − x₀‖ for each method
weak_error = jnp.linalg.norm(initial_star[0] - truth_start)  # → ()
strong_error = jnp.linalg.norm(strong_initial[0] - truth_start)  # → ()
# mean of η*, against the missing forcing 0.5
mean_model_error = einx.mean("s t n ->", model_error_star)  # → ()
print(f"x₀ error: strong {strong_error:.2f}, weak {weak_error:.2f}")
print(f"mean η* {mean_model_error:.2f}")
```

It prints `x₀ error: strong 2.81, weak 0.25` and `mean η* 0.46`: the strong analysis moves $x_0$ to make up for the missing forcing, while the weak analysis puts it into $\eta$.

The model error covariance $Q$ is a separate input. Three common
parameterisations:

- **Diagonal** — `lineax.DiagonalLinearOperator(variances)`. White
  Gaussian model noise. Simple but rarely realistic.
- **Matérn spatial** — `gaussx.ImplicitKernelOperator(matern, coords)` with a Matérn kernel function.
  Spatially correlated model error.
- **Markovian time-correlated** — Ornstein-Uhlenbeck $Q_t$ with
  correlation timescale $\tau$. The error decorrelates over $\tau$
  steps, which is physically motivated.

Choice of $Q$ is the single most impactful tuning choice in
weak-constraint 4DVar — too tight and the result reverts to strong;
too loose and the analysis is dominated by observations to the
exclusion of dynamics.

## Computational cost

The control vector dimension grows with $T$:

$$\dim(\text{control}) = \dim(x_0) + T \cdot \dim(x_t).$$

For $T = 100$ and a 2D field with $n = 10^4$, this is $\sim 10^6$
control variables instead of $10^4$. The minimisation is correspondingly
heavier. Two mitigations:

- **Lower-rank model-error parameterisation.** Restrict $\eta_t$ to a
  low-dimensional subspace (POD modes, learned latent space). vardax
  does not provide this out of the box; users supply a custom
  reparameterisation via an `eqx.Module`.
- **Shorter sub-windows.** Run weak-constraint 4DVar over a short
  sub-window where $\eta_t$ dimensionality is manageable, cycle the
  result via `pipekit_cycle.SmootherCycle`. Operational ECMWF uses
  this pattern.

## Linear-Gaussian agreement

For $T = 0$, $\eta$ is empty and `WeakFourDVar` reduces to 3DVar; the block continues the one above with the first time only, and `tests/test_linear_gaussian_agreement.py` enforces the agreement for every method:

```python
# Shapes: S = 1 window; T = 1 time; N = 8 sites

# today (vardax)
# y₀ only
first_time = vdx.Batch1D(input=batch.input[:, :1], mask=batch.mask[:, :1])
# today (vardax)
# x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b), over OI's (T, N) state
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 8)),
    prior_cov_op=lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 8), jnp.float32)),
    obs_cov_op=lx.DiagonalLinearOperator(jnp.full((1, 8), 0.01)),
)
closed_form = optimal_interpolation(first_time)  # → (S, T, N)
# Q is unused when T = 0
weak_initial, weak_model_error = model(first_time)  # → (S, N), (S, 0, N)
print(bool(jnp.allclose(closed_form[:, 0], weak_initial, atol=1e-3)))
print(weak_model_error.shape)
```

It prints `True` and `(1, 0, 8)`: $\eta$ is empty when $T = 0$.

For $T > 0$ and very tight $Q$ (precision $Q^{-1} \to \infty$), $\eta_t
\to 0$ and `WeakFourDVar` reduces to `StrongFourDVar`. This is the
weak-to-strong limit, also testable.

## Posterior

The posterior is over the augmented control vector $(x_0,
\boldsymbol{\eta})$. The Laplace approximation gives a Gaussian on
this space. Marginalising to the state trajectory requires applying
the linearised forward through the joint posterior — vardax provides

```python
# Shapes: N = 8 sites; T = 6 times

# today (vardax)
# P* = [(Λ_GN)⁻¹]ₓ₀ₓ₀, Λ_GN the Gauss–Newton precision over (x₀, η)
posterior = vdx.LaplaceCovariance()(
    (initial_star[0], model_error_star[0]), model.as_analysis_step(), batch
)
# P* 1, one matrix-free product on the state
spread = posterior.cov.mv(jnp.ones(8))  # (N,) → (N,)
print(posterior.mean.shape, spread.shape)
```

It prints `(8,) (8,)`: the mean is $x_0^*$ and the covariance is the $x_0$ block of the inverse augmented precision, a lazy operator on the state, not the whole augmented covariance; `model.posterior(batch)` returns the same.

For the marginal posterior on the state trajectory $\{x_t\}_{t=0}^T$,
apply the forward at each time and propagate the covariance. This is
implementation-heavy and rarely needed; ensemble approaches via
`filterax` are usually preferable when full trajectory uncertainty
matters.

## When `WeakFourDVar` is the right answer

Use `WeakFourDVar` when:

- Multi-timestep observations
- Model is known to have biases (climatological drift, missing physics)
- Long assimilation windows where strong-constraint analyses produce
  obvious misfits at later times
- You have a reasonable model-error covariance $Q_t$ (often from
  comparing the free model to higher-fidelity runs or to reanalyses)

If model error is small (within obs noise), use `StrongFourDVar`
(chapter 6) — the extra control variables aren't worth the cost.

If you'd rather replace the parametric prior with a learned one,
`FourDVarNet` (chapter 9) is the alternative. The learned prior
absorbs the same kind of model error that $\eta_t$ would, without an
explicit augmented control.

## See also

- Chapter 6 — `StrongFourDVar` (the no-model-error baseline)
- Chapter 9 — `FourDVarNet` (learned alternative to weak-constraint)
- Chapter 13 — posterior covariance (augmented-state case)

## References

- Trémolet, Y. (2006). *Accounting for an imperfect model in 4D-Var.*
  QJRMS 132(621).
- Fisher, M., Leutbecher, M., & Kelly, G. A. (2005). *On the equivalence
  between Kalman smoothing and weak-constraint four-dimensional
  variational data assimilation.* QJRMS 131(613).
- Daescu, D. N., & Todling, R. (2010). *Adjoint sensitivity of the
  model forecast to data assimilation system error covariance
  parameters.* QJRMS 136(653).
