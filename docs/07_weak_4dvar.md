---
numbering:
  enumerator: "7.%s"
---

# Weak-constraint 4DVar

Strong-constraint 4DVar (chapter 6) treats the forward model as exact. For real systems this is wrong, as models have biases, missing physics and discretisation error. Over long assimilation windows the trajectory drifts away from the observations because the model itself is wrong, even when the analysis is good. Weak-constraint 4DVar admits this model error as an additional control variable.

## Cost function

The state evolves under noisy dynamics,

$$\vect{x}_t = f(\vect{x}_{t-1}) + \gvect{\eta}_t, \text{ with } \gvect{\eta}_t \sim \mathcal{N}(\mathbf{0}, \gmat{\Sigma}_{\eta,t})$$ (eq-weak-model)

where $f$ is the free model's single step from $t - 1$ to $t$, $\gvect{\eta}_t \in \mathbb{R}^{D_x}$ the model error of that step and $\gmat{\Sigma}_{\eta,t} \in \mathbb{R}^{D_x \times D_x}$ its covariance. The control vector is augmented from $\vect{x}_0$ to $(\vect{x}_0, \gvect{\eta}_1, \ldots, \gvect{\eta}_T)$, and the strong cost [](#eq-strong4dvar-cost) gains a model-error term:

$$\mathcal{J}(\vect{x}_0, \gvect{\eta}_{1:T}) = \underbrace{\frac{1}{2} \Vert \vect{x}_0 - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}}}_{\text{background}} + \underbrace{\frac{1}{2} \sum_{t=0}^{T} \Vert \vect{y}_t - h_t(\vect{x}_t) \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}}_{\text{observations}} + \underbrace{\frac{1}{2} \sum_{t=1}^{T} \Vert \gvect{\eta}_t \Vert^2_{\gmat{\Sigma}_{\eta,t}^{-1}}}_{\text{model error}}$$ (eq-weak4dvar-cost)

where the trajectory $\vect{x}_t \in \mathbb{R}^{D_x}$ is recovered from $(\vect{x}_0, \gvect{\eta}_{1:T})$ by stepping [](#eq-weak-model) forward from $\vect{x}_0$.
The other shapes are those of chapter 6: $\gvect{\mu}_x \in \mathbb{R}^{D_x}$, $\vect{y}_t \in \mathbb{R}^{D_y}$, $h_t: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$, $\gmat{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}$ and $\gmat{\Sigma}_{\varepsilon,t} \in \mathbb{R}^{D_y \times D_y}$.

## Why the model-error term matters

Two failure modes of strong-constraint 4DVar:

- **Climatological drift.** The free model's climatology differs from reality. Long windows produce $\vect{x}_T$ far from $\vect{y}_T$ regardless of $\vect{x}_0$, and the analysis tries to compensate by moving $\vect{x}_0$, which damages the early-time fit.
- **Missing fast physics.** The model resolves the slow manifold but
  not the fast (cloud microphysics, sub-grid turbulence). Strong 4DVar
  attributes fast-physics errors to $\vect{x}_0$, which is the wrong place.

Allowing $\gvect{\eta}_t$ to absorb these errors per-step gives a better fit at
every time without distorting $\vect{x}_0$.

## Implementation in vardax

The example below has $D_x = 8$ sites and a window of $T = 5$ steps, which holds $T + 1 = 6$ times.
The free model decays every site, $f(\vect{x}) = 0.8 \vect{x}$, but the truth also gains a constant $0.5$ per step, $\vect{x}_t = 0.8 \vect{x}_{t-1} + 0.5$, which the free model misses.
Every site is observed at every time, so $D_y = D_x$, with $\gmat{\Sigma}_\varepsilon = 0.01 \mat{I}$; the background is $\gvect{\mu}_x = \mathbf{0}$ with $\gmat{\Sigma}_x = \mat{I}$, and the model-error covariance is $\gmat{\Sigma}_\eta = 0.04 \mat{I}$.

```{code-block} python
:caption: Minimises the weak-constraint cost [](#eq-weak4dvar-cost) and, for comparison, the strong cost [](#eq-strong4dvar-cost) on a window whose free model misses a constant forcing.
:label: code-weak4dvar-analysis
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 window; T = 5 steps, so T + 1 = 6 times; D_x = D_y = 8 sites

key_truth, key_noise = jax.random.split(jax.random.PRNGKey(0))


# user
class Decay(eqx.Module):
    """The free model f(x) = 0.8 x, missing the truth's forcing."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # f(x) = 0.8 x
        return 0.8 * state


# user
def truth_step(
    state: Float[Array, " n"], _: None
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """One step of the true dynamics, with the forcing the free model misses."""
    # xₜ = 0.8 xₜ₋₁ + 0.5
    next_state = 0.8 * state + 0.5  # (D_x,)
    return next_state, next_state


psd = lx.positive_semidefinite_tag
state_structure = jax.ShapeDtypeStruct((8,), jnp.float32)
# Σ_x = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# Σ_ε = 0.01 I, tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(0.01 * prior_cov, psd)  # (D_y, D_y) PSD op
# Σ_η = 0.04 I
model_err_cov = lx.TaggedLinearOperator(0.04 * prior_cov, psd)  # (D_x, D_x) PSD op
# μ_x = 0
background = jnp.zeros(8)  # (D_x,)
# x₀ ~ 𝒩(0, I)
truth_start = jax.random.normal(key_truth, (8,))  # (D_x,)
# x₁ … x_T under the true dynamics
_, truth_later = jax.lax.scan(truth_step, truth_start, length=5)  # (T, D_x)
# [x₀; x₁ … x_T]
truth = einx.id("n, t n -> (1 + t) n", truth_start, truth_later)  # (T + 1, D_x)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (6, 8))  # (T + 1, D_y)
# yₜ = xₜ + 0.1 εₜ
obs = truth + 0.1 * obs_noise  # (T + 1, D_y)

# today (vardax)
# 𝒥(x₀, η) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖yₜ − xₜ‖²_Σ_ε⁻¹ + ½ ∑ₜ ‖ηₜ‖²_Σ_η⁻¹
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
obs_window = einx.id("t n -> 1 t n", obs)  # (T + 1, D_y) → (S, T + 1, D_y)
# mₜ = 1 everywhere
all_observed = jnp.ones((1, 6, 8))  # (S, T + 1, D_y)
# today (vardax)
batch = vdx.Batch1D(input=obs_window, mask=all_observed)
# today (vardax)
# (x₀⋆, η⋆) = argmin 𝒥(x₀, η): the analysis IC and the model-error trajectory
initial_star, model_error_star = model(batch)  # → (S, D_x), (S, T, D_x)

# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖yₜ − fₜ(x₀)‖²_Σ_ε⁻¹, no model-error term
strong = vdx.StrongFourDVar(
    forward=Decay(),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
)
# today (vardax)
# x₀⋆ = argmin 𝒥(x₀)
strong_initial = strong(batch)  # → (S, D_x)
# x₀⋆ − x₀ for each method
weak_offset = initial_star[0] - truth_start  # (D_x,)
strong_offset = strong_initial[0] - truth_start  # (D_x,)
# ‖x₀⋆ − x₀‖
weak_error = jnp.linalg.norm(weak_offset)  # → ()
strong_error = jnp.linalg.norm(strong_offset)  # → ()
# mean of η⋆, against the missing forcing 0.5
mean_model_error = einx.mean("s t n ->", model_error_star)  # → ()
print(f"x₀ error: strong {strong_error:.2f}, weak {weak_error:.2f}")
print(f"mean η⋆ {mean_model_error:.2f}")
```

It prints `x₀ error: strong 2.81, weak 0.25` and `mean η⋆ 0.46`: the strong analysis moves $\vect{x}_0$ to make up for the missing forcing, while the weak analysis puts it into $\gvect{\eta}$.

The model error covariance $\gmat{\Sigma}_\eta$ is a separate input. Three common
parameterisations:

- **Diagonal**: `lineax.DiagonalLinearOperator(variances)`, white Gaussian model noise. Simple but rarely realistic.
- **Matérn spatial**: `kernellib.ImplicitKernelOperator(matern, coords)` with a Matérn kernel function, for spatially correlated model error.
- **Markovian time-correlated**: Ornstein-Uhlenbeck $\gmat{\Sigma}_{\eta,t}$ with correlation timescale $\tau$. The error decorrelates over $\tau$ steps, which is physically motivated.

The choice of $\gmat{\Sigma}_\eta$ is the single most important tuning choice in weak-constraint 4DVar. Too tight, and the result reverts to strong; too loose, and the observations dominate the analysis to the exclusion of the dynamics.

## Computational cost

The control vector dimension grows with $T$:

$$D_\text{control} = D_x + T D_x$$ (eq-weak-control-dim)

For $T = 100$ and a 2D field with $D_x = 10^4$, this is $\sim 10^6$
control variables instead of $10^4$. The minimisation is correspondingly
heavier. Two mitigations:

- **Lower-rank model-error parameterisation.** Restrict $\gvect{\eta}_t$ to a
  low-dimensional subspace (POD modes, learned latent space). vardax
  does not provide this out of the box; users supply a custom
  reparameterisation via an `eqx.Module`.
- **Shorter sub-windows.** Run weak-constraint 4DVar over a short
  sub-window where $\gvect{\eta}_t$ dimensionality is manageable, cycle the
  result via `pipekit_cycle.SmootherCycle`. Operational ECMWF uses
  this pattern.

## Linear-Gaussian agreement

For $T = 0$, $\gvect{\eta}_{1:T}$ is empty and [](#eq-weak4dvar-cost) is the 3DVar cost [](#eq-3dvar-cost), whose minimum for linear $h$ is the BLUE [](#eq-blue-mean); this is the Decision D14 invariant of [chapter 4](04_oi_blue.md#validation). The block continues the one above with the first time only:

```{code-block} python
:caption: Checks that the weak-constraint minimum of [](#eq-weak4dvar-cost) with T = 0 equals the BLUE [](#eq-blue-mean) and that η is empty.
:label: code-weak4dvar-oi-agreement
# Shapes: S = 1 window; T = 0 steps, so T + 1 = 1 time; D_x = D_y = 8 sites

# y₀ and m₀ only
first_obs = batch.input[:, :1]  # (S, T + 1, D_y)
first_mask = batch.mask[:, :1]  # (S, T + 1, D_y)
# today (vardax)
first_time = vdx.Batch1D(input=first_obs, mask=first_mask)
# Σ_x = I and Σ_ε = 0.01 I over OI's (T + 1, D_x) state
field_structure = jax.ShapeDtypeStruct((1, 8), jnp.float32)
field_prior_cov = lx.IdentityLinearOperator(field_structure)  # ((T+1) D_x)² op
field_obs_diag = jnp.full((1, 8), 0.01)  # (T + 1, D_y)
field_obs_cov = lx.DiagonalLinearOperator(field_obs_diag)  # ((T+1) D_y)² op
# today (vardax)
# p(x) = 𝒩(0, Σ_x), p(y ∣ x) = 𝒩(x, Σ_ε)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 8)),
    prior_cov_op=field_prior_cov,
    obs_cov_op=field_obs_cov,
)
# today (vardax)
# x⋆ = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x)
closed_form = optimal_interpolation(first_time)  # → (S, T + 1, D_x)
# today (vardax)
# (x₀⋆, η⋆) = argmin 𝒥(x₀, η); Σ_η is unused when T = 0
weak_initial, weak_model_error = model(first_time)  # → (S, D_x), (S, T, D_x)
print(bool(jnp.allclose(closed_form[:, 0], weak_initial, atol=1e-3)))
print(weak_model_error.shape)
```

It prints `True` and `(1, 0, 8)`: $\gvect{\eta}_{1:T}$ is empty when $T = 0$.

For $T > 0$ and a very tight $\gmat{\Sigma}_\eta$ (precision $\gmat{\Sigma}_\eta^{-1} \to \infty$), $\gvect{\eta}_t \to \mathbf{0}$ and [](#eq-weak4dvar-cost) reduces to [](#eq-strong4dvar-cost), so `WeakFourDVar` reduces to `StrongFourDVar`. This weak-to-strong limit can be tested too.

## Posterior

The posterior is over the augmented control vector $(\vect{x}_0, \gvect{\eta}_{1:T})$, and the Laplace approximation [](#eq-laplace-cov) of chapter 13 gives a Gaussian on this space, whose precision gains $\gmat{\Sigma}_\eta^{-1}$ on each $\gvect{\eta}_t$. `LaplaceCovariance` returns its $\vect{x}_0$ block:

```{code-block} python
:caption: Builds the x₀ block of the Laplace covariance [](#eq-laplace-cov) over the augmented control (x₀, η) of [](#eq-weak4dvar-cost) and applies it to one vector.
:label: code-weak4dvar-laplace
# Shapes: D_x = 8 sites; T = 5 steps

# today (vardax)
laplace = vdx.LaplaceCovariance()
# today (vardax)
analysis_step = model.as_analysis_step()
# (x₀⋆, η⋆) for the one window
control_star = (initial_star[0], model_error_star[0])  # (D_x,), (T, D_x)
# today (vardax)
# Σ_x|y = [Λ_w⁻¹]ₓ₀ₓ₀, Λ_w the Gauss-Newton precision over w = (x₀, η)
posterior = laplace(control_star, analysis_step, batch)
# 1
ones = jnp.ones(8)  # (D_x,)
# Σ_x|y 1, one matrix-free product on the state
spread = posterior.cov.mv(ones)  # (D_x,) → (D_x,)
print(posterior.mean.shape, spread.shape)
```

It prints `(8,) (8,)`: the mean is $\vect{x}_0^\star$ and the covariance is the $\vect{x}_0$ block of the inverse augmented precision, a lazy operator on the state, not the whole augmented covariance; `model.posterior(batch)` returns the same.

For the marginal posterior on the state trajectory $\vect{x}_{0:T}$,
apply the forward at each time and propagate the covariance. This is
implementation-heavy and rarely needed; ensemble approaches via
`filterax` are usually preferable when full trajectory uncertainty
matters.

## When `WeakFourDVar` is the right answer

Use `WeakFourDVar` for multi-time observations when the model is known to have biases (the two failure modes above), strong-constraint analyses misfit the later times of a long window, and you have a reasonable model-error covariance $\gmat{\Sigma}_{\eta,t}$ (often from comparing the free model to higher-fidelity runs or to reanalyses). If model error is small (within the observation noise), use `StrongFourDVar` (chapter 6), since the extra control variables are not worth the cost.

If you'd rather replace the parametric prior with a learned one,
`FourDVarNet` (chapter 9) is the alternative. The learned prior
absorbs the same kind of model error that $\gvect{\eta}_t$ would, without an
explicit augmented control.

## See also

- Chapter 6: `StrongFourDVar` (the no-model-error baseline)
- Chapter 9: `FourDVarNet` (learned alternative to weak-constraint)
- Chapter 13: posterior covariance (augmented-state case)
- Chapter 23: reduced weak-constraint 4DVar, a model error in a basis ([](#eq-reduced-weak4dvar-cost))

## References

- Trémolet, Y. (2006). *Accounting for an imperfect model in 4D-Var.*
  QJRMS 132(621).
- Fisher, M., Leutbecher, M., & Kelly, G. A. (2005). *On the equivalence
  between Kalman smoothing and weak-constraint four-dimensional
  variational data assimilation.* QJRMS 131(613).
- Daescu, D. N., & Todling, R. (2010). *Adjoint sensitivity of the
  model forecast to data assimilation system error covariance
  parameters.* QJRMS 136(653).
