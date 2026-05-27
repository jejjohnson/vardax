# Weak-constraint 4DVar

Strong-constraint 4DVar (chapter 6) treats the forward model as exact.
For real systems this is wrong — models have biases, missing physics,
and discretisation error. Over long assimilation windows the trajectory
drifts away from observations not because the analysis is bad but
because the model itself is. Weak-constraint 4DVar acknowledges this
by admitting model error as an additional control variable.

## Cost function

The state evolves under a noisy dynamics:

$$
x_t = M_t^\text{free}(x_{t-1}) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t).
$$

The control vector is augmented from $x_0$ to $(x_0, \eta_1, \ldots,
\eta_T)$. The cost adds a model-error term:

$$
J(x_0, \boldsymbol{\eta}) =
  \underbrace{\tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}}_{\text{background}} +
  \underbrace{\tfrac{1}{2} \sum_t \|y_t - H_t(x_t)\|^2_{R_t^{-1}}}_{\text{observations}} +
  \underbrace{\tfrac{1}{2} \sum_t \|\eta_t\|^2_{Q_t^{-1}}}_{\text{model error}}.
$$

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

```python
import diffrax as dfx
import optimistix as optx
from vardax.models import WeakFourDVar

model = WeakFourDVar(
    forward=somax_model,                              # M_t^free
    obs_op=obs_op,
    prior_mean=x_b,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
    model_err_cov_op=Q_op,                            # Q for η_t
    minimiser=optx.NonlinearCG(rtol=1e-5),
    forward_adjoint=dfx.RecursiveCheckpointAdjoint(),
)

# Returns both the analysis IC and the model-error trajectory
x_0_star, eta_star = model(batch)
```

The model error covariance $Q$ is a separate input. Three common
parameterisations:

- **Diagonal** — `lineax.DiagonalLinearOperator(variances)`. White
  Gaussian model noise. Simple but rarely realistic.
- **Matérn spatial** — `gaussx.MaternLinearOperator(coords, ...)`.
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

$$
\dim(\text{control}) = \dim(x_0) + T \cdot \dim(x_t).
$$

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

For $T = 0$, $\eta$ is empty and `WeakFourDVar` reduces to 3DVar:

```python
def test_weak_4dvar_recovers_oi_at_T_zero():
    oi = OptimalInterpolation(linear_H, x_b, B_op, R_op)
    weak = WeakFourDVar(
        forward=static_forward, obs_op=linear_H,
        prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
        model_err_cov_op=Q_op,                     # ignored when T = 0
        minimiser=optx.NonlinearCG(rtol=1e-8),
    )
    batch = make_linear_gaussian_batch(T=0)
    x_oi = oi(batch)
    x_weak, eta_weak = weak(batch)
    assert jnp.allclose(x_oi, x_weak, atol=1e-3)
    # η is empty when T = 0
```

For $T > 0$ and very tight $Q$ (precision $Q^{-1} \to \infty$), $\eta_t
\to 0$ and `WeakFourDVar` reduces to `StrongFourDVar`. This is the
weak-to-strong limit, also testable.

## Posterior

The posterior is over the augmented control vector $(x_0,
\boldsymbol{\eta})$. The Laplace approximation gives a Gaussian on
this space. Marginalising to the state trajectory requires applying
the linearised forward through the joint posterior — vardax provides

```python
from vardax.posterior import LaplaceCovariance

posterior_aug = LaplaceCovariance()(
    (x_0_star, eta_star), model.as_analysis_step(), batch,
)
# posterior_aug.mean: (x_0_star, eta_star)
# posterior_aug.cov:  block AbstractLinearOperator over the augmented space
```

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
