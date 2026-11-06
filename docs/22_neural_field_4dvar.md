---
numbering:
  enumerator: "22.%s"
---

# 22 · Neural-Field 4D-Var

Strong-constraint 4D-Var [](#eq-strong4dvar-cost) controls the initial state and reaches every later time by rolling a model forward [](#eq-rollout). `SpaceTimeFieldVar` drops the rollout: the whole assimilation window is one continuous field

$$x(t, \vect{s}) = g_\text{field}(t, \vect{s}; \gvect{\theta}),$$ (eq-neural-field)

where $t \in \mathbb{R}$ is time, $\vect{s} \in \mathbb{R}^{D_s}$ the spatial position, $x(t, \vect{s}) \in \mathbb{R}$ the field and $g_\text{field}$ a coordinate network whose weights $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ are the control, fitted to the observations with the dynamics as a *soft* constraint (Oh 2025).
The analysis is a function, not a grid: it can be evaluated at any
$(t, \vect{s})$ and differentiated there.

## 22.1 Running example: SSH from along-track altimetry

Satellite altimeters sample sea-surface height (SSH) along narrow
ground tracks, a few thousand scattered $(t_i, \vect{s}_i, y_i)$
points per day with errors $\sigma_i$, and nothing in between. A
gridded 4D-Var first interpolates those points onto the model grid
([`GridPointObs`](xref:api#vardax.GridPointObs)) and then needs a background
covariance $\gmat{\Sigma}_x$ to spread their information. A neural field reads the
tracks as they are ([`PointObsBatch`](xref:api#vardax.PointObsBatch)), and the
quasi-geostrophic equations, imposed at collocation points, carry the
information along the flow instead of $\gmat{\Sigma}_x$.

## 22.2 The cost

With $N_\text{obs}$ observations $y_i$ at $(t_i, \vect{s}_i)$ with errors $\sigma_i$ and
validity flags $m_i$ (zero on padding rows), a PDE residual $r$ that
vanishes for the true flow, collocation points
$\vect{c}_1, \ldots, \vect{c}_{N_c}$ and weights $\beta_\text{res}, \beta \ge 0$,

$$\mathcal{J}(\gvect{\theta}) = \underbrace{\frac{1}{2} \sum_{i=1}^{N_\text{obs}} m_i \frac{(y_i - x(t_i, \vect{s}_i; \gvect{\theta}))^2}{\sigma_i^2}}_{\mathcal{J}_o,\ \texttt{point\_obs\_cost}} + \underbrace{\frac{\beta_\text{res}}{2 N_c} \sum_{j=1}^{N_c} \Vert r(\vect{c}_j; \gvect{\theta}) \Vert^2}_{\mathcal{J}_\text{res},\ \texttt{ResidualPenalty}} + \underbrace{\frac{\beta}{2} \Vert \gvect{\theta} \Vert^2}_{\text{weight decay}}$$ (eq-neural-field-cost)

where $r(\vect{c}; \gvect{\theta}) \in \mathbb{R}^{D_r}$ is the PDE operator applied to the field $x(\cdot; \gvect{\theta})$ and evaluated at $\vect{c} \in \mathbb{R}^{1 + D_s}$, and $y_i, \sigma_i \in \mathbb{R}$, $m_i \in \{0, 1\}$.
The first term is the point-observation cost [](#eq-point-obs-cost) with the field $x(\cdot; \gvect{\theta})$ as the observation operator.
There is **no background term**. What plays the role of $\gmat{\Sigma}_x$ is the
network's spectral bias: gradient descent from a random initialisation
fits smooth, low-frequency functions first (Rahaman et al. 2019), and
the weight decay keeps $\gvect{\theta}$ near the origin. The PDE term is the
physics-informed loss of Raissi et al. (2019); its weight $\beta_\text{res}$
trades fidelity to the equations against fidelity to the data.

Each term is a merged vardax building block:

| Term | Object |
|---|---|
| $x(t_i, \vect{s}_i; \gvect{\theta})$ at the observations, $m_i$-masked and $\sigma_i$-whitened | `FieldPointObs` + `point_obs_cost` ([chapter 11](11_observation_operators.md)) |
| $r(\vect{c}_j; \gvect{\theta})$ at collocation points | `ResidualPenalty(residual_fn, collocation, weight, sampler)`; residuals in `vardax.examples` |
| control split, $\frac{\beta}{2} \Vert \gvect{\theta} \Vert^2$ | `NeuralFieldParameterization` (its `filter_spec` keeps frozen arrays out of $\gvect{\theta}$) |
| $\min_{\gvect{\theta}} \mathcal{J}$ | `OptaxMinimiser` (Adam, early stopping, optional L-BFGS polish) |

## 22.3 Usage

Coordinates put time first, $(t, s_1, s_2)$, for the field, the observations
and the collocation points alike (the convention of the residuals in
`vardax.examples`):

A small SIREN is fitted to 64 noisy samples of the decaying Taylor-Green vortex $\psi(t, \vect{s}) = \sin s_1 \sin s_2 e^{-2 \nu t}$ with viscosity $\nu = 0.1$, which solves the unforced vorticity equation exactly; the example runs on a CPU in seconds. A real analysis uses a wider and deeper network, more collocation points and a step budget in the tens of thousands.

```{code-block} python
:caption: Minimises the neural-field cost [](#eq-neural-field-cost) for a SIREN field [](#eq-neural-field) fitted to point observations [](#eq-point-obs-cost) of a Taylor-Green vortex, then evaluates it and runs it as a cycle step.
:label: code-neural-field-taylor-green
import einx
import geonnax as gnx
import jax
import jax.numpy as jnp
import optax
import vardax as vdx
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: N_obs = 64 observations; N_c = 32 collocation points; N_grid = 25 grid points
# Coordinates are (t, s₁, s₂), time first

key = jax.random.PRNGKey(0)
key_init, key_coords, key_noise, key_fit, key_cycle = jax.random.split(key, 5)
viscosity = 0.1
# the window (t, s₁, s₂) ∈ [0, 1] × [0, π]²
box_low = jnp.array([0.0, 0.0, 0.0])  # (3,)
box_high = jnp.array([1.0, jnp.pi, jnp.pi])  # (3,)


# user
def sample_box(key: PRNGKeyArray) -> Float[Array, "c 3"]:
    # cⱼ ~ U([0, 1] × [0, π]²)
    return jax.random.uniform(key, (32, 3), minval=box_low, maxval=box_high)


# user
def taylor_green(point: Float[Array, " 3"]) -> Float[Array, ""]:
    # −2νt
    exponent = -2.0 * viscosity * point[0]  # → ()
    # e^{−2νt}
    decay = jnp.exp(exponent)  # → ()
    # sin s₁
    sin_1 = jnp.sin(point[1])  # → ()
    # sin s₂
    sin_2 = jnp.sin(point[2])  # → ()
    # ψ(t, s) = sin s₁ sin s₂ e^{−2νt}
    return sin_1 * sin_2 * decay


batched_taylor_green = jax.vmap(taylor_green)
# (tᵢ, sᵢ) ~ U(window)
obs_coords = jax.random.uniform(  # (N_obs, 3)
    key_coords, (64, 3), minval=box_low, maxval=box_high
)
# σ εᵢ, with σ = 0.05
obs_noise = 0.05 * jax.random.normal(key_noise, (64,))  # (N_obs,)
# user
# ψ(tᵢ, sᵢ)
obs_signal = batched_taylor_green(obs_coords)  # (N_obs, 3) → (N_obs,)
# yᵢ = ψ(tᵢ, sᵢ) + σ εᵢ
obs_values = obs_signal + obs_noise  # (N_obs,)
# σᵢ = 0.05
obs_sigma = jnp.full((64,), 0.05)  # (N_obs,)
# mᵢ = 1
obs_valid = jnp.ones((64,), dtype=bool)  # (N_obs,)
# today (vardax)
obs = vdx.PointObsBatch(
    coords=obs_coords, values=obs_values, sigma=obs_sigma, valid=obs_valid
)

# today (geonnax)
# g_field(·; θ): (t, s₁, s₂) ↦ ψ
initial_field = gnx.SIREN.init(3, 16, 1, depth=2, first_omega=3.0, key=key_init)
# today (vardax)
# r = ∂ₜω + u·∇ω − ν Δω, with ω = Δψ and u = ∇^⊥ψ
vorticity_residual = vdx.examples.navier_stokes_vorticity_2d(viscosity)
# today (vardax)
# 𝒥_res(θ) = β_res / (2 N_c) Σⱼ ‖r(cⱼ; θ)‖², with the cⱼ drawn once per analysis
physics_penalty = vdx.ResidualPenalty(
    residual_fn=vorticity_residual,
    collocation=None,
    sampler=sample_box,
    weight=1e-2,
)
# today (optax)
adam = optax.adam(1e-2)
# today (vardax)
minimiser = vdx.OptaxMinimiser(adam, max_steps=300)
# today (vardax)
# 𝒥(θ) = 𝒥_o(θ) + 𝒥_res(θ) + β/2 ‖θ‖²
analysis = vdx.SpaceTimeFieldVar(
    field=initial_field,
    residual=physics_penalty,
    weight_decay=1e-6,
    minimiser=minimiser,
)
# today (vardax)
# θ⋆ = argmin_θ 𝒥(θ)
result = analysis(obs, key=key_fit)

# sₙ on a 5 × 5 grid of [0, π]²
axis_points = jnp.linspace(0.0, jnp.pi, 5)  # (5,)
grid = einx.id("a, b -> (a b) (1 + 1)", axis_points, axis_points)  # (N_grid, 2)
snapshot_time = jnp.float32(0.5)  # → ()
# (t, sₙ) with t = 0.5
points = einx.id(", n d -> n (1 + d)", snapshot_time, grid)  # (N_grid, 2) → (N_grid, 3)
# today (vardax)
# x(0.5, sₙ; θ⋆)
field_values = result.evaluate(points)  # (N_grid, 3) → (N_grid,)
# today (vardax)
# x₀ = x(t₀, sₙ; θ⋆)
initial_condition = result.initial_condition(grid)  # (N_grid, 2) → (N_grid,)
# today (vardax)
# the collocation points are drawn once here and reused by every cycle
step = analysis.as_analysis_step(grid, key=key_cycle)
# today (vardax)
# x₀ = x(t₀, sₙ; θ⋆) for this cycle; without a grid the step returns the result
cycle_state = step(None, obs)  # → (N_grid,)
# user
# ψ(0.5, sₙ)
truth = batched_taylor_green(points)  # (N_grid, 3) → (N_grid,)
# x − ψ
field_error = field_values - truth  # (N_grid,)
# ‖x − ψ‖
error_norm = jnp.linalg.norm(field_error)  # → ()
# ‖ψ‖
truth_norm = jnp.linalg.norm(truth)  # → ()
# ‖x − ψ‖ / ‖ψ‖ on the grid
relative_error = error_norm / truth_norm  # → ()
print(f"{int(result.num_steps)} steps, ‖∇𝒥‖ = {result.grad_norm:.0f}")
print(f"relative error {relative_error:.2f}, cycle state {cycle_state.shape}")
```

It prints `300 steps, ‖∇𝒥‖ = 141` and `relative error 0.14, cycle state (25,)`; the large gradient norm is the step budget running out, not convergence.

`result.grad_norm` reports $\Vert \nabla_{\gvect{\theta}} \mathcal{J}(\gvect{\theta}^\star) \Vert$: a
first-order optimiser stopped by its step budget is not at a stationary
point, which matters if you differentiate through the analysis.

A **separable** field (one small network per axis, combined by a
low-rank product, as in Oh 2025) evaluates a whole space-time grid in
one pass and is much faster for dense collocation; any single-example
`eqx.Module` of $(t, s)$ works as `field`.

### Collocation points

A `ResidualPenalty` with a `sampler` draws its points **once per
analysis**, from `key`, and holds them fixed while the minimiser runs.
The minimiser assumes a deterministic cost: its patience test and
best-iterate tracking compare costs across steps, and the implicit
adjoint differentiates the stationarity condition
$\nabla_{\gvect{\theta}} \mathcal{J}(\gvect{\theta}^\star) = 0$. Resampling every step (a stochastic
gradient of the expected residual) would break all three, so it is left
to a custom training loop; draw enough points to resolve the residual,
or refit with fresh keys.

In a cycle, `as_analysis_step(grid, key=key)` draws the points once, at
construction, and every cycle reuses them. `pipekit_cycle.DACycle`
passes no time, so each cycle's `PointObsBatch` must use
**window-relative** times in the same window as the collocation points;
the step refuses observation times (and a $t_0$) outside the collocation
time range.

### Choosing $\beta_\text{res}$

The observation term is a sum over $N_\text{obs}$ points weighted by
$1 / \sigma^2$, the residual term a mean over the collocation points, so
$\beta_\text{res}$ trades off against roughly $N_\text{obs} / (2 \sigma^2)$ and has
to grow with it. The 1-D advection test uses $\beta_\text{res} = 10^3$ for 100
observations with $\sigma = 0.05$ ($N_\text{obs} / (2 \sigma^2) = 2 \times 10^4$).

### Uncertainty: a deep ensemble

`analysis.ensemble(obs, field_init, n_members, key=key)` refits from
`n_members` random initialisations (`field_init: key -> field`), each
with its own collocation draw, under `eqx.filter_vmap`, and returns one
result whose arrays carry a leading member axis. The spread of the
members is the deep-ensemble heuristic of Lakshminarayanan et al. (2017):
small where observations pin the field down, large away from them
(`tests/test_spacetime_field.py` checks both). A Laplace push-forward
$\mat{J}_x \gmat{\Sigma}_\theta \mat{J}_x^\top$ with $\mat{J}_x = \partial \vect{x} / \partial \gvect{\theta} \in \mathbb{R}^{N_\text{grid} \times D_\theta}$ (the field values $\vect{x}$ at $N_\text{grid}$ points) and
$\gmat{\Sigma}_\theta \in \mathbb{R}^{D_\theta \times D_\theta}$ a Gauss-Newton inverse in weight space is deferred, as in
[design/latent_da.md](design/latent_da.md) §13.

## 22.4 The linear-Gaussian limit

If $x(t, \vect{s}; \vect{z}) = \gvect{\phi}(t, \vect{s})^\top \vect{z}$ is linear in fixed features $\gvect{\phi} \in \mathbb{R}^{D_z}$ with coefficients $\vect{z} \in \mathbb{R}^{D_z}$ and
$\beta_\text{res} = 0$, the cost is

$$\mathcal{J}(\vect{z}) = \frac{1}{2} \Vert \vect{y} - \mat{H} \vect{z} \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}} + \frac{\beta}{2} \Vert \vect{z} \Vert^2, \text{ with } H_{im} = \phi_m(t_i, \vect{s}_i), \gmat{\Sigma}_\varepsilon = \op{diag}(\sigma_i^2)$$ (eq-linear-field-cost)

where $\vect{y} \in \mathbb{R}^{D_y}$ stacks the $D_y = N_\text{obs}$ observations, $\mat{H} \in \mathbb{R}^{D_y \times D_z}$ and $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$.
This is the 3D-Var cost [](#eq-3dvar-cost) for the coefficients $\vect{z}$ with prior covariance $\gmat{\Sigma}_z = \beta^{-1} \mat{I}$, zero prior mean and the features $\mat{H}$ as the observation operator. Its minimiser is the BLUE mean [](#eq-blue-mean), $\vect{z}^\star = \beta^{-1} \mat{H}^\top (\beta^{-1} \mat{H} \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1} \vect{y}$, the posterior mean of Bayesian linear regression. The test suite checks
that `SpaceTimeFieldVar` and `OptimalInterpolation` agree there, the same
correctness gate as Decision D14 for the classical methods.

## 22.5 When to use it, and when strong 4D-Var

| | `SpaceTimeFieldVar` | `StrongFourDVar` (+ `NeuralFieldParameterization`) |
|---|---|---|
| Control | the whole space-time field $x(t, \vect{s}; \gvect{\theta})$ | the initial state $\vect{x}_0 = g_\text{dec}(\gvect{\theta})$ |
| Dynamics | soft: $\beta_\text{res}$-weighted residual at collocation points | hard: $\vect{x}_t = f_t(\vect{x}_0)$ |
| Needs | a differentiable PDE residual | a `ForwardModel` and its adjoint |
| Cost per iteration | residual at $N_c$ points, parallel in time | a sequential rollout and its adjoint |
| Output | a continuous field; $x(t_0, \cdot)$ for forecasting | $\vect{x}_0$ on the grid; the trajectory by rollout |
| Prior | spectral bias + weight decay, no $\gmat{\Sigma}_x$ | $\gmat{\Sigma}_x$, or the parameterization's penalty |

Use the neural field when observations are scattered in space and time,
when no tangent-linear/adjoint model is at hand but the equations are
known, or when the window is long enough that a sequential rollout
dominates the cost. Prefer strong 4D-Var when the model is trusted
(the residual weight then has to grow without bound to match it) and
when a forecast from the analysis must be consistent with the model
step by step. The two meet in the **strong-constraint variant**: fit
only $x(t_0, \cdot)$ as a field and roll it out,
`StrongFourDVar(..., parameterization=NeuralFieldParameterization(field, coords, alpha))`
([chapter 6](06_strong_4dvar.md#parameterized-initial-states)).

## References

- Oh, Y. (2025). On the effect of neural field reparameterization for
  4DVAR. arXiv:2509.21751.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
  Physics-informed neural networks. *J. Comput. Phys.* 378, 686-707.
- Rahaman, N., et al. (2019). On the spectral bias of neural networks.
  *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
  scalable predictive uncertainty estimation using deep ensembles.
  *NeurIPS*.
