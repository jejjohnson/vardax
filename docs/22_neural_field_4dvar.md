# 22 · Neural-Field 4D-Var

Strong-constraint 4D-Var ([chapter 6](06_strong_4dvar.md)) controls the
initial state and reaches every later time by rolling a model forward.
`SpaceTimeFieldVar` drops the rollout: the whole assimilation window is
one continuous field

$$u(t, s) = f_\theta(t, s),$$

a coordinate network whose weights $\theta$ are the control, fitted to
the observations with the dynamics as a *soft* constraint (Oh 2025).
The analysis is a function, not a grid: it can be evaluated at any
$(t, s)$ and differentiated there.

## 22.1 Running example: SSH from along-track altimetry

Satellite altimeters sample sea-surface height (SSH) along narrow
ground tracks, a few thousand scattered $(t_k, x_k, y_k, \eta_k)$
points per day with errors $\sigma_k$, and nothing in between. A
gridded 4D-Var first interpolates those points onto the model grid
([`GridPointObs`](api/obs_operators.md)) and then needs a background
covariance $B$ to spread their information. A neural field reads the
tracks as they are ([`PointObsBatch`](api/protocols.md)), and the
quasi-geostrophic equations, imposed at collocation points, carry the
information along the flow instead of $B$.

## 22.2 The cost

With observations $y_k$ at $(t_k, s_k)$ with errors $\sigma_k$ and
validity flags $m_k$ (zero on padding rows), a PDE operator
$\mathcal N$ with $\mathcal N[u] = 0$ for the true flow, collocation
points $c_1, \dots, c_{N_c}$ and weights $\lambda, \alpha \ge 0$,

$$J(\theta) = \underbrace{\frac{1}{2}\sum_k m_k \frac{\left(y_k - f_\theta(t_k, s_k)\right)^2}{\sigma_k^2}}_{\texttt{point\_obs\_cost}} + \underbrace{\frac{\lambda}{2 N_c}\sum_{j=1}^{N_c} \left\lVert \mathcal N[f_\theta](c_j) \right\rVert^2}_{\texttt{ResidualPenalty}} + \underbrace{\frac{\alpha}{2}\lVert\theta\rVert^2}_{\text{weight decay}} .$$

There is **no background term**. What plays the role of $B$ is the
network's spectral bias: gradient descent from a random initialisation
fits smooth, low-frequency functions first (Rahaman et al. 2019), and
the weight decay keeps $\theta$ near the origin. The PDE term is the
physics-informed loss of Raissi et al. (2019); its weight $\lambda$
trades fidelity to the equations against fidelity to the data.

Each term is a merged vardax building block:

| Term | Object |
|---|---|
| $f_\theta$ at the observations, $m_k$-masked and $\sigma_k$-whitened | `FieldPointObs` + `point_obs_cost` ([chapter 11](11_observation_operators.md)) |
| $\mathcal N[f_\theta]$ at collocation points | `ResidualPenalty(residual_fn, collocation, weight, sampler)`; residuals in `vardax.examples` |
| control split, $\frac{\alpha}{2}\lVert\theta\rVert^2$ | `NeuralFieldParameterization` (its `filter_spec` keeps frozen arrays out of $\theta$) |
| $\min_\theta J$ | `OptaxMinimiser` (Adam, early stopping, optional L-BFGS polish) |

## 22.3 Usage

Coordinates put time first, $(t, x, y)$, for the field, the observations
and the collocation points alike (the convention of the residuals in
`vardax.examples`):

The block below fits a small SIREN to 64 noisy samples of the decaying Taylor–Green vortex $\psi = \sin x \sin y e^{-2 \nu t}$ with $\nu = 0.1$, which solves the unforced vorticity equation exactly; it runs on a CPU in seconds. A real analysis uses a wider and deeper network, more collocation points and a step budget in the tens of thousands.

```python
import einx
import geonnax
import jax
import jax.numpy as jnp
import optax
import vardax as vdx
from jaxtyping import Array, Float, PRNGKeyArray
from vardax.examples import navier_stokes_vorticity_2d

# Shapes: K = 64 observations; N_c = 32 collocation points; N = 25 grid points

key_init, key_coords, key_noise, key_fit, key_cycle = jax.random.split(
    jax.random.PRNGKey(0), 5
)
viscosity = 0.1
# the window (t, x, y) ∈ [0, 1] × [0, π]²
box_low = jnp.array([0.0, 0.0, 0.0])  # (3,)
box_high = jnp.array([1.0, jnp.pi, jnp.pi])  # (3,)


# user
def sample_box(key: PRNGKeyArray) -> Float[Array, "c 3"]:
    # c_j ~ U([0, 1] × [0, π]²)
    return jax.random.uniform(key, (32, 3), minval=box_low, maxval=box_high)


# user
def taylor_green(point: Float[Array, " 3"]) -> Float[Array, ""]:
    # ψ(t, x, y) = sin x sin y e^{−2νt}
    decay = jnp.exp(-2.0 * viscosity * point[0])  # → ()
    return jnp.sin(point[1]) * jnp.sin(point[2]) * decay


# (t_k, x_k, y_k) ~ U(window)
obs_coords = jax.random.uniform(  # (K, 3)
    key_coords, (64, 3), minval=box_low, maxval=box_high
)
# y_k = ψ(t_k, s_k) + σ ε_k, with σ = 0.05
obs_noise = 0.05 * jax.random.normal(key_noise, (64,))  # (K,)
obs_values = jax.vmap(taylor_green)(obs_coords) + obs_noise  # (K, 3) → (K,)
# today (vardax)
obs = vdx.PointObsBatch(
    coords=obs_coords,
    values=obs_values,
    sigma=jnp.full((64,), 0.05),
    valid=jnp.ones((64,), dtype=bool),
)

# today (geonnax)
# f_θ: (t, x, y) ↦ ψ
initial_field = geonnax.SIREN.init(3, 16, 1, depth=2, first_omega=3.0, key=key_init)
# today (vardax)
# 𝒩[ψ] = ∂ₜω + u·∇ω − ν Δω, with ω = Δψ and u = ∇^⊥ψ
vorticity_residual = navier_stokes_vorticity_2d(viscosity)
# today (vardax)
# λ/(2 N_c) Σⱼ ‖𝒩[f_θ](c_j)‖², with the c_j drawn once per analysis
physics_penalty = vdx.ResidualPenalty(
    residual_fn=vorticity_residual,
    collocation=None,
    sampler=sample_box,
    weight=1e-2,
)
# today (vardax)
minimiser = vdx.OptaxMinimiser(optax.adam(1e-2), max_steps=300)
# today (vardax)
# J(θ) = J_o(θ) + λ/(2 N_c) Σⱼ ‖𝒩[f_θ](c_j)‖² + α/2 ‖θ‖²
analysis = vdx.SpaceTimeFieldVar(
    field=initial_field,
    residual=physics_penalty,
    weight_decay=1e-6,
    minimiser=minimiser,
)
# θ* = argmin_θ J(θ)
result = analysis(obs, key=key_fit)

# s_j on a 5 × 5 grid of [0, π]²
axis_points = jnp.linspace(0.0, jnp.pi, 5)  # (5,)
grid = einx.id("a, b -> (a b) (1 + 1)", axis_points, axis_points)  # (N, 2)
# (t, s_j) with t = 0.5
points = einx.id(", n d -> n (1 + d)", jnp.float32(0.5), grid)  # (N, 2) → (N, 3)
# today (vardax)
# u(0.5, s_j) = f_θ*(0.5, s_j)
field_values = result.evaluate(points)  # (N, 3) → (N,)
# today (vardax)
# x₀ = u(t₀, s_j)
initial_condition = result.initial_condition(grid)  # (N, 2) → (N,)
# today (vardax)
# the collocation points are drawn once here and reused by every cycle
step = analysis.as_analysis_step(grid, key=key_cycle)
# x₀ = u(t₀, s_j) for this cycle; without a grid the step returns the result
cycle_state = step(None, obs)  # → (N,)
truth = jax.vmap(taylor_green)(points)  # (N, 3) → (N,)
# ‖u − ψ‖ / ‖ψ‖ on the grid
relative_error = jnp.linalg.norm(field_values - truth) / jnp.linalg.norm(truth)
print(f"{int(result.num_steps)} steps, ‖∇J‖ = {result.grad_norm:.0f}")
print(f"relative error {relative_error:.2f}, cycle state {cycle_state.shape}")
```

It prints `300 steps, ‖∇J‖ = 141` and `relative error 0.14, cycle state (25,)`; the large gradient norm is the step budget running out, not convergence.

`result.grad_norm` reports $\lVert\nabla J(\theta^\star)\rVert$: a
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
$\nabla J(\theta^\star) = 0$. Resampling every step (a stochastic
gradient of the expected residual) would break all three, so it is left
to a custom training loop; draw enough points to resolve the residual,
or refit with fresh keys.

In a cycle, `as_analysis_step(grid, key=key)` draws the points once, at
construction, and every cycle reuses them. `pipekit_cycle.DACycle`
passes no time, so each cycle's `PointObsBatch` must use
**window-relative** times in the same window as the collocation points;
the step refuses observation times (and a $t_0$) outside the collocation
time range.

### Choosing $\lambda$

The observation term is a sum over $N_\text{obs}$ points weighted by
$1/\sigma^2$, the residual term a mean over the collocation points, so
$\lambda$ trades off against roughly $N_\text{obs}/(2\sigma^2)$ and has
to grow with it. The 1-D advection test uses $\lambda = 10^3$ for 100
observations with $\sigma = 0.05$ ($N_\text{obs}/(2\sigma^2) = 2\times10^4$).

### Uncertainty: a deep ensemble

`ana.ensemble(obs, field_init, n_members, key=key)` refits from
`n_members` random initialisations (`field_init: key -> field`), each
with its own collocation draw, under `eqx.filter_vmap`, and returns one
result whose arrays carry a leading member axis. The spread of the
members is the deep-ensemble heuristic of Lakshminarayanan et al. (2017):
small where observations pin the field down, large away from them
(`tests/test_spacetime_field.py` checks both). A Laplace push-forward
$J \Sigma_\theta J^\top$ with $J = \partial u/\partial\theta$ and
$\Sigma_\theta$ a Gauss–Newton inverse in weight space is deferred, as in
[design/latent_da.md](design/latent_da.md) §13.

## 22.4 The linear-Gaussian limit

If $f_\theta(t, s) = \phi(t, s)^\top w$ is linear in fixed features and
$\lambda = 0$, the cost is

$$J(w) = \frac{1}{2} \lVert y - \Phi w \rVert^2_{R^{-1}} + \frac{\alpha}{2}\lVert w \rVert^2, \text{ with } \Phi_{km} = \phi_m(t_k, s_k), R = \mathrm{diag}(\sigma_k^2),$$

the 3D-Var cost for the coefficients with $B = \alpha^{-1} I$, $x_b = 0$
and $H = \Phi$; its minimiser is the OI / BLUE analysis
$w^\star = \alpha^{-1}\Phi^\top(\alpha^{-1}\Phi\Phi^\top + R)^{-1} y$,
the posterior mean of Bayesian linear regression. The test suite checks
that `SpaceTimeFieldVar` and `OptimalInterpolation` agree there, the same
correctness gate as Decision D14 for the classical methods.

## 22.5 When to use it, and when strong 4D-Var

| | `SpaceTimeFieldVar` | `StrongFourDVar` (+ `NeuralFieldParameterization`) |
|---|---|---|
| Control | the whole space-time field $f_\theta(t, s)$ | the initial state $x_0 = \psi(\theta)$ |
| Dynamics | soft: $\lambda$-weighted residual at collocation points | hard: $x_t = \mathcal M^t(x_0)$ |
| Needs | a differentiable PDE residual | a `ForwardModel` and its adjoint |
| Cost per iteration | residual at $N_c$ points, parallel in time | a sequential rollout and its adjoint |
| Output | a continuous field; $u(t_0, \cdot)$ for forecasting | $x_0$ on the grid; the trajectory by rollout |
| Prior | spectral bias + weight decay, no $B$ | $B$, or the parameterization's penalty |

Use the neural field when observations are scattered in space and time,
when no tangent-linear/adjoint model is at hand but the equations are
known, or when the window is long enough that a sequential rollout
dominates the cost. Prefer strong 4D-Var when the model is trusted
(the residual weight then has to grow without bound to match it) and
when a forecast from the analysis must be consistent with the model
step by step. The two meet in the **strong-constraint variant**: fit
only $u(t_0, \cdot)$ as a field and roll it out,
`StrongFourDVar(..., parameterization=NeuralFieldParameterization(field, coords, alpha))`
([chapter 6](06_strong_4dvar.md#parameterized-initial-states)).

## References

- Oh, Y. (2025). On the effect of neural field reparameterization for
  4DVAR. arXiv:2509.21751.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
  Physics-informed neural networks. *J. Comput. Phys.* 378, 686–707.
- Rahaman, N., et al. (2019). On the spectral bias of neural networks.
  *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
  scalable predictive uncertainty estimation using deep ensembles.
  *NeurIPS*.
