---
numbering:
  enumerator: "6.%s"
---

# Strong-constraint 4DVar

When observations are spread across an assimilation window and the state evolves through known dynamics, 3DVar is no longer enough, because we need to relate observations at later times to the initial state through the forward model. Strong-constraint 4DVar treats the dynamics as exact and makes the initial state $x_0$ the control variable.

## Cost function

Given observations $y_t$ at times $t = 0, 1, \ldots, T$, dynamics $M_t$ propagating $x_0$ to time $t$, and an observation operator $H_t$ with error covariance $R_t$ at each time, the cost is the 3DVar cost [](#eq-3dvar-cost) summed over the window,

$$J(x_0) = \frac{1}{2} \|x_0 - x_b\|^2_{B^{-1}} + \frac{1}{2} \sum_{t=0}^{T} \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}$$ (eq-strong4dvar-cost)

**Strong-constraint** means the model is treated as exact: any $x_0$ determines the trajectory [](#eq-rollout) exactly, and model error is not admitted (`WeakFourDVar` in chapter 7 admits it). The gradient combines the background term with the adjoint of every forward step,

$$\nabla J(x_0) = B^{-1}(x_0 - x_b) + \sum_{t=0}^{T} (M'_t)^\top H'_t{}^\top R_t^{-1} (H_t(M_t(x_0)) - y_t)$$ (eq-strong4dvar-grad)

where $M'_t$ is the tangent-linear model of $M_t$ at $x_0$ and $H'_t$ the Jacobian of $H_t$ at $M_t(x_0)$. In hand-coded operational 4DVar, $(M'_t)^\top$ is the adjoint model, historically a major engineering effort (Talagrand and Courtier 1987, Errico 1997). With `diffrax`, it is a constructor argument.

## Implementation in vardax

The example below is Lorenz-96 [](#eq-lorenz96) on $N = 40$ sites with forcing $F = 8$, one RK4 step of $\Delta t = 0.05$ per call, over a window of $T = 11$ times (10 steps).
The truth $x_\text{true}$ is a state on the attractor; the background is $x_b = x_\text{true} + L \xi$ with $B = L L^\top$ a Gaussian correlation of length scale $\ell = 2$ sites (plus a $10^{-3}$ jitter), and every second site is observed at every time, $y_t = M_t(x_\text{true}) + \varepsilon_t$, with $R = I$.
$H_t$ is the identity on the observed sites, so the mask $m_t$ does the selection.

```{code-block} python
:caption: Builds the Lorenz-96 twin experiment and minimises the strong-constraint cost [](#eq-strong4dvar-cost) with NonlinearCG.
:label: code-strong4dvar-analysis
import dataclasses

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: S = 1 window; T = 11 times; N = 40 sites

key_spinup, key_background, key_noise, key_field = jax.random.split(
    jax.random.PRNGKey(0), 4
)


# user
class Lorenz96(eqx.Module):
    """Lorenz-96 with forcing F = 8, one RK4 step of length dt per call."""

    dt: float = 0.05

    def tendency(self, state: Float[Array, " n"]) -> Float[Array, " n"]:
        # xₖ₊₁
        east = jnp.roll(state, -1)
        # xₖ₋₂
        far_west = jnp.roll(state, 2)
        # xₖ₋₁
        west = jnp.roll(state, 1)
        # (xₖ₊₁ − xₖ₋₂) xₖ₋₁
        advection = (east - far_west) * west
        # dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + F
        return advection - state + 8.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # k₁ = f(x)
        k1 = self.tendency(state)
        # k₂ = f(x + Δt/2 k₁)
        k2 = self.tendency(state + 0.5 * dt * k1)
        # k₃ = f(x + Δt/2 k₂)
        k3 = self.tendency(state + 0.5 * dt * k2)
        # k₄ = f(x + Δt k₃)
        k4 = self.tendency(state + dt * k3)
        # x ← x + Δt/6 (k₁ + 2k₂ + 2k₃ + k₄)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


forward = Lorenz96()


# user
def rollout(state: Float[Array, " n"], n_steps: int) -> Float[Array, "t n"]:
    """Return the trajectory x₀, M(x₀), …, Mⁿ(x₀)."""

    def advance(
        carry: Float[Array, " n"], _: None
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        # xₜ₊₁ = M(xₜ)
        next_state = forward.step(carry, forward.dt)
        return next_state, next_state

    _, later_states = jax.lax.scan(advance, state, length=n_steps)
    # [x₀; x₁ … xₙ]
    return einx.id("n, t n -> (1 + t) n", state, later_states)


site = jnp.arange(40.0)  # (N,)
# i − j
signed_offset = einx.subtract("i, j -> i j", site, site)  # (N, N)
# |i − j|
offset = jnp.abs(signed_offset)  # (N, N)
# dᵢⱼ = min(|i − j|, N − |i − j|), the distance round the ring
distance = jnp.minimum(offset, 40.0 - offset)  # (N, N)
# exp(−dᵢⱼ² / 2ℓ²), ℓ = 2 sites
correlation = jnp.exp(-0.5 * (distance / 2.0) ** 2)  # (N, N)
# 10⁻³ δᵢⱼ
jitter = 1e-3 * jnp.eye(40)  # (N, N)
# Bᵢⱼ = exp(−dᵢⱼ² / 2ℓ²) + 10⁻³ δᵢⱼ
background_corr = correlation + jitter  # (N, N)
# B
prior_cov = lx.MatrixLinearOperator(  # (N, N) PSD op
    background_corr, lx.positive_semidefinite_tag
)
# R = I
obs_structure = jax.ShapeDtypeStruct((40,), jnp.float32)
obs_cov = lx.IdentityLinearOperator(obs_structure)  # (N, N) op

# δ ~ 𝒩(0, I)
spinup_noise = jax.random.normal(key_spinup, (40,))  # (N,)
# F + δ
spinup_start = 8.0 + spinup_noise  # (N,)
# x_true = M²⁰⁰(F + δ): a state on the attractor
spinup = rollout(spinup_start, 200)  # (N,) → (201, N)
truth_start = spinup[-1]  # (N,)
# B = L Lᵀ
background_chol = jnp.linalg.cholesky(background_corr)  # (N, N)
# ξ ~ 𝒩(0, I)
background_noise = jax.random.normal(key_background, (40,))  # (N,)
# L ξ
background_error = einx.dot(  # (N,)
    "i j, j -> i", background_chol, background_noise
)
# x_b = x_true + L ξ
background = truth_start + background_error  # (N,)
# xₜ = Mₜ(x_true), t = 0 … 10
truth_trajectory = rollout(truth_start, 10)  # (N,) → (T, N)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (11, 40))  # (T, N)
# yₜ = Mₜ(x_true) + εₜ
obs = truth_trajectory + obs_noise  # (T, N)
# mₜ = 1 on every second site
mask = jnp.zeros((11, 40)).at[:, ::2].set(1.0)  # (T, N)

# today (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖mₜ ⊙ (yₜ − Mₜ(x₀))‖²_{R⁻¹}
model = vdx.StrongFourDVar(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
    # recompute each step in reverse mode instead of storing it
    checkpoint=True,
    # this B is too ill-conditioned for the default CG solve
    cov_solver=lx.Cholesky(),
)
obs_window = einx.id("t n -> 1 t n", obs)  # (T, N) → (S, T, N)
mask_window = einx.id("t n -> 1 t n", mask)  # (T, N) → (S, T, N)
# today (vardax)
batch = vdx.Batch1D(input=obs_window, mask=mask_window)
# today (vardax)
# x₀* = argmin J(x₀)
analysis = model(batch)  # → (S, N)


# user
def rmse(state: Float[Array, " n"]) -> Float[Array, ""]:
    """Root-mean-square error of a state against x_true."""
    # (x − x_true)²
    squared_error = (state - truth_start) ** 2  # (N,)
    # mean (x − x_true)²
    mean_squared_error = einx.mean("n ->", squared_error)  # → ()
    # √(mean (x − x_true)²)
    return jnp.sqrt(mean_squared_error)  # → ()


print(f"RMSE of x_b {rmse(background):.2f}, of x₀* {rmse(analysis[0]):.2f}")
```

It prints `RMSE of x_b 1.44, of x₀* 0.63`.
With the default `lx.CG` covariance solver this $B$ (condition number about $5 \times 10^3$) makes the $B^{-1}$ solve return non-finite values, so the block passes `cov_solver=lx.Cholesky()`.

Three slots control the gradient:

- **`checkpoint`** rematerialises the rollout in reverse mode (`True`, or a block length `k`), for any per-step forward; the gradient is the same and the memory smaller (chapter 3).
- **`forward_adjoint`** is a `diffrax.AbstractAdjoint` for a forward that integrates with `diffrax` and implements `with_adjoint(adjoint)` ([`SupportsForwardAdjoint`](xref:api#vardax.SupportsForwardAdjoint)), for example `dfx.BacksolveAdjoint()` for constant memory over long windows. Any other forward raises `ValueError`.
- **`minimiser_adjoint`** controls how gradients flow back through the outer minimiser, as for `ThreeDVar` (chapter 5).

## Large controls: first-order minimisers

`minimiser=` takes an `optimistix.AbstractMinimiser` (the default is `optx.BFGS(rtol=1e-6, atol=1e-6)`; chapter 5 lists the choices) or any [`Minimiser`](xref:api#vardax.Minimiser): a callable `minimiser(cost_fn, x0, batch) -> x_star` over a pytree control. BFGS keeps a dense inverse-Hessian estimate, which is not viable when the control is $10^5$ network weights. `OptaxMinimiser` runs a first-order `optax` optimiser instead, in a `jax.lax.while_loop`, and stops at `max_steps`, at a gradient norm below `gtol`, at a step below `atol + rtol·‖x‖`, or after `patience` steps whose cost is worse than the best by more than round-off. (There is no relative cost-change test: next to a large irreducible misfit the cost change rounds to zero in float32 long before $x$ converges.) An optional `polish` optimiser (e.g. `optax.lbfgs()`), with its own tight tolerances, finishes from there.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) of [](#code-strong4dvar-analysis) again with Adam, then L-BFGS.
:label: code-strong4dvar-optax
import optax

# today (vardax)
minimiser = vdx.OptaxMinimiser(
    optimizer=optax.adam(1e-2),
    max_steps=2000,
    gtol=1e-4,
    patience=200,
    polish=optax.lbfgs(),
    polish_steps=200,
)
optax_model = dataclasses.replace(model, minimiser=minimiser)
# today (vardax)
# x₀* = argmin J(x₀), by Adam then L-BFGS
optax_analysis = optax_model(batch)  # → (S, N)
print(f"RMSE of x₀* {rmse(optax_analysis[0]):.2f}")
```

It prints `RMSE of x₀* 0.26`; the NonlinearCG run above stops at its 200-step cap before it converges.

`minimiser_adjoint` still applies. Its default, `ImplicitAdjoint`, is
exact only at a stationary point, $\nabla J(x^\star) = 0$, which a
first-order optimiser stopped early does not guarantee: `OptaxMinimiser`
reports $\lVert \nabla J(x^\star) \rVert_2$ in `MinimiseResult.grad_norm`;
check it before trusting gradients through the analysis. `result.cost`
is differentiable (envelope theorem). `OptimistixMinimiser()` is the default made explicit
and returns exactly what the default model returns.

## Parameterized initial states

`parameterization=` takes any [`StateParameterization`](xref:api#vardax.StateParameterization): the control becomes a pytree $\theta$, the initial state $x_0 = \psi(\theta)$, and its penalty replaces the background term of [](#eq-strong4dvar-cost),

$$J(\theta) = \mathrm{pen}(\theta) + \frac{1}{2} \sum_t \lVert y_t - H_t(M_t(\psi(\theta)))\rVert^2_{R_t^{-1}}$$ (eq-param-cost)

where $\psi$ maps the control to the initial state and $\mathrm{pen}$ is the penalty on the control.

`LinearParameterization(basis, x_b)` is the reduced-basis case
($\psi(X) = x_b + \Phi X$, $\mathrm{pen}(X) = \frac{1}{2} X^\top Q^{-1} X$,
identical to `ReducedBasisFourDVar(basis=...)`).
`NeuralFieldParameterization(field, coords, weight_decay)` makes $x_0$ a
coordinate network evaluated on the grid, with
$\mathrm{pen}(\theta) = \frac{\alpha}{2}\lVert\theta\rVert^2$ and no $B$
(the network's spectral bias is the prior); its weights are a control
for `OptaxMinimiser`, not BFGS. Frozen arrays inside the field (random
Fourier frequencies read through `stop_gradient`) must be excluded from
the control with `filter_spec=`, or the weight decay shrinks them.

The SIREN below has no frozen arrays, so the default `filter_spec` is right.

```{code-block} python
:caption: Minimises the parameterised cost [](#eq-param-cost) with x₀ a SIREN field on the ring of sites.
:label: code-strong4dvar-neural-field
import geonnax as gnx

# Shapes: S = 1 window; T = 11 times; N = 40 sites

# λₙ = 2π n / N
angle = 2 * jnp.pi * site / 40  # (N,)
# cos λₙ
cos_angle = jnp.cos(angle)  # (N,)
# sin λₙ
sin_angle = jnp.sin(angle)  # (N,)
# sₙ = (cos λₙ, sin λₙ), the sites on the unit circle
coords = einx.id("n, n -> n (1 + 1)", cos_angle, sin_angle)  # (N,) → (N, 2)
# today (geonnax)
# f_θ: ℝ² → ℝ, three hidden layers of width 32
siren = gnx.SIREN.init(2, 32, 1, depth=3, key=key_field)
# today (vardax)
# x₀ = ψ(θ) = [f_θ(sₙ)]ₙ, pen(θ) = α/2 ‖θ‖², α = 10⁻⁴
field_param = vdx.NeuralFieldParameterization(siren, coords, 1e-4)
# today (vardax)
field_minimiser = vdx.OptaxMinimiser(optax.adam(1e-2), max_steps=2000)
# J(θ) = pen(θ) + ½ Σₜ ‖mₜ ⊙ (yₜ − Mₜ(ψ(θ)))‖²_{R⁻¹}
field_model = dataclasses.replace(
    model, parameterization=field_param, minimiser=field_minimiser
)
# today (vardax)
# ψ(θ*), θ* = argmin J(θ)
field_analysis = field_model(batch)  # → (S, N)
print(f"RMSE of ψ(θ*) {rmse(field_analysis[0]):.2f}")
```

It prints `RMSE of ψ(θ*) 1.82`, worse than the background: the network starts from its random initialisation, not from $x_b$.

## Reduced-basis 4DVar

`ReducedBasisFourDVar(basis=...)` minimises this cost with the control moved into a basis, $x_0 = x_b + \Phi u$ with $P \ll N$ coefficients; it is the reduced strong-constraint rung of chapter [23](23_reduced_order.md), which owns its cost [](#eq-reduced-basis-cost), its gradient and the Lorenz-96 comparison against this chapter's full control.

## The adjoint problem

The gradient [](#eq-strong4dvar-grad) reads "integrate the trajectory forward, then propagate the observation residuals backward through the adjoint of the dynamics." In equations, this is the backward sweep

$$\lambda_T = H'_T{}^\top R_T^{-1} (H_T(x_T) - y_T)$$ (eq-adjoint-terminal)

$$\lambda_{t-1} = (M'_{t \leftarrow t-1})^\top \lambda_t + H'_{t-1}{}^\top R_{t-1}^{-1} (H_{t-1}(x_{t-1}) - y_{t-1})$$ (eq-adjoint-recursion)

$$\nabla_{x_0} J = B^{-1}(x_0 - x_b) + \lambda_0$$ (eq-adjoint-grad)

where $\lambda_t$ is the adjoint state at time $t$ and $M'_{t \leftarrow t-1}$ is the tangent-linear model of the single step from $t - 1$ to $t$, so that $M'_t$ in [](#eq-strong4dvar-grad) is the product of these one-step tangents. This is exactly what `diffrax.RecursiveCheckpointAdjoint` or
`BacksolveAdjoint` compute under the hood. Vardax just composes them
into the cost-gradient call.

## Algorithm

Written out, the cost [](#eq-strong4dvar-cost) rolls the trajectory forward, sums the whitened misfit over the window and adds the background term, and `optimistix.minimise` differentiates through the rollout by reverse mode, which is the backward sweep [](#eq-adjoint-terminal) to [](#eq-adjoint-grad).
`StrongFourDVar` builds the same cost (with optional `jax.checkpoint` blocks and a `forward_adjoint`), so the hand-written one reaches the same minimum.
The block continues the one above.

```{code-block} python
:caption: Writes out the strong-constraint cost [](#eq-strong4dvar-cost) and minimises it with optimistix, whose reverse mode is the adjoint sweep [](#eq-adjoint-terminal) to [](#eq-adjoint-grad).
:label: code-strong4dvar-hand-cost
# Shapes: T = 11 times; N = 40 sites

cov_solver = lx.Cholesky()


# user
def whiten(residual: Float[Array, " n"]) -> Float[Array, " n"]:
    """Apply R⁻¹ to one time's residual with a lineax solve."""
    # R⁻¹ r
    solution = lx.linear_solve(obs_cov, residual, solver=cov_solver)
    return solution.value  # (N,)


# user
def strong_cost(
    initial_state: Float[Array, " n"],
    window: tuple[Float[Array, "t n"], Float[Array, "t n"]],
) -> Float[Array, ""]:
    """Strong-constraint cost J(x₀) over one masked window."""
    window_obs, window_mask = window
    # xₜ = Mₜ(x₀), t = 0 … T − 1
    trajectory = rollout(initial_state, 10)  # (N,) → (T, N)
    # yₜ − Hₜ(xₜ), Hₜ = I
    innovation = window_obs - trajectory  # (T, N)
    # rₜ = mₜ ⊙ (yₜ − Hₜ(xₜ))
    residual = window_mask * innovation  # (T, N)
    # R⁻¹ rₜ at every time
    weighted_residual = jax.vmap(whiten)(residual)  # (T, N)
    # rₜᵀ R⁻¹ rₜ summed over t
    obs_misfit = einx.dot("t n, t n ->", residual, weighted_residual)  # → ()
    # J_o = ½ Σₜ rₜᵀ R⁻¹ rₜ
    obs_term = 0.5 * obs_misfit  # → ()
    # δx = x₀ − x_b
    increment = initial_state - background  # (N,)
    # B⁻¹ δx
    prior_solve = lx.linear_solve(prior_cov, increment, solver=cov_solver)
    # δxᵀ B⁻¹ δx
    prior_misfit = einx.dot("n, n ->", increment, prior_solve.value)  # → ()
    # J_b = ½ δxᵀ B⁻¹ δx
    background_term = 0.5 * prior_misfit  # → ()
    # J(x₀) = J_b + J_o
    return background_term + obs_term  # → ()


# today (optimistix)
# x₀* = argmin J(x₀), started at x_b
solution = optx.minimise(
    strong_cost,
    optx.NonlinearCG(rtol=1e-5, atol=1e-5),
    background,
    args=(obs, mask),
    adjoint=optx.ImplicitAdjoint(),
    max_steps=200,
    throw=False,
)
print(bool(jnp.allclose(solution.value, analysis[0], atol=1e-3)))

# today (vardax)
# x₀* = argmin J(x₀), with J(x₀*) and ‖∇J(x₀*)‖ as diagnostics
result = minimiser.run(strong_cost, background, (obs, mask))
print(f"J = {result.cost:.1f}, ‖∇J‖ = {result.grad_norm:.1e}")
```

It prints `True` and `J = 133.5, ‖∇J‖ = 1.6e-04`.

## Linear-Gaussian agreement

When $H_t$ and $M_t$ are linear and $T = 0$, [](#eq-strong4dvar-cost) is the 3DVar cost and its minimum is the BLUE [](#eq-blue-mean), so `StrongFourDVar` must agree with `OptimalInterpolation` (the Decision D14 invariant, see [chapter 4](04_oi_blue.md#validation)). The block continues the one above with a single-time window and $M = I$:

```{code-block} python
:caption: Checks that the strong-constraint minimum of [](#eq-strong4dvar-cost) with T = 0 and M = I equals the BLUE [](#eq-blue-mean).
:label: code-strong4dvar-oi-agreement
# Shapes: S = 1 window; T = 1 time; N = 40 sites


# user
class Static(eqx.Module):
    """M(x) = x, so a single-time window reduces 4DVar to OI."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # M(x) = x
        return state


psd = lx.positive_semidefinite_tag
# diag(B) = 2
prior_diag = jnp.full(40, 2.0)  # (N,)
# diag(R) = 1/2
obs_diag = jnp.full(40, 0.5)  # (N,)
# B = 2 I, tagged PSD so the default CG solve accepts it
state_prior_cov = lx.TaggedLinearOperator(  # (N, N) PSD op
    lx.DiagonalLinearOperator(prior_diag), psd
)
# R = I / 2
state_obs_cov = lx.TaggedLinearOperator(  # (N, N) PSD op
    lx.DiagonalLinearOperator(obs_diag), psd
)
# the same B and R over OI's (T, N) state
field_prior_diag = jnp.full((1, 40), 2.0)  # (T, N)
field_obs_diag = jnp.full((1, 40), 0.5)  # (T, N)
field_prior_cov = lx.TaggedLinearOperator(  # (T N, T N) PSD op
    lx.DiagonalLinearOperator(field_prior_diag), psd
)
field_obs_cov = lx.TaggedLinearOperator(  # (T N, T N) PSD op
    lx.DiagonalLinearOperator(field_obs_diag), psd
)
# y₀ and m₀ only
first_obs = batch.input[:, :1]  # (S, T, N)
first_mask = batch.mask[:, :1]  # (S, T, N)
# today (vardax)
first_time = vdx.Batch1D(input=first_obs, mask=first_mask)
# today (vardax)
# J(x₀) = ½ ‖x₀‖²_{B⁻¹} + ½ ‖m₀ ⊙ (y₀ − x₀)‖²_{R⁻¹}
strong_static = vdx.StrongFourDVar(
    forward=Static(),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(40),
    prior_cov_op=state_prior_cov,
    obs_cov_op=state_obs_cov,
)
# today (vardax)
# p(x) = 𝒩(0, B), p(y | x) = 𝒩(m ⊙ x, R)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 40)),
    prior_cov_op=field_prior_cov,
    obs_cov_op=field_obs_cov,
)
# today (vardax)
# x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
closed_form = optimal_interpolation(first_time)  # → (S, T, N)
# today (vardax)
# x₀* = argmin J(x₀)
iterative = strong_static(first_time)  # → (S, N)
print(bool(jnp.allclose(closed_form[:, 0], iterative, atol=1e-3)))
```

It prints `True`.

## When the window is long

For $T = 100$+ assimilation cycles, the forward trajectory dominates
memory. Four strategies:

1. **`checkpoint=True` (or a block length).** Works with any per-step forward, with 3 to 4 times less gradient memory on Lorenz-96 (chapter 3).
2. **`dfx.BacksolveAdjoint()`** (via `forward_adjoint=`, for a [`SupportsForwardAdjoint`](xref:api#vardax.SupportsForwardAdjoint) forward) solves the adjoint ODE backwards in time, with $O(1)$ memory in the trajectory length. Reverse-time stiffness can require an implicit integrator (`dfx.Kvaerno5`, `dfx.ImplicitEuler`).
3. **`dfx.RecursiveCheckpointAdjoint(checkpoints=N)`** (also via `forward_adjoint=`) stores $N$ checkpoints and recomputes the segments between them, with $O(\sqrt{T})$ memory.
4. **Shorter windows.** Run multiple `StrongFourDVar` analyses with
   shorter windows and cycle them via `pipekit_cycle.DACycle`. This is
   what operational systems do.

For most research-scale problems with $T \le 20$, the defaults
(plain reverse mode, the forward's own adjoint) are fine.

## When the dynamics are nonlinear

The 4DVar cost [](#eq-strong4dvar-cost) is nonconvex when $M_t$ is nonlinear, so multiple local minima exist (and matter, especially for chaotic systems like Lorenz). Strategies:

- **Multi-start.** Minimise from several perturbed first guesses and
  keep the lowest cost. vardax has no helper for this yet: the models
  start every minimisation from $x_b$, so a multi-start needs a first
  guess separate from the background.
- **Quasi-static gradient.** Use a warm-start from a previous cycle
  (the usual operational pattern).
- **Switch to `IncrementalFourDVar`** (chapter 8): the outer Gauss-Newton iterations relinearise around the current iterate, which is more robust than NonlinearCG on the full nonlinear cost.

## Posterior

Pair the analysis with a `PosteriorAdapter`; `LaplaceCovariance` returns the Gauss-Newton covariance [](#eq-gn-cov) of chapter 13 as a lazy operator. The block continues the one above.
It passes `solver=lx.Cholesky()` because the default CG solve of $B^{-1}$ reaches its step cap on this $B$; `model.posterior(batch, adapter=laplace)` is the same call.

```{code-block} python
:caption: Builds the Gauss-Newton posterior covariance [](#eq-gn-cov) at the analysis of [](#code-strong4dvar-analysis) and applies it to one vector.
:label: code-strong4dvar-laplace
# Shapes: N = 40 sites

# today (vardax)
laplace = vdx.LaplaceCovariance(solver=lx.Cholesky())
# today (vardax)
analysis_step = model.as_analysis_step()
# today (vardax)
# P* = (B⁻¹ + Σₜ (Hₜ′ Mₜ′)ᵀ R⁻¹ (Hₜ′ Mₜ′))⁻¹ at x₀*
posterior = laplace(analysis[0], analysis_step, batch)
# 1
ones = jnp.ones(40)  # (N,)
# P* 1, one matrix-free product
spread = posterior.cov.mv(ones)  # (N,) → (N,)
print(posterior.mean.shape, spread.shape)
```

It prints `(40,) (40,)`.

Each product with [](#eq-gn-cov) is a Krylov solve (`lineax.CG`) against the Gauss-Newton Hessian [](#eq-gn-hessian), and each of its mat-vecs is a tangent-linear and adjoint pass through the trajectory, so a query is expensive: a typical `n_krylov = 50` costs 50 such passes. `IncrementalFourDVar` reuses the operator its inner loop already solves with (chapter 8).

## When `StrongFourDVar` is the right answer

Use `StrongFourDVar` for multi-time observations with dynamics when model error is small (within the noise floor of the observations) and you want an optimistix or optax minimiser directly on the full nonlinear cost. For operational 4DVar with structured priors, `IncrementalFourDVar` solves the same problem faster; chapter 8 says when to switch. If model error matters, use `WeakFourDVar` (chapter 7).

## See also

- Chapter 3: dynamical model and `forward_adjoint` options
- Chapter 7: `WeakFourDVar` (admits model error)
- Chapter 8: `IncrementalFourDVar` (operational fast path)
- Chapter 12: adjoint composition

## References

- Talagrand, O., & Courtier, P. (1987). *Variational assimilation of
  meteorological observations with the adjoint vorticity equation.*
  QJRMS 113(478).
- Le Dimet, F.-X., & Talagrand, O. (1986). *Variational algorithms for
  analysis and assimilation of meteorological observations.* Tellus A
  38(2).
- Errico, R. M. (1997). *What is an adjoint model?* BAMS 78(11).
