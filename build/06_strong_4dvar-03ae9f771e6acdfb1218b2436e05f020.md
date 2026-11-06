---
numbering:
  enumerator: "6.%s"
---

# Strong-constraint 4DVar

When observations are spread across an assimilation window and the state evolves through known dynamics, 3DVar is no longer enough, because we need to relate observations at later times to the initial state through the forward model. Strong-constraint 4DVar treats the dynamics as exact and makes the initial state $\vect{x}_0$ the control variable.

## Cost function

Given observations $\vect{y}_t$ at times $t = 0, 1, \ldots, T$, a forward model $f$ whose $t$-fold composition $f_t$ propagates $\vect{x}_0$ to time $t$ (so $\vect{x}_t = f_t(\vect{x}_0)$ and $f_0$ is the identity), and an observation operator $h_t$ with error covariance $\gmat{\Sigma}_{\varepsilon,t}$ at each time, the cost is the 3DVar cost [](#eq-3dvar-cost) summed over the window,

$$\mathcal{J}(\vect{x}_0) = \frac{1}{2} \Vert \vect{x}_0 - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{y}_t - h_t(f_t(\vect{x}_0)) \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}$$ (eq-strong4dvar-cost)

**Strong-constraint** means the model is treated as exact: any $\vect{x}_0$ determines the trajectory [](#eq-rollout) exactly, and model error is not admitted (`WeakFourDVar` in chapter 7 admits it). The gradient combines the background term with the adjoint of every forward step,

$$\nabla_{\vect{x}_0} \mathcal{J}(\vect{x}_0) = \gmat{\Sigma}_x^{-1}(\vect{x}_0 - \gvect{\mu}_x) + \sum_{t=0}^{T} \mat{F}_{t \leftarrow 0}^\top \mat{H}_t^\top \gmat{\Sigma}_{\varepsilon,t}^{-1} (h_t(f_t(\vect{x}_0)) - \vect{y}_t)$$ (eq-strong4dvar-grad)

where $\mat{F}_{t \leftarrow 0}$ is the tangent-linear model of $f_t$ at $\vect{x}_0$ and $\mat{H}_t$ the Jacobian of $h_t$ at $f_t(\vect{x}_0)$.
The shapes are $\vect{x}_0, \gvect{\mu}_x \in \mathbb{R}^{D_x}$, $\vect{y}_t \in \mathbb{R}^{D_y}$, $f_t: \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$, $h_t: \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$, $\gmat{\Sigma}_x, \mat{F}_{t \leftarrow 0} \in \mathbb{R}^{D_x \times D_x}$, $\gmat{\Sigma}_{\varepsilon,t} \in \mathbb{R}^{D_y \times D_y}$ and $\mat{H}_t \in \mathbb{R}^{D_y \times D_x}$.
In hand-coded operational 4DVar, $\mat{F}_{t \leftarrow 0}^\top$ is the adjoint model, historically a major engineering effort (Talagrand and Courtier 1987, Errico 1997). With `diffrax`, it is a constructor argument.

## Implementation in vardax

The example below is Lorenz-96 [](#eq-lorenz96) on $D_x = 40$ sites with forcing $u = 8$, one RK4 step of $\Delta t = 0.05$ per call, over a window of $T = 10$ steps, which holds $T + 1 = 11$ times.
The truth $\vect{x}_\text{true}$ is a state on the attractor; the background is $\gvect{\mu}_x = \vect{x}_\text{true} + \mat{L}_x \gvect{\xi}$ with $\gmat{\Sigma}_x = \mat{L}_x \mat{L}_x^\top$ a Gaussian correlation of length scale $\ell = 2$ sites (plus a $10^{-3}$ jitter) and $\gvect{\xi} \sim \mathcal{N}(\mathbf{0}, \mat{I})$, and every second site is observed at every time, $\vect{y}_t = f_t(\vect{x}_\text{true}) + \gvect{\varepsilon}_t$, with $\gmat{\Sigma}_\varepsilon = \mat{I}$ and $D_y = D_x$.
$h_t$ is the identity on the observed sites, so the mask $\vect{m}_t$ does the selection.

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

# Shapes: S = 1 window; T = 10 steps, so T + 1 = 11 times; D_x = D_y = 40 sites

key_spinup, key_background, key_noise, key_field = jax.random.split(
    jax.random.PRNGKey(0), 4
)


# user
class Lorenz96(eqx.Module):
    """Lorenz-96 with forcing u = 8, one RK4 step of length dt per call."""

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
        # dxₖ/dt = (xₖ₊₁ − xₖ₋₂) xₖ₋₁ − xₖ + u
        return advection - state + 8.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # k₁ = tendency(x)
        k1 = self.tendency(state)
        # k₂ = tendency(x + Δt/2 k₁)
        k2 = self.tendency(state + 0.5 * dt * k1)
        # k₃ = tendency(x + Δt/2 k₂)
        k3 = self.tendency(state + 0.5 * dt * k2)
        # k₄ = tendency(x + Δt k₃)
        k4 = self.tendency(state + dt * k3)
        # f(x) = x + Δt/6 (k₁ + 2k₂ + 2k₃ + k₄)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


forward = Lorenz96()


# user
def rollout(state: Float[Array, " n"], n_steps: int) -> Float[Array, "t n"]:
    """Return the trajectory x₀, f(x₀), …, fⁿ(x₀)."""

    def advance(
        carry: Float[Array, " n"], _: None
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        # xₜ₊₁ = f(xₜ)
        next_state = forward.step(carry, forward.dt)
        return next_state, next_state

    _, later_states = jax.lax.scan(advance, state, length=n_steps)
    # [x₀; x₁ … xₙ]
    return einx.id("n, t n -> (1 + t) n", state, later_states)


site = jnp.arange(40.0)  # (D_x,)
# i − j
signed_offset = einx.subtract("i, j -> i j", site, site)  # (D_x, D_x)
# |i − j|
offset = jnp.abs(signed_offset)  # (D_x, D_x)
# dᵢⱼ = min(|i − j|, D_x − |i − j|), the distance round the ring
distance = jnp.minimum(offset, 40.0 - offset)  # (D_x, D_x)
# exp(−dᵢⱼ² / 2ℓ²), ℓ = 2 sites
correlation = jnp.exp(-0.5 * (distance / 2.0) ** 2)  # (D_x, D_x)
# 10⁻³ δᵢⱼ
jitter = 1e-3 * jnp.eye(40)  # (D_x, D_x)
# [Σ_x]ᵢⱼ = exp(−dᵢⱼ² / 2ℓ²) + 10⁻³ δᵢⱼ
background_corr = correlation + jitter  # (D_x, D_x)
# Σ_x
prior_cov = lx.MatrixLinearOperator(  # (D_x, D_x) PSD op
    background_corr, lx.positive_semidefinite_tag
)
# Σ_ε = I
obs_structure = jax.ShapeDtypeStruct((40,), jnp.float32)
obs_cov = lx.IdentityLinearOperator(obs_structure)  # (D_y, D_y) op

# δ ~ 𝒩(0, I)
spinup_noise = jax.random.normal(key_spinup, (40,))  # (D_x,)
# u + δ
spinup_start = 8.0 + spinup_noise  # (D_x,)
# x_true = f²⁰⁰(u + δ): a state on the attractor
spinup = rollout(spinup_start, 200)  # (D_x,) → (201, D_x)
truth_start = spinup[-1]  # (D_x,)
# Σ_x = L_x L_xᵀ
background_chol = jnp.linalg.cholesky(background_corr)  # (D_x, D_x)
# ξ ~ 𝒩(0, I)
background_noise = jax.random.normal(key_background, (40,))  # (D_x,)
# L_x ξ
background_error = einx.dot(  # (D_x,)
    "i j, j -> i", background_chol, background_noise
)
# μ_x = x_true + L_x ξ
background = truth_start + background_error  # (D_x,)
# xₜ = fₜ(x_true), t = 0 … T
truth_trajectory = rollout(truth_start, 10)  # (D_x,) → (T + 1, D_x)
# εₜ ~ 𝒩(0, I)
obs_noise = jax.random.normal(key_noise, (11, 40))  # (T + 1, D_y)
# yₜ = fₜ(x_true) + εₜ
obs = truth_trajectory + obs_noise  # (T + 1, D_y)
# mₜ = 1 on every second site
mask = jnp.zeros((11, 40)).at[:, ::2].set(1.0)  # (T + 1, D_y)

# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖mₜ ⊙ (yₜ − fₜ(x₀))‖²_Σ_ε⁻¹
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
obs_window = einx.id("t n -> 1 t n", obs)  # (T + 1, D_y) → (S, T + 1, D_y)
mask_window = einx.id("t n -> 1 t n", mask)  # (T + 1, D_y) → (S, T + 1, D_y)
# today (vardax)
batch = vdx.Batch1D(input=obs_window, mask=mask_window)
# today (vardax)
# x₀⋆ = argmin 𝒥(x₀)
analysis = model(batch)  # → (S, D_x)


# user
def rmse(state: Float[Array, " n"]) -> Float[Array, ""]:
    """Root-mean-square error of a state against x_true."""
    # (x − x_true)²
    squared_error = (state - truth_start) ** 2  # (D_x,)
    # mean (x − x_true)²
    mean_squared_error = einx.mean("n ->", squared_error)  # → ()
    # √(mean (x − x_true)²)
    return jnp.sqrt(mean_squared_error)  # → ()


print(f"RMSE of μ_x {rmse(background):.2f}, of x₀⋆ {rmse(analysis[0]):.2f}")
```

It prints `RMSE of μ_x 1.44, of x₀⋆ 0.63`.
With the default `lx.CG` covariance solver this $\gmat{\Sigma}_x$ (condition number about $5 \times 10^3$) makes the $\gmat{\Sigma}_x^{-1}$ solve return non-finite values, so the block passes `cov_solver=lx.Cholesky()`.

Three slots control the gradient:

- **`checkpoint`** rematerialises the rollout in reverse mode (`True`, or a block length `k`), for any per-step forward; the gradient is the same and the memory smaller (chapter 3).
- **`forward_adjoint`** is a `diffrax.AbstractAdjoint` for a forward that integrates with `diffrax` and implements `with_adjoint(adjoint)` ([`SupportsForwardAdjoint`](xref:api#vardax.SupportsForwardAdjoint)), for example `dfx.BacksolveAdjoint()` for constant memory over long windows. Any other forward raises `ValueError`.
- **`minimiser_adjoint`** controls how gradients flow back through the outer minimiser, as for `ThreeDVar` (chapter 5).

## Large controls: first-order minimisers

`minimiser=` takes an `optimistix.AbstractMinimiser` (the default is `optx.BFGS(rtol=1e-6, atol=1e-6)`; chapter 5 lists the choices) or any [`Minimiser`](xref:api#vardax.Minimiser): a callable `minimiser(cost_fn, x0, batch) -> x_star` over a pytree control. BFGS keeps a dense inverse-Hessian estimate, which is not viable when the control is $10^5$ network weights. `OptaxMinimiser` runs a first-order `optax` optimiser instead, in a `jax.lax.while_loop`, and stops at `max_steps`, at a gradient norm below `gtol`, at a step below `atol + rtol·‖x‖`, or after `patience` steps whose cost is worse than the best by more than round-off. (There is no relative cost-change test: next to a large irreducible misfit the cost change rounds to zero in float32 long before $\vect{x}$ converges.) An optional `polish` optimiser (e.g. `optax.lbfgs()`), with its own tight tolerances, finishes from there.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) of [](#code-strong4dvar-analysis) again with Adam, then L-BFGS.
:label: code-strong4dvar-optax
import optax

# Shapes: S = 1 window; D_x = 40 sites

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
# x₀⋆ = argmin 𝒥(x₀), by Adam then L-BFGS
optax_analysis = optax_model(batch)  # → (S, D_x)
print(f"RMSE of x₀⋆ {rmse(optax_analysis[0]):.2f}")
```

It prints `RMSE of x₀⋆ 0.26`; the NonlinearCG run above stops at its 200-step cap before it converges.

`minimiser_adjoint` still applies. Its default, `ImplicitAdjoint`, is
exact only at a stationary point, $\nabla_{\gvect{\theta}} \mathcal{J}(\gvect{\theta}^\star) = \mathbf{0}$, which a
first-order optimiser stopped early does not guarantee: `OptaxMinimiser`
reports $\Vert \nabla_{\gvect{\theta}} \mathcal{J}(\gvect{\theta}^\star) \Vert_2$ in `MinimiseResult.grad_norm`;
check it before trusting gradients through the analysis. `result.cost`
is differentiable (envelope theorem). `OptimistixMinimiser()` is the default made explicit
and returns exactly what the default model returns.

## Parameterized initial states

`parameterization=` takes any [`StateParameterization`](xref:api#vardax.StateParameterization): the control becomes a pytree $\gvect{\theta} \in \mathbb{R}^{D_\theta}$, the initial state $\vect{x}_0 = g_\text{dec}(\gvect{\theta})$, and its penalty replaces the background term of [](#eq-strong4dvar-cost),

$$\mathcal{J}(\gvect{\theta}) = \mathcal{R}(\gvect{\theta}) + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{y}_t - h_t(f_t(g_\text{dec}(\gvect{\theta}))) \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}$$ (eq-param-cost)

where $g_\text{dec}: \mathbb{R}^{D_\theta} \to \mathbb{R}^{D_x}$ maps the control to the initial state and $\mathcal{R}$ is the penalty on the control.

`LinearParameterization(basis, background)` is the reduced-basis case
($g_\text{dec}(\vect{z}) = \gvect{\mu}_x + \gmat{\Phi} \vect{z}$, $\mathcal{R}(\vect{z}) = \frac{1}{2} \vect{z}^\top \gmat{\Sigma}_z^{-1} \vect{z}$,
with the coefficients $\vect{z} \in \mathbb{R}^{D_z}$, the basis $\gmat{\Phi} \in \mathbb{R}^{D_x \times D_z}$ and $\gmat{\Sigma}_z \in \mathbb{R}^{D_z \times D_z}$,
identical to `ReducedBasisFourDVar(basis=...)`).
`NeuralFieldParameterization(field, coords, weight_decay)` makes $\vect{x}_0$ a
coordinate network evaluated on the grid, with
$\mathcal{R}(\gvect{\theta}) = \frac{\beta}{2} \Vert \gvect{\theta} \Vert_2^2$ and no $\gmat{\Sigma}_x$
(the network's spectral bias is the prior); its weights are a control
for `OptaxMinimiser`, not BFGS. Frozen arrays inside the field (random
Fourier frequencies read through `stop_gradient`) must be excluded from
the control with `filter_spec=`, or the weight decay shrinks them.

The SIREN below has no frozen arrays, so the default `filter_spec` is right.

```{code-block} python
:caption: Minimises the parameterised cost [](#eq-param-cost) with x₀ a SIREN field on the ring of sites.
:label: code-strong4dvar-neural-field
import geonnax as gnx

# Shapes: S = 1 window; T = 10 steps, so T + 1 = 11 times; D_x = 40 sites;
# D_s = 2 coordinates

# ϑₙ = 2π n / D_x
angle = 2 * jnp.pi * site / 40  # (D_x,)
# cos ϑₙ
cos_angle = jnp.cos(angle)  # (D_x,)
# sin ϑₙ
sin_angle = jnp.sin(angle)  # (D_x,)
# sₙ = (cos ϑₙ, sin ϑₙ), the sites on the unit circle
coords = einx.id("n, n -> n (1 + 1)", cos_angle, sin_angle)  # (D_x,) → (D_x, D_s)
# today (geonnax)
# g_field(s; θ): ℝ² → ℝ, three hidden layers of width 32
siren = gnx.SIREN.init(2, 32, 1, depth=3, key=key_field)
# today (vardax)
# x₀ = g_dec(θ) = [g_field(sₙ; θ)]ₙ, ℛ(θ) = β/2 ‖θ‖², β = 10⁻⁴
field_param = vdx.NeuralFieldParameterization(siren, coords, 1e-4)
# today (vardax)
field_minimiser = vdx.OptaxMinimiser(optax.adam(1e-2), max_steps=2000)
# 𝒥(θ) = ℛ(θ) + ½ ∑ₜ ‖mₜ ⊙ (yₜ − fₜ(g_dec(θ)))‖²_Σ_ε⁻¹
field_model = dataclasses.replace(
    model, parameterization=field_param, minimiser=field_minimiser
)
# today (vardax)
# g_dec(θ⋆), θ⋆ = argmin 𝒥(θ)
field_analysis = field_model(batch)  # → (S, D_x)
print(f"RMSE of g_dec(θ⋆) {rmse(field_analysis[0]):.2f}")
```

It prints `RMSE of g_dec(θ⋆) 1.82`, worse than the background: the network starts from its random initialisation, not from $\gvect{\mu}_x$.

## Reduced-basis 4DVar

`ReducedBasisFourDVar(basis=...)` minimises this cost with the control moved into a basis, $\vect{x}_0 = \gvect{\mu}_x + \gmat{\Phi} \vect{z}$ with $D_z \ll D_x$ coefficients; it is the reduced strong-constraint rung of chapter [23](23_reduced_order.md), which owns its cost [](#eq-reduced-basis-cost), its gradient and the Lorenz-96 comparison against this chapter's full control.

## The adjoint problem

The gradient [](#eq-strong4dvar-grad) reads "integrate the trajectory forward, then propagate the observation residuals backward through the adjoint of the dynamics." In equations, this is the backward sweep

$$\vect{x}^\dagger_T = \mat{H}_T^\top \gmat{\Sigma}_{\varepsilon,T}^{-1} (h_T(\vect{x}_T) - \vect{y}_T)$$ (eq-adjoint-terminal)

$$\vect{x}^\dagger_{t-1} = \mat{F}_t^\top \vect{x}^\dagger_t + \mat{H}_{t-1}^\top \gmat{\Sigma}_{\varepsilon,t-1}^{-1} (h_{t-1}(\vect{x}_{t-1}) - \vect{y}_{t-1})$$ (eq-adjoint-recursion)

$$\nabla_{\vect{x}_0} \mathcal{J} = \gmat{\Sigma}_x^{-1}(\vect{x}_0 - \gvect{\mu}_x) + \vect{x}^\dagger_0$$ (eq-adjoint-grad)

where $\vect{x}^\dagger_t \in \mathbb{R}^{D_x}$ is the adjoint state at time $t$ and $\mat{F}_t = \partial f / \partial \vect{x}$ at $\vect{x}_{t-1}$ is the tangent-linear model of the single step from $t - 1$ to $t$, so that $\mat{F}_{t \leftarrow 0} = \mat{F}_t \ldots \mat{F}_1$ in [](#eq-strong4dvar-grad) is the product of these one-step tangents. This is exactly what `diffrax.RecursiveCheckpointAdjoint` or
`BacksolveAdjoint` compute under the hood. Vardax just composes them
into the cost-gradient call.

## Algorithm

Written out, the cost [](#eq-strong4dvar-cost) rolls the trajectory forward, sums the whitened misfit over the window and adds the background term, and `optimistix.minimise` differentiates through the rollout by reverse mode, which is the backward sweep [](#eq-adjoint-terminal) to [](#eq-adjoint-grad).
`StrongFourDVar` builds the same cost (with optional `jax.checkpoint` blocks and a `forward_adjoint`), so the hand-written one reaches the same minimum.
The block continues the one above.

```{code-block} python
:caption: Writes out the strong-constraint cost [](#eq-strong4dvar-cost) and minimises it with optimistix, whose reverse mode is the adjoint sweep [](#eq-adjoint-terminal) to [](#eq-adjoint-grad).
:label: code-strong4dvar-hand-cost
# Shapes: T = 10 steps, so T + 1 = 11 times; D_x = D_y = 40 sites

cov_solver = lx.Cholesky()


# user
def whiten(residual: Float[Array, " n"]) -> Float[Array, " n"]:
    """Apply Σ_ε⁻¹ to one time's residual with a lineax solve."""
    # Σ_ε⁻¹ r
    solution = lx.linear_solve(obs_cov, residual, solver=cov_solver)
    return solution.value  # (D_y,)


# user
def strong_cost(
    initial_state: Float[Array, " n"],
    window: tuple[Float[Array, "t n"], Float[Array, "t n"]],
) -> Float[Array, ""]:
    """Strong-constraint cost 𝒥(x₀) over one masked window."""
    window_obs, window_mask = window
    # xₜ = fₜ(x₀), t = 0 … T
    trajectory = rollout(initial_state, 10)  # (D_x,) → (T + 1, D_x)
    # yₜ − hₜ(xₜ), hₜ = I
    innovation = window_obs - trajectory  # (T + 1, D_y)
    # rₜ = mₜ ⊙ (yₜ − hₜ(xₜ))
    residual = window_mask * innovation  # (T + 1, D_y)
    # Σ_ε⁻¹ rₜ at every time
    weighted_residual = jax.vmap(whiten)(residual)  # (T + 1, D_y)
    # rₜᵀ Σ_ε⁻¹ rₜ summed over t
    obs_misfit = einx.dot("t n, t n ->", residual, weighted_residual)  # → ()
    # 𝒥_o = ½ ∑ₜ rₜᵀ Σ_ε⁻¹ rₜ
    obs_term = 0.5 * obs_misfit  # → ()
    # δx = x₀ − μ_x
    increment = initial_state - background  # (D_x,)
    # Σ_x⁻¹ δx
    prior_solve = lx.linear_solve(prior_cov, increment, solver=cov_solver)
    # δxᵀ Σ_x⁻¹ δx
    prior_misfit = einx.dot("n, n ->", increment, prior_solve.value)  # → ()
    # 𝒥_b = ½ δxᵀ Σ_x⁻¹ δx
    background_term = 0.5 * prior_misfit  # → ()
    # 𝒥(x₀) = 𝒥_b + 𝒥_o
    return background_term + obs_term  # → ()


# today (optimistix)
# x₀⋆ = argmin 𝒥(x₀), started at μ_x
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
# x₀⋆ = argmin 𝒥(x₀), with 𝒥(x₀⋆) and ‖∇𝒥(x₀⋆)‖ as diagnostics
result = minimiser.run(strong_cost, background, (obs, mask))
print(f"𝒥 = {result.cost:.1f}, ‖∇𝒥‖ = {result.grad_norm:.1e}")
```

It prints `True` and `𝒥 = 133.5, ‖∇𝒥‖ = 1.6e-04`.

## Linear-Gaussian agreement

When $h_t$ and $f$ are linear and $T = 0$, [](#eq-strong4dvar-cost) is the 3DVar cost and its minimum is the BLUE [](#eq-blue-mean), so `StrongFourDVar` must agree with `OptimalInterpolation` (the Decision D14 invariant, see [chapter 4](04_oi_blue.md#validation)). The block continues the one above with a single-time window and $f(\vect{x}) = \vect{x}$:

```{code-block} python
:caption: Checks that the strong-constraint minimum of [](#eq-strong4dvar-cost) with T = 0 and f = I equals the BLUE [](#eq-blue-mean).
:label: code-strong4dvar-oi-agreement
# Shapes: S = 1 window; T = 0 steps, so T + 1 = 1 time; D_x = D_y = 40 sites


# user
class Static(eqx.Module):
    """f(x) = x, so a single-time window reduces 4DVar to OI."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # f(x) = x
        return state


psd = lx.positive_semidefinite_tag
# diag(Σ_x) = 2
prior_diag = jnp.full(40, 2.0)  # (D_x,)
# diag(Σ_ε) = 1/2
obs_diag = jnp.full(40, 0.5)  # (D_y,)
# Σ_x = 2 I, tagged PSD so the default CG solve accepts it
state_prior_cov = lx.TaggedLinearOperator(  # (D_x, D_x) PSD op
    lx.DiagonalLinearOperator(prior_diag), psd
)
# Σ_ε = I / 2
state_obs_cov = lx.TaggedLinearOperator(  # (D_y, D_y) PSD op
    lx.DiagonalLinearOperator(obs_diag), psd
)
# the same Σ_x and Σ_ε over OI's (T + 1, D_x) state
field_prior_diag = jnp.full((1, 40), 2.0)  # (T + 1, D_x)
field_obs_diag = jnp.full((1, 40), 0.5)  # (T + 1, D_y)
field_prior_cov = lx.TaggedLinearOperator(  # ((T+1) D_x, (T+1) D_x) PSD op
    lx.DiagonalLinearOperator(field_prior_diag), psd
)
field_obs_cov = lx.TaggedLinearOperator(  # ((T+1) D_y, (T+1) D_y) PSD op
    lx.DiagonalLinearOperator(field_obs_diag), psd
)
# y₀ and m₀ only
first_obs = batch.input[:, :1]  # (S, T + 1, D_y)
first_mask = batch.mask[:, :1]  # (S, T + 1, D_y)
# today (vardax)
first_time = vdx.Batch1D(input=first_obs, mask=first_mask)
# today (vardax)
# 𝒥(x₀) = ½ ‖x₀‖²_Σ_x⁻¹ + ½ ‖m₀ ⊙ (y₀ − x₀)‖²_Σ_ε⁻¹
strong_static = vdx.StrongFourDVar(
    forward=Static(),
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros(40),
    prior_cov_op=state_prior_cov,
    obs_cov_op=state_obs_cov,
)
# today (vardax)
# p(x) = 𝒩(0, Σ_x), p(y ∣ x) = 𝒩(m ⊙ x, Σ_ε)
optimal_interpolation = vdx.OptimalInterpolation(
    obs_op=vdx.MaskedIdentity(),
    prior_mean=jnp.zeros((1, 40)),
    prior_cov_op=field_prior_cov,
    obs_cov_op=field_obs_cov,
)
# today (vardax)
# x⋆ = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x)
closed_form = optimal_interpolation(first_time)  # → (S, T + 1, D_x)
# today (vardax)
# x₀⋆ = argmin 𝒥(x₀)
iterative = strong_static(first_time)  # → (S, D_x)
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

The 4DVar cost [](#eq-strong4dvar-cost) is nonconvex when $f$ is nonlinear, so multiple local minima exist (and matter, especially for chaotic systems like Lorenz). Strategies:

- **Multi-start.** Minimise from several perturbed first guesses and
  keep the lowest cost. vardax has no helper for this yet: the models
  start every minimisation from $\gvect{\mu}_x$, so a multi-start needs a first
  guess separate from the background.
- **Quasi-static gradient.** Use a warm-start from a previous cycle
  (the usual operational pattern).
- **Switch to `IncrementalFourDVar`** (chapter 8): the outer Gauss-Newton iterations relinearise around the current iterate, which is more robust than NonlinearCG on the full nonlinear cost.

## Posterior

Pair the analysis with a `PosteriorAdapter`; `LaplaceCovariance` returns the Gauss-Newton covariance [](#eq-gn-cov) of chapter 13 as a lazy operator. The block continues the one above.
It passes `solver=lx.Cholesky()` because the default CG solve of $\gmat{\Sigma}_x^{-1}$ reaches its step cap on this $\gmat{\Sigma}_x$; `model.posterior(batch, adapter=laplace)` is the same call.

```{code-block} python
:caption: Builds the Gauss-Newton posterior covariance [](#eq-gn-cov) at the analysis of [](#code-strong4dvar-analysis) and applies it to one vector.
:label: code-strong4dvar-laplace
# Shapes: D_x = 40 sites

# today (vardax)
laplace = vdx.LaplaceCovariance(solver=lx.Cholesky())
# today (vardax)
analysis_step = model.as_analysis_step()
# today (vardax)
# Σ_x|y = (Σ_x⁻¹ + ∑ₜ (Hₜ F_t←0)ᵀ Σ_ε⁻¹ (Hₜ F_t←0))⁻¹ at x₀⋆
posterior = laplace(analysis[0], analysis_step, batch)
# 1
ones = jnp.ones(40)  # (D_x,)
# Σ_x|y 1, one matrix-free product
spread = posterior.cov.mv(ones)  # (D_x,) → (D_x,)
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
