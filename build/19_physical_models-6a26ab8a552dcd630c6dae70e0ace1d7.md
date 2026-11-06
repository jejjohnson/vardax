---
numbering:
  enumerator: "19.%s"
---

# Physical Models & ODE Priors

Ported and condensed from the `mfourdvar` physical-models notes
(`content/physical_models/`), rewired to the shipped vardax API.

Chapter [3](03_dynamical_model.md) established the interface: vardax does not own dynamics, it owns the seam (`pipekit_cycle.ForwardModel`) and the adjoint composition. This chapter answers the two questions that seam leaves open in practice: **which physical model** should sit behind it at each stage of a project, and **how an ODE right-hand side becomes a prior**, through the `DynamicalPrior` family (Decision D18), which turns "the state should obey the dynamics" into a differentiable cost term.

## Choosing a testbed model

Every assimilation method in this library is exercised against a
ladder of physical models before it touches real data. Four selection
criteria drive the choice of rung:

- **Chaotic**: the model must exhibit sensitive dependence on initial conditions, otherwise the assimilation problem is trivially easy and says nothing about real geophysical use.
- **Coupled**: for parameter-estimation and parameterisation-learning studies, the model needs a term that can be withheld (a missing forcing, or an unobserved fast state in a multi-level system).
- **2-D spatiotemporal structure**: convolutional priors and the 2-D model classes (`FourDVarNet2D`, `Batch2D`) need genuinely two-dimensional fields.
- **Scale**: eventually the method must survive state dimensions where dense covariances are impossible and matrix-free structure (chapter [13](13_posterior_covariance.md)) is mandatory.

### The ladder

| Model | State | Structure | Good for | Not for |
|---|---|---|---|---|
| Lorenz-63 | $\mathbb{R}^3$ | none | smoke tests, visualisation | anything spatial |
| Lorenz-96 | $\mathbb{R}^{D_x}$ (ring) | 1-D periodic | prototyping, interpretability, low engineering cost | 2-D structure, scale |
| Lorenz-96 two-level | $\mathbb{R}^{D_x + D_x N_\text{fast}}$ | 1-D, two timescales | coupled parameterisation learning, partial observation | 2-D structure |
| Shallow water | $(h, u, v)$ on a grid | 2-D | wave dynamics, linearisation studies | eddy statistics |
| Stacked quasi-geostrophy | $(q_k, \psi_k)$, $N_Z$ layers | 2-D, multi-layer | mesoscale turbulence, realistic SSH proxies | full-physics fidelity |
| Ocean GCM (NEMO, MOM6, …) | full ocean state | 3-D | production reanalysis | differentiable end-to-end use |

vardax ships the first two rungs (`Lorenz63`, `Lorenz96`, `simulate_lorenz63`, `simulate_lorenz96`); everything above them lives in dedicated model libraries (`somax` for geophysical fluids) and plugs in through `ForwardModel` (Decision D7).

### Lorenz-96, one and two levels

The single-level system on a periodic ring of $D_x$ variables, [](#eq-lorenz96), is chaotic for $F = 8$, cheap, and interpretable, which makes it the standard prototyping rung used throughout the [Lorenz examples](15_lorenz_examples.md). Its two-level extension couples each slow variable $x_i$ (the $X_k$ of [](#eq-lorenz96)) to $N_\text{fast}$ fast variables $y_j$:

$$\begin{aligned} \frac{\mathrm{d} x_i}{\mathrm{d} t} &= (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F - \frac{h c}{b} \sum_{j} y_j, \\ \frac{\mathrm{d} y_j}{\mathrm{d} t} &= -b c (y_{j+2} - y_{j-1}) y_{j+1} - c y_j + \frac{h c}{b} x_{\lceil j / N_\text{fast} \rceil}. \end{aligned}$$ (eq-lorenz96-two-level)

where $x_i$ for $i = 1, \ldots, D_x$ are the slow and $y_j$ for $j = 1, \ldots, D_x N_\text{fast}$ the fast variables (scalars, not the observations), $F$ is the forcing, $h$ the coupling strength, $c$ the ratio of fast to slow time scales and $b$ the ratio of slow to fast amplitudes, and the sum runs over the $N_\text{fast}$ fast variables attached to $x_i$.

Observing only $x$ while the fast $y$ dynamics act as unresolved
physics is the minimal faithful model of the parameterisation-learning
problem: the coupling term is exactly the kind of "missing physics"
a learnable ODE parameter $\gvect{\theta}$ (below) is meant to absorb.

### Shallow water

The linearised shallow-water system for height $h$ and velocities
$(u, v)$ on a rotating plane,

$$\begin{aligned} \partial_t h &+ H \left(\partial_x u + \partial_y v \right) = 0, \\ \partial_t u &- f_0 v = -g \partial_x h - \kappa u, \\ \partial_t v &+ f_0 u = -g \partial_y h - \kappa v, \end{aligned}$$ (eq-shallow-water)

where $(x, y)$ are the horizontal coordinates, $H$ is the mean depth, $f_0$ the Coriolis parameter, $g$ gravity and $\kappa$ a linear drag. It is the first genuinely 2-D rung: wave propagation, geostrophic adjustment, and a clean linear operator to test tangent-linear and adjoint machinery (chapter [12](12_adjoint_methods.md)) against an analytic reference.

### Stacked quasi-geostrophy

For mesoscale ocean turbulence, the regime behind the [SSH application](21_oceanbench.md), the workhorse is multi-layer QG in vorticity-streamfunction form, with $N_Z$ stacked isopycnal layers:

$$\partial_t q_k + (u_k q_k)_x + (v_k q_k)_y = F_k + D_k, \text{ with } q = \frac{1}{f_0} \nabla_H^2 \psi - f_0 \mat{A} \psi + \beta (y - y_0) + \tilde{\vect{D}},$$ (eq-qg)

where $q_k$ and $\psi_k$ are the potential vorticity and streamfunction of layer $k$, $(u_k, v_k)$ its velocity, $f_0$ and $\beta$ the Coriolis parameter and its meridional gradient, $F_k$ and $D_k$ the per-layer forcing and dissipation, $\tilde{\vect{D}}$ the dynamic topography, and $\mat{A} \in \mathbb{R}^{N_Z \times N_Z}$ the tri-diagonal layer-coupling matrix built from layer depths $H_k$ and reduced gravities $g_k'$. QG produces realistic eddy fields at a fraction of a GCM's cost, which is why the OSSE ground truths in chapter [21](21_oceanbench.md) are QG or NEMO simulations.

### Ocean GCMs

Full GCMs (NEMO, MOM6) anchor the top of the ladder, but converting such systems wholesale into differentiable models is a massive engineering effort (attempted rebuilds exist, e.g. Veros, and autodiff conversions of individual cores), and back-propagating through an entire GCM is rarely feasible or even useful in a learning loop. The practical route the mfourdvar notes converge on, and the one
this library's boundaries assume, is **component surrogacy**: train
fast differentiable emulators of individual subsystems and compose
them behind `ForwardModel`, keeping the full GCM for producing
training data and reference reanalyses.

## From model to prior: the `DynamicalPrior` family

A physical model enters the variational problem in one of two roles. As the **forward operator** it generates the trajectory that the observation term scores, which is the strong-constraint pattern [](#eq-strong4dvar-cost). As a **prior** it penalises state sequences that disobey the dynamics while the state itself remains free, which is the weak-constraint pattern [](#eq-weak4dvar-cost). The
`DynamicalPrior` classes (ported from mfourdvar, Decision D18) package
a diffrax-compatible ODE right-hand side
$f(t, \vect{x}; \gvect{\theta})$ for both roles.

### Two residuals

`DynIncrements` scores local, one-step consistency: each state is
integrated a single step and compared to its successor,

$$\mathcal{J}_p(\vect{x}_{0:T}; \gvect{\theta}) = \sum_{t=0}^{T-1} \Vert \vect{x}_{t+1} - \Psi_{\Delta t}(\vect{x}_t; \gvect{\theta}) \Vert^2$$ (eq-residual-one-step)

where $\vect{x}_{0:T} \in \mathbb{R}^{(T + 1) \times D_x}$ is the state sequence over the window of $T$ steps (so $T + 1$ times), $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ the ODE parameters, $f: \mathbb{R} \times \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ the right-hand side and $\Psi_{\Delta t}(\cdot; \gvect{\theta}): \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ integrates $f$ over one step $\Delta t$.

`DynTrajectory` scores global consistency: one rollout from the
initial state, compared along the whole window,

$$\mathcal{J}_p(\vect{x}_{0:T}; \gvect{\theta}) = \sum_{t=0}^{T} \Vert \vect{x}_t - \Psi_{t \Delta t}(\vect{x}_0; \gvect{\theta}) \Vert^2$$ (eq-residual-rollout)

where $\Psi_{t \Delta t}(\cdot; \gvect{\theta})$ integrates $f$ from the initial time over $t$ steps of length $\Delta t$, the rollout [](#eq-rollout).

The increment form tolerates model error accumulating over the window
(weak-constraint flavour); the trajectory form is the hard-constraint
propagation used by
[`strong_variational_cost`](xref:api#vardax.strong_variational_cost).

The example below runs on a ring of $D_x = 8$ sites over a window of $T = 10$ steps, which holds $T + 1 = 11$ times.

```{code-block} python
:caption: Rolls Lorenz-96 [](#eq-lorenz96) out with [](#eq-rollout) and scores the rollout and a perturbed copy with the increment residual [](#eq-residual-one-step).
:label: code-physical-l96-residual
import jax
import jax.numpy as jnp
import vardax as vdx

# Shapes: T = 10 steps, so T + 1 = 11 times; D_x = 8 ring sites

key = jax.random.PRNGKey(0)
key_initial, key_perturb = jax.random.split(key)

# today (vardax)
# f(t, x) = (x_{i+1} − x_{i−2}) x_{i−1} − x_i + F, with F = 8
rhs = vdx.Lorenz96(forcing=8.0)
# t₀, …, t₁₀ on [0, 0.5]
times = jnp.linspace(0.0, 0.5, 11)  # (T + 1,)
# ε ~ 𝒩(0, I)
initial_noise = jax.random.normal(key_initial, (8,))  # (D_x,)
# x₀ = F + ε
initial_state = 8.0 + initial_noise  # (D_x,)

# today (vardax)
rollout = vdx.DynTrajectory(model=rhs)
# today (vardax)
# xₜ = Ψ_{tΔt}(x₀)
trajectory = rollout(initial_state, times)  # (D_x,) → (T + 1, D_x)

# today (vardax)
prior = vdx.DynIncrements(model=rhs)
# today (vardax)
# 𝒥_p(x) = Σₜ ‖x_{t+1} − Ψ_Δt(xₜ)‖²
consistent_residual = prior.loss(trajectory, times)  # → ()
# δ ~ 𝒩(0, I)
perturbation = jax.random.normal(key_perturb, (11, 8))  # (T + 1, D_x)
# x_pert = x + 0.1 δ
perturbed = trajectory + 0.1 * perturbation  # (T + 1, D_x)
# today (vardax)
# 𝒥_p(x_pert)
perturbed_residual = prior.loss(perturbed, times)  # → ()
print(f"𝒥_p(x) = {consistent_residual:.1e}, 𝒥_p(x_pert) = {perturbed_residual:.2f}")
```

It prints `𝒥_p(x) = 3.3e-08, 𝒥_p(x_pert) = 1.93`: the rollout is consistent with the dynamics to solver tolerance and the perturbed sequence is not.

Solver, step-size controller, and adjoint are pluggable
(`solver=`, `stepsize=`, `adjoint=`); the defaults are `Tsit5` with an
adaptive PID controller, falling back to a constant step for
fixed-step solvers, and `RecursiveCheckpointAdjoint` for reverse-mode memory control, the same knobs discussed in chapter [12](12_adjoint_methods.md).

### Learnable physics

The ODE parameters $\gvect{\theta}$ thread through every call as `params`
(diffrax `args`), and gradients flow through the solve:

`Lorenz96` holds its forcing as a field and ignores `args`, so a learnable forcing needs a right-hand side that reads $\gvect{\theta}$ from `args`.

```{code-block} python
:caption: Differentiates the increment residual [](#eq-residual-one-step) in the forcing $\gvect{\theta} = F$ of a learnable Lorenz-96 right-hand side.
:label: code-physical-learnable-forcing
from jaxtyping import Array, Float

# Shapes: T = 10 steps, so T + 1 = 11 times; D_x = 8 ring sites


# user
def lorenz96_rhs(
    time: Float[Array, ""], state: Float[Array, " n"], forcing: Float[Array, ""]
) -> Float[Array, " n"]:
    # x_{i+1}
    state_ahead = jnp.roll(state, -1)
    # x_{i−2}
    state_two_behind = jnp.roll(state, 2)
    # x_{i−1}
    state_behind = jnp.roll(state, 1)
    # x_{i+1} − x_{i−2}
    difference = state_ahead - state_two_behind
    # (x_{i+1} − x_{i−2}) x_{i−1}
    advection = difference * state_behind
    # (x_{i+1} − x_{i−2}) x_{i−1} − x_i
    damped = advection - state
    # dx_i/dt = (x_{i+1} − x_{i−2}) x_{i−1} − x_i + θ, with θ = F from args
    return damped + forcing


# today (vardax)
learnable_prior = vdx.DynIncrements(model=lorenz96_rhs)


# user
def residual_in_forcing(forcing: Float[Array, ""]) -> Float[Array, ""]:
    # today (vardax)
    # 𝒥_p(x; θ) = Σₜ ‖x_{t+1} − Ψ_Δt(xₜ; θ)‖²
    return learnable_prior.loss(trajectory, times, params=forcing)


# ∂𝒥_p(x; θ)/∂θ
residual_gradient = jax.grad(residual_in_forcing)
trial_forcing = jnp.float32(7.0)  # → ()
# ∂𝒥_p(x; θ)/∂θ at θ = 7, for the trajectory generated with F = 8
forcing_gradient = residual_gradient(trial_forcing)  # → ()
print(f"d𝒥_p/dθ at θ = 7: {forcing_gradient:.2f}")
```

It prints `d𝒥_p/dθ at θ = 7: -0.38`; the negative gradient points towards the forcing $F = 8$ that generated the trajectory.

This is the parameter-estimation seam: fit $\gvect{\theta}$ (a forcing, a
drag coefficient, a neural closure's weights) by minimising the
dynamical residual of observed or analysed trajectories.
Notebook [09](notebooks/09_param_estimation_L63.ipynb) walks through it on
Lorenz-63, including joint state-parameter estimation and an
identifiability check.

### One prior, three seams

The same object plugs into all three integration points of the
library:

```{code-block} python
:caption: Uses one dynamical prior as a temporal residual [](#eq-residual-one-step), as the prior term of the 4DVarNet-style cost [](#eq-4dvarnet-cost), and as a forward model.
:label: code-physical-three-seams
import einx

# Shapes: S = 1 sample; T = 10 steps, so T + 1 = 11 times; D_x = 8 ring sites

# 1. TemporalPrior, the native two-argument seam
# today (vardax)
# 𝒥_p(x_pert) = Σₜ ‖x_{t+1} − Ψ_Δt(xₜ)‖², on x_pert
temporal_residual = prior.loss(perturbed, times)  # → ()

# 2. Prior: bind the time grid and drop it into the weak-constraint cost
trajectory_window = einx.id(  # (T + 1, D_x) → (S, T + 1, D_x)
    "t n -> 1 t n", trajectory
)
# m = 1 everywhere
full_mask = jnp.ones((1, 11, 8))  # (S, T + 1, D_x)
# today (vardax)
batch = vdx.Batch1D(input=trajectory_window, mask=full_mask)
# today (vardax)
# x̂ = [x₀, Ψ_Δt(x₀), …, Ψ_Δt(x_{T−1})]
bound_prior = prior.bind(times)
batched_prior = jax.vmap(bound_prior)
perturbed_window = einx.id(  # (T + 1, D_x) → (S, T + 1, D_x)
    "t n -> 1 t n", perturbed
)
# today (vardax)
# 𝒥(x) = α_o 𝒥_o + α_p 𝒥_p
# 𝒥_o = mean((m ⊙ (x − y))²), 𝒥_p = mean((x − x̂)²)
weak_cost = vdx.variational_cost(perturbed_window, batch, batched_prior)  # → ()

# 3. pipekit ForwardModel, to drive strong 4DVar or a DA cycle
# today (vardax)
forward_model = prior.as_forward_model(dt=0.05)
# today (vardax)
# x₁ = Ψ_Δt(x₀), with Δt = 0.05
next_state = forward_model.step(initial_state, forward_model.dt)  # (D_x,) → (D_x,)
print(f"{temporal_residual:.2f} {weak_cost:.4f} {next_state.shape}")
```

It prints `1.93 0.0169 (8,)`.

`bind(ts)` closes over the window's time grid so the dynamical prior satisfies the one-argument [`Prior`](xref:api#vardax.Prior) protocol, turning the $\Vert \vect{x} - \hat{\vect{x}} \Vert^2$ term (with $\hat{\vect{x}}$ the prior's output, $g_\text{ae}(\vect{x}; \gvect{\theta})$ in 4DVarNet) of [`variational_cost`](xref:api#vardax.variational_cost) ([](#eq-4dvarnet-cost)) into a weak-constraint dynamical residual with no API change. `as_forward_model(dt)` adapts
the wrapped ODE to `pipekit_cycle.ForwardModel` (autonomous dynamics
assumed), so the same physics can serve as the forward operator of
[`StrongFourDVar`](06_strong_4dvar.md) or a `pipekit_cycle.DACycle`.
