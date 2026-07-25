# Physical Models & ODE Priors

Ported and condensed from the `mfourdvar` physical-models notes
(`content/physical_models/`), rewired to the shipped vardax API.

Chapter [3](03_dynamical_model.md) established the interface: vardax
does not own dynamics, it owns the seam (`pipekit_cycle.ForwardModel`)
and the adjoint composition. This chapter answers the two questions
that seam leaves open in practice: **which physical model** should sit
behind it at each stage of a project, and **how an ODE right-hand side
becomes a prior** — the `DynamicalPrior` family (Decision D18), which
turns "the state should obey the dynamics" into a differentiable cost
term.

## Choosing a testbed model

Every assimilation method in this library is exercised against a
ladder of physical models before it touches real data. Four selection
criteria drive the choice of rung:

- **Chaotic** — the model must exhibit sensitive dependence on initial
  conditions, otherwise the assimilation problem is trivially easy and
  says nothing about real geophysical use.
- **Coupled** — for parameter-estimation and parameterisation-learning
  studies, the model needs a term that can be withheld (a missing
  forcing, or an unobserved fast state in a multi-level system).
- **2-D spatiotemporal structure** — convolutional priors and the 2-D
  model classes (`FourDVarNet2D`, `Batch2D`) need genuinely
  two-dimensional fields.
- **Scale** — eventually the method must survive state dimensions
  where dense covariances are impossible and matrix-free structure
  (chapter [13](13_posterior_covariance.md)) is mandatory.

### The ladder

| Model | State | Structure | Good for | Not for |
|---|---|---|---|---|
| Lorenz-63 | $\mathbb{R}^3$ | none | smoke tests, visualisation | anything spatial |
| Lorenz-96 | $\mathbb{R}^N$ (ring) | 1-D periodic | prototyping, interpretability, low engineering cost | 2-D structure, scale |
| Lorenz-96 two-level | $\mathbb{R}^{N + NJ}$ | 1-D, two timescales | coupled parameterisation learning, partial observation | 2-D structure |
| Shallow water | $(h, u, v)$ on a grid | 2-D | wave dynamics, linearisation studies | eddy statistics |
| Stacked quasi-geostrophy | $(q_k, \psi_k)$, $N_Z$ layers | 2-D, multi-layer | mesoscale turbulence, realistic SSH proxies | full-physics fidelity |
| Ocean GCM (NEMO, MOM6, …) | full ocean state | 3-D | production reanalysis | differentiable end-to-end use |

vardax ships the first two rungs (`Lorenz63`, `Lorenz96`,
`simulate_lorenz63`, `simulate_lorenz96`); everything above them lives
in dedicated model libraries (`somax` for geophysical fluids) and
plugs in through `ForwardModel` — Decision D7.

### Lorenz-96, one and two levels

The single-level system on a periodic ring of $N$ variables,

$$
\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\,x_{i-1} - x_i + F,
$$

is chaotic for $F = 8$, cheap, and interpretable — the standard
prototyping rung, used throughout the
[Lorenz examples](15_lorenz_examples.md). Its two-level extension
couples each slow variable $x_i$ to $J$ fast variables $y_j$:

$$
\begin{aligned}
\frac{dx_i}{dt} &= (x_{i+1} - x_{i-2})\,x_{i-1} - x_i + F
                  - \frac{h c}{b} \sum_{j} y_j, \\
\frac{dy_j}{dt} &= -b c\,(y_{j+2} - y_{j-1})\,y_{j+1} - c\,y_j
                  + \frac{h c}{b}\, x_{\lceil j/J \rceil}.
\end{aligned}
$$

Observing only $x$ while the fast $y$ dynamics act as unresolved
physics is the minimal faithful model of the parameterisation-learning
problem: the coupling term is exactly the kind of "missing physics"
a learnable ODE parameter $\theta$ (below) is meant to absorb.

### Shallow water

The linearised shallow-water system for height $h$ and velocities
$(u, v)$ on a rotating plane,

$$
\begin{aligned}
\partial_t h &+ H \left(\partial_x u + \partial_y v \right) = 0, \\
\partial_t u &- f v = -g\, \partial_x h - \kappa u, \\
\partial_t v &+ f u = -g\, \partial_y h - \kappa v,
\end{aligned}
$$

is the first genuinely 2-D rung: wave propagation, geostrophic
adjustment, and a clean linear operator to test tangent-linear /
adjoint machinery (chapter [12](12_adjoint_methods.md)) against an
analytic reference.

### Stacked quasi-geostrophy

For mesoscale ocean turbulence — the regime behind the
[SSH application](21_oceanbench.md) — the workhorse is multi-layer QG
in vorticity–streamfunction form, with $N_Z$ stacked isopycnal
layers:

$$
\partial_t q_k + (u_k q_k)_x + (v_k q_k)_y = F_k + D_k,
\qquad
q = \frac{1}{f_0} \nabla_H^2 \psi - f_0 \mathbf{A} \psi
    + \beta (y - y_0) + \tilde{\mathbf{D}},
$$

where $F_k$, $D_k$ are per-layer forcing and dissipation,
$\tilde{\mathbf{D}}$ is dynamic topography, and $\mathbf{A}$ is the
tri-diagonal layer-coupling matrix built from layer depths $H_k$ and
reduced gravities $g_k'$. QG produces realistic eddy fields at a
fraction of a GCM's cost, which is why the OSSE ground truths in
chapter [21](21_oceanbench.md) are QG or NEMO simulations.

### Ocean GCMs

Full GCMs (NEMO, MOM6) anchor the top of the ladder, but converting
such systems wholesale into differentiable models is a massive
engineering effort (attempted rebuilds exist — e.g. Veros — and
autodiff conversions of individual cores), and back-propagating
through an entire GCM is rarely feasible or even useful in a learning
loop. The practical route the mfourdvar notes converge on, and the one
this library's boundaries assume, is **component surrogacy**: train
fast differentiable emulators of individual subsystems and compose
them behind `ForwardModel`, keeping the full GCM for producing
training data and reference reanalyses.

## From model to prior: the `DynamicalPrior` family

A physical model enters the variational problem in one of two roles.
As the **forward operator** it generates the trajectory that the
observation term scores — the strong-constraint pattern of chapter
[6](06_strong_4dvar.md). As a **prior** it penalises state sequences
that disobey the dynamics while the state itself remains free — the
weak-constraint pattern of chapter [7](07_weak_4dvar.md). The
`DynamicalPrior` classes (ported from mfourdvar, Decision D18) package
a diffrax-compatible ODE right-hand side
$f(t, x; \theta)$ for both roles.

### Two residuals

`DynIncrements` scores local, one-step consistency: each state is
integrated a single step and compared to its successor,

$$
R(u; \theta) = \sum_t \bigl\| u_{t+1} - \varphi_{\Delta t}(u_t; \theta) \bigr\|^2 .
$$

`DynTrajectory` scores global consistency: one rollout from the
initial state, compared along the whole window,

$$
R(u; \theta) = \sum_t \bigl\| u_t - \varphi_t(u_0; \theta) \bigr\|^2 .
$$

The increment form tolerates model error accumulating over the window
(weak-constraint flavour); the trajectory form is the hard-constraint
propagation used by
[`strong_variational_cost`](api/costs_priors.md).

```python
import jax.numpy as jnp
from vardax import DynIncrements, DynTrajectory, Lorenz96

rhs = Lorenz96(F=8.0)                     # any f(t, y, args) -> dy/dt
ts = jnp.linspace(0.0, 0.5, 11)

prior = DynIncrements(model=rhs)
r = prior.loss(x, ts)                     # Σₜ ‖x_{t+1} − φ_Δt(x_t)‖²

rollout = DynTrajectory(model=rhs)
traj = rollout(x0, ts)                    # (T, N) trajectory from x0
```

Solver, step-size controller, and adjoint are pluggable
(`solver=`, `stepsize=`, `adjoint=`); the defaults are `Tsit5` with an
adaptive PID controller, falling back to a constant step for
fixed-step solvers, and `RecursiveCheckpointAdjoint` for reverse-mode
memory control — the same knobs discussed in chapter
[12](12_adjoint_methods.md).

### Learnable physics

The ODE parameters $\theta$ thread through every call as `params`
(diffrax `args`), and gradients flow through the solve:

```python
import jax

prior = DynIncrements(model=rhs)
grad_theta = jax.grad(lambda p: prior.loss(x, ts, params=p))(theta_0)
```

This is the parameter-estimation seam: fit $\theta$ (a forcing, a
drag coefficient, a neural closure's weights) by minimising the
dynamical residual of observed or analysed trajectories.

### One prior, three seams

The same object plugs into all three integration points of the
library:

```python
# 1. TemporalPrior — native two-argument seam
prior.loss(x, ts)

# 2. Prior — bind the time grid, drop into the weak-constraint cost
from vardax import variational_cost
cost = variational_cost(x, batch, jax.vmap(prior.bind(ts)))

# 3. pipekit ForwardModel — drive strong 4DVar or a DA cycle
fwd = prior.as_forward_model(dt=0.05)     # .step / .dt / .state_signature
```

`bind(ts)` closes over the window's time grid so the dynamical prior
satisfies the one-argument [`Prior`](api/protocols.md) protocol —
turning the $\lVert x - \varphi(x)\rVert^2$ term of
[`variational_cost`](api/costs_priors.md) into a weak-constraint
dynamical residual with no API change. `as_forward_model(dt)` adapts
the wrapped ODE to `pipekit_cycle.ForwardModel` (autonomous dynamics
assumed), so the same physics can serve as the forward operator of
[`StrongFourDVar`](06_strong_4dvar.md) or a `pipekit_cycle.DACycle`.
