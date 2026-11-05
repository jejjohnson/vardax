---
numbering:
  enumerator: "3.%s"
---

# Dynamical Model

For 4D analysis, observations $\vect{y}_t \in \mathbb{R}^{D_y}$ at times $t = 0, 1, \ldots, T$ must be connected to the initial state $\vect{x}_0 \in \mathbb{R}^{D_x}$ through the rollout [](#eq-rollout), where $f_t$ propagates the state forward from time $0$ to time $t$. The dynamics enter the cost as the composition $h_t \circ f_t$.

vardax does not own dynamics; those come from `somax` (geophysical fluids), `plumax` (atmospheric transport and methane) or any user code. What vardax owns is the **interface** to dynamics (the `pipekit_cycle.ForwardModel` protocol) and the **adjoint composition** (delegated to `diffrax.AbstractAdjoint`).

## The `ForwardModel` protocol

A forward supplies `step(state, dt)`, a default `dt` and a `state_signature`. The Lorenz-63 model below, with state $\vect{x} = (x_1, x_2, x_3) \in \mathbb{R}^{D_x}$ standing for the $(x, y, z)$ of [](#eq-lorenz63) and the vector field $f(t, \vect{x})$ of that equation (written with its arguments, to tell it from the rollout $f_t$), with the parameters $\sigma$, $\rho$ and $\beta$ of Lorenz-63, advances by forward Euler steps and satisfies the protocol structurally, and the check prints `True`; the later blocks on this page build on it.

```{code-block} python
:caption: Defines Lorenz-63 [](#eq-lorenz63) as a forward-Euler forward model and checks that it satisfies the protocol.
:label: code-dynamical-model-lorenz63
import equinox as eqx
import jax.numpy as jnp
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: D_x = 3 state components


# user
class Lorenz63(eqx.Module):
    """Lorenz-63 dynamics, advanced by forward Euler steps."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01

    def vector_field(self, state: Float[Array, " state"]) -> Float[Array, " state"]:
        x1, x2, x3 = state
        # f(t, x) = (σ (x₂ − x₁), x₁ (ρ − x₃) − x₂, x₁ x₂ − β x₃)
        return jnp.stack(
            [
                self.sigma * (x2 - x1),
                x1 * (self.rho - x3) - x2,
                x1 * x2 - self.beta * x3,
            ]
        )

    def step(self, state: Float[Array, " state"], dt: float) -> Float[Array, " state"]:
        """Advance state by dt."""
        # Ψ_Δt(x) ≈ x + Δt f(t, x), a forward Euler step
        return state + dt * self.vector_field(state)

    @property
    def state_signature(self) -> None:
        return None


dynamics = Lorenz63()
# today (pipekit_cycle)
print(isinstance(dynamics, pc.ForwardModel))
```

`somax.ShallowWaterModel`, `plumax.GaussianPlumeForward`, and
user-defined dynamics all satisfy this protocol natively. From vardax's
perspective they're interchangeable.

For multi-step rollouts, vardax composes `step` calls into a trajectory with a `lax.scan`; a forward that is an ODE integrates each step with `diffrax.diffeqsolve` itself. The choice of ODE integrator
(`Tsit5`, `Dopri5`, `Heun`, …) is on the dynamics, not on vardax.

## Why the adjoint matters

The strong-constraint 4DVar cost [](#eq-strong4dvar-cost) sums an observation term over every time. Minimising it over $\vect{x}_0$ requires the gradient [](#eq-strong4dvar-grad), which by the chain rule pulls each weighted residual $\mat{H}_t^\top \gmat{\Sigma}_{\varepsilon,t}^{-1} (h_t(f_t(\vect{x}_0)) - \vect{y}_t)$ back to the initial time through $\mat{F}_t^\top$, where $\mat{F}_t \in \mathbb{R}^{D_x \times D_x}$ and $\mat{H}_t \in \mathbb{R}^{D_y \times D_x}$ are the tangent-linear dynamics and observation operator.

The transpose $\mat{F}_t^\top$ is the **adjoint model**, which was historically hand-coded line by line (Talagrand and Courtier 1987, Errico 1997). Modern numerical libraries make this far less painful: `diffrax` provides the adjoint of any `diffeqsolve` call automatically, in several flavours.

## diffrax adjoint methods

vardax exposes `forward_adjoint: diffrax.AbstractAdjoint` as a
constructor slot on `StrongFourDVar`, `WeakFourDVar`,
`IncrementalFourDVar` and `ReducedBasisFourDVar`. It reaches `diffrax`
through the forward: the model calls `forward.with_adjoint(adjoint)`
(the [`SupportsForwardAdjoint`](xref:api#vardax.SupportsForwardAdjoint) protocol) and rolls
out the forward it returns. A per-step forward has no integrator to hand
the adjoint to, so passing `forward_adjoint=` with one raises
`ValueError`; bound its memory with `checkpoint=` instead (below).

| Adjoint | Memory | Time | Convergence requirement | When |
|---|---|---|---|---|
| `diffrax.RecursiveCheckpointAdjoint(checkpoints=N_ckpt)` | $O(N_\text{ckpt})$ | $O(\log T)$ extra forward solves | None | Default; balanced |
| `diffrax.BacksolveAdjoint()` | $O(1)$ | Backwards ODE solve | None (but reverse-time stability matters) | Long windows, memory-constrained |
| `diffrax.ForwardMode()` | $O(D_\theta)$ in parameters | One extra forward per param | None | Few parameters (parameter estimation) |
| `diffrax.DirectAdjoint()` | $O(T)$ steps | Standard reverse | None | Short windows where memory is cheap |

**Default**: `RecursiveCheckpointAdjoint`. Standard autodiff with recursive checkpointing, which balances memory and recompute. It supports reverse mode only, so `as_forward_model(dt)` differentiates each one-step solve with `DirectAdjoint` instead, which `IncrementalFourDVar`'s forward-mode tangent-linear model needs; pass `as_forward_model(dt, adjoint=...)` to choose another.

**For long assimilation windows**: `BacksolveAdjoint`. Solves the continuous adjoint ODE [](#eq-adjoint-ode) backwards in time, with constant memory in the forward-trajectory length. Caveat: the reverse-time ODE may be stiff even when the forward ODE is not (diffusion-dominated problems, for example). If forward uses `Tsit5`, backward should typically use the same, and if the reverse problem is stiff, switch to an implicit solver; the `diffrax` docs list stable integrators.

**For parameter estimation**: `ForwardMode`. When you're sensitive to a
small number of parameters (e.g. a single emission rate $Q$ in a
methane Tier I problem), forward sensitivity is cheaper than reverse.

**For debugging**: `DirectAdjoint`. Plain reverse mode through the `lax.scan` of forward steps, slow but transparent. Its $O(T)$ memory is fine for $T = 10$ steps but painful for $T = 1000$, so it is the wrong choice for long windows.

## When the dynamics are linear

If $f_t$ is linear, the tangent-linear model [](#eq-tlm-forward) is just $f_t$ itself (no linearisation needed). For `IncrementalFourDVar` and operational 4DVar, the inner loop is in this situation: the *outer* nonlinear model is wrapped, but the *inner* loop operates on its tangent linear, computed once per outer iteration via `jax.linearize` and reused across many CG iterations.

The block below linearises one forward Euler step $\Psi_{\Delta t}$ at $\gvect{\mu}_x = (1, 1, 1)$, pulls back a cotangent $\vect{r} \in \mathbb{R}^{D_x}$ through its Jacobian $\mat{F} \in \mathbb{R}^{D_x \times D_x}$ and prints `True`.

```{code-block} python
:caption: Linearises one Lorenz-63 step as in [](#eq-tlm-forward) and checks the lineax adjoint step against the dense Jacobian.
:label: code-dynamical-model-tangent-linear
import einx
import jax
import lineax as lx

# Shapes: D_x = 3 state components

state_structure = jax.ShapeDtypeStruct((3,), jnp.float32)
# μ_x, the outer iterate
background = jnp.array([1.0, 1.0, 1.0])  # (D_x,)
# r, a cotangent to pull back
cotangent = jnp.array([1.0, -1.0, 0.5])  # (D_x,)


# user
def one_step(state: Float[Array, " state"]) -> Float[Array, " state"]:
    # Ψ_Δt(x) = x + Δt f(t, x)
    return dynamics.step(state, dynamics.dt)  # (D_x,)


# today (jax)
# δx ↦ F δx, the tangent-linear step at μ_x
_, tangent_step = jax.linearize(one_step, background)
# today (lineax)
# F as an operator, so lineax derives Fᵀ
tangent_op = lx.FunctionLinearOperator(tangent_step, state_structure)  # (D_x, D_x) op
# today (lineax)
adjoint_op = tangent_op.transpose()  # (D_x, D_x) op
# Fᵀ r, the adjoint step
adjoint_step = adjoint_op.mv(cotangent)  # (D_x,) → (D_x,)
# today (jax)
jacobian_fn = jax.jacfwd(one_step)
# F, formed densely to check
step_jacobian = jacobian_fn(background)  # (D_x,) → (D_x, D_x)
# Fᵀ r = ∑ᵢ Fᵢⱼ rᵢ
expected_adjoint = einx.dot(  # (D_x,)
    "next prev, next -> prev", step_jacobian, cotangent
)
print(bool(jnp.allclose(adjoint_step, expected_adjoint)))
```

For the unwrapped `StrongFourDVar` and `WeakFourDVar`, the full
nonlinear $f_t$ is integrated each outer step and `diffrax`'s adjoint
handles the gradient computation.

## When the dynamics are stochastic

For SDEs (Langevin dynamics in Lagrangian transport, stochastic parameterisations), `diffrax` still applies: `diffeqsolve` handles SDE integration via `EulerMaruyama`, `Heun`, etc., and the adjoint methods work in the SDE setting too (with caveats around non-deterministic reverse integration).

vardax does not currently expose dedicated SDE-DA classes; the existing
4DVar variants work as long as the SDE realisations are seeded
deterministically (e.g. via a `jax.random.PRNGKey` carried through the
batch).

## Forward model rollout in practice

For typical 4DVar with between $T = 10$ and $T = 100$ assimilation cycles and a shallow-water `ForwardModel`, the rollout [](#eq-rollout) is one `diffeqsolve` saved at every step. The toy below rolls Lorenz-63 out for $T = 100$ steps of $\Delta t = 0.01$ and differentiates $\frac{1}{2} \Vert \vect{x}_T \Vert^2$, the squared norm of the final state, with respect to $\vect{x}_0$ with two adjoints; it continues the block above.

```{code-block} python
:caption: Differentiates the rollout [](#eq-rollout) through the discrete adjoint and through the adjoint ODE [](#eq-adjoint-ode).
:label: code-dynamical-model-rollout-adjoints
import diffrax as dfx

# Shapes: T = 100 steps, so T + 1 = 101 times; D_x = 3 state components


# user
def rollout(
    initial_state: Float[Array, " state"],
    forward: Lorenz63,
    n_steps: int,
    dt: float,
    *,
    adjoint: dfx.AbstractAdjoint,
) -> Float[Array, "time state"]:
    """Integrate forward and return the trajectory."""
    # today (diffrax)
    # dx/dt = f(t, x)
    term = dfx.ODETerm(lambda time, state, args: forward.vector_field(state))
    # t_k = k Δt, k = 0, …, T
    save_times = dt * jnp.arange(n_steps + 1)  # (T + 1,)
    # today (diffrax)
    # x(t_k) = x₀ + ∫₀^{t_k} f(t, x(t)) dt
    solution = dfx.diffeqsolve(
        term,
        dfx.Tsit5(),
        t0=0.0,
        t1=n_steps * dt,
        dt0=dt,
        y0=initial_state,
        saveat=dfx.SaveAt(ts=save_times),
        adjoint=adjoint,
    )
    return solution.ys  # (T + 1, D_x)


# user
def final_state_cost(
    initial_state: Float[Array, " state"], adjoint: dfx.AbstractAdjoint
) -> Float[Array, ""]:
    # user
    # x(t_k) from x₀, k = 0, …, T
    trajectory = rollout(initial_state, dynamics, 100, 0.01, adjoint=adjoint)
    # x_T
    final_state = trajectory[-1]  # (D_x,)
    # ½ ‖x_T‖²
    return 0.5 * einx.dot("state, state ->", final_state, final_state)  # → ()


# today (jax)
cost_grad = jax.grad(final_state_cost)
# today (diffrax)
# ∇_x₀ by discretise-then-optimise, with recursive checkpoints
checkpoint_grad = cost_grad(background, dfx.RecursiveCheckpointAdjoint())  # (D_x,)
# today (diffrax)
# ∇_x₀ by optimise-then-discretise, solving the adjoint ODE backwards
backsolve_grad = cost_grad(background, dfx.BacksolveAdjoint())  # (D_x,)
print(bool(jnp.allclose(checkpoint_grad, backsolve_grad, rtol=1e-2)))
```

It prints `True`. The two gradients differ by about 0.1%, because `BacksolveAdjoint` discretises the continuous adjoint ODE while `RecursiveCheckpointAdjoint` differentiates the discrete solve exactly.

Inside vardax, the rollout is composed inside the cost function. Give the forward a `with_adjoint` method that stores the adjoint and passes it to `diffeqsolve`; the model's `forward_adjoint` slot then calls it, so that when `jax.grad(J)(x_0)` is called, the gradient flows back through the chosen adjoint path.

With $h_t$ the identity, `StrongFourDVar` recovers the initial state $\vect{x}_\text{true} = (1.2, 0.8, 1.1)$ from a noise-free window of $T + 1 = 5$ times, starting from $\gvect{\mu}_x$, once with the forward's own adjoint and once with `forward_adjoint=dfx.BacksolveAdjoint()`:

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) with two forward adjoints and compares the analyses.
:label: code-dynamical-model-strong4dvar
import dataclasses
from collections.abc import Callable

import vardax as vdx

# Shapes: N_batch = 1 window; T = 4 steps, so T + 1 = 5 times; D_x = 3 components


# user
@dataclasses.dataclass(frozen=True)
class DiffraxForward:
    vector_field: Callable[[Float[Array, " state"]], Float[Array, " state"]]
    dt: float
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()

    def step(
        self, state: Float[Array, " state"], dt: float
    ) -> Float[Array, " state"]:
        # today (diffrax)
        # dx/dt = f(t, x)
        term = dfx.ODETerm(lambda time, state, args: self.vector_field(state))
        # today (diffrax)
        # Ψ_Δt(x), the flow of dx/dt = f(t, x) over one step
        solution = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=state,
            adjoint=self.adjoint,
        )
        return solution.ys[-1]  # (D_x,)

    def with_adjoint(self, adjoint: dfx.AbstractAdjoint) -> "DiffraxForward":
        return dataclasses.replace(self, adjoint=adjoint)


forward = DiffraxForward(vector_field=dynamics.vector_field, dt=0.01)
# x_true, the initial state to recover
true_initial = jnp.array([1.2, 0.8, 1.1])  # (D_x,)
# user
# yₜ = xₜ rolled out from x_true, observed everywhere without noise
window_obs = rollout(  # (D_x,) → (T + 1, D_x)
    true_initial, dynamics, 4, 0.01, adjoint=dfx.RecursiveCheckpointAdjoint()
)
# today (lineax)
# Σ_x = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# today (lineax)
# Σ_ε = 0.01 I, noise s.d. 0.1; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (D_x, D_x) op
    0.01 * prior_cov, lx.positive_semidefinite_tag
)
# today (vardax)
window = vdx.Batch1D(
    input=einx.id(  # (T + 1, D_x) → (N_batch, T + 1, D_x)
        "time state -> 1 time state", window_obs
    ),
    mask=jnp.ones((1, 5, 3)),
)

# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ₜ ‖yₜ − hₜ(xₜ)‖²_Σ_ε⁻¹, xₜ rolled out from x₀
default_4dvar = vdx.StrongFourDVar(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)
# today (vardax)
# the same cost, differentiated through the adjoint ODE
backsolve_4dvar = vdx.StrongFourDVar(
    forward=forward,
    obs_op=vdx.MaskedIdentity(),
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    forward_adjoint=dfx.BacksolveAdjoint(),
)
# x₀⋆ = argmin 𝒥(x₀)
default_analysis = default_4dvar(window)  # → (N_batch, D_x)
# x₀⋆ again, through the adjoint ODE
backsolve_analysis = backsolve_4dvar(window)  # → (N_batch, D_x)

print(default_analysis[0], backsolve_analysis[0])
print(bool(jnp.allclose(default_analysis, backsolve_analysis, atol=1e-3)))
```

Both analyses print `[1.199159  0.8008326 1.0997877]`, and the check prints `True`: the adjoint changes how the gradient is computed, not the analysis.

## Per-step forwards: `checkpoint=`

Most forwards are a single `step(x, dt)` (an RK4 step of a finite-volume
model, a learned emulator). Reverse mode through a `lax.scan` of such
steps stores every intermediate of every step. The 4DVar models'
`checkpoint=` slot wraps blocks of steps in `jax.checkpoint`, so the
backward pass recomputes them instead: `True` checkpoints every step, an
integer `k` every block of `k` steps. The gradient is unchanged.

On Lorenz-96 ($D_x = 40$, RK4, $T = 500$ steps, float64, CPU) the compiled
gradient's temporary memory drops from 1.61 MB to 0.49 MB with
`checkpoint=True` and to 0.37 MB with `checkpoint=10`. What remains is
mostly the trajectory itself, which the observation cost reads at every
time.

## Performance considerations

- **Vectorise over the batch dimension**, not the time dimension. Time
  is intrinsically sequential; batch is embarrassingly parallel. Use
  `eqx.filter_vmap` over the batch axis.
- **Checkpoint sparingly with RecursiveCheckpointAdjoint.** The default
  number of checkpoints (`int(log2(n_steps)) + 1`) is usually right.
  Override only if profiling shows memory pressure.

## See also

- Chapter 6: `StrongFourDVar` (uses `forward_adjoint` for the dynamics rollout)
- Chapter 7: `WeakFourDVar` (the augmented control includes $\gvect{\eta}_t$)
- Chapter 8: `IncrementalFourDVar` (linearises once per outer iteration)
- Chapter 12: adjoint composition, with `diffrax` and `optimistix` working together
- Design doc: [`design/decisions.md#d15`](design/decisions.md#d15-lean-on-optimistix-diffrax-adjoints-not-in-house-grad-modes)
