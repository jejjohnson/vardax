---
numbering:
  enumerator: "3.%s"
---

# Dynamical Model

For 4D analysis, observations $y_t$ at times $t = 0, 1, \ldots, T$ must be connected to the initial state $x_0$ through the rollout [](#eq-rollout), where $M_t$ propagates the state forward from time $0$ to time $t$. The dynamics enter the cost as the composition $H_t \circ M_t$.

vardax does not own dynamics; those come from `somax` (geophysical fluids), `plumax` (atmospheric transport and methane) or any user code. What vardax owns is the **interface** to dynamics (the `pipekit_cycle.ForwardModel` protocol) and the **adjoint composition** (delegated to `diffrax.AbstractAdjoint`).

## The `ForwardModel` protocol

A forward supplies `step(state, dt)`, a default `dt` and a `state_signature`. The Lorenz-63 model below, with state $s = (x, y, z)$ and the vector field $f$ of [](#eq-lorenz63), advances by forward Euler steps and satisfies the protocol structurally, and the check prints `True`; the later blocks on this page build on it.

```{code-block} python
:caption: Defines Lorenz-63 [](#eq-lorenz63) as a forward-Euler forward model and checks that it satisfies the protocol.
:label: code-dynamical-model-lorenz63
import equinox as eqx
import jax.numpy as jnp
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N = 3 state components


# user
class Lorenz63(eqx.Module):
    """Lorenz-63 dynamics, advanced by forward Euler steps."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01

    def vector_field(self, state: Float[Array, " n"]) -> Float[Array, " n"]:
        x, y, z = state
        # f(s) = (σ (y − x), x (ρ − z) − y, x y − β z)
        return jnp.stack(
            [self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]
        )

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        """Advance state by dt."""
        # sₖ₊₁ = sₖ + Δt f(sₖ)
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

The strong-constraint 4DVar cost [](#eq-strong4dvar-cost) sums an observation term over every time. Minimising it over $x_0$ requires the gradient [](#eq-strong4dvar-grad), which by the chain rule pulls each weighted residual $H'_t{}^\top R_t^{-1} (H_t(M_t(x_0)) - y_t)$ back to the initial time through $(M'_t)^\top$, where $M'_t$ and $H'_t$ are the tangent-linear dynamics and observation operator.

The transpose $(M'_t)^\top$ is the **adjoint model**, which was historically hand-coded line by line (Talagrand and Courtier 1987, Errico 1997). Modern numerical libraries make this far less painful: `diffrax` provides the adjoint of any `diffeqsolve` call automatically, in several flavours.

## diffrax adjoint methods

vardax exposes `forward_adjoint: diffrax.AbstractAdjoint` as a
constructor slot on `StrongFourDVar`, `WeakFourDVar`,
`IncrementalFourDVar` and `ReducedBasisFourDVar`. It reaches `diffrax`
through the forward: the model calls `forward.with_adjoint(adjoint)`
(the [`SupportsForwardAdjoint`](api/protocols.md) protocol) and rolls
out the forward it returns. A per-step forward has no integrator to hand
the adjoint to, so passing `forward_adjoint=` with one raises
`ValueError`; bound its memory with `checkpoint=` instead (below).

| Adjoint | Memory | Time | Convergence requirement | When |
|---|---|---|---|---|
| `diffrax.RecursiveCheckpointAdjoint(checkpoints=N)` | $O(N)$ | $O(\log K)$ extra forward solves | None | Default; balanced |
| `diffrax.BacksolveAdjoint()` | $O(1)$ | Backwards ODE solve | None (but reverse-time stability matters) | Long windows, memory-constrained |
| `diffrax.ForwardMode()` | $O(P)$ in parameters | One extra forward per param | None | Few parameters (parameter estimation) |
| `diffrax.DirectAdjoint()` | $O(K)$ steps | Standard reverse | None | Short windows where memory is cheap |

**Default**: `RecursiveCheckpointAdjoint`. Standard autodiff with recursive checkpointing, which balances memory and recompute.

**For long assimilation windows**: `BacksolveAdjoint`. Solves the continuous adjoint ODE [](#eq-adjoint-ode) backwards in time, with constant memory in the forward-trajectory length. Caveat: the reverse-time ODE may be stiff even when the forward ODE is not (diffusion-dominated problems, for example). If forward uses `Tsit5`, backward should typically use the same, and if the reverse problem is stiff, switch to an implicit solver; the `diffrax` docs list stable integrators.

**For parameter estimation**: `ForwardMode`. When you're sensitive to a
small number of parameters (e.g. a single emission rate $Q$ in a
methane Tier I problem), forward sensitivity is cheaper than reverse.

**For debugging**: `DirectAdjoint`. Plain reverse mode through the `lax.scan` of forward steps, slow but transparent. Its $O(K)$ memory is fine for $K = 10$ steps but painful for $K = 1000$, so it is the wrong choice for long windows.

## When the dynamics are linear

If $M_t$ is linear, the tangent-linear model [](#eq-tlm-forward) is just $M_t$ itself (no linearisation needed). For `IncrementalFourDVar` and operational 4DVar, the inner loop is in this situation: the *outer* nonlinear model is wrapped, but the *inner* loop operates on its tangent linear, computed once per outer iteration via `jax.linearize` and reused across many CG iterations.

The block below linearises at $x_b = (1, 1, 1)$, pulls back a cotangent $r$ and prints `True`.

```{code-block} python
:caption: Linearises one Lorenz-63 step as in [](#eq-tlm-forward) and checks the lineax adjoint step against the dense Jacobian.
:label: code-dynamical-model-tangent-linear
import einx
import jax
import lineax as lx

# Shapes: N = 3 state components

state_structure = jax.ShapeDtypeStruct((3,), jnp.float32)
# x_b, the outer iterate
background = jnp.array([1.0, 1.0, 1.0])  # (N,)
# r, a cotangent to pull back
cotangent = jnp.array([1.0, -1.0, 0.5])  # (N,)


# user
def one_step(state: Float[Array, " n"]) -> Float[Array, " n"]:
    # M(s) = s + Δt f(s)
    return dynamics.step(state, dynamics.dt)  # (N,)


# today (jax)
# δx ↦ M′(x_b) δx, the tangent-linear step
_, tangent_step = jax.linearize(one_step, background)
# today (lineax)
# M′ as an operator, so lineax derives M′ᵀ
tangent_op = lx.FunctionLinearOperator(tangent_step, state_structure)  # (N, N) op
# today (lineax)
adjoint_op = tangent_op.transpose()  # (N, N) op
# M′ᵀ r, the adjoint step
adjoint_step = adjoint_op.mv(cotangent)  # (N,) → (N,)
# today (jax)
jacobian_fn = jax.jacfwd(one_step)
# M′(x_b), formed densely to check
step_jacobian = jacobian_fn(background)  # (N,) → (N, N)
# M′ᵀ r = Σᵢ M′ᵢⱼ rᵢ
expected_adjoint = einx.dot("i j, i -> j", step_jacobian, cotangent)  # (N,)
print(bool(jnp.allclose(adjoint_step, expected_adjoint)))
```

For the unwrapped `StrongFourDVar` and `WeakFourDVar`, the full
nonlinear $M_t$ is integrated each outer step and `diffrax`'s adjoint
handles the gradient computation.

## When the dynamics are stochastic

For SDEs (Langevin dynamics in Lagrangian transport, stochastic parameterisations), `diffrax` still applies: `diffeqsolve` handles SDE integration via `EulerMaruyama`, `Heun`, etc., and the adjoint methods work in the SDE setting too (with caveats around non-deterministic reverse integration).

vardax does not currently expose dedicated SDE-DA classes; the existing
4DVar variants work as long as the SDE realisations are seeded
deterministically (e.g. via a `jax.random.PRNGKey` carried through the
batch).

## Forward model rollout in practice

For typical 4DVar with between $T = 10$ and $T = 100$ assimilation cycles and a shallow-water `ForwardModel`, the rollout [](#eq-rollout) is one `diffeqsolve` saved at every step. The toy below rolls Lorenz-63 out for $K = 100$ steps of $\Delta t = 0.01$ and differentiates $\frac{1}{2} \|s(t_K)\|^2$, the squared norm of the final state, with two adjoints; it continues the block above.

```{code-block} python
:caption: Differentiates the rollout [](#eq-rollout) through the discrete adjoint and through the adjoint ODE [](#eq-adjoint-ode).
:label: code-dynamical-model-rollout-adjoints
import diffrax as dfx

# Shapes: K = 100 steps; N = 3 state components


# user
def rollout(
    initial_state: Float[Array, " n"],
    forward: Lorenz63,
    n_steps: int,
    dt: float,
    *,
    adjoint: dfx.AbstractAdjoint,
) -> Float[Array, "k n"]:
    """Integrate forward and return the trajectory."""
    # today (diffrax)
    # ds/dt = f(s)
    term = dfx.ODETerm(lambda time, state, args: forward.vector_field(state))
    # tₖ = k Δt, k = 0, …, K
    save_times = dt * jnp.arange(n_steps + 1)  # (K + 1,)
    # today (diffrax)
    # s(tₖ) = s₀ + ∫₀^{tₖ} f(s) dt
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
    return solution.ys  # (K + 1, N)


# user
def final_state_cost(
    initial_state: Float[Array, " n"], adjoint: dfx.AbstractAdjoint
) -> Float[Array, ""]:
    # user
    # s(tₖ) = Mₖ(s₀), k = 0, …, K
    trajectory = rollout(initial_state, dynamics, 100, 0.01, adjoint=adjoint)
    # s(t_K)
    final_state = trajectory[-1]  # (N,)
    # ½ ‖s(t_K)‖²
    return 0.5 * einx.dot("n, n ->", final_state, final_state)  # → ()


# today (jax)
cost_grad = jax.grad(final_state_cost)
# today (diffrax)
# ∇ₛ₀ by discretise-then-optimise, with recursive checkpoints
checkpoint_grad = cost_grad(background, dfx.RecursiveCheckpointAdjoint())  # (N,)
# today (diffrax)
# ∇ₛ₀ by optimise-then-discretise, solving the adjoint ODE backwards
backsolve_grad = cost_grad(background, dfx.BacksolveAdjoint())  # (N,)
print(bool(jnp.allclose(checkpoint_grad, backsolve_grad, rtol=1e-2)))
```

It prints `True`. The two gradients differ by about 0.1%, because `BacksolveAdjoint` discretises the continuous adjoint ODE while `RecursiveCheckpointAdjoint` differentiates the discrete solve exactly.

Inside vardax, the rollout is composed inside the cost function. Give the forward a `with_adjoint` method that stores the adjoint and passes it to `diffeqsolve`; the model's `forward_adjoint` slot then calls it, so that when `jax.grad(J)(x_0)` is called, the gradient flows back through the chosen adjoint path.

With $H_t$ the identity, `StrongFourDVar` recovers the initial state $x_\text{true} = (1.2, 0.8, 1.1)$ from a noise-free window of $T = 5$ times, starting from $x_b$, once with the forward's own adjoint and once with `forward_adjoint=dfx.BacksolveAdjoint()`:

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) with two forward adjoints and compares the analyses.
:label: code-dynamical-model-strong4dvar
import dataclasses
from collections.abc import Callable

import vardax as vdx

# Shapes: S = 1 window; T = 5 times; N = 3 state components


# user
@dataclasses.dataclass(frozen=True)
class DiffraxForward:
    vector_field: Callable[[Float[Array, " n"]], Float[Array, " n"]]
    dt: float
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # today (diffrax)
        # ds/dt = f(s)
        term = dfx.ODETerm(lambda time, state, args: self.vector_field(state))
        # today (diffrax)
        # s(t + Δt) = s(t) + ∫ f(s) dt over one step
        solution = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=state,
            adjoint=self.adjoint,
        )
        return solution.ys[-1]  # (N,)

    def with_adjoint(self, adjoint: dfx.AbstractAdjoint) -> "DiffraxForward":
        return dataclasses.replace(self, adjoint=adjoint)


forward = DiffraxForward(vector_field=dynamics.vector_field, dt=0.01)
# x_true(t₀), the initial state to recover
true_initial = jnp.array([1.2, 0.8, 1.1])  # (N,)
# user
# yₜ = Mₜ(x_true), observed everywhere without noise
window_obs = rollout(  # (N,) → (T, N)
    true_initial, dynamics, 4, 0.01, adjoint=dfx.RecursiveCheckpointAdjoint()
)
# today (lineax)
# B = I
prior_cov = lx.IdentityLinearOperator(state_structure)  # (N, N) op
# today (lineax)
# R = σ² I, σ = 0.1; tagged PSD so the default CG solve accepts it
obs_cov = lx.TaggedLinearOperator(  # (N, N) op
    0.01 * prior_cov, lx.positive_semidefinite_tag
)
# today (vardax)
window = vdx.Batch1D(
    input=einx.id("t n -> 1 t n", window_obs),  # (T, N) → (S, T, N)
    mask=jnp.ones((1, 5, 3)),
)

# today (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σₜ ‖yₜ − Mₜ(x₀)‖²_{R⁻¹}
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
# x₀* = argmin J(x₀)
default_analysis = default_4dvar(window)  # → (S, N)
# x₀* again, through the adjoint ODE
backsolve_analysis = backsolve_4dvar(window)  # → (S, N)

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

On Lorenz-96 ($N = 40$, RK4, 500 steps, float64, CPU) the compiled
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
- Chapter 7: `WeakFourDVar` (the augmented control includes $\eta_t$)
- Chapter 8: `IncrementalFourDVar` (linearises once per outer iteration)
- Chapter 12: adjoint composition, with `diffrax` and `optimistix` working together
- Design doc: [`design/decisions.md#d15`](design/decisions.md#d15-lean-on-optimistix-diffrax-adjoints-not-in-house-grad-modes)
