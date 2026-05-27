# Dynamical Model

For 4D analysis, observations $y_t$ at times $t = 0, 1, \ldots, T$ must
be connected to the initial state $x_0$ through the dynamics:

$$
x_t = M_t(x_0),
$$

where $M_t$ propagates the state forward from $t_0$ to $t_t$. The
dynamics enter the cost as the composition $H_t \circ M_t$.

vardax does not own dynamics — that's `somax` (geophysical fluids),
`plumax` (atmospheric transport / methane), or any user code. What
vardax owns is the **interface** to dynamics (the
`pipekit_cycle.ForwardModel` protocol) and the **adjoint composition**
(delegated to `diffrax.AbstractAdjoint`).

## The `ForwardModel` protocol

```python
from pipekit_cycle import ForwardModel

class MyDynamics(eqx.Module):
    def step(self, state: Array, dt: float) -> Array:
        """Advance state by dt."""
        ...

    @property
    def dt(self) -> float: ...

    @property
    def state_signature(self) -> Signature: ...
```

`somax.ShallowWaterModel`, `plumax.GaussianPlumeForward`, and
user-defined dynamics all satisfy this protocol natively. From vardax's
perspective they're interchangeable.

For multi-step rollouts, vardax composes `step` calls into a
trajectory via `diffrax.diffeqsolve` (when the model is an ODE) or
direct iteration (when it's discrete). The choice of ODE integrator
(`Tsit5`, `Dopri5`, `Heun`, …) is on the dynamics, not on vardax.

## Why the adjoint matters

The 4DVar cost involves a sum over time:

$$
J(x_0) = \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}
       + \tfrac{1}{2} \sum_t \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}.
$$

Minimising over $x_0$ requires $\nabla_{x_0} J$, which by the chain
rule pulls back through every $M_t$:

$$
\nabla_{x_0} J(x_0) = B^{-1}(x_0 - x_b)
                    + \sum_t (M'_t)^\top H'_t{}^\top R_t^{-1} (H_t(M_t(x_0)) - y_t).
$$

The transpose $(M'_t)^\top$ is the **adjoint model** — historically
hand-coded line by line (Talagrand and Courtier 1987, Errico 1997).
Modern numerical libraries make this far less painful: `diffrax`
provides the adjoint of any `diffeqsolve` call automatically, in
several flavours.

## diffrax adjoint methods

vardax exposes `forward_adjoint: diffrax.AbstractAdjoint` as a
constructor slot on `StrongFourDVar`, `WeakFourDVar`,
`IncrementalFourDVar`, and `FourDVarNet` (when the prior is a
`DynamicalPrior`). The choice flows directly through to `diffrax`.

| Adjoint | Memory | Time | Convergence requirement | When |
|---|---|---|---|---|
| `diffrax.RecursiveCheckpointAdjoint(checkpoints=N)` | $O(N)$ | $O(\log K)$ extra forward solves | None | Default; balanced |
| `diffrax.BacksolveAdjoint()` | $O(1)$ | Backwards ODE solve | None (but reverse-time stability matters) | Long windows, memory-constrained |
| `diffrax.ForwardMode()` | $O(P)$ in parameters | One extra forward per param | None | Few parameters (parameter estimation) |
| `diffrax.DirectAdjoint()` | $O(K)$ steps | Standard reverse | None | Short windows where memory is cheap |

**Default**: `RecursiveCheckpointAdjoint`. Standard autodiff with
recursive checkpointing — balances memory and recompute.

**For long assimilation windows**: `BacksolveAdjoint`. Solves the
continuous adjoint ODE backwards in time. Constant memory in the
forward-trajectory length. Caveat: the reverse-time ODE may be stiff
even when the forward ODE is not; consult `diffrax` docs for stable
integrators.

**For parameter estimation**: `ForwardMode`. When you're sensitive to a
small number of parameters (e.g. a single emission rate $Q$ in a
methane Tier I problem), forward sensitivity is cheaper than reverse.

**For debugging**: `DirectAdjoint`. Plain reverse-mode through the
`lax.scan` of forward steps. Slow but transparent.

## When the dynamics are linear

If $M_t$ is linear, the tangent-linear is just $M_t$ itself (no
linearisation needed). For `IncrementalFourDVar` and operational 4DVar,
this is the usual case — the *outer* nonlinear model is wrapped, but
the *inner* loop operates on its tangent linear, computed once per
outer iteration via `jax.linearize` and reused across many CG
iterations.

```python
import jax

# At outer iterate x_b, compute the tangent-linear of M
_, M_lin = jax.linearize(lambda s: forward_model.step(s, dt), x_b)
# M_lin(dx) returns M'_t · dx; M_lin can be transposed via lineax
```

For the unwrapped `StrongFourDVar` and `WeakFourDVar`, the full
nonlinear $M_t$ is integrated each outer step and `diffrax`'s adjoint
handles the gradient computation.

## When the dynamics are stochastic

For SDEs (Langevin dynamics in Lagrangian transport, stochastic
parameterisations), `diffrax` still applies — `diffeqsolve` handles
SDE integration via `EulerMaruyama`, `Heun`, etc., and the adjoint
methods work in the SDE setting too (with appropriate caveats around
non-deterministic reverse integration).

vardax does not currently expose dedicated SDE-DA classes; the existing
4DVar variants work as long as the SDE realisations are seeded
deterministically (e.g. via a `jax.random.PRNGKey` carried through the
batch).

## Forward model rollout in practice

For typical 4DVar with $T = 10$–$100$ assimilation cycles and a
shallow-water `ForwardModel`:

```python
import jax.numpy as jnp
import diffrax as dfx

def rollout(x_0, forward, n_steps, dt, *, adjoint):
    """Integrate forward and return the trajectory."""
    sol = dfx.diffeqsolve(
        terms=dfx.ODETerm(lambda t, y, args: forward.vector_field(y, t)),
        solver=dfx.Tsit5(),
        t0=0.0, t1=n_steps * dt, dt0=dt,
        y0=x_0,
        saveat=dfx.SaveAt(ts=jnp.arange(0, n_steps + 1) * dt),
        adjoint=adjoint,
    )
    return sol.ys     # shape (n_steps + 1, *state_shape)
```

Inside vardax, the rollout is composed inside the cost function. The
`forward_adjoint` slot is threaded into `diffeqsolve` so that when
`jax.grad(J)(x_0)` is called, the gradient flows back through the
chosen adjoint path.

## Performance considerations

- **Vectorise over the batch dimension**, not the time dimension. Time
  is intrinsically sequential; batch is embarrassingly parallel. Use
  `eqx.filter_vmap` over the batch axis.
- **Checkpoint sparingly with RecursiveCheckpointAdjoint.** The default
  number of checkpoints (`int(log2(n_steps)) + 1`) is usually right.
  Override only if profiling shows memory pressure.
- **BacksolveAdjoint requires reverse-stable solvers.** If forward uses
  `Tsit5`, backward should typically use the same. Some problems (e.g.
  diffusion-dominated) become stiff in reverse; switch to an implicit
  solver if so.
- **DirectAdjoint is the wrong choice for long windows.** $O(K)$ memory
  is fine for $K = 10$, painful for $K = 1000$.

## See also

- Chapter 6 — `StrongFourDVar` (uses `forward_adjoint` for the dynamics rollout)
- Chapter 7 — `WeakFourDVar` (augmented control includes $\eta_t$)
- Chapter 8 — `IncrementalFourDVar` (linearises once per outer iteration)
- Chapter 12 — adjoint composition: `diffrax` + `optimistix` working together
- Design doc: [`design/decisions.md#d15`](design/decisions.md#d15-lean-on-optimistix-diffrax-adjoints-not-in-house-grad-modes)
