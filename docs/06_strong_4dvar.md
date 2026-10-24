# Strong-constraint 4DVar

When observations are spread across an assimilation window and the
state evolves through known dynamics, 3DVar is no longer enough — we
need to relate observations at later times to the initial state via
the forward model. Strong-constraint 4DVar treats the dynamics as
exact and makes the initial state $x_0$ the control variable.

## Cost function

Given observations $y_t$ at times $t = 0, 1, \ldots, T$, dynamics $M_t$
propagating $x_0$ to time $t$, and obs operator $H_t$ at each time:

$$
J(x_0) =
  \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}} +
  \tfrac{1}{2} \sum_{t=0}^{T} \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}.
$$

**Strong-constraint** means the model is treated as exact: any $x_0$
determines the trajectory $x_t = M_t(x_0)$ exactly. Model error is not
admitted. (If you need model error, use `WeakFourDVar` — chapter 7.)

The gradient combines the background term with the adjoint of every
forward step:

$$
\nabla J(x_0) = B^{-1}(x_0 - x_b) +
                \sum_{t=0}^{T} (M'_t)^\top H'_t{}^\top R_t^{-1} (H_t(M_t(x_0)) - y_t).
$$

In hand-coded operational 4DVar, $(M'_t)^\top$ is the adjoint model —
historically a major engineering effort (Talagrand and Courtier 1987,
Errico 1997). With `diffrax`, it's a constructor argument.

## Implementation in vardax

```python
import diffrax as dfx
import optimistix as optx
from vardax import StrongFourDVar

model = StrongFourDVar(
    forward=somax_model,                              # pipekit_cycle.ForwardModel
    obs_op=MaskedIdentity(),                          # or AveragingKernel, etc.
    prior_mean=x_b,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
    minimiser=optx.NonlinearCG(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
    checkpoint=True,                                  # recompute steps in reverse mode
)
x_0_star = model(batch)
```

Three slots control the gradient:

- **`checkpoint`** — rematerialises the rollout in reverse mode
  (`True`, or a block length `k`), for any per-step forward. Same
  gradient, less memory (chapter 3).
- **`forward_adjoint`** — a `diffrax.AbstractAdjoint` for a forward that
  integrates with `diffrax` and implements `with_adjoint(adjoint)`
  ([`SupportsForwardAdjoint`](api/protocols.md)); for example
  `dfx.BacksolveAdjoint()` for constant memory over long windows. Any
  other forward raises `ValueError`.
- **`minimiser_adjoint`** — controls how gradients flow back through
  the outer minimiser if the model itself is differentiated through
  (e.g., for training hyperparameters of $B$). `ImplicitAdjoint` is
  exact at the optimum with $O(1)$ memory.

## Large controls: first-order minimisers

`minimiser=` takes an `optimistix.AbstractMinimiser` (the default is
`optx.BFGS(rtol=1e-6, atol=1e-6)`) or any [`Minimiser`](api/protocols.md):
a callable `minimiser(cost_fn, x0, batch) -> x_star` over a pytree
control. BFGS keeps a dense inverse-Hessian estimate, which is not viable
when the control is $10^5$ network weights. `OptaxMinimiser` runs a
first-order `optax` optimiser instead, in a `jax.lax.while_loop`, and
stops at `max_steps`, at a gradient norm below `gtol`, at a step below
`atol + rtol·‖x‖`, or after `patience` steps whose cost is worse than the
best by more than round-off. (There is no relative cost-change test: next
to a large irreducible misfit the cost change rounds to zero in float32
long before $x$ converges.) An optional `polish` optimiser (e.g.
`optax.lbfgs()`), with its own tight tolerances, finishes from there:

```python
import optax
import vardax

minimiser = vardax.OptaxMinimiser(
    optimizer=optax.adam(1e-3), max_steps=5000, gtol=1e-6, patience=200,
    polish=optax.lbfgs(), polish_steps=200,
)
model = vardax.StrongFourDVar(..., minimiser=minimiser)
result = minimiser.run(cost_fn, theta0, batch)   # value, cost, grad_norm, num_steps
```

`minimiser_adjoint` still applies. Its default, `ImplicitAdjoint`, is
exact only at a stationary point, $\nabla J(x^\star) = 0$, which a
first-order optimiser stopped early does not guarantee: `OptaxMinimiser`
reports $\lVert \nabla J(x^\star) \rVert_2$ in `MinimiseResult.grad_norm`;
check it before trusting gradients through the analysis. `result.cost`
is differentiable (envelope theorem). `OptimistixMinimiser()` is the default made explicit
and returns exactly what the default model returns.

## Parameterized initial states

`parameterization=` takes any
[`StateParameterization`](api/protocols.md): the control becomes a
pytree $\theta$, the initial state $x_0 = \psi(\theta)$, and its
penalty replaces the background term,

$$
J(\theta) = \mathrm{pen}(\theta)
  + \tfrac{1}{2} \sum_t \lVert y_t - H_t(M^t(\psi(\theta)))\rVert^2_{R_t^{-1}}.
$$

`LinearParameterization(basis, x_b)` is the reduced-basis case
($\psi(X) = x_b + \Phi X$, $\mathrm{pen}(X) = \tfrac12 X^\top Q^{-1} X$,
identical to `ReducedBasisFourDVar(basis=...)`).
`NeuralFieldParameterization(field, coords, weight_decay)` makes $x_0$ a
coordinate network evaluated on the grid, with
$\mathrm{pen}(\theta) = \tfrac{\alpha}{2}\lVert\theta\rVert^2$ and no $B$
(the network's spectral bias is the prior); its weights are a control
for `OptaxMinimiser`, not BFGS. Frozen arrays inside the field (random
Fourier frequencies read through `stop_gradient`) must be excluded from
the control with `filter_spec=`, or the weight decay shrinks them:

```python
import geonnax, optax, vardax

siren = geonnax.SIREN.init(2, 32, 1, depth=3, key=key)   # (cos, sin) on the ring
model = vardax.StrongFourDVar(
    ..., parameterization=vardax.NeuralFieldParameterization(siren, coords, 1e-4),
    minimiser=vardax.OptaxMinimiser(optax.adam(1e-3), max_steps=2000),
)
```

## Reduced-basis 4DVar

`ReducedBasisFourDVar` is strong-constraint 4DVar with the control moved
into a low-dimensional basis. Let $\Phi \in \mathbb{R}^{N \times P}$ be a
basis matrix with $P \ll N$ columns, $X \in \mathbb{R}^P$ its
coefficients, and $Q \in \mathbb{R}^{P \times P}$ their prior covariance
(diagonal for every built-in basis). The initial state is
$x_0 = x_b + \Phi X$, and the cost is

$$
J(X) = \tfrac{1}{2} X^\top Q^{-1} X
  + \tfrac{1}{2} \sum_t \lVert m_t \odot \big(y_t - H_t(M^t(x_b + \Phi X))\big)\rVert^2_{R_t^{-1}}.
$$

It is minimised directly, with no linearisation, outer loop or inner CG
(that is [incremental 4DVar](08_incremental_4dvar.md)). The gradient is
the state-space adjoint projected onto the basis,

$$
\nabla_X J = Q^{-1} X + \Phi^\top \lambda_0,
$$

where $\lambda_0 = -\sum_t M_t'^\top H_t'^\top R_t^{-1} d_t$ is the
adjoint state at $t = 0$ (the backward sweep below) and
$d_t = m_t \odot (y_t - H_t(M^t(x_0)))$. One cost-and-gradient
evaluation costs one forward and one adjoint run, as in state space;
the gain is that the minimiser works in $P$ dimensions. When $P$ is a few
tens, forward-mode differentiation (`jax.jacfwd`, $P$ tangent runs) also
becomes affordable, and the full Hessian $\nabla^2_X J$ is a
$P \times P$ matrix.

Every [`ReducedBasis`](api/protocols.md) carries its own $Q$:

| Constructor | Columns of $\Phi$ | $Q$ |
|---|---|---|
| `rbf_basis` | Gaussian or Wendland bumps at chosen centres | `variance` (scalar or per mode) |
| `fourier_basis` | Laplacian eigenfunctions on a box | `variance`, times a kernel's spectral density if given (HSGP) |
| `wavelet_basis` | orthonormal DWT atoms | `variance` per scale |
| `eof_basis` | leading EOFs of a sample matrix | each mode's sample variance (`"explained"`) |
| `linear_basis` | any $(N, P)$ matrix | `variance` |
| `composite_basis` | concatenated families | block-diagonal |

With `basis=None` the class minimises the state-space cost and is the
same as `StrongFourDVar`; `basis=` is equivalent to
`StrongFourDVar(parameterization=LinearParameterization(basis, x_b))`.

**Lorenz-96 example.** $N = 40$, RK4 with $\Delta t = 0.05$ over 10
steps, every second site observed at every time, $\sigma_o = \sigma_b = 1$,
Gaussian-correlated $B$ (length scale 2 sites). The basis is the leading
$P$ eigenvectors of $B$, the EOFs of the background error:

| Control | Share of $\mathrm{tr}\,B$ | Analysis RMSE of $x_0$ |
|---|---:|---:|
| background $x_b$ | — | 0.96 |
| $P = 8$ | 79 % | 0.57 |
| $P = 16$ | 99 % | 0.31 |
| full state, $N = 40$ | 100 % | 0.28 |

Two practical notes. First, scale the basis so that $Q = I$
($\Phi = U_P \Lambda_P^{1/2}$ for eigenpairs $(U_P, \Lambda_P)$ of $B$):
the cost is unchanged, and BFGS no longer has to learn a spread of
$Q^{-1}$ spanning several decades. Second, a chaotic forward can return
non-finite values during an aggressive line-search trial step, and the
masked $R$-solve raises on them; if that happens, cap the step, shorten
the window, or start from a better first guess.

```python
import jax.numpy as jnp
import vardax

eigvals, eigvecs = jnp.linalg.eigh(B)                 # ascending
whitened = eigvecs[:, -16:] * jnp.sqrt(eigvals[-16:])  # Φ = U_P Λ_P^{1/2}
model = vardax.ReducedBasisFourDVar(
    forward=l96, obs_op=vardax.MaskedIdentity(), prior_mean=x_b,
    prior_cov_op=B_op, obs_cov_op=R_op,
    basis=vardax.linear_basis(whitened, variance=1.0),
)
x0_star = model(batch)
```

`ReducedBasisFourDVar` was called `Incremental4DVar` until vardax 0.1.10.
The old name still works and emits a `DeprecationWarning`.

## The adjoint problem

The gradient computation reads "integrate the trajectory forward,
then propagate the observation residuals backward through the adjoint
of the dynamics." In equations, this is the backward sweep

$$
\lambda_T = H'_T{}^\top R_T^{-1} (H_T(x_T) - y_T)
$$

$$
\lambda_{t-1} = (M'_t)^\top \lambda_t + H'_{t-1}{}^\top R_{t-1}^{-1} (H_{t-1}(x_{t-1}) - y_{t-1})
$$

$$
\nabla_{x_0} J = B^{-1}(x_0 - x_b) + \lambda_0.
$$

This is exactly what `diffrax.RecursiveCheckpointAdjoint` or
`BacksolveAdjoint` compute under the hood. Vardax just composes them
into the cost-gradient call.

## Algorithm

```
Inputs:
   x_b, B_op, R_op           — prior and observation-error covariances
   forward                   — pipekit_cycle.ForwardModel (M_t)
   obs_op                    — pipekit_cycle.ObservationOperator (H_t)
   batch                     — observations y_t for t = 0, …, T
   minimiser                 — optimistix.AbstractMinimiser or vardax.Minimiser
   minimiser_adjoint         — optimistix.AbstractAdjoint
   forward_adjoint           — diffrax.AbstractAdjoint

def cost_fn(x_0, batch):
    # Forward rollout via diffrax with forward_adjoint
    trajectory = rollout(x_0, forward, n_steps=T, adjoint=forward_adjoint)

    # Observation cost summed over time
    j_obs = 0
    for t in range(T + 1):
        residual = batch.y[t] - obs_op(trajectory[t])
        j_obs = j_obs + 0.5 * residual @ R_inv_op @ residual

    # Background term
    j_b = 0.5 * (x_0 - x_b) @ B_inv_op @ (x_0 - x_b)
    return j_b + j_obs

# Iterate
result = optimistix.minimise(
    fn=cost_fn, solver=minimiser, y0=x_b, args=batch,
    adjoint=minimiser_adjoint,
)
x_0_star = result.value
```

## Linear-Gaussian agreement

When $H_t$ and $M_t$ are linear and $T = 0$, `StrongFourDVar` reduces
to `OptimalInterpolation`. The conformance test enforces this:

```python
def test_strong_4dvar_recovers_oi():
    oi = OptimalInterpolation(linear_H, x_b, B_op, R_op)
    strong = StrongFourDVar(
        forward=static_forward,                # M_t = I for T = 0
        obs_op=linear_H,
        prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
        minimiser=optx.NonlinearCG(rtol=1e-8),
    )
    batch = make_linear_gaussian_batch(T=0)
    assert jnp.allclose(oi(batch), strong(batch), atol=1e-3)
```

Part of the Decision D14 invariant.

## When the window is long

For $T = 100$+ assimilation cycles, the forward trajectory dominates
memory. Four strategies:

1. **`checkpoint=True` (or a block length).** Works with any per-step
   forward: reverse mode recomputes each block of steps instead of
   storing its intermediates. 3–4× less gradient memory on Lorenz-96
   (chapter 3).
2. **`dfx.BacksolveAdjoint()`** (via `forward_adjoint=`, for a
   [`SupportsForwardAdjoint`](api/protocols.md) forward) — solve the adjoint ODE backwards in
   time. $O(1)$ memory in the trajectory length. Caveat: reverse-time
   stiffness can require an implicit integrator (`dfx.Kvaerno5`,
   `dfx.ImplicitEuler`).
3. **`dfx.RecursiveCheckpointAdjoint(checkpoints=N)`** (also via
   `forward_adjoint=`) — store $N$ checkpoints, recompute segments between them. $O(\sqrt{T})$ memory.
4. **Shorter windows.** Run multiple `StrongFourDVar` analyses with
   shorter windows and cycle them via `pipekit_cycle.DACycle`. This is
   what operational systems do.

For most research-scale problems with $T \le 20$, the defaults
(plain reverse mode, the forward's own adjoint) are fine.

## When the dynamics are nonlinear

The 4DVar cost $J$ is nonconvex when $M_t$ is nonlinear — multiple
local minima exist (and matter, especially for chaotic systems like
Lorenz). Strategies:

- **Multi-start.** Minimise from several perturbed first guesses and
  keep the lowest cost. vardax has no helper for this yet: the models
  start every minimisation from $x_b$, so a multi-start needs a first
  guess separate from the background.
- **Quasi-static gradient.** Use a warm-start from a previous cycle
  (the usual operational pattern).
- **Switch to `IncrementalFourDVar`** (chapter 8) — the outer
  Gauss-Newton iterations relinearise around the current iterate,
  which is more robust than NonlinearCG on the full nonlinear cost.

## Posterior

Pair the analysis with a `PosteriorAdapter`:

```python
from vardax import LaplaceCovariance

posterior = LaplaceCovariance()(x_0_star, model.as_analysis_step(), batch)
```

The Laplace covariance involves $(B^{-1} + \sum_t (M'_t H'_t)^\top R_t^{-1} (M'_t H'_t))^{-1}$,
computed by Krylov / Lanczos via `lineax.CG`. The mat-vec includes a
tangent-linear-plus-adjoint pass through the trajectory, so it's
expensive per query — `n_krylov = 50` typical mat-vec count gives
posterior diagonal at the cost of 50 trajectory tangent + adjoint
passes.

For operational use, `IncrementalFourDVar` is preferred — it produces
the Gauss-Newton Hessian as a side effect of the last outer iteration,
so posterior is essentially free.

## When `StrongFourDVar` is the right answer

Use `StrongFourDVar` when:

- Multi-time observations with dynamics
- Model error is small (within the noise floor of the observations)
- General-purpose 4DVar — when you want to try `optimistix.BFGS` or
  `NonlinearCG` directly on the full nonlinear cost without the
  incremental machinery

For production / operational 4DVar with structured priors, use `IncrementalFourDVar` (chapter 8) instead — it's
the same problem, solved faster.

If model error matters, use `WeakFourDVar` (chapter 7).

## See also

- Chapter 3 — dynamical model and `forward_adjoint` options
- Chapter 7 — `WeakFourDVar` (admits model error)
- Chapter 8 — `IncrementalFourDVar` (operational fast path)
- Chapter 12 — adjoint composition

## References

- Talagrand, O., & Courtier, P. (1987). *Variational assimilation of
  meteorological observations with the adjoint vorticity equation.*
  QJRMS 113(478).
- Le Dimet, F.-X., & Talagrand, O. (1986). *Variational algorithms for
  analysis and assimilation of meteorological observations.* Tellus A
  38(2).
- Errico, R. M. (1997). *What is an adjoint model?* BAMS 78(11).
