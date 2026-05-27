---
status: draft
version: 0.4.0
---

# Layer 2 — Models

Seven peer classes, each satisfying `pipekit_cycle.AnalysisStep` via
`.as_analysis_step()`. None inherits from any other; the differences sit
in the algorithm, not in the type hierarchy.

| Class | Method | Decision |
|---|---|---|
| `OptimalInterpolation` | BLUE / OI — closed-form linear-Gaussian | D14, D16 |
| `ThreeDVar` | 3D variational, nonlinear $H$, single time | D14 |
| `StrongFourDVar` | 4DVar, control = $x_0$, exact dynamics | D14 |
| `WeakFourDVar` | 4DVar, control = $(x_0, \eta_1, \ldots, \eta_T)$ | D14 |
| `IncrementalFourDVar` | GN outer + CG inner + CVT (fast `StrongFourDVar`) | D11 |
| `FourDVarNet` | Learned $\varphi_\theta$ + learned $\Phi_\phi$ | D3, D6 |
| `AmortizedPosterior` | Direct $q_\phi(x \mid y)$ head | D12 |

All seven accept the same `Batch*` types and the same observation
operator family. The user picks the method that matches the regime.

---

## `OptimalInterpolation` — BLUE / OI

### Mathematical formulation

The Best Linear Unbiased Estimator for linear-Gaussian state estimation:

$$x^* = x_b + K(y - H x_b), \qquad K = B H^\top (H B H^\top + R)^{-1}$$

with posterior covariance

$$P^* = (B^{-1} + H^\top R^{-1} H)^{-1} = (I - K H) B.$$

Choice of form (Sherman-Morrison-Woodbury): use the $R$-space form when
the observation dimension $m$ is smaller than the state dimension $n$;
the $B$-space form otherwise. `gaussx` handles structured $B$ and $R$
without materialising dense matrices.

### Class contract

```python
class OptimalInterpolation(eqx.Module):
    """BLUE / OI — closed-form linear-Gaussian analysis.

    Requires a linear observation operator. Use ThreeDVar for nonlinear
    H. Posterior covariance is included in the result without a
    separate PosteriorAdapter call.
    """
    obs_op: ObservationOperator                # linear (validated in __init__)
    prior_mean: Array                          # x_b
    prior_cov_op: AbstractLinearOperator       # B (gaussx-friendly)
    obs_cov_op: AbstractLinearOperator         # R (gaussx-friendly)

    def __call__(self, batch: Batch) -> Array:
        """Return x*. No iteration, no convergence criterion."""
        ...

    def posterior(self, batch: Batch) -> Posterior:
        """Closed-form (mean, cov, provenance) — no PosteriorAdapter needed."""
        ...

    def as_analysis_step(self) -> AnalysisStep:
        ...
```

### When to use

- $H$ is linear (or the linearisation around $x_b$ is good enough)
- $B$, $R$, $H$ are Gaussian (or close enough that the posterior is
  unimodal Gaussian)
- Static field (no dynamics) or a single timestep

If any of these fail, `ThreeDVar` (nonlinear $H$) or
`StrongFourDVar` / `IncrementalFourDVar` (dynamics) is the right choice.
`OptimalInterpolation.__init__` validates linearity and refuses
nonlinear obs operators with a clear error message.

---

## `ThreeDVar` — 3D Variational

### Mathematical formulation

Minimise

$$J(x) = \tfrac{1}{2} \|x - x_b\|^2_{B^{-1}} + \tfrac{1}{2} \|y - H(x)\|^2_{R^{-1}}$$

over $x$. For nonlinear $H$, solve iteratively via Gauss-Newton, BFGS,
or NonlinearCG (chosen by the user via the `minimiser` slot).

In the linear-Gaussian limit, `ThreeDVar` recovers `OptimalInterpolation`
exactly — the conformance test suite verifies this agreement.

### Class contract

```python
class ThreeDVar(eqx.Module):
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...
```

### When to use

- Nonlinear $H$ (e.g. averaging kernel + nonlinear retrieval)
- Single timestep (snapshot inversion)
- Posterior assumed Gaussian-around-MAP (`LaplaceCovariance` adapter)

---

## `StrongFourDVar` — Strong-constraint 4DVar

### Mathematical formulation

Control variable is the initial state $x_0$ alone; dynamics $M_t$ are
treated as exact:

$$J(x_0) = \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}} + \tfrac{1}{2} \sum_{t=0}^{T} \|y_t - H_t(M_t(x_0))\|^2_{R_t^{-1}}$$

Gradients of $J$ with respect to $x_0$ are computed by integrating the
adjoint of $M_t$ backwards in time — vardax delegates this to
`diffrax.AbstractAdjoint` (see Decision D15). The default
`RecursiveCheckpointAdjoint` is autodiff with recursive checkpointing;
`BacksolveAdjoint` solves the continuous adjoint ODE in reverse time
(constant memory, good for long windows).

### Class contract

```python
class StrongFourDVar(eqx.Module):
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def as_analysis_step(self) -> AnalysisStep: ...
```

### When to use

- Multi-timestep observations of a state with known dynamics
- Model error is small enough to ignore
- General-purpose 4DVar (use `IncrementalFourDVar` for the operational
  fast path)

---

## `WeakFourDVar` — Weak-constraint 4DVar

### Mathematical formulation

Control vector is augmented to include per-step model error:

$$\boldsymbol{\eta} = (\eta_1, \ldots, \eta_T), \quad x_t = M_t^\text{free}(x_{t-1}) + \eta_t$$

with cost

$$J(x_0, \boldsymbol{\eta}) = \tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}} + \tfrac{1}{2} \sum_t \|y_t - H_t(x_t)\|^2_{R_t^{-1}} + \tfrac{1}{2} \sum_t \|\eta_t\|^2_{Q_t^{-1}}$$

The model-error covariance $Q_t$ is a separate input.

### Class contract

```python
class WeakFourDVar(eqx.Module):
    forward: ForwardModel                            # M_t^free
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator             # B
    obs_cov_op: AbstractLinearOperator               # R
    model_err_cov_op: AbstractLinearOperator         # Q
    minimiser: optimistix.AbstractMinimiser
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> tuple[Array, Array]:
        """Return (x_0*, η*) — analysis initial condition and model error."""
        ...
    def as_analysis_step(self) -> AnalysisStep: ...
```

### When to use

- Multi-timestep observations with non-negligible model error
- Long assimilation windows where dynamics drift away from observations
- Climatological or seasonal mismatch between $M_t^\text{free}$ and
  reality

`WeakFourDVar` is computationally heavier than `StrongFourDVar` because
the control vector dimension scales with $T$. Use sparingly; consider
`StrongFourDVar` with a learned $\varphi_\theta$ as a cheaper
alternative.

---

## `IncrementalFourDVar` — Operational Fast Path

### Mathematical formulation

Functionally equivalent to `StrongFourDVar`, but uses Gauss-Newton outer
iterations on the full nonlinear cost with CG / Lanczos inner iterations
on the linearised quadratic subproblem. At each outer iterate $x_b^{(k)}$,
linearise $M_t$ and $H_t$ via `jax.linearize`:

$$\delta J(\delta x) = \tfrac{1}{2} \|\delta x\|^2_{B^{-1}} + \tfrac{1}{2} \sum_t \|d_t - H'_t M'_t \delta x\|^2_{R_t^{-1}}$$

with innovation $d_t = y_t - H_t(M_t(x_b^{(k)}))$. Solve for
$\delta x^*$ by CG, update $x_b^{(k+1)} = x_b^{(k)} + \delta x^*$,
relinearise.

**Control-variable transform** (CVT): set $\chi = B^{-1/2}(\delta x)$;
the prior term becomes $\|\chi\|^2$ and the inner CG operates on a
well-conditioned problem. $B^{1/2}$ comes from
`gaussx.MaternLinearOperator.half()`.

### Class contract

```python
class IncrementalFourDVar(eqx.Module):
    forward: ForwardModel
    obs_op: ObservationOperator
    prior_mean: Array
    prior_cov_op: AbstractLinearOperator
    obs_cov_op: AbstractLinearOperator
    config: IncrementalConfig
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Array: ...
    def posterior(self, batch: Batch) -> Posterior:
        """Reuse the Hessian from the last outer iteration for GN-Hessian UQ."""
        ...
    def as_analysis_step(self) -> AnalysisStep: ...
```

`IncrementalConfig` carries `n_outer`, `n_inner`, `cg_atol`, `cg_rtol`,
`cvt: bool` (default `True`).

### When to use

- Operational 4DVar (ECMWF-style)
- Long assimilation windows where general `StrongFourDVar` would be slow
- When you want posterior covariance via Gauss-Newton Hessian without a
  separate adapter pass (the Hessian is already assembled in the last
  outer iteration)

---

## `FourDVarNet` — Learned 4DVar

### Mathematical formulation

Replace the Gaussian prior $\|x - x_b\|^2_{B^{-1}}$ with a learned prior
$\|x - \varphi_\theta(x)\|^2$, where $\varphi_\theta$ is an autoencoder
(`BilinAE`, `ConvAE`, `MLPAE`). Replace the standard gradient-descent
inner solver with a learned gradient modulator $\Phi_\phi$:

$$x_{k+1} = x_k - \Phi_\phi(\nabla_x J(x_k),\; h_k)$$

with $J(x) = \alpha_\text{obs} \|H(x) - y\|^2 + \alpha_\text{prior} \|x - \varphi_\theta(x)\|^2$.

Training minimises reconstruction MSE against ground truth:

$$\mathcal{L}(\theta, \phi) = \|x^*(\theta, \phi) - x_\text{true}\|^2.$$

The training gradient $\nabla_{\theta, \phi} \mathcal{L}$ flows through
the inner solver via the `solver_adjoint` slot (an
`optimistix.AbstractAdjoint`).

### Class contract

```python
class FourDVarNet(eqx.Module):
    prior: Prior                       # φ_θ
    obs_op: ObservationOperator
    grad_mod: GradModulator            # Φ_φ
    config: SolverConfig
    solver_adjoint: optimistix.AbstractAdjoint = RecursiveCheckpointAdjoint()
    forward_adjoint: diffrax.AbstractAdjoint = RecursiveCheckpointAdjoint()
    # forward_adjoint used only when prior is a DynamicalPrior

    def __call__(self, batch: Batch) -> Array:
        """Training interface: batch → x_reconstructed."""
        ...
    def as_analysis_step(self) -> AnalysisStep: ...
```

### Adjoint choices

| Adjoint | Memory | Convergence requirement | Notes |
|---|---|---|---|
| `optimistix.RecursiveCheckpointAdjoint()` | $O(K)$ checkpoints | None | Default; standard backprop with recursive checkpointing |
| `vardax.adjoints.OneStepAdjoint()` | $O(1)$ | None | Bolte et al. 2023; only the last step differentiable |
| `optimistix.ImplicitAdjoint()` | $O(1)$ | Yes | IFT-based at the fixed point; exact at convergence |

`OneStepAdjoint` is the v0.3 `"one_step"` mode promoted to an
`optimistix.AbstractAdjoint`. Vardax ships it in
`vardax._src.adjoints.one_step` with the goal of upstreaming once stable
(Decisions D6, D15).

### When to use

- Research / benchmarks where ground-truth $x_\text{true}$ is available
  for training
- Data-rich regimes (the learned $\varphi_\theta$ shines)
- Settings where the classical $B$ is hard to specify or known to be
  inadequate

---

## `AmortizedPosterior` — Direct head

### Mathematical formulation

Learn a conditional density $q_\phi(x \mid y)$ that approximates
$p(x \mid y)$. Train on simulated pairs from the prior and forward:

$$\phi^* = \underset{\phi}{\arg\min}\;
  \mathbb{E}_{(x, y) \sim p(x, y)}\;
  \mathrm{KL}\big(q_\phi(\cdot \mid y) \,\|\, p(\cdot \mid y)\big).$$

Head variants: conditional normalising flow (`gauss_flows`), score-based
diffusion, regression (Gaussian heads).

### Class contract

```python
class AmortizedPosterior(eqx.Module):
    encoder: eqx.Module                  # y, mask → conditioning context
    head: eqx.Module                     # context → posterior params / samples
    config: AmortizedConfig

    def __call__(self, batch: Batch) -> Array:
        """Return MAP / mode of q_φ(x | y)."""
        ...
    def sample(self, batch: Batch, key, n: int) -> Array: ...
    def log_prob(self, x: Array, batch: Batch) -> Scalar: ...
    def as_analysis_step(self) -> AnalysisStep: ...
```

### When to use

- Real-time inference (single forward pass, sub-second)
- Many independent events (catalog reprocessing)
- After validating against a `StrongFourDVar` / `IncrementalFourDVar`
  oracle on a held-out set (six-step cycle gates, Decision D12)

---

## `AnalysisStep` adapter (Decision D8)

All seven classes expose `.as_analysis_step()`:

```python
# Returned callable satisfies pipekit_cycle.AnalysisStep:
def analysis_fn(forecast, obs, *, obs_op, obs_err_cov) -> analysis: ...
```

The adapter shells the model's `__call__` to match the pipekit-cycle
analysis signature. Use it in `pipekit_cycle.DACycle`:

```python
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_model,
    obs_op=AveragingKernel(...),
    analysis_step=any_of_the_seven.as_analysis_step(),
    obs_source=load_obs_op,
    n_steps=n_assimilation_windows,
)
```

The same pipeline accepts `OptimalInterpolation`, `ThreeDVar`,
`StrongFourDVar`, `WeakFourDVar`, `IncrementalFourDVar`, `FourDVarNet`,
or `AmortizedPosterior` — what changes is the inference algorithm, not
the orchestration code.

---

## Linear-Gaussian baseline (Decision D14 invariant)

In the linear-Gaussian limit (linear $H$, Gaussian $B$ and $R$), all
seven methods must agree:

```
OptimalInterpolation(batch)
  ≈ ThreeDVar(batch)                                # iterative GN
  ≈ StrongFourDVar(batch)         (T = 0)
  ≈ WeakFourDVar(batch)           (T = 0, η = 0)
  ≈ IncrementalFourDVar(batch)    (T = 0)
  ≈ FourDVarNet(batch)            (IdentityPrior, n_steps large)
  ≈ AmortizedPosterior(batch)     (trained to convergence on this regime)
```

`tests/test_linear_gaussian_agreement.py` enforces this — it is the
canonical correctness baseline. If a new analysis method disagrees with
`OptimalInterpolation` in the linear-Gaussian limit, something is wrong.

---

## Training utilities

### `train_step` and `eval_step`

Library code (Decision D5) — encode correct differentiation through
whichever inner algorithm the model uses (4DVarNet inner solver,
amortized head, …):

```python
@eqx.filter_jit
def train_step(model, batch, optimizer, opt_state):
    loss, grads = eqx.filter_value_and_grad(train_loss_fn)(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def eval_step(model, batch) -> dict:
    x_recon = model(batch)
    return {"reconstruction": x_recon,
            "mse": jnp.mean((x_recon - batch.target) ** 2)}
```

Only `FourDVarNet` and `AmortizedPosterior` have learned parameters and
therefore use `train_step` / `eval_step`. The classical methods are
non-learnable — there are no parameters to optimise over (the prior
covariance $B$ is supplied, not learned).

### Integration with `pipekit-train` (`[train]` extra)

```python
from pipekit_train import MSE, EarlyStopping, Checkpoint, TrainingLoop
from vardax.training import train_step

loop = TrainingLoop(
    dataset=my_dataset,
    model_op=JaxModelOp(model),
    loss=MSE(),
    callbacks=[EarlyStopping(patience=10), Checkpoint(registry, every_n=1000)],
)
trained_model_op, trained_state = loop(JaxModelOp(model), TrainerCarryState(...))
```

---

## Posterior interface

Every model can be paired with a `PosteriorAdapter` (Decision D10):

```python
posterior_adapter = LaplaceCovariance()
# or GaussNewtonHessian(n_krylov=50)
# or EnsembleCovariance(n_members=32)

analysis = model(batch)
posterior = posterior_adapter(analysis, model, batch)
# Posterior(mean=..., cov=AbstractLinearOperator, samples=None, provenance={...})
```

`OptimalInterpolation` and `IncrementalFourDVar` additionally expose
`.posterior(batch)` directly, since their algorithms compute the
posterior covariance as part of the analysis (closed-form for OI,
GN-Hessian-reuse for incremental).

See [`../posterior.md`](../posterior.md) for the full contract.

---

*For ecosystem integration (somax, plumax, gaussx, filterax,
pipekit-cycle, coordax), see
[../examples/integration.md](../examples/integration.md). For end-to-end
walkthroughs (methane single-overpass, SSH 4DVarNet), see
[../examples/use_cases.md](../examples/use_cases.md).*
