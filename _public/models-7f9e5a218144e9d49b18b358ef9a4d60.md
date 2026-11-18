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
| `ThreeDVar` | 3D variational, nonlinear $h$, single time | D14 |
| `StrongFourDVar` | 4DVar, control = $\vect{x}_0$, exact dynamics | D14 |
| `WeakFourDVar` | 4DVar, control = $(\vect{x}_0, \gvect{\eta}_1, \ldots, \gvect{\eta}_T)$ | D14 |
| `IncrementalFourDVar` | GN outer + CG inner + CVT (fast `StrongFourDVar`) | D11 |
| `FourDVarNet` | Learned $g_\text{ae}$ + learned $g_\text{mod}$ | D3, D6 |
| `AmortizedPosterior` | Direct $q(\vect{x} \mid \vect{y}; \gvect{\nu})$ head | D12 |

All seven accept the same `Batch*` types and the same observation
operator family. The user picks the method that matches the regime.

---

## `OptimalInterpolation` — BLUE / OI

### Mathematical formulation

The Best Linear Unbiased Estimator for linear-Gaussian state estimation:

$$\vect{x}^\star = \gvect{\mu}_x + \mat{G} (\vect{y} - \mat{H} \gvect{\mu}_x), \qquad \mat{G} = \gmat{\Sigma}_x \mat{H}^\top (\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1}$$

with posterior covariance

$$\gmat{\Sigma}_{x \mid y} = (\gmat{\Sigma}_x^{-1} + \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H})^{-1} = (\mat{I} - \mat{G} \mat{H}) \gmat{\Sigma}_x.$$

Choice of form (Sherman-Morrison-Woodbury): use the $\gmat{\Sigma}_\varepsilon$-space form when
the observation dimension $D_y$ is smaller than the state dimension $D_x$;
the $\gmat{\Sigma}_x$-space form otherwise. `gaussx` handles structured $\gmat{\Sigma}_x$ and $\gmat{\Sigma}_\varepsilon$
without materialising dense matrices.

### Class contract

```{code-block} python
:caption: Declares `OptimalInterpolation`, the closed-form BLUE analysis [](#eq-blue-mean) with its posterior covariance [](#eq-blue-cov).
:label: code-design-api-models-oi
import equinox as eqx
import lineax as lx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): also ships cg_atol, cg_rtol, cg_max_steps and solver fields,
# and posterior(batch, adapter=None)
class OptimalInterpolation(eqx.Module):
    """BLUE / OI, closed-form linear-Gaussian analysis.

    Requires a linear observation operator. Use ThreeDVar for nonlinear
    H. Posterior covariance is included in the result without a
    separate PosteriorAdapter call.
    """

    # H, linear (validated in __init__)
    obs_op: pc.ObservationOperator
    # μ_x
    prior_mean: Float[Array, " N"]
    # Σ_x (gaussx-friendly)
    prior_cov_op: lx.AbstractLinearOperator
    # Σ_ε (gaussx-friendly)
    obs_cov_op: lx.AbstractLinearOperator

    def __call__(self, batch: Batch) -> Float[Array, "B N"]:
        """Return x⋆. No iteration, no convergence criterion."""
        # x⋆ = μ_x + G (y − H μ_x)
        ...

    def posterior(self, batch: Batch) -> vdx.Posterior:
        """Closed-form (mean, cov, provenance), no PosteriorAdapter needed."""
        # Σ_x|y = (Σ_x⁻¹ + Hᵀ Σ_ε⁻¹ H)⁻¹
        ...

    def as_analysis_step(self) -> pc.AnalysisStep: ...
```

### When to use

- $h$ is linear (or the linearisation around $\gvect{\mu}_x$ is good enough)
- $\gmat{\Sigma}_x$, $\gmat{\Sigma}_\varepsilon$, $h$ are Gaussian (or close enough that the posterior is
  unimodal Gaussian)
- Static field (no dynamics) or a single timestep

If any of these fail, `ThreeDVar` (nonlinear $h$) or
`StrongFourDVar` / `IncrementalFourDVar` (dynamics) is the right choice.
`OptimalInterpolation.__init__` validates linearity and refuses
nonlinear obs operators with a clear error message.

---

## `ThreeDVar` — 3D Variational

### Mathematical formulation

Minimise

$$\mathcal{J}(\vect{x}) = \frac{1}{2} \Vert \vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \Vert \vect{y} - h(\vect{x}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}$$

over $\vect{x}$. For nonlinear $h$, solve iteratively via Gauss-Newton, BFGS,
or NonlinearCG (chosen by the user via the `minimiser` slot).

In the linear-Gaussian limit, `ThreeDVar` recovers `OptimalInterpolation`
exactly — the conformance test suite verifies this agreement.

### Class contract

```{code-block} python
:caption: Declares `ThreeDVar`, which minimises the 3DVar cost [](#eq-3dvar-cost) with an optimistix minimiser.
:label: code-design-api-models-threedvar
import equinox as eqx
import lineax as lx
import optimistix as optx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): also ships cov_solver, max_steps and parameterization fields
class ThreeDVar(eqx.Module):
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    # today (optimistix)
    minimiser: optx.AbstractMinimiser
    # today (optimistix)
    minimiser_adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()

    # x⋆ = argmin_x 𝒥(x)
    def __call__(self, batch: Batch) -> Float[Array, "B N"]: ...

    def as_analysis_step(self) -> pc.AnalysisStep: ...
```

### When to use

- Nonlinear $h$ (e.g. averaging kernel + nonlinear retrieval)
- Single timestep (snapshot inversion)
- Posterior assumed Gaussian-around-MAP (`LaplaceCovariance` adapter)

---

## `StrongFourDVar` — Strong-constraint 4DVar

### Mathematical formulation

Control variable is the initial state $\vect{x}_0$ alone; dynamics $f_t$ are
treated as exact:

$$\mathcal{J}(\vect{x}_0) = \frac{1}{2} \Vert \vect{x}_0 - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{y}_t - h_t(f_t(\vect{x}_0)) \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}$$

Gradients of $\mathcal{J}$ with respect to $\vect{x}_0$ are computed by integrating the
adjoint of $f_t$ backwards in time — vardax delegates this to
`diffrax.AbstractAdjoint` (see Decision D15). The default
`RecursiveCheckpointAdjoint` is autodiff with recursive checkpointing;
`BacksolveAdjoint` solves the continuous adjoint ODE in reverse time
(constant memory, good for long windows).

### Class contract

```{code-block} python
:caption: Declares `StrongFourDVar`, which minimises the strong-constraint cost [](#eq-strong4dvar-cost) with the gradient [](#eq-strong4dvar-grad) taken through a diffrax adjoint.
:label: code-design-api-models-strong4dvar
import diffrax
import equinox as eqx
import lineax as lx
import optimistix as optx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): also ships cov_solver, max_steps, parameterization and checkpoint
class StrongFourDVar(eqx.Module):
    # today (pipekit_cycle)
    forward: pc.ForwardModel
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    # today (optimistix)
    minimiser: optx.AbstractMinimiser
    # today (optimistix)
    minimiser_adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()
    # today (diffrax)
    forward_adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

    # x₀⋆ = argmin_{x₀} 𝒥(x₀)
    def __call__(self, batch: Batch) -> Float[Array, "B N"]: ...

    def as_analysis_step(self) -> pc.AnalysisStep: ...
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

$$\gvect{\eta} = (\gvect{\eta}_1, \ldots, \gvect{\eta}_T), \quad \vect{x}_t = f_t^\text{free}(\vect{x}_{t-1}) + \gvect{\eta}_t$$

with cost

$$\mathcal{J}(\vect{x}_0, \gvect{\eta}) = \frac{1}{2} \Vert \vect{x}_0 - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \sum_t \Vert \vect{y}_t - h_t(\vect{x}_t) \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}} + \frac{1}{2} \sum_t \Vert \gvect{\eta}_t \Vert^2_{\gmat{\Sigma}_{\eta,t}^{-1}}$$

The model-error covariance $\gmat{\Sigma}_{\eta,t}$ is a separate input.

### Class contract

```{code-block} python
:caption: Declares `WeakFourDVar`, which minimises the weak-constraint cost [](#eq-weak4dvar-cost) over x₀ and the model errors ηₜ of [](#eq-weak-model).
:label: code-design-api-models-weak4dvar
import diffrax
import equinox as eqx
import lineax as lx
import optimistix as optx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; T = window length; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): also ships cov_solver, max_steps, parameterization and checkpoint
class WeakFourDVar(eqx.Module):
    # today (pipekit_cycle)
    # f_t^free
    forward: pc.ForwardModel
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    prior_mean: Float[Array, " N"]
    # Σ_x
    prior_cov_op: lx.AbstractLinearOperator
    # Σ_ε
    obs_cov_op: lx.AbstractLinearOperator
    # Σ_η
    model_err_cov_op: lx.AbstractLinearOperator
    # today (optimistix)
    minimiser: optx.AbstractMinimiser
    # today (optimistix)
    minimiser_adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()
    # today (diffrax)
    forward_adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

    def __call__(
        self, batch: Batch
    ) -> tuple[Float[Array, "B N"], Float[Array, "B T N"]]:
        """Return (x₀⋆, η⋆): analysis initial condition and model error."""
        # (x₀⋆, η⋆) = argmin 𝒥(x₀, η)
        ...

    def as_analysis_step(self) -> pc.AnalysisStep: ...
```

### When to use

- Multi-timestep observations with non-negligible model error
- Long assimilation windows where dynamics drift away from observations
- Climatological or seasonal mismatch between $f_t^\text{free}$ and
  reality

`WeakFourDVar` is computationally heavier than `StrongFourDVar` because
the control vector dimension scales with $T$. Use sparingly; consider
`StrongFourDVar` with a learned $g_\text{ae}$ as a cheaper
alternative.

---

## `IncrementalFourDVar` — Operational Fast Path

### Mathematical formulation

Functionally equivalent to `StrongFourDVar`, but uses Gauss-Newton outer
iterations on the full nonlinear cost with CG / Lanczos inner iterations
on the linearised quadratic subproblem. At each outer iterate $\vect{x}^{(i)}$
(with $\vect{x}^{(0)} = \gvect{\mu}_x$; the background $\gvect{\mu}_x$ stays fixed), linearise $f_t$
and $h_t$ via `jax.linearize`:

$$\delta\mathcal{J}(\delta\vect{x}) = \frac{1}{2} \Vert \vect{x}^{(i)} + \delta\vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \sum_t \Vert \vect{r}_t^{(i)} - \mat{H}_t \mat{F}_{t \leftarrow 0} \delta\vect{x} \Vert^2_{\gmat{\Sigma}_{\varepsilon,t}^{-1}}$$

with innovation $\vect{r}_t^{(i)} = \vect{y}_t - h_t(f_t(\vect{x}^{(i)}))$. Solve
$\mat{H}_\text{GN} \delta\vect{x}^\star = -\nabla \mathcal{J}(\vect{x}^{(i)})$ by CG (the
right-hand side keeps the background misfit $-\gmat{\Sigma}_x^{-1}(\vect{x}^{(i)} - \gvect{\mu}_x)$),
update $\vect{x}^{(i+1)} = \vect{x}^{(i)} + \delta\vect{x}^\star$, relinearise.

**Control-variable transform** (CVT): set $\vect{z} = \mat{L}_x^{-1} \delta\vect{x}$;
the prior term becomes $\Vert \vect{z} \Vert^2$ and the inner CG operates on a
well-conditioned problem. $\mat{L}_x = \gmat{\Sigma}_x^{1/2}$ comes from
`gaussx.MaternLinearOperator.half()`.

### Class contract

```{code-block} python
:caption: Declares `IncrementalFourDVar`, which minimises the incremental cost [](#eq-incremental-cost) by Gauss-Newton steps [](#eq-gn-system) and reuses the Hessian for the covariance [](#eq-gn-cov).
:label: code-design-api-models-incremental4dvar
import diffrax
import equinox as eqx
import lineax as lx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): also ships prior_sqrt_op and checkpoint fields,
# and posterior(batch, adapter=None)
class IncrementalFourDVar(eqx.Module):
    # today (pipekit_cycle)
    forward: pc.ForwardModel
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    prior_mean: Float[Array, " N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    # today (vardax)
    config: vdx.IncrementalConfig
    # today (diffrax)
    forward_adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

    # 𝐇_GN δx⋆ = −∇𝒥(x⁽ⁱ⁾), x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ + δx⋆
    def __call__(self, batch: Batch) -> Float[Array, "B N"]: ...

    def posterior(self, batch: Batch) -> vdx.Posterior:
        """Reuse the Hessian from the last outer iteration for GN-Hessian UQ."""
        # Σ_x|y ≈ 𝐇_GN⁻¹
        ...

    def as_analysis_step(self) -> pc.AnalysisStep: ...
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

Replace the Gaussian prior $\Vert \vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}}$ with a learned prior
$\Vert \vect{x} - g_\text{ae}(\vect{x}; \gvect{\theta}) \Vert^2$, where $g_\text{ae}$ is an autoencoder
(`BilinAE`, `ConvAE`, `MLPAE`). Replace the standard gradient-descent
inner solver with a learned gradient modulator $g_\text{mod}$:

$$\vect{x}^{(i+1)} = \vect{x}^{(i)} - g_\text{mod}(\nabla_{\vect{x}} \mathcal{J}(\vect{x}^{(i)}), \vect{c}^{(i)})$$

with $\mathcal{J}(\vect{x}) = \alpha_\text{obs} \Vert h(\vect{x}) - \vect{y} \Vert^2 + \alpha_\text{prior} \Vert \vect{x} - g_\text{ae}(\vect{x}; \gvect{\theta}) \Vert^2$.

Training minimises reconstruction MSE against ground truth:

$$\mathcal{L}(\gvect{\theta}, \gvect{\theta}_\text{mod}) = \Vert \vect{x}^\star(\gvect{\theta}, \gvect{\theta}_\text{mod}) - \vect{x}_\text{true} \Vert^2.$$

The training gradient $\nabla_{\gvect{\theta}, \gvect{\theta}_\text{mod}} \mathcal{L}$ flows through
the inner solver via the `solver_adjoint` slot (an
`optimistix.AbstractAdjoint`).

### Class contract

```{code-block} python
:caption: Declares the generic `FourDVarNet`, which iterates the learned update [](#eq-4dvarnet-update) on the cost [](#eq-4dvarnet-cost) and trains on the loss [](#eq-4dvarnet-loss).
:label: code-design-api-models-fourdvarnet
import diffrax
import equinox as eqx
import optimistix as optx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; T = time; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# proposed (D3, D6): ships as FourDVarNet1D / FourDVarNet2D, with
# n_solver_steps, alpha and prior_weight fields and no forward_adjoint
class FourDVarNet(eqx.Module):
    # today (vardax)
    # g_ae
    prior: vdx.Prior
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    # today (vardax)
    # g_mod
    grad_mod: vdx.GradModulator
    # proposed (components.md)
    config: SolverConfig
    # today (optimistix)
    solver_adjoint: optx.AbstractAdjoint = optx.RecursiveCheckpointAdjoint()
    # today (diffrax)
    # used only when prior is a DynamicalPrior
    forward_adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

    def __call__(self, batch: Batch) -> Float[Array, "B T N"]:
        """Training interface: batch → x_reconstructed."""
        # x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − g_mod(∇ₓ𝒥(x⁽ⁱ⁾), c⁽ⁱ⁾)
        ...

    def as_analysis_step(self) -> pc.AnalysisStep: ...
```

### Adjoint choices

| Adjoint | Memory | Convergence requirement | Notes |
|---|---|---|---|
| `optimistix.RecursiveCheckpointAdjoint()` | $O(N_\text{iter})$ checkpoints | None | Default; standard backprop with recursive checkpointing |
| `vardax.adjoints.OneStepAdjoint()` | $O(1)$ | None | Bolte et al. 2023; only the last step differentiable |
| `optimistix.ImplicitAdjoint()` | $O(1)$ | Yes | IFT-based at the fixed point; exact at convergence |

`OneStepAdjoint` is the v0.3 `"one_step"` mode promoted to an
`optimistix.AbstractAdjoint`. Vardax ships it in
`vardax._src.adjoints.one_step` with the goal of upstreaming once stable
(Decisions D6, D15).

### When to use

- Research / benchmarks where ground-truth $\vect{x}_\text{true}$ is available
  for training
- Data-rich regimes (the learned $g_\text{ae}$ shines)
- Settings where the classical $\gmat{\Sigma}_x$ is hard to specify or known to be
  inadequate

---

## `AmortizedPosterior` — Direct head

### Mathematical formulation

Learn a conditional density $q(\vect{x} \mid \vect{y}; \gvect{\nu})$ that approximates
$p(\vect{x} \mid \vect{y})$. Train on simulated pairs from the prior and forward:

$$\gvect{\nu}^\star = \underset{\gvect{\nu}}{\op{argmin}} \mathbb{E}_{(\vect{x}, \vect{y}) \sim p(\vect{x}, \vect{y})} \mathrm{KL}\left(q(\cdot \mid \vect{y}; \gvect{\nu}) \Vert p(\cdot \mid \vect{y})\right).$$

Head variants: conditional normalising flow (`gauss_flows`), score-based
diffusion, regression (Gaussian heads).

### Class contract

```{code-block} python
:caption: Declares `AmortizedPosterior`, the conditional density q(x ∣ y; ν) trained on the amortized objective [](#eq-amortized-objective).
:label: code-design-api-models-amortized
import equinox as eqx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: B = batch; S = samples; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): shipped sample takes n: int | None = None,
# and log_prob returns one value per batch item
class AmortizedPosterior(eqx.Module):
    # y, mask → conditioning context
    encoder: eqx.Module
    # context → posterior params / samples
    head: eqx.Module
    # today* (vardax)
    config: vdx.AmortizedConfig

    def __call__(self, batch: Batch) -> Float[Array, "B N"]:
        """Return MAP / mode of q(x | y; ν)."""
        # x̂ = mode q(x ∣ y; ν)
        ...

    # xˢ ~ q(x ∣ y; ν)
    def sample(
        self, batch: Batch, key: PRNGKeyArray, n: int
    ) -> Float[Array, "B S N"]: ...

    # log q(x ∣ y; ν)
    def log_prob(self, x: Float[Array, "B N"], batch: Batch) -> Float[Array, ""]: ...

    def as_analysis_step(self) -> pc.AnalysisStep: ...
```

### When to use

- Real-time inference (single forward pass, sub-second)
- Many independent events (catalog reprocessing)
- After validating against a `StrongFourDVar` / `IncrementalFourDVar`
  oracle on a held-out set (six-step cycle gates, Decision D12)

---

## `AnalysisStep` adapter (Decision D8)

All seven classes expose `.as_analysis_step()`:

```{code-block} python
:caption: Shows the pipekit-cycle analysis signature that `.as_analysis_step()` returns, mapping a forecast μ_x and observations y to an analysis such as [](#eq-blue-mean).
:label: code-design-api-models-analysis-step
import lineax as lx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N = state size; M = observation size


# today (pipekit_cycle): the pc.AnalysisStep call signature
def analysis_fn(
    forecast: Float[Array, " N"],
    obs: Float[Array, " M"],
    *,
    obs_op: pc.ObservationOperator,
    obs_err_cov: lx.AbstractLinearOperator,
) -> Float[Array, " N"]: ...
```

The adapter shells the model's `__call__` to match the pipekit-cycle
analysis signature. The `OptimalInterpolation` and `ThreeDVar` adapters
analyse around `forecast` (the background $\gvect{\mu}_x$, not the model's
`prior_mean`) and use the cycle's `obs_op` and `obs_err_cov` ($h$ and $\gmat{\Sigma}_\varepsilon$;
`None` falls back to the model's own), keeping the model's $\gmat{\Sigma}_x$. Use it in
`pipekit_cycle.DACycle`:

```{code-block} python
:caption: Runs any of the seven analysis steps inside a pipekit-cycle `DACycle`, alternating the rollout [](#eq-rollout) with an analysis.
:label: code-design-api-models-da-cycle
import pipekit_cycle as pc
import vardax as vdx

# Shapes: no arrays (configuration)

# today (pipekit_cycle)
da_cycle = pc.DACycle(
    forward_model=somax_model,
    # today* (vardax)
    obs_op=vdx.AveragingKernel(...),
    # today (vardax)
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

In the linear-Gaussian limit (linear $h$, Gaussian $\gmat{\Sigma}_x$ and $\gmat{\Sigma}_\varepsilon$), all
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

```{code-block} python
:caption: Defines the training and evaluation steps that minimise the reconstruction loss [](#eq-4dvarnet-loss) by differentiating through the inner solver.
:label: code-design-api-models-train-step
import einx
import equinox as eqx
import optax
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: B = batch; T = time; N = state size

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today (vardax)
@eqx.filter_jit
def train_step(
    model: eqx.Module,
    batch: Batch,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
    # today (vardax)
    # ℒ(θ, ν), ∇_{θ, ν} ℒ
    loss, grads = eqx.filter_value_and_grad(vdx.train_loss_fn)(model, batch)
    # today (optax)
    updates, opt_state = optimizer.update(grads, opt_state)
    # (θ, ν) ← (θ, ν) + Δ
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# today* (vardax): shipped eval_step returns the scalar loss, not a dict
def eval_step(model: eqx.Module, batch: Batch) -> dict[str, Array]:
    # x⋆(θ, ν)
    x_recon = model(batch)  # → (B, T, N)
    # x⋆ − x_true
    residual = x_recon - batch.target  # (B, T, N)
    # (x⋆ − x_true)²
    squared = residual**2  # (B, T, N)
    # ℒ = mean (x⋆ − x_true)²
    mse = einx.mean("b t n ->", squared)  # (B, T, N) → ()
    return {"reconstruction": x_recon, "mse": mse}
```

Only `FourDVarNet` and `AmortizedPosterior` have learned parameters and
therefore use `train_step` / `eval_step`. The classical methods are
non-learnable — there are no parameters to optimise over (the prior
covariance $\gmat{\Sigma}_x$ is supplied, not learned).

### Integration with `pipekit-train` (`[train]` extra)

```{code-block} python
:caption: Runs the training step for the loss [](#eq-4dvarnet-loss) inside a `pipekit_train.TrainingLoop` (the `[train]` extra).
:label: code-design-api-models-pipekit-train
import pipekit_jax as pj
import pipekit_train as pt
import vardax as vdx

# Shapes: no arrays (configuration)

# today (vardax): vdx.train_step; there is no vardax.training module
train_step = vdx.train_step

# not pinned (pipekit_train)
loop = pt.TrainingLoop(
    dataset=my_dataset,
    # not pinned (pipekit_jax)
    model_op=pj.JaxModelOp(model),
    # not pinned (pipekit_train)
    loss=pt.MSE(),
    # not pinned (pipekit_train)
    callbacks=[pt.EarlyStopping(patience=10), pt.Checkpoint(registry, every_n=1000)],
)
# not pinned (pipekit_train)
trained_model_op, trained_state = loop(pj.JaxModelOp(model), pt.TrainerCarryState(...))
```

---

## Posterior interface

Every model can be paired with a `PosteriorAdapter` (Decision D10):

```{code-block} python
:caption: Pairs a model with a posterior adapter that returns the Laplace covariance [](#eq-laplace-cov), or the Gauss-Newton [](#eq-gn-cov) or ensemble [](#eq-ensemble-cov) alternatives.
:label: code-design-api-models-posterior-adapter
import vardax as vdx

# Shapes: B = batch; N = state size; M = ensemble members

# today (vardax)
posterior_adapter = vdx.LaplaceCovariance()
# alternatives, both today (vardax):
# vdx.GaussNewtonHessian(n_krylov=50)
# vdx.EnsembleCovariance(n_members=32)

# user
# x⋆ = argmin 𝒥(x)
analysis = model(batch)  # → (B, N)
# today (vardax)
# Σ_x|y = (∇²𝒥(x⋆))⁻¹
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
