---
status: draft
version: 0.3.0
---

# Layer 2 — Models

End-to-end VarDA models that compose Layer 1 operators with Layer 0
primitives. Three families:

| Family | Purpose | Decision |
|---|---|---|
| `VarDANet*` | Learned 4DVarNet with prior + ConvLSTM grad modulator | D3 |
| `IncrementalVarDA*` | Operational 4DVar with CVT + Gauss-Newton + CG | D11 |
| `AmortizedVarDA*` | Direct $q_\phi(x \mid y)$ head (flow / score / regression) | D12 |

All satisfy `pipekit_cycle.AnalysisStep` via `.as_analysis_step()` (D8).
All accept the same `Batch*` and `ObservationOperator` types — only the
inference algorithm differs.

---

## `VarDANet*` — Learned 4DVarNet

### Mathematical formulation

VarDANet solves the variational DA problem:

$$x^* = \underset{x}{\arg\min}\; J(x) = \alpha_\text{obs}\|H(x) - y\|^2_{R^{-1}} + \alpha_\text{prior}\|x - \varphi(x)\|^2$$

via a learned iterative solver with gradient modulation:

$$x_{k+1} = x_k - \Phi_\theta(\nabla_x J(x_k),\; h_k), \quad k = 0, \ldots, K-1$$

where $\Phi_\theta$ is the learned gradient modulator (ConvLSTM, attention,
MLP) with parameters $\theta$.

### Training objective

$$\mathcal{L}(\theta, \psi) = \|x^*(\theta, \psi) - x_\text{true}\|^2$$

where $\psi$ are prior parameters. Training gradient $\nabla_{\theta, \psi}
\mathcal{L}$ flows through the inner solver via `grad_mode` ∈
{`"unrolled"`, `"one_step"`, `"implicit"`}.

### Class contract

```python
class VarDANet(eqx.Module):
    prior: Prior
    obs_op: ObservationOperator
    grad_mod: GradModulator
    config: SolverConfig

    def __call__(self, batch: Batch) -> Array:
        """Training interface: batch → x_reconstructed."""
        ...

    def as_analysis_step(self) -> AnalysisStep:
        """Operational interface: returns AnalysisStep-compliant callable."""
        ...
```

### Dimensional variants

| Class | Spatial dims | Default grad mod |
|---|---|---|
| `VarDANet1D` | `(B, T, N)` | `ConvLSTMGradMod1D` |
| `VarDANet2D` | `(B, T, H, W)` | `ConvLSTMGradMod2D` |
| `VarDANet3D` | `(B, T, D, H, W)` | `ConvLSTMGradMod3D` (planned) |

---

## `IncrementalVarDA*` — Operational 4DVar (Decision D11)

### Mathematical formulation

Operational incremental 4DVar with control-variable transform. The
three-term cost:

$$J(x_0) = \frac{1}{2}\|x_0 - x_b\|^2_{B^{-1}} + \frac{1}{2}\sum_t \|y_t - H_t(M_t(x_0))\|^2_{R^{-1}}$$

is solved iteratively. At each outer iteration, linearise around the current
$x_b$:

$$\delta J(\delta x) = \frac{1}{2}\|\delta x\|^2_{B^{-1}} + \frac{1}{2}\sum_t \|d_t - H'_t M'_t \delta x\|^2_{R^{-1}}$$

with $d_t = y_t - H_t(M_t(x_b))$, $H'_t = \partial H / \partial x$,
$M'_t = \partial M / \partial x$ from `jax.linearize`. Inner CG / Lanczos
on the tangent-linear cost.

**Control-variable transform.** With $\chi = B^{-1/2}(\delta x)$, the prior
cost becomes $\|\chi\|^2$ (identity Gaussian), preconditioning CG.

### Class contract

```python
class IncrementalVarDA(eqx.Module):
    forward: ForwardModel            # tangent-linear via jax.linearize
    obs_op: ObservationOperator
    prior_mean: Array                # x_b
    prior_cov_op: AbstractLinearOperator  # B (gaussx Matérn)
    obs_cov_op: AbstractLinearOperator    # R (gaussx diagonal / block-diag)
    config: IncrementalConfig

    def __call__(self, batch: Batch) -> Array:
        """Run incremental 4DVar: x_b + sum of inner increments."""
        ...

    def as_analysis_step(self) -> AnalysisStep:
        ...
```

### Dimensional variants

| Class | Spatial dims | Notes |
|---|---|---|
| `IncrementalVarDA2D` | `(B, T, H, W)` | Default for SSH, SST, methane column |
| `IncrementalVarDA3D` | `(B, T, D, H, W)` | Volumetric — methane Eulerian, ocean primitive equations |

### Training

`IncrementalVarDA*` is **not learned** by default — no parameters to train.
The forward model is supplied; the prior is `prior_mean + B`. Users may
train the prior covariance hyperparameters (e.g. Matérn $\ell$, $\sigma$) via
`vardax.training.train_hyperparams` (planned, Epic 5).

---

## `AmortizedVarDA` — Direct posterior head (Decision D12)

### Mathematical formulation

Learn a direct head $q_\phi(x \mid y)$ that approximates the Bayesian
posterior:

$$\phi^* = \underset{\phi}{\arg\min}\; \mathbb{E}_{(x, y) \sim p(x, y)}\; \mathrm{KL}\big(q_\phi(\cdot \mid y)\,\|\,p(\cdot \mid y)\big)$$

For samplable forward + prior, the training distribution comes from
simulation: draw $x \sim p(x)$, simulate $y \sim p(y \mid x) = H(\text{Forward}(x)) + \varepsilon$,
optimise $q_\phi$ against these pairs.

**Head variants:**

- **Conditional flow** (`head_type="flow"`) — `gauss_flows` conditional
  normalising flow. Exact density, exact samples.
- **Score-based** (`head_type="score"`) — learned $s_\phi(x, t \mid y)$.
  Sampling via reverse SDE.
- **Regression** (`head_type="regression"`) — direct $\mu_\phi(y)$ /
  $\sigma_\phi(y)$ heads. Cheapest, restricted to Gaussian families.

### Class contract

```python
class AmortizedVarDA(eqx.Module):
    encoder: eqx.Module                  # y, mask → conditioning context
    head: eqx.Module                     # context → q_φ parameters / samples
    config: AmortizedConfig

    def __call__(self, batch: Batch) -> Array:
        """Return MAP / mode of q_φ(x | y)."""
        ...

    def sample(self, batch: Batch, key, n: int) -> Array:
        """Draw n posterior samples q_φ(x | y)."""
        ...

    def log_prob(self, x: Array, batch: Batch) -> Scalar:
        """Evaluate log q_φ(x | y). May raise NotImplementedError for
        score-based heads where exact density is unavailable."""
        ...

    def as_analysis_step(self) -> AnalysisStep:
        ...
```

### Validation gates (Decision D12)

`AmortizedVarDA` is only useful if its output matches a slower oracle.
`tests/test_six_step_validation.py` enforces:

- Posterior agreement: amortized MAP within $1\sigma_\text{post}$ of
  `IncrementalVarDA` MAP on held-out events.
- Adjoint calibration: $\|\partial \text{amortized} / \partial x - \partial \text{physics} / \partial x\|_\text{op} < 5\%$
  before promotion.
- SBC: rank histograms uniform across parameters.

---

## `AnalysisStep` adapter (Decision D8)

All three model families expose `.as_analysis_step()`:

```python
# Returned callable satisfies pipekit_cycle.AnalysisStep:
def analysis_fn(forecast, obs, *, obs_op, obs_err_cov) -> analysis: ...

# Equivalent to:
batch = Batch2D(input=obs, mask=...mask_from(obs)..., target=None,
                obs_err=jnp.sqrt(jnp.diag(obs_err_cov)))
return model(batch)
```

The adapter shells the model's `__call__` to match the pipekit-cycle
analysis signature. Use it directly in `pipekit_cycle.DACycle` /
`SmootherCycle`:

```python
import pipekit_cycle as pc

da_cycle = pc.DACycle(
    forward_model=somax_model,        # any ForwardModel
    obs_op=my_obs_op,                  # any ObservationOperator (e.g. AveragingKernel)
    analysis_step=my_model.as_analysis_step(),  # vardax model
    obs_source=load_obs_op,
    n_steps=n_assimilation_windows,
)
```

---

## Training utilities

### `train_step`

Library code (D5) — encodes correct differentiation through the inner
solver:

```python
@eqx.filter_jit
def train_step(model, batch, optimizer, opt_state):
    loss, grads = eqx.filter_value_and_grad(train_loss_fn)(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
```

### `eval_step`

```python
def eval_step(model, batch) -> dict:
    x_recon = model(batch)
    return {"reconstruction": x_recon,
            "mse": jnp.mean((x_recon - batch.target)**2)}
```

### Integration with `pipekit-train` (optional `[train]` extra)

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

`train_step` is the inner primitive; `TrainingLoop` is the orchestration
wrapper. Use either depending on how much control you need.

---

## Posterior interface

Every model can be paired with a `PosteriorAdapter`:

```python
posterior_adapter = LaplaceCovariance()
# or GaussNewtonHessian(n_krylov=50)
# or EnsembleCovariance(n_members=32)

analysis = model(batch)
posterior = posterior_adapter(analysis, model, batch)
# Posterior(mean=..., cov=AbstractLinearOperator, samples=None, provenance={...})

# Export to population model (Decision D10):
mark = GaussianMarkLikelihood(posterior, event_metadata).to_dict()
```

See [`../posterior.md`](../posterior.md) for the full posterior contract.

---

*For ecosystem integration (somax, plumax, gaussx, filterax, pipekit-cycle,
coordax), see [../examples/integration.md](../examples/integration.md). For
end-to-end walkthroughs (methane single-overpass, SSH 4DVarNet), see
[../examples/use_cases.md](../examples/use_cases.md).*
