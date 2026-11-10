---
status: draft
version: 0.4.0
---

# Layer 0 — Primitives

Pure JAX functions. Stateless, differentiable, no `eqx.Module`, no
`optimistix`. The mathematical building blocks that Layer 1 operators
compose.

v0.4 removes the in-house gradient-mode primitives (`unrolled_solve`,
`one_step_solve`, `implicit_solve`) in favour of upstream
`optimistix.AbstractAdjoint` and `diffrax.AbstractAdjoint` (Decision
D15). The remaining primitives are mathematical, not differentiation
strategies.

---

## Cost terms

### `obs_cost(x, y_obs, mask, obs_operator=None, R_inv_op=None)`

Observation cost:

$$\mathcal{J}_o(\vect{x}) = \frac{1}{2} \Vert h(\vect{x}) - \vect{y} \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}.$$

When `obs_operator=None`, $h$ defaults to masked identity. When
`R_inv_op=None`, $\gmat{\Sigma}_\varepsilon^{-1}$ defaults to $\mat{I} / \vert \Omega \vert$ (averaged MSE).

### `prior_cost(x, prior_mean=None, B_inv_op=None, prior_fn=None)`

Two distinct meanings depending on which kwargs are passed:

**Classical** (`prior_mean` and `B_inv_op` supplied):

$$\mathcal{J}_p(\vect{x}) = \frac{1}{2} \Vert \vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}}.$$

**Learned** (`prior_fn` supplied — `FourDVarNet`):

$$\mathcal{J}_p(\vect{x}) = \Vert \vect{x} - g_\text{ae}(\vect{x}; \gvect{\theta}) \Vert^2.$$

The two forms cannot be mixed — pass one set of arguments or the other.

### `model_error_cost(eta, Q_inv_op)`

Weak-constraint 4DVar model-error term:

$$\mathcal{J}_\eta(\gvect{\eta}) = \frac{1}{2} \sum_{t=1}^{T} \Vert \gvect{\eta}_t \Vert^2_{\gmat{\Sigma}_{\eta,t}^{-1}}.$$

### `variational_cost(x, batch, prior_fn, obs_operator, alpha_obs=1.0, alpha_prior=1.0)`

Weak-constraint variational cost (learned-prior form):

$$\mathcal{J}(\vect{x}) = \alpha_\text{obs} \mathcal{J}_o(\vect{x}) + \alpha_\text{prior} \mathcal{J}_p(\vect{x}).$$

Used by `FourDVarNet`.

### `threedvar_cost(x, batch, obs_operator, prior_mean, B_inv_op, R_inv_op)`

Classical 3DVar cost:

$$\mathcal{J}(\vect{x}) = \frac{1}{2} \Vert \vect{x} - \gvect{\mu}_x \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \Vert \vect{y} - h(\vect{x}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}.$$

### `incremental_cost(δx, x_b, batch, forward_lin, obs_op_lin, B_inv_op, R_inv_op)`

Linearised incremental cost for the 4DVar inner loop:

$$\delta\mathcal{J}(\delta\vect{x}) = \frac{1}{2} \Vert \delta\vect{x} \Vert^2_{\gmat{\Sigma}_x^{-1}} + \frac{1}{2} \sum_t \Vert \vect{y}_t - h_t(\gvect{\mu}_x) - \mat{H}_t \mat{F}_t \delta\vect{x} \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}.$$

$\mat{H}_t$ and $\mat{F}_t$ are tangent-linear operators at the outer iterate
$\gvect{\mu}_x$.

---

## Closed-form analysis

### `blue_analysis(x_b, y, B_op, R_op, H_op)`

BLUE / OI in closed form (Decision D16):

$$\vect{x}^\star = \gvect{\mu}_x + \gmat{\Sigma}_x \mat{H}^\top (\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1} (\vect{y} - \mat{H} \gvect{\mu}_x).$$

Returns `(x_star, posterior_cov_op)` where `posterior_cov_op` is an
`AbstractLinearOperator` representing $\gmat{\Sigma}_{x \mid y}$ — not materialised by
default. `gaussx` exploits structure in $\gmat{\Sigma}_x$, $\gmat{\Sigma}_\varepsilon$, $\mat{H}$.

```{code-block} python
:caption: Declares the planned closed-form BLUE primitive, returning the mean [](#eq-blue-mean) and the covariance operator [](#eq-blue-cov).
:label: code-design-api-primitives-blue
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size; M = observation size


# proposed (D16): OptimalInterpolation ships this analysis, not a standalone primitive
def blue_analysis(
    x_b: Float[Array, " N"],
    y: Float[Array, " M"],
    B_op: lx.AbstractLinearOperator,
    R_op: lx.AbstractLinearOperator,
    H_op: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    # x⋆ = μ_x + Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ (y − H μ_x)
    # Σ_x|y = (Σ_x⁻¹ + Hᵀ Σ_ε⁻¹ H)⁻¹, returned as an (N, N) operator
    ...
```

The implementation picks between the $\gmat{\Sigma}_x$-space form (Sherman-Morrison-
Woodbury, useful when state is high-dimensional) and the $\gmat{\Sigma}_\varepsilon$-space form
(useful when observation count is low) based on dimensions.

---

## Control-variable transform

### `cvt_transform(x, x_b, B_half_op)` and `cvt_inverse(chi, x_b, B_half_op)`

CVT: $\vect{z} = \mat{L}_x^{-1} (\vect{x} - \gvect{\mu}_x)$, with inverse $\vect{x} = \gvect{\mu}_x + \mat{L}_x \vect{z}$, where $\mat{L}_x = \gmat{\Sigma}_x^{1/2}$.

$\mat{L}_x$ comes from `gaussx.MaternLinearOperator.half()` when the prior
is Matérn; falls back to Cholesky factorisation for arbitrary
`AbstractLinearOperator`.

```{code-block} python
:caption: Declares the planned control-variable transform z = L_x⁻¹(x − μ_x) of [](#eq-cvt) and its inverse.
:label: code-design-api-primitives-cvt
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size


# proposed (D11): IncrementalFourDVar applies the CVT internally (config.cvt)
def cvt_transform(
    x: Float[Array, " N"], x_b: Float[Array, " N"], B_half_op: lx.AbstractLinearOperator
) -> Float[Array, " N"]:
    # z = L_x⁻¹ (x − μ_x)
    ...


# proposed (D11)
def cvt_inverse(
    chi: Float[Array, " N"],
    x_b: Float[Array, " N"],
    B_half_op: lx.AbstractLinearOperator,
) -> Float[Array, " N"]:
    # x = μ_x + L_x z
    ...
```

In CVT coordinates the prior cost is $\Vert \vect{z} \Vert^2$ (identity Gaussian),
which preconditions the CG inner loop in `IncrementalFourDVar`.

---

## Tangent-linear and adjoint

### `tangent_linear(fn, x)`

Wraps `jax.linearize` to return an `AbstractLinearOperator`:

```{code-block} python
:caption: Defines the planned tangent-linear primitive, which wraps `jax.linearize` as the operator F of [](#eq-tlm-forward).
:label: code-design-api-primitives-tangent-linear
from collections.abc import Callable

import jax
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size


# proposed (primitives.md)
def tangent_linear(
    fn: Callable[[Float[Array, " N"]], Float[Array, " N"]], x: Float[Array, " N"]
) -> lx.AbstractLinearOperator:
    """Linearise fn at x. Returns operator suitable for incremental 4DVar."""
    # today (jax)
    # δx ↦ F(x) δx
    _, jvp = jax.linearize(fn, x)
    # today* (lineax): the shipped keyword is input_structure, a ShapeDtypeStruct
    return lx.FunctionLinearOperator(jvp, in_structure=x)
```

The transpose / adjoint comes for free via `lineax`.

---

## Posterior primitives

### `laplace_covariance(x_star, cost_grad_fn, B_inv_op, R_inv_op)`

Laplace approximation at MAP:

$$\gmat{\Sigma}_{x \mid y} = \left(\mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H} + \gmat{\Sigma}_x^{-1}\right)^{-1}.$$

Returns `AbstractLinearOperator` — supports mat-vec via `lineax.CG`, no
materialisation.

### `gauss_newton_hessian(x_star, batch, forward, obs_op, B_op, R_op, n_krylov=50)`

Gauss-Newton Hessian at MAP, via Krylov / Lanczos. Returns
`AbstractLinearOperator` representing $\nabla^2 \mathcal{J}(\vect{x}^\star)$ for posterior
inversion.

---

## Training primitives

### `reconstruction_loss(pred, target)`

$$\mathcal{L}(\gvect{\theta}) = \Vert \vect{x}^\star(\gvect{\theta}) - \vect{x}_\text{true} \Vert^2.$$

### `train_loss_fn(model, batch)`

Forward through `model(batch) → x*`, then
`reconstruction_loss(x*, batch.target)`. Gradients flow through the
inner solver according to `model.solver_adjoint`.

### `train_step(model, batch, optimizer, opt_state)`

One outer training step: loss → backprop → optimiser update. Encodes
the correctness-critical differentiation pattern (Decisions D5, D15).

```{code-block} python
:caption: Declares the training step that updates the model parameters on the reconstruction loss [](#eq-4dvarnet-loss).
:label: code-design-api-primitives-train-step
import equinox as eqx
import optax
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: no arrays beyond the batch; loss → ()

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D


# today (vardax)
def train_step(
    model: eqx.Module,
    batch: Batch,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]: ...
```

---

## Inner-solver primitives (used by `IncrementalFourDVar`)

### `gauss_newton_inner(δx0, x_b, batch, forward_lin, obs_op_lin, B_inv_op, R_inv_op, n_inner, cg_atol, cg_rtol)`

Inner CG solve over the linearised quadratic cost:

```{code-block} python
:caption: Declares the planned inner CG solve of the Gauss-Newton system [](#eq-gn-system) for the incremental cost [](#eq-incremental-cost).
:label: code-design-api-primitives-gn-inner
import lineax as lx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size

# today (vardax)
Batch = vdx.Batch1D


# proposed (D11): IncrementalFourDVar runs this loop internally
def gauss_newton_inner(
    dx0: Float[Array, " N"],
    x_b: Float[Array, " N"],
    batch: Batch,
    forward_lin: lx.AbstractLinearOperator,
    obs_op_lin: lx.AbstractLinearOperator,
    B_inv_op: lx.AbstractLinearOperator,
    R_inv_op: lx.AbstractLinearOperator,
    n_inner: int,
    cg_atol: float,
    cg_rtol: float,
) -> Float[Array, " N"]:
    # 𝐇_GN δx⋆ = −∇𝒥, solved by CG
    ...
```

### `incremental_outer(x0, batch, forward, obs_op, B_op, R_op, config)`

Full incremental 4DVar — GN outer + CG inner with optional CVT:

```{code-block} python
:caption: Declares the planned Gauss-Newton outer loop that relinearises and minimises the incremental cost [](#eq-incremental-cost) through [](#eq-gn-system).
:label: code-design-api-primitives-incremental-outer
import lineax as lx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size

# today (vardax)
Batch = vdx.Batch1D


# proposed (D11): IncrementalFourDVar runs this loop internally
def incremental_outer(
    x0: Float[Array, " N"],
    batch: Batch,
    forward: pc.ForwardModel,
    obs_op: pc.ObservationOperator,
    B_op: lx.AbstractLinearOperator,
    R_op: lx.AbstractLinearOperator,
    # today (vardax)
    config: vdx.IncrementalConfig,
) -> Float[Array, " N"]:
    # x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ + δx⋆, i = 0 … n_outer − 1
    ...
```

Each outer iteration relinearises $h$, $f$ at current $\gvect{\mu}_x$, solves the
quadratic with `gauss_newton_inner`, updates $\gvect{\mu}_x$.

---

## Removed in v0.4

The following Layer 0 primitives from v0.3 are removed:

- `solver_step(x, grad_J, grad_mod_fn, carry)` — moved into `FourDVarNet` internals
- `unrolled_solve(x0, cost_fn, grad_mod_fn, n_steps, carry0)` — replaced by `optimistix.RecursiveCheckpointAdjoint`
- `one_step_solve(...)` — replaced by `vardax.adjoints.OneStepAdjoint` (targeting upstream)
- `implicit_solve(...)` — replaced by `optimistix.ImplicitAdjoint`
- `GradMode` literal type — removed from public API

If you need the gradient-strategy behaviour, configure
`model.solver_adjoint` (for `FourDVarNet`) or `model.minimiser_adjoint`
(for classical methods).

---

## Method-to-primitive map

| Method | Primary cost | Closed-form | Inner solver |
|---|---|---|---|
| `OptimalInterpolation` | — | `blue_analysis` | (none) |
| `ThreeDVar` | `threedvar_cost` | — | `optimistix.GaussNewton` (or BFGS / NCG) |
| `StrongFourDVar` | `StrongConstraintCost` | — | `optimistix.AbstractMinimiser` |
| `WeakFourDVar` | `WeakConstraintCost` | — | `optimistix.AbstractMinimiser` |
| `IncrementalFourDVar` | `incremental_cost` | — | `gauss_newton_inner` + `incremental_outer` |
| `FourDVarNet` | `variational_cost` (learned) | — | `solver_step` (learned, internal) |
| `AmortizedPosterior` | training-time KL | — | (none — single forward pass) |
