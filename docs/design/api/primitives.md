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

$$J_\text{obs}(x) = \tfrac{1}{2} \|H(x) - y\|^2_{R^{-1}}.$$

When `obs_operator=None`, $H$ defaults to masked identity. When
`R_inv_op=None`, $R^{-1}$ defaults to $I / |\Omega|$ (averaged MSE).

### `prior_cost(x, prior_mean=None, B_inv_op=None, prior_fn=None)`

Two distinct meanings depending on which kwargs are passed:

**Classical** (`prior_mean` and `B_inv_op` supplied):

$$J_\text{prior}(x) = \tfrac{1}{2} \|x - x_b\|^2_{B^{-1}}.$$

**Learned** (`prior_fn` supplied — `FourDVarNet`):

$$J_\text{prior}(x) = \|x - \varphi_\theta(x)\|^2.$$

The two forms cannot be mixed — pass one set of arguments or the other.

### `model_error_cost(eta, Q_inv_op)`

Weak-constraint 4DVar model-error term:

$$J_\eta(\boldsymbol{\eta}) = \tfrac{1}{2} \sum_{t=1}^{T} \|\eta_t\|^2_{Q_t^{-1}}.$$

### `variational_cost(x, batch, prior_fn, obs_operator, alpha_obs=1.0, alpha_prior=1.0)`

Weak-constraint variational cost (learned-prior form):

$$J(x) = \alpha_\text{obs} J_\text{obs}(x) + \alpha_\text{prior} J_\text{prior}(x).$$

Used by `FourDVarNet`.

### `threedvar_cost(x, batch, obs_operator, prior_mean, B_inv_op, R_inv_op)`

Classical 3DVar cost:

$$J(x) = \tfrac{1}{2} \|x - x_b\|^2_{B^{-1}} + \tfrac{1}{2} \|y - H(x)\|^2_{R^{-1}}.$$

### `incremental_cost(δx, x_b, batch, forward_lin, obs_op_lin, B_inv_op, R_inv_op)`

Linearised incremental cost for the 4DVar inner loop:

$$J_\text{inc}(\delta x) = \tfrac{1}{2} \|\delta x\|^2_{B^{-1}}
   + \tfrac{1}{2} \sum_t \|y_t - H_t(x_b) - H'_t M'_t \delta x\|^2_{R^{-1}}.$$

$H'_t$ and $M'_t$ are tangent-linear operators at the outer iterate
$x_b$.

---

## Closed-form analysis

### `blue_analysis(x_b, y, B_op, R_op, H_op)`

BLUE / OI in closed form (Decision D16):

$$x^* = x_b + B H^\top (H B H^\top + R)^{-1} (y - H x_b).$$

Returns `(x_star, posterior_cov_op)` where `posterior_cov_op` is an
`AbstractLinearOperator` representing $P^*$ — not materialised by
default. `gaussx` exploits structure in $B$, $R$, $H$.

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
    # x* = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
    # P* = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹, returned as an (N, N) operator
    ...
```

The implementation picks between the $B$-space form (Sherman-Morrison-
Woodbury, useful when state is high-dimensional) and the $R$-space form
(useful when observation count is low) based on dimensions.

---

## Control-variable transform

### `cvt_transform(x, x_b, B_half_op)` and `cvt_inverse(chi, x_b, B_half_op)`

CVT: $\chi = B^{-1/2}(x - x_b)$, with inverse $x = x_b + B^{1/2}\chi$.

$B^{1/2}$ comes from `gaussx.MaternLinearOperator.half()` when the prior
is Matérn; falls back to Cholesky factorisation for arbitrary
`AbstractLinearOperator`.

```{code-block} python
:caption: Declares the planned control-variable transform χ = B⁻¹ᐟ²(x − x_b) of [](#eq-cvt) and its inverse.
:label: code-design-api-primitives-cvt
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size


# proposed (D11): IncrementalFourDVar applies the CVT internally (config.cvt)
def cvt_transform(
    x: Float[Array, " N"], x_b: Float[Array, " N"], B_half_op: lx.AbstractLinearOperator
) -> Float[Array, " N"]:
    # χ = B⁻¹ᐟ² (x − x_b)
    ...


# proposed (D11)
def cvt_inverse(
    chi: Float[Array, " N"],
    x_b: Float[Array, " N"],
    B_half_op: lx.AbstractLinearOperator,
) -> Float[Array, " N"]:
    # x = x_b + B¹ᐟ² χ
    ...
```

In CVT coordinates the prior cost is $\|\chi\|^2$ (identity Gaussian),
which preconditions the CG inner loop in `IncrementalFourDVar`.

---

## Tangent-linear and adjoint

### `tangent_linear(fn, x)`

Wraps `jax.linearize` to return an `AbstractLinearOperator`:

```{code-block} python
:caption: Defines the planned tangent-linear primitive, which wraps `jax.linearize` as the operator M′ of [](#eq-tlm-forward).
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
    # δx ↦ M′(x) δx
    _, jvp = jax.linearize(fn, x)
    # today* (lineax): the shipped keyword is input_structure, a ShapeDtypeStruct
    return lx.FunctionLinearOperator(jvp, in_structure=x)
```

The transpose / adjoint comes for free via `lineax`.

---

## Posterior primitives

### `laplace_covariance(x_star, cost_grad_fn, B_inv_op, R_inv_op)`

Laplace approximation at MAP:

$$P^* = \big((H')^\top R^{-1} H' + B^{-1}\big)^{-1}.$$

Returns `AbstractLinearOperator` — supports mat-vec via `lineax.CG`, no
materialisation.

### `gauss_newton_hessian(x_star, batch, forward, obs_op, B_op, R_op, n_krylov=50)`

Gauss-Newton Hessian at MAP, via Krylov / Lanczos. Returns
`AbstractLinearOperator` representing $J''(x^*)$ for posterior
inversion.

---

## Training primitives

### `reconstruction_loss(pred, target)`

$$\mathcal{L}(\theta) = \|x^*(\theta) - x_\text{true}\|^2.$$

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
    # 𝐇_GN δx* = −∇J, solved by CG
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
    # x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + δx*, k = 0 … n_outer − 1
    ...
```

Each outer iteration relinearises $H$, $M$ at current $x_b$, solves the
quadratic with `gauss_newton_inner`, updates $x_b$.

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
