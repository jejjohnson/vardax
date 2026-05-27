---
status: draft
version: 0.3.0
---

# Layer 0 — Primitives

Pure JAX functions. Stateless, differentiable, no `eqx.Module`, no
`optimistix`. These are the mathematical building blocks that Layer 1
operators compose.

---

## Cost functions

### `obs_cost(x, y_obs, mask, obs_operator=None)`

Observation cost:

$$J_\text{obs}(x) = \frac{1}{|\Omega|} \|H(x) - y\|^2_{R^{-1}}$$

where $H$ is the observation operator, $y$ the observations, $\Omega$ the
mask, and $R^{-1}$ the inverse observation-error covariance. When
`obs_operator=None`, $H$ defaults to masked identity.

```python
def obs_cost(x, y_obs, mask, obs_operator=None, obs_err_inv=None) -> Scalar: ...
```

### `prior_cost(x, prior_fn)`

Prior / regularisation cost:

$$J_\text{prior}(x) = \|x - \varphi(x)\|^2_{B^{-1}}$$

where $\varphi$ is the prior model and $B^{-1}$ the inverse prior covariance
(defaults to identity for AE-style priors; non-trivial for
incremental 4DVar).

```python
def prior_cost(x, prior_fn, prior_cov_inv=None) -> Scalar: ...
```

### `variational_cost(x, batch, prior_fn, obs_operator, alpha_obs=1.0, alpha_prior=1.0)`

Weak-constraint variational cost:

$$J(x) = \alpha_\text{obs} J_\text{obs}(x) + \alpha_\text{prior} J_\text{prior}(x)$$

```python
def variational_cost(x, batch, prior_fn, obs_operator,
                     alpha_obs=1.0, alpha_prior=1.0) -> Scalar: ...
```

### `variational_cost_grad(x, ...)`

$$\nabla_x J(x) = \alpha_\text{obs} \nabla_x J_\text{obs} + \alpha_\text{prior} \nabla_x J_\text{prior}$$

Returned via `jax.grad(variational_cost)` — no separate function needed in
public API, kept here for documentation.

### `incremental_cost(δx, x_b, batch, forward_lin, obs_op_lin, B_inv_op, R_inv_op)`

Linearised incremental 4DVar cost (Decision D11):

$$J_\text{inc}(\delta x) = \frac{1}{2}\|\delta x\|^2_{B^{-1}}
   + \frac{1}{2}\sum_t \|y_t - H_t(x_b) - H_t' \cdot M_t' \cdot \delta x\|^2_{R^{-1}}$$

where $H_t'$ and $M_t'$ are tangent-linear obs and forward operators at the
outer iterate $x_b$. Solved iteratively by CG / Lanczos in the inner loop.

```python
def incremental_cost(dx, x_b, batch, forward_lin, obs_op_lin,
                     B_inv_op, R_inv_op) -> Scalar: ...
```

---

## Control-variable transform (CVT)

### `cvt_transform(x, B_half_op)` and `cvt_inverse(chi, B_half_op)`

CVT: $\chi = B^{-1/2}(x - x_b)$, with inverse $x = x_b + B^{1/2}\chi$.

$B^{1/2}$ comes from a `gaussx.MaternLinearOperator` factorisation when the
prior is Matérn (default in `IncrementalConfig`). Falls back to Cholesky
factorisation for arbitrary `AbstractLinearOperator`.

```python
def cvt_transform(x, x_b, B_half_op) -> chi: ...
def cvt_inverse(chi, x_b, B_half_op) -> x: ...
```

In CVT coordinates the prior cost becomes $\|\chi\|^2$ (identity Gaussian),
which preconditions the CG inner loop.

---

## Solver steps (inner loop)

### `solver_step(x, grad_J, grad_modulator_fn, carry)`

One step of the inner minimisation:

$$x_{k+1} = x_k - \Phi(\nabla_x J(x_k),\; \text{carry}_k)$$

where $\Phi$ is the gradient modulator. Returns `(x_new, carry_new)`.

```python
def solver_step(x, grad_J, grad_modulator_fn, carry) -> tuple[Array, Any]: ...
```

### `unrolled_solve(x0, cost_fn, grad_mod_fn, n_steps, carry0)`

Unrolled differentiation via `jax.lax.scan`:

$$x_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(x_0)$$

**Memory:** $O(K)$. Standard backprop through all K steps.

### `one_step_solve(x0, cost_fn, grad_mod_fn, n_steps, carry0)`

One-step differentiation (Bolte et al., NeurIPS 2023):

$K-1$ steps with `jax.lax.stop_gradient`, then one differentiable step.

$$x_{K-1} = \mathrm{sg}(f_{K-1} \circ \cdots \circ f_1(x_0)), \quad x_K = f_K(x_{K-1})$$

**Memory:** $O(1)$. No convergence requirement.

### `implicit_solve(x0, cost_fn, grad_mod_fn, n_steps, carry0)`

Implicit differentiation via `optimistix.FixedPointIteration`. At fixed
point $x^* = f(x^*)$, the training gradient flows through the implicit
function theorem:

$$\frac{dx^*}{d\theta} = (I - \partial f / \partial x)^{-1} \frac{\partial f}{\partial \theta}$$

**Memory:** $O(1)$. Requires convergence.

### `gauss_newton_inner(δx0, x_b, batch, forward_lin, obs_op_lin, B_inv_op, R_inv_op, n_inner, cg_atol, cg_rtol)`

Inner-loop solver for incremental 4DVar (Decision D11). Solves the
quadratic linearised problem via `lineax.CG`:

$$\delta x^* = \underset{\delta x}{\arg\min}\; J_\text{inc}(\delta x)$$

Returns the increment $\delta x^*$ to apply at the next outer iteration:
$x_b \leftarrow x_b + \delta x^*$.

```python
def gauss_newton_inner(dx0, x_b, batch, forward_lin, obs_op_lin,
                       B_inv_op, R_inv_op, n_inner, cg_atol, cg_rtol) -> dx_star: ...
```

### `incremental_outer(x0, batch, forward, obs_op, B_op, R_op, config)`

Full incremental 4DVar — Gauss-Newton outer + CG inner with optional CVT:

```python
def incremental_outer(x0, batch, forward, obs_op, B_op, R_op,
                      config: IncrementalConfig) -> Array: ...
```

Each outer iteration: relinearise $H, M$ at current $x_b$, solve the
quadratic with `gauss_newton_inner`, update $x_b$.

---

## Posterior primitives

### `laplace_covariance(x_star, cost_grad_fn, B_inv_op, R_inv_op)`

Laplace approximation at MAP:

$$P^* = (H^\top R^{-1} H + B^{-1})^{-1}$$

Returns an `AbstractLinearOperator` (not materialised — supports
mat-vec / log-det via `lineax`).

### `gauss_newton_hessian(x_star, batch, forward, obs_op, B_op, R_op, n_krylov=50)`

Gauss-Newton Hessian via Krylov / Lanczos at MAP. Returns
`AbstractLinearOperator` representing $J''(x^*)$ for posterior inversion.

---

## Training primitives

### `reconstruction_loss(pred, target)`

$$\mathcal{L}(\theta) = \|x^*(\theta) - x_\text{true}\|^2$$

### `train_loss_fn(model, batch)`

Forward through `model(batch) → x*`, then `reconstruction_loss(x*, batch.target)`.
Wired to propagate gradients through the inner solver according to the
model's `grad_mode`.

### `train_step(model, batch, optimizer, opt_state)`

One outer training step: loss → backprop through solver → optimiser update.
Encodes the correctness-critical differentiation pattern (Decision D5).

```python
def train_step(model, batch, optimizer, opt_state) -> tuple[model, opt_state, loss]: ...
```

---

## Gradient mode summary

| Mode | Function | Memory | Convergence? | Used by | Reference |
|---|---|---|---|---|---|
| `"unrolled"` | `unrolled_solve` | $O(K)$ | No | `VarDANet*` | Standard backprop |
| `"one_step"` | `one_step_solve` | $O(1)$ | No | `VarDANet*` | Bolte et al. (2023) |
| `"implicit"` | `implicit_solve` | $O(1)$ | Yes | `VarDANet*` | IFT via optimistix |
| `"incremental"` | `incremental_outer` | $O(\text{n\_outer})$ | At each outer | `IncrementalVarDA*` | Courtier et al. (1994) |
| (amortized) | direct head | $O(1)$ | N/A | `AmortizedVarDA*` | Conditional flow / score |
