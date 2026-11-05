---
numbering:
  enumerator: "2.%s"
---

# Observation Model

The observation operator $h : \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ maps the state $\vect{x} \in \mathbb{R}^{D_x}$ to predicted observations $\hat{\vect{y}} = h(\vect{x}) \in \mathbb{R}^{D_y}$. Real instruments rarely measure the state directly; they smooth, average, mask or project it. Getting $h$ right is what separates an inversion that recovers the truth from one that returns a biased fantasy.

## The likelihood model

vardax assumes Gaussian observation errors,

$$\vect{y} = h(\vect{x}) + \gvect{\varepsilon}, \text{ with } \gvect{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \gmat{\Sigma}_\varepsilon)$$ (eq-obs-model)

where $\vect{y} \in \mathbb{R}^{D_y}$ are the observations, $\gvect{\varepsilon} \in \mathbb{R}^{D_y}$ is the observation noise and $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$ is the observation-error covariance, so $p(\vect{y} \mid \vect{x}) = \mathcal{N}(\vect{y}; h(\vect{x}), \gmat{\Sigma}_\varepsilon)$. The observation cost in the variational problem is the negative log-likelihood up to a constant,

$$\mathcal{J}_o(\vect{x}) = \frac{1}{2} \Vert \vect{y} - h(\vect{x}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}} = \frac{1}{2} (\vect{y} - h(\vect{x}))^\top \gmat{\Sigma}_\varepsilon^{-1} (\vect{y} - h(\vect{x}))$$ (eq-obs-cost)

Non-Gaussian likelihoods (Student-$t$, censored, count data) require
either a non-Gaussian extension of $\mathcal{J}_o$ or amortized
inference with a flow / score-based head (chapter 10).

## The `ObservationOperator` protocol

Every vardax observation operator satisfies
`pipekit_cycle.ObservationOperator`, restated here with typed signatures:

```{code-block} python
:caption: Restates the observation-operator protocol: the forward map of [](#eq-obs-model) and its tangent linear.
:label: code-obs-model-protocol
from typing import Protocol

import lineax as lx
from jaxtyping import Array, Float

# Shapes: D_x state cells; D_y observations


# today* (pipekit_cycle)
# the shipped protocol types both members as Any
class ObservationOperator(Protocol):
    # ŷ = h(x)
    def __call__(self, state: Float[Array, " state"]) -> Float[Array, " obs"]:
        """Forward: h(x)."""

    # H = J_h(x), the tangent-linear operator at x
    def linearize(self, state: Float[Array, " state"]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator H for use in 4DVar inner loops."""
```

The shapes are $\vect{x} \in \mathbb{R}^{D_x}$, $\hat{\vect{y}} \in \mathbb{R}^{D_y}$ and $\mat{H} \in \mathbb{R}^{D_y \times D_x}$, the Jacobian of $h$ at $\vect{x}$.

The `linearize()` method returns a `lineax.AbstractLinearOperator`
(typically `JacobianLinearOperator` via autodiff, but structured
overrides are common). The transpose / adjoint comes for free from
`lineax`.

This contract is the same regardless of whether the operator is a
masked identity, an averaging kernel, or a multi-instrument fusion.

## Five operator families

### Masked identity

In the simplest case we observe the state where the mask is non-zero,

$$h(\vect{x}) = \vect{m} \odot \vect{x}, \text{ with } \vect{m} \in \{0, 1\}^{D_y}$$ (eq-masked-identity)

where $\odot$ is the element-wise product and $D_y = D_x$.

Tangent-linear: $\mat{H} = \mat{M} = \op{diag}(\vect{m})$, with $\mat{M} \in \mathbb{R}^{D_y \times D_y}$. Use case: SSH altimetry
along-track gaps, SST with cloud masks, in-situ samples on a grid.

The mask is passed per call, so the models apply it to $\mat{H}$ themselves. The examples on this page build on each other, starting from a toy state on $D_x = 6$ cells.

```{code-block} python
:caption: Computes the masked identity [](#eq-masked-identity) on a toy state of six cells.
:label: code-obs-model-masked-identity
import jax.numpy as jnp
import lineax as lx
import vardax as vdx

# Shapes: D_x = 6 state cells; D_y = D_x

# x, a toy state
state = jnp.linspace(1.0, 2.0, 6)  # (D_x,)
# m ∈ {0, 1}^D_y, cells 2 and 5 unobserved
mask = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])  # (D_y,)

# today (vardax)
masked_obs = vdx.MaskedIdentity()
# h(x) = m ⊙ x
gappy_obs = masked_obs(state, mask=mask)  # (D_x,) → (D_y,)
print(gappy_obs)
```

It prints `[1.  1.2 0.  1.6 1.8 0. ]`.

### Linear projection

A pre-computed projection from state to observation space is

$$h(\vect{x}) = \mat{H} \vect{x}$$ (eq-linear-obs)

where $\mat{H} \in \mathbb{R}^{D_y \times D_x}$ is a fixed matrix.

Use case: Lagrangian footprint matrix from a particle simulator
(observation at receptor = footprint-weighted source field), spectral
filtering, spatial interpolation. The matrix is stored as an
`AbstractLinearOperator` so structure (sparse, low-rank, Kronecker) can
be exploited.

Here the footprint matrix $\mat{H}$ maps the six cells to $D_y = 3$ receptors, each averaging two neighbouring cells.

```{code-block} python
:caption: Computes the linear projection [](#eq-linear-obs) through a footprint matrix of three receptors.
:label: code-obs-model-linear
# Shapes: D_x = 6 state cells; D_y = 3 receptors

# (½ ½), the average of two neighbouring cells
pair_average = jnp.array([[0.5, 0.5]])  # (1, 2)
# H = I₃ ⊗ (½ ½), each receptor averages two neighbouring cells
footprint = jnp.kron(jnp.eye(3), pair_average)  # (D_y, D_x)
# today (lineax)
footprint_op = lx.MatrixLinearOperator(footprint)  # (D_y, D_x) op
# today (vardax)
footprint_obs = vdx.LinearObs(operator=footprint_op)
# h(x) = H x
receptor_obs = footprint_obs(state)  # (D_x,) → (D_y,)
print(receptor_obs)
```

It prints `[1.1 1.5 1.9]`.

### Averaging kernel: RTM L2 products

For satellite L2 products derived from optimal-estimation retrievals
(TROPOMI CH₄, EMIT CH₄, OCO CO₂, MOPITT CO, …), the L2 retrieval is
*not* a direct sample of the mixing-ratio profile $\vect{x}$. The retrieval
applies a smoothing matrix $\mat{J}_{\hat{x}}$ (the averaging kernel, $\mat{A}$ in the retrieval literature) and falls back to
a retrieval prior $\gvect{\mu}_\text{ret}$ where the signal is weak:

$$\hat{\vect{y}} = \mat{J}_{\hat{x}} (\vect{w} \odot \vect{x} + (1 - \vect{w}) \odot \gvect{\mu}_\text{ret})$$ (eq-averaging-kernel)

where the symbols are

| Symbol | Description |
|---|---|
| $\vect{x} \in \mathbb{R}^{D_x}$ | Model state (profile, surface field) |
| $\gvect{\mu}_\text{ret} \in \mathbb{R}^{D_x}$ | Retrieval prior from L2 metadata |
| $\vect{w} \in \mathbb{R}^{D_x}$ | Weighting vector (often pressure-weighted) |
| $\mat{J}_{\hat{x}} \in \mathbb{R}^{D_x \times D_x}$ | Averaging kernel matrix |

The retrieved $\hat{\vect{y}}$ lives in the state's space, so $D_y = D_x$. Tangent-linear: $\mat{H} = \mat{J}_{\hat{x}} \op{diag}(\vect{w})$.

**Skipping the averaging kernel is the most common cause of bias in
operational satellite inversions.** Treating an L2 retrieval as a
direct measurement of the truth gives systematically wrong posteriors,
especially when the retrieval prior $\gvect{\mu}_\text{ret}$ differs from the inversion
prior $\gvect{\mu}_x$. vardax exposes `AveragingKernel` as a first-class operator
because this is a day-one requirement, not a future feature.

The toy kernel $\mat{J}_{\hat{x}}$ below is tridiagonal, the prior $\gvect{\mu}_\text{ret}$ is flat and the sensitivity $\vect{w}$ falls from 1 to 0.2 with height. The check of the tangent linear against $\mat{J}_{\hat{x}} \op{diag}(\vect{w})$ formed densely prints `True`.

```{code-block} python
:caption: Computes the averaging-kernel operator [](#eq-averaging-kernel) and checks its tangent linear against the dense J diag(w).
:label: code-obs-model-averaging-kernel
import einx

# Shapes: D_x = 6 levels; D_y = D_x

# the two nearest neighbours of each level
neighbours = jnp.eye(6, k=1) + jnp.eye(6, k=-1)  # (D_x, D_x)
# J_x̂ = ½ I + ¼ neighbours, a tridiagonal smoothing kernel
kernel = 0.5 * jnp.eye(6) + 0.25 * neighbours  # (D_x, D_x)
# μ_ret, a flat retrieval prior
retrieval_prior = jnp.full(6, 1.8)  # (D_x,)
# w, sensitivity falling off with height
weighting_vector = jnp.linspace(1.0, 0.2, 6)  # (D_x,)
# today (lineax)
kernel_op = lx.MatrixLinearOperator(kernel)  # (D_x, D_x) op

# today (vardax)
averaging_kernel = vdx.AveragingKernel(
    kernel_matrix=kernel_op,
    prior_profile=retrieval_prior,
    pressure_weights=weighting_vector,
)
# ŷ = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_ret)
retrieved = averaging_kernel(state)  # (D_x,) → (D_y,)
# today (vardax)
# H = J_x̂ diag(w)
kernel_tangent = averaging_kernel.linearize(state)  # (D_y, D_x) op
# J_x̂ diag(w), formed densely to check
dense_tangent = einx.multiply(  # (D_y, D_x)
    "row col, col -> row col", kernel, weighting_vector
)
print(bool(jnp.allclose(kernel_tangent.as_matrix(), dense_tangent)))
```

When $\mat{J}_{\hat{x}}$ has structure (Kronecker for separable kernels, low-rank for
under-determined inversions, banded for compactly-supported kernels),
store it as a `gaussx.AbstractLinearOperator`. The tangent linear
$\mat{J}_{\hat{x}} \op{diag}(\vect{w})$ is then computed without ever materialising
the dense $\mat{J}_{\hat{x}}$.

### Multi-instrument fusion

Operational satellite work combines multiple instruments. Each instrument $i = 1, \ldots, N_\text{inst}$ has its own $h_i : \mathbb{R}^{D_x} \to \mathbb{R}^{D_{y,i}}$, mask $\vect{m}_i \in \{0, 1\}^{D_{y,i}}$, error covariance $\gmat{\Sigma}_{\varepsilon,i} \in \mathbb{R}^{D_{y,i} \times D_{y,i}}$ and optionally an averaging kernel $(\mat{J}_{\hat{x},i}, \gvect{\mu}_{\text{ret},i}, \vect{w}_i)$. The fused observation cost composes [](#eq-obs-cost) at the likelihood level,

$$\mathcal{J}_o(\vect{x}) = \sum_{i=1}^{N_\text{inst}} \alpha_i \cdot \frac{1}{2} \Vert \vect{m}_i \odot (\vect{y}_i - h_i(\vect{x})) \Vert^2_{\gmat{\Sigma}_{\varepsilon,i}^{-1}}$$ (eq-fusion-cost)

where $N_\text{inst}$ is the number of instruments and the scalar $\alpha_i$ is a per-instrument weight. The weight defaults to uniform; tuning it is for hierarchical bias correction (Epic 9). Each `InstrumentSpec` carries its own operator, mask and $\gmat{\Sigma}_{\varepsilon,i}$. Quality masks zero-weight unreliable pixels, so they contribute zero log-likelihood rather than being dropped, which lets us audit the effective per-instrument observation count.

The registry below reuses the three operators above: TROPOMI through the averaging kernel, EMIT through the masked identity and GHGSat through the footprint. It evaluates each $h_i(\vect{x})$ but does not form the cost itself.

```{code-block} python
:caption: Evaluates each instrument operator of the fusion cost [](#eq-fusion-cost) through a registry of three instruments.
:label: code-obs-model-fusion
import jax

# Shapes: D_x = 6 state cells; D_y,TROPOMI = D_y,EMIT = 6; D_y,GHGSat = 3

state_structure = jax.ShapeDtypeStruct((6,), jnp.float32)
receptor_structure = jax.ShapeDtypeStruct((3,), jnp.float32)
# today (lineax)
state_identity = lx.IdentityLinearOperator(state_structure)  # (D_x, D_x) op
# today (lineax)
receptor_identity = lx.IdentityLinearOperator(receptor_structure)  # (3, 3) op
# Σ_ε,TROPOMI = 0.04 I
tropomi_cov = 0.04 * state_identity  # (D_y,TROPOMI, D_y,TROPOMI) op
# Σ_ε,EMIT = 0.01 I
emit_cov = 0.01 * state_identity  # (D_y,EMIT, D_y,EMIT) op
# Σ_ε,GHGSat = 0.0025 I
ghgsat_cov = 0.0025 * receptor_identity  # (D_y,GHGSat, D_y,GHGSat) op

# today (vardax)
registry = vdx.InstrumentRegistry(
    entries={
        "TROPOMI": vdx.InstrumentSpec(
            obs_op=averaging_kernel,
            mask=jnp.ones(6),
            obs_cov_op=tropomi_cov,
            instrument_id="TROPOMI",
        ),
        "EMIT": vdx.InstrumentSpec(
            obs_op=masked_obs,
            mask=mask,
            obs_cov_op=emit_cov,
            instrument_id="EMIT",
        ),
        "GHGSat": vdx.InstrumentSpec(
            obs_op=footprint_obs,
            mask=jnp.ones(3),
            obs_cov_op=ghgsat_cov,
            instrument_id="GHGSat",
        ),
    }
)
# today (vardax)
fusion = vdx.MultiInstrumentFusion(registry=registry)
# {i: hᵢ(x)}
predicted = fusion(state)  # (D_x,) → {i: (D_y,i,)}
# today (vardax)
stacked_op = fusion.to_observation_operator()
# [h_EMIT(x); h_GHGSat(x); h_TROPOMI(x)], in sorted id order
stacked = stacked_op(state)  # (D_x,) → (D_y,EMIT + D_y,GHGSat + D_y,TROPOMI,)
print({name: obs.shape for name, obs in predicted.items()}, stacked.shape)
```

It prints `{'TROPOMI': (6,), 'EMIT': (6,), 'GHGSat': (3,)} (15,)`.

`MultiInstrumentFusion.__call__` returns `dict[str, Array]` (per-instrument predicted observations), and `MultiInstrumentFusion.cost` sums the weighted per-instrument terms. For strict `pipekit_cycle.ObservationOperator` contexts, the `.to_observation_operator()` adapter flattens to a single concatenated output, and its `linearize` is the Jacobian of that map, the per-instrument $\mat{H}_i$ stacked row-wise. Pass `fusion.obs_cov_op()` as the model's $\gmat{\Sigma}_\varepsilon$ with that adapter: its blocks are $\gmat{\Sigma}_{\varepsilon,i} / \alpha_i$, so the model minimises the same weighted cost.

### Any differentiable model: parameter estimation

When the control is not the state but a source, a forcing or a parameter vector $\vect{q} \in \mathbb{R}^{D_q}$, the observation operator is a downstream model run $h(\vect{q})$: a transport model followed by its receptors, a radiative-transfer forward, a neural emulator. The cost $\mathcal{J}(\vect{q})$ is then exactly the 3DVar cost [](#eq-3dvar-cost) with $\vect{x} = \vect{q}$ and prior mean $\gvect{\mu}_q$ in place of $\gvect{\mu}_x$, or OI (chapter 4) when $h$ is linear. `CallableObs` wraps any differentiable JAX function as that operator; `linearize(x)` is a matrix-free `lineax.JacobianLinearOperator`, so the tangent-linear map is a JVP and its adjoint a VJP.

In the example below the control is $D_q = 3$ unconstrained emission parameters $\vect{q}$, and $h(\vect{q}) = \mat{H}_\text{tr} \op{softplus}(\vect{q})$ maps them to $D_y = 6$ receptors through a transport matrix $\mat{H}_\text{tr} \in \mathbb{R}^{D_y \times D_q}$ that decays with the distance between receptor and source. The observations are noise-free, $\gmat{\Sigma}_x = \mat{I}$ and $\gmat{\Sigma}_\varepsilon = 10^{-2} \mat{I}$.

```{code-block} python
:caption: Minimises the 3DVar cost [](#eq-3dvar-cost) over emission parameters q, with a transport model as the observation operator.
:label: code-obs-model-parameter-3dvar
import optimistix as optx

# Shapes: N_batch = 1; T = 0 steps, so T + 1 = 1 time; D_q = 3 sources;
# D_y = 6 receptors

# H_tr, receptor i sees source j with weight exp(−|i − 2j|)
receptor_index = jnp.arange(6.0)  # (D_y,)
source_index = 2.0 * jnp.arange(3.0)  # (D_q,)
offset = einx.subtract(  # (D_y, D_q)
    "obs, source -> obs source", receptor_index, source_index
)
# |i − 2j|
distance = jnp.abs(offset)  # (D_y, D_q)
# H_tr = exp(−|i − 2j|)
transport = jnp.exp(-distance)  # (D_y, D_q)


# user
def run_model(source: Float[Array, "time source"]) -> Float[Array, "time obs"]:
    # e = log(1 + exp(q)), positive emission rates
    emission = jax.nn.softplus(source)  # (T + 1, D_q)
    # h(q) = H_tr softplus(q)
    return einx.dot(  # (T + 1, D_y)
        "time source, obs source -> time obs", emission, transport
    )


# μ_q = 0, q_true = (0.5, −0.3, 0.2)
source_background = jnp.zeros((1, 3))  # (T + 1, D_q)
true_source = jnp.array([[0.5, -0.3, 0.2]])  # (T + 1, D_q)
# y = h(q_true), noise-free
receptor_data = run_model(true_source)  # (T + 1, D_y)

source_structure = jax.ShapeDtypeStruct((1, 3), jnp.float32)
receptor_data_structure = jax.ShapeDtypeStruct((1, 6), jnp.float32)
# today (lineax)
# Σ_x = I
source_cov = lx.IdentityLinearOperator(source_structure)  # (T+1)D_q square op
# today (lineax)
receptor_data_identity = lx.IdentityLinearOperator(receptor_data_structure)
# today (lineax)
# Σ_ε = 10⁻² I; tagged PSD so the default CG solve accepts it
receptor_cov = lx.TaggedLinearOperator(  # ((T + 1) D_y,) * 2 op
    1e-2 * receptor_data_identity, lx.positive_semidefinite_tag
)
# today (vardax)
# h(q) = H_tr softplus(q)
model_obs = vdx.CallableObs(fn=run_model)
# today (vardax)
# 𝒥(q) = ½ ‖q − μ_q‖²_Σ_x⁻¹ + ½ ‖y − h(q)‖²_Σ_ε⁻¹
parameter_3dvar = vdx.ThreeDVar(
    obs_op=model_obs,
    prior_mean=source_background,
    prior_cov_op=source_cov,
    obs_cov_op=receptor_cov,
    minimiser=optx.BFGS(rtol=1e-6, atol=1e-6),
)
# today (vardax)
receptor_batch = vdx.Batch1D(
    input=einx.id(  # (T + 1, D_y) → (N_batch, T + 1, D_y)
        "time obs -> 1 time obs", receptor_data
    ),
    mask=jnp.ones((1, 1, 6)),
)
# q⋆ = argmin_q 𝒥(q)
source_analysis = parameter_3dvar(receptor_batch)  # → (N_batch, T + 1, D_q)
print(source_analysis[0])
```

It prints `[[ 0.4838582  -0.27665472  0.19014639]]`, close to $\vect{q}_\text{true} = (0.5, -0.3, 0.2)$; the prior pulls each rate slightly towards $\gvect{\mu}_q = \mathbf{0}$.

The observation shape need not match the state shape. Prefer
`LinearObs` when you already hold an explicit or structured matrix you
want to reuse.

For a model that defines only a reverse-mode rule (a `jax.custom_vjp`),
pass `CallableObs(fn=..., jac="bwd")`: forward-mode linearisation raises
on those as soon as an analysis or a posterior needs $\mat{H}$. Matrix-free,
lineax then applies $\mat{H} \vect{v}$ by transposing the VJP, which fails when the
backward pass holds a loop JAX cannot transpose. A `diffrax` solve is
the common case: `RecursiveCheckpointAdjoint` runs a `while_loop`. Add
`materialise=True` and `linearize` forms the Jacobian once, with one VJP
per observation; that is cheap when the control is small, as a source or
parameter vector usually is. The toy `simulate` below integrates $\mathrm{d}\vect{c}/\mathrm{d}t = \vect{q} - \vect{c}$ for the concentration $\vect{c} \in \mathbb{R}^{D_q}$ from $\vect{c}(0) = \mathbf{0}$ to $t = 1$, so $h(\vect{q}) = \vect{c}(1)$ and $\mat{H} = (1 - e^{-1}) \mat{I}$, and the block prints `True`:

```{code-block} python
:caption: Materialises the tangent linear of a diffrax-solve observation operator [](#eq-obs-model) with one VJP per observation.
:label: code-obs-model-materialise
import diffrax as dfx

# Shapes: T = 0 steps, so T + 1 = 1 time; D_q = 3 sources


# user
def simulate(source: Float[Array, "time source"]) -> Float[Array, "time source"]:
    # today (diffrax)
    # dc/dt = q − c
    term = dfx.ODETerm(lambda time, conc, rate: rate - conc)
    # today (diffrax)
    # c(1) from c(0) = 0, reverse mode by RecursiveCheckpointAdjoint
    solution = dfx.diffeqsolve(
        term,
        dfx.Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=jnp.zeros_like(source),
        args=source,
    )
    return solution.ys[-1]  # (T + 1, D_q)


# today (vardax)
simulator_obs = vdx.CallableObs(fn=simulate, jac="bwd", materialise=True)
# today (vardax)
# H = ∂c(1)/∂q = (1 − e⁻¹) I
simulator_tangent = simulator_obs.linearize(true_source)  # ((T + 1) D_q,) * 2 op
# (1 − e⁻¹) I, the exact tangent
exact_tangent = (1.0 - jnp.exp(-1.0)) * jnp.eye(3)  # ((T + 1) D_q,) * 2
print(bool(jnp.allclose(simulator_tangent.as_matrix(), exact_tangent, atol=1e-4)))
```

## Tangent-linear and adjoint contract

For incremental 4DVar (chapter 8), every observation operator must expose `linearize`, whose operator applies $\mat{H}$ with `mv` and $\mat{H}^\top$ with `transpose().mv`. The pair must pass the adjoint test

$$\langle \mat{H} \vect{u}, \vect{v} \rangle = \langle \vect{u}, \mat{H}^\top \vect{v} \rangle \text{ for all } \vect{u}, \vect{v}$$ (eq-adjoint-test)

up to floating-point tolerance, where $\vect{u} \in \mathbb{R}^{D_x}$ is a state-space vector and $\vect{v} \in \mathbb{R}^{D_y}$ an observation-space vector. Failing this test means the gradient flowing back through $h$ is wrong, and any downstream optimiser will take wrong steps. The block below draws random $\vect{u} = \delta\vect{x}$ and $\vect{v} = \vect{r}$ and prints `True`; `tests/test_obs_operators.py` runs the same test for `AveragingKernel`.

```{code-block} python
:caption: Checks the adjoint test [](#eq-adjoint-test) for the averaging kernel.
:label: code-obs-model-adjoint-test
# Shapes: D_x = 6 levels; D_y = D_x

key = jax.random.PRNGKey(0)
key_tangent, key_residual = jax.random.split(key)

# δx ~ 𝒩(0, I)
increment = jax.random.normal(key_tangent, (6,))  # (D_x,)
# r ~ 𝒩(0, I)
residual = jax.random.normal(key_residual, (6,))  # (D_y,)

# today (vardax)
# H = J_h(x)
kernel_tangent = averaging_kernel.linearize(state)  # (D_y, D_x) op
# H δx, the forward tangent-linear model
tangent_increment = kernel_tangent.mv(increment)  # (D_x,) → (D_y,)
# Hᵀ r, the adjoint
adjoint_residual = kernel_tangent.transpose().mv(residual)  # (D_y,) → (D_x,)
# ⟨H δx, r⟩
forward_inner = einx.dot("obs, obs ->", tangent_increment, residual)  # → ()
# ⟨δx, Hᵀ r⟩
adjoint_inner = einx.dot("state, state ->", increment, adjoint_residual)  # → ()
print(bool(jnp.isclose(forward_inner, adjoint_inner, atol=1e-5)))
```

## Observation-error covariance $\gmat{\Sigma}_\varepsilon$

$\gmat{\Sigma}_\varepsilon$ is supplied as a `lineax.AbstractLinearOperator`. Common
parameterisations:

- **Diagonal**: `lineax.DiagonalLinearOperator(variances)`. The default for heteroscedastic retrievals (per-pixel σ from L2 metadata).
- **Block-diagonal**: `gaussx.BlockDiag(*instrument_covs)`. Per-instrument diagonals when fusing.
- **Structured**: `kernellib.ImplicitKernelOperator(matern, coords)` if observation errors are spatially correlated (rare, but it matters for high-resolution imaging spectrometers).

The models' solves reject gaussx operators until they are wrapped as a `lx.FunctionLinearOperator`, as [chapter 4](04_oi_blue.md#why-the-matrices-are-never-materialised) does for $\gmat{\Sigma}_x$.

The choice flows into [](#eq-obs-cost) via $\gmat{\Sigma}_\varepsilon^{-1}$, which is applied lazily: the inverse is never materialised, only applied to vectors via `lineax.CG`.

## When the observation model is wrong

A common failure mode: $h$ is misspecified (wrong averaging kernel,
missing instrument bias, ignored representation error). The MAP
$\vect{x}^\star$ then converges to a wrong answer with a confident-looking
posterior. Diagnostics:

- **Innovation statistics** $\vect{r} = \vect{y} - h(\gvect{\mu}_x)$. Should be roughly mean-zero
  and consistent with $\sqrt{\op{diag}(\mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon)}$. Large
  systematic offsets indicate bias.
- **Posterior vs prior log-evidence ratio.** If the data move the
  posterior much further than $\sigma_\text{post}$ from the prior, the
  $\gmat{\Sigma}_x$ or $\gmat{\Sigma}_\varepsilon$ assumptions are wrong.
- **Cross-instrument disagreement.** Per-instrument residual statistics
  should be consistent; large disagreements flag uncorrected bias.

Epic 9 adds joint bias estimation for the cross-instrument case.

## See also

- Chapter 3: the dynamical model and its adjoint
- Chapter 4: `OptimalInterpolation` (the linear-$h$ case)
- Chapter 8: `IncrementalFourDVar` (uses `linearize()` for tangent-linear observations)
- Chapter 11: the observation operator family in detail
- Design doc: [`design/api/observation_operators.md`](design/api/observation_operators.md)
