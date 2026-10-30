---
numbering:
  enumerator: "2.%s"
---

# Observation Model

The observation operator $H : \mathbb{R}^n \to \mathbb{R}^m$ maps the state $x$ to predicted observations $\hat{y} = H(x)$. Real instruments rarely measure the state directly; they smooth, average, mask or project it. Getting $H$ right is what separates an inversion that recovers the truth from one that returns a biased fantasy.

## The likelihood model

vardax assumes Gaussian observation errors,

$$y = H(x) + \varepsilon, \text{ with } \varepsilon \sim \mathcal{N}(0, R)$$ (eq-obs-model)

where $y \in \mathbb{R}^m$ are the observations and $R$ is the observation-error covariance, so $p(y \mid x) = \mathcal{N}(y; H(x), R)$. The observation cost in the variational problem is the negative log-likelihood up to a constant,

$$J_\text{obs}(x) = \frac{1}{2} \|y - H(x)\|^2_{R^{-1}} = \frac{1}{2} (y - H(x))^\top R^{-1} (y - H(x))$$ (eq-obs-cost)

Non-Gaussian likelihoods (Student-$t$, censored, count data) require
either a non-Gaussian extension of $J_\text{obs}$ or amortized
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

# Shapes: N state cells; M observations


# today* (pipekit_cycle)
# the shipped protocol types both members as Any
class ObservationOperator(Protocol):
    # ŷ = H(x)
    def __call__(self, state: Float[Array, " n"]) -> Float[Array, " m"]:
        """Forward: H(x)."""

    # H′(x), the tangent-linear operator at x
    def linearize(self, state: Float[Array, " n"]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator H'(x) for use in 4DVar inner loops."""
```

The `linearize()` method returns a `lineax.AbstractLinearOperator`
(typically `JacobianLinearOperator` via autodiff, but structured
overrides are common). The transpose / adjoint comes for free from
`lineax`.

This contract is the same regardless of whether the operator is a
masked identity, an averaging kernel, or a multi-instrument fusion.

## Five operator families

### Masked identity

In the simplest case we observe the state where the mask is non-zero,

$$H(x) = m \odot x, \text{ with } m \in \{0, 1\}^n$$ (eq-masked-identity)

where $\odot$ is the element-wise product.

Tangent-linear: $H'(x) = \mathrm{diag}(m)$. Use case: SSH altimetry
along-track gaps, SST with cloud masks, in-situ samples on a grid.

The mask is passed per call, so the models apply it to $H'$ themselves. The examples on this page build on each other, starting from a toy state on $N = 6$ cells.

```{code-block} python
:caption: Computes the masked identity [](#eq-masked-identity) on a toy state of six cells.
:label: code-obs-model-masked-identity
import jax.numpy as jnp
import lineax as lx
import vardax as vdx

# Shapes: N = 6 state cells

# x, a toy state
state = jnp.linspace(1.0, 2.0, 6)  # (N,)
# m ∈ {0, 1}ᴺ, cells 2 and 5 unobserved
mask = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])  # (N,)

# today (vardax)
masked_obs = vdx.MaskedIdentity()
# H(x) = m ⊙ x
gappy_obs = masked_obs(state, mask=mask)  # (N,) → (N,)
print(gappy_obs)
```

It prints `[1.  1.2 0.  1.6 1.8 0. ]`.

### Linear projection

A pre-computed projection from state to observation space is

$$H(x) = H_\text{mat} \cdot x$$ (eq-linear-obs)

where $H_\text{mat} \in \mathbb{R}^{m \times n}$ is a fixed matrix.

Use case: Lagrangian footprint matrix from a particle simulator
(observation at receptor = footprint-weighted source field), spectral
filtering, spatial interpolation. The matrix is stored as an
`AbstractLinearOperator` so structure (sparse, low-rank, Kronecker) can
be exploited.

Here the footprint $F = H_\text{mat}$ maps the six cells to $M = 3$ receptors, each averaging two neighbouring cells.

```{code-block} python
:caption: Computes the linear projection [](#eq-linear-obs) through a footprint matrix of three receptors.
:label: code-obs-model-linear
# Shapes: N = 6 state cells; M = 3 receptors

# (½ ½), the average of two neighbouring cells
pair_average = jnp.array([[0.5, 0.5]])  # (1, 2)
# F = I₃ ⊗ (½ ½), each receptor averages two neighbouring cells
footprint = jnp.kron(jnp.eye(3), pair_average)  # (M, N)
# today (lineax)
footprint_op = lx.MatrixLinearOperator(footprint)  # (M, N) op
# today (vardax)
footprint_obs = vdx.LinearObs(operator=footprint_op)
# H(x) = F x
receptor_obs = footprint_obs(state)  # (N,) → (M,)
print(receptor_obs)
```

It prints `[1.1 1.5 1.9]`.

### Averaging kernel: RTM L2 products

For satellite L2 products derived from optimal-estimation retrievals
(TROPOMI CH₄, EMIT CH₄, OCO CO₂, MOPITT CO, …), the L2 retrieval is
*not* a direct sample of the mixing-ratio profile $x$. The retrieval
applies a smoothing matrix $A$ (the averaging kernel) and falls back to
a retrieval prior $x_a$ where the signal is weak:

$$\hat{y} = A (h \odot x + (1 - h) \odot x_a)$$ (eq-averaging-kernel)

where the symbols are

| Symbol | Description |
|---|---|
| $x \in \mathbb{R}^N$ | Model state (profile, surface field) |
| $x_a \in \mathbb{R}^N$ | Retrieval prior from L2 metadata |
| $h \in \mathbb{R}^N$ | Weighting vector (often pressure-weighted) |
| $A \in \mathbb{R}^{N \times N}$ | Averaging kernel matrix |

Tangent-linear: $H'(x) = A \cdot \mathrm{diag}(h)$.

**Skipping the averaging kernel is the most common cause of bias in
operational satellite inversions.** Treating an L2 retrieval as a
direct measurement of the truth gives systematically wrong posteriors,
especially when the retrieval prior $x_a$ differs from the inversion
prior $x_b$. vardax exposes `AveragingKernel` as a first-class operator
because this is a day-one requirement, not a future feature.

The toy kernel $A$ below is tridiagonal, the prior $x_a$ is flat and the sensitivity $h$ falls from 1 to 0.2 with height. The check of the tangent linear against $A \mathrm{diag}(h)$ formed densely prints `True`.

```{code-block} python
:caption: Computes the averaging-kernel operator [](#eq-averaging-kernel) and checks its tangent linear against the dense A diag(h).
:label: code-obs-model-averaging-kernel
import einx

# Shapes: N = 6 levels

# the two nearest neighbours of each level
neighbours = jnp.eye(6, k=1) + jnp.eye(6, k=-1)  # (N, N)
# A = ½ I + ¼ neighbours, a tridiagonal smoothing kernel
kernel = 0.5 * jnp.eye(6) + 0.25 * neighbours  # (N, N)
# x_a, a flat retrieval prior
retrieval_prior = jnp.full(6, 1.8)  # (N,)
# h, sensitivity falling off with height
weighting_vector = jnp.linspace(1.0, 0.2, 6)  # (N,)
# today (lineax)
kernel_op = lx.MatrixLinearOperator(kernel)  # (N, N) op

# today (vardax)
averaging_kernel = vdx.AveragingKernel(
    kernel_matrix=kernel_op,
    prior_profile=retrieval_prior,
    pressure_weights=weighting_vector,
)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a)
retrieved = averaging_kernel(state)  # (N,) → (N,)
# today (vardax)
# H′(x) = A diag(h)
kernel_tangent = averaging_kernel.linearize(state)  # (N, N) op
# A diag(h), formed densely to check
dense_tangent = einx.multiply("i j, j -> i j", kernel, weighting_vector)  # (N, N)
print(bool(jnp.allclose(kernel_tangent.as_matrix(), dense_tangent)))
```

When $A$ has structure (Kronecker for separable kernels, low-rank for
under-determined inversions, banded for compactly-supported kernels),
store it as a `gaussx.AbstractLinearOperator`. The tangent linear
$A \cdot \mathrm{diag}(h)$ is then computed without ever materialising
the dense $A$.

### Multi-instrument fusion

Operational satellite work combines multiple instruments. Each instrument has its own $H_i$, mask $m_i$, error covariance $R_i$ and optionally an averaging kernel $(A_i, x_{a,i}, h_i)$. The fused observation cost composes [](#eq-obs-cost) at the likelihood level,

$$J_\text{obs}(x) = \sum_{i \in \mathcal{I}} \alpha_i \cdot \frac{1}{2} \|m_i \odot (y_i - H_i(x))\|^2_{R_i^{-1}}$$ (eq-fusion-cost)

where $\mathcal{I}$ is the set of instruments and $\alpha_i$ is a per-instrument weight. The weight defaults to uniform; tuning it is for hierarchical bias correction (Epic 9). Each `InstrumentSpec` carries its own operator, mask and $R_i$. Quality masks zero-weight unreliable pixels, so they contribute zero log-likelihood rather than being dropped, which lets us audit the effective per-instrument observation count.

The registry below reuses the three operators above: TROPOMI through the averaging kernel, EMIT through the masked identity and GHGSat through the footprint. It evaluates each $H_i(x)$ but does not form the cost itself.

```{code-block} python
:caption: Evaluates each instrument operator of the fusion cost [](#eq-fusion-cost) through a registry of three instruments.
:label: code-obs-model-fusion
import jax

# Shapes: N = 6 state cells; M = 3 receptors

state_structure = jax.ShapeDtypeStruct((6,), jnp.float32)
receptor_structure = jax.ShapeDtypeStruct((3,), jnp.float32)
# today (lineax)
state_identity = lx.IdentityLinearOperator(state_structure)  # (N, N) op
# today (lineax)
receptor_identity = lx.IdentityLinearOperator(receptor_structure)  # (M, M) op
# R_TROPOMI = 0.04 I
tropomi_cov = 0.04 * state_identity  # (N, N) op
# R_EMIT = 0.01 I
emit_cov = 0.01 * state_identity  # (N, N) op
# R_GHGSat = 0.0025 I
ghgsat_cov = 0.0025 * receptor_identity  # (M, M) op

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
# {i: Hᵢ(x)}
predicted = fusion(state)  # (N,) → {i: (Mᵢ,)}
# today (vardax)
stacked_op = fusion.to_observation_operator()
# [H_EMIT(x); H_GHGSat(x); H_TROPOMI(x)], in sorted id order
stacked = stacked_op(state)  # (N,) → (N + M + N,)
print({name: obs.shape for name, obs in predicted.items()}, stacked.shape)
```

It prints `{'TROPOMI': (6,), 'EMIT': (6,), 'GHGSat': (3,)} (15,)`.

`MultiInstrumentFusion.__call__` returns `dict[str, Array]` (per-instrument predicted observations), and `MultiInstrumentFusion.cost` sums the weighted per-instrument terms. For strict `pipekit_cycle.ObservationOperator` contexts, the `.to_observation_operator()` adapter flattens to a single concatenated output, and its `linearize` is the Jacobian of that map, the per-instrument $H_i'$ stacked row-wise. Pass `fusion.obs_cov_op()` as the model's $R$ with that adapter: its blocks are $R_i / \alpha_i$, so the model minimises the same weighted cost.

### Any differentiable model: parameter estimation

When the control is not the state but a source, a forcing or a parameter vector $q$, the observation operator is a downstream model run $\mathcal{G}(q)$: a transport model followed by its receptors, a radiative-transfer forward, a neural emulator. The cost $J(q)$ is then exactly the 3DVar cost [](#eq-3dvar-cost) with $x = q$, $x_b = q_b$ and $H = \mathcal{G}$, or OI (chapter 4) when $\mathcal{G}$ is linear. `CallableObs` wraps any differentiable JAX function as that operator; `linearize(x)` is a matrix-free `lineax.JacobianLinearOperator`, so the tangent-linear map is a JVP and its adjoint a VJP.

In the example below the control is $P = 3$ unconstrained emission parameters $q$, and $\mathcal{G}(q) = G \operatorname{softplus}(q)$ maps them to $M = 6$ receptors through a transport matrix $G$ that decays with the distance between receptor and source. The observations are noise-free, $B = I$ and $R = 10^{-2} I$.

```{code-block} python
:caption: Minimises the 3DVar cost [](#eq-3dvar-cost) over emission parameters q, with a transport model as the observation operator.
:label: code-obs-model-parameter-3dvar
import optimistix as optx

# Shapes: S = 1 sample; T = 1 time; P = 3 sources; M = 6 receptors

# G, receptor m sees source p with weight exp(−|m − 2p|)
receptor_index = jnp.arange(6.0)  # (M,)
source_index = 2.0 * jnp.arange(3.0)  # (P,)
offset = einx.subtract("m, p -> m p", receptor_index, source_index)  # (M, P)
# |m − 2p|
distance = jnp.abs(offset)  # (M, P)
# G = exp(−|m − 2p|)
transport = jnp.exp(-distance)  # (M, P)


# user
def run_model(source: Float[Array, "t p"]) -> Float[Array, "t m"]:
    # e = log(1 + exp(q)), positive emission rates
    emission = jax.nn.softplus(source)  # (T, P)
    # 𝒢(q) = G softplus(q)
    return einx.dot("t p, m p -> t m", emission, transport)  # (T, M)


# q_b = 0, q_true = (0.5, −0.3, 0.2)
source_background = jnp.zeros((1, 3))  # (T, P)
true_source = jnp.array([[0.5, -0.3, 0.2]])  # (T, P)
# y = 𝒢(q_true), noise-free
receptor_data = run_model(true_source)  # (T, M)

source_structure = jax.ShapeDtypeStruct((1, 3), jnp.float32)
receptor_data_structure = jax.ShapeDtypeStruct((1, 6), jnp.float32)
# today (lineax)
# B = I
source_cov = lx.IdentityLinearOperator(source_structure)  # (T P, T P) op
# today (lineax)
receptor_data_identity = lx.IdentityLinearOperator(receptor_data_structure)
# today (lineax)
# R = 10⁻² I; tagged PSD so the default CG solve accepts it
receptor_cov = lx.TaggedLinearOperator(  # (T M, T M) op
    1e-2 * receptor_data_identity, lx.positive_semidefinite_tag
)
# today (vardax)
# H(q) = 𝒢(q)
model_obs = vdx.CallableObs(fn=run_model)
# today (vardax)
# J(q) = ½ ‖q − q_b‖²_{B⁻¹} + ½ ‖y − 𝒢(q)‖²_{R⁻¹}
parameter_3dvar = vdx.ThreeDVar(
    obs_op=model_obs,
    prior_mean=source_background,
    prior_cov_op=source_cov,
    obs_cov_op=receptor_cov,
    minimiser=optx.BFGS(rtol=1e-6, atol=1e-6),
)
# today (vardax)
receptor_batch = vdx.Batch1D(
    input=einx.id("t m -> 1 t m", receptor_data),  # (T, M) → (S, T, M)
    mask=jnp.ones((1, 1, 6)),
)
# q* = argmin_q J(q)
source_analysis = parameter_3dvar(receptor_batch)  # → (S, T, P)
print(source_analysis[0])
```

It prints `[[ 0.4838582  -0.27665472  0.19014639]]`, close to $q_\text{true} = (0.5, -0.3, 0.2)$; the prior pulls each rate slightly towards $q_b = 0$.

The observation shape need not match the state shape. Prefer
`LinearObs` when you already hold an explicit or structured matrix you
want to reuse.

For a model that defines only a reverse-mode rule (a `jax.custom_vjp`),
pass `CallableObs(fn=..., jac="bwd")`: forward-mode linearisation raises
on those as soon as an analysis or a posterior needs $H'$. Matrix-free,
lineax then applies $H'v$ by transposing the VJP, which fails when the
backward pass holds a loop JAX cannot transpose. A `diffrax` solve is
the common case: `RecursiveCheckpointAdjoint` runs a `while_loop`. Add
`materialise=True` and `linearize` forms the Jacobian once, with one VJP
per observation; that is cheap when the control is small, as a source or
parameter vector usually is. The toy `simulate` below integrates $dc/dt = q - c$ to $t = 1$, so $H'(q) = (1 - e^{-1}) I$, and the block prints `True`:

```{code-block} python
:caption: Materialises the tangent linear of a diffrax-solve observation operator [](#eq-obs-model) with one VJP per observation.
:label: code-obs-model-materialise
import diffrax as dfx

# Shapes: T = 1 time; P = 3 sources


# user
def simulate(source: Float[Array, "t p"]) -> Float[Array, "t p"]:
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
    return solution.ys[-1]  # (T, P)


# today (vardax)
simulator_obs = vdx.CallableObs(fn=simulate, jac="bwd", materialise=True)
# today (vardax)
# H′(q) = ∂c(1)/∂q = (1 − e⁻¹) I
simulator_tangent = simulator_obs.linearize(true_source)  # (T P, T P) op
# (1 − e⁻¹) I, the exact tangent
exact_tangent = (1.0 - jnp.exp(-1.0)) * jnp.eye(3)  # (T P, T P)
print(bool(jnp.allclose(simulator_tangent.as_matrix(), exact_tangent, atol=1e-4)))
```

## Tangent-linear and adjoint contract

For incremental 4DVar (chapter 8), every observation operator must expose `linearize`, whose operator applies $H'$ with `mv` and $H'^\top$ with `transpose().mv`. The pair must pass the adjoint test

$$\langle H' u, v \rangle = \langle u, (H')^\top v \rangle \text{ for all } u, v$$ (eq-adjoint-test)

up to floating-point tolerance, where $u$ is a state-space vector and $v$ an observation-space vector. Failing this test means the gradient flowing back through $H$ is wrong, and any downstream optimiser will take wrong steps. The block below draws random $u = \delta x$ and $v = r$ and prints `True`; `tests/test_obs_operators.py` runs the same test for `AveragingKernel`.

```{code-block} python
:caption: Checks the adjoint test [](#eq-adjoint-test) for the averaging kernel.
:label: code-obs-model-adjoint-test
# Shapes: N = 6 levels

key = jax.random.PRNGKey(0)
key_tangent, key_residual = jax.random.split(key)

# δx ~ 𝒩(0, I)
increment = jax.random.normal(key_tangent, (6,))  # (N,)
# r ~ 𝒩(0, I)
residual = jax.random.normal(key_residual, (6,))  # (N,)

# today (vardax)
# H′(x)
kernel_tangent = averaging_kernel.linearize(state)  # (N, N) op
# H′ δx, the forward tangent-linear model
tangent_increment = kernel_tangent.mv(increment)  # (N,) → (N,)
# H′ᵀ r, the adjoint
adjoint_residual = kernel_tangent.transpose().mv(residual)  # (N,) → (N,)
# ⟨H′ δx, r⟩
forward_inner = einx.dot("n, n ->", tangent_increment, residual)  # → ()
# ⟨δx, H′ᵀ r⟩
adjoint_inner = einx.dot("n, n ->", increment, adjoint_residual)  # → ()
print(bool(jnp.isclose(forward_inner, adjoint_inner, atol=1e-5)))
```

## Observation-error covariance $R$

$R$ is supplied as a `lineax.AbstractLinearOperator`. Common
parameterisations:

- **Diagonal**: `lineax.DiagonalLinearOperator(variances)`. The default for heteroscedastic retrievals (per-pixel σ from L2 metadata).
- **Block-diagonal**: `gaussx.BlockDiag(*instrument_covs)`. Per-instrument diagonals when fusing.
- **Structured**: `kernellib.ImplicitKernelOperator(matern, coords)` if observation errors are spatially correlated (rare, but it matters for high-resolution imaging spectrometers).

The models' solves reject gaussx operators until they are wrapped as a `lx.FunctionLinearOperator`, as [chapter 4](04_oi_blue.md#why-the-matrices-are-never-materialised) does for $B$.

The choice flows into [](#eq-obs-cost) via $R^{-1}$, which is applied lazily: the inverse is never materialised, only applied to vectors via `lineax.CG`.

## When the observation model is wrong

A common failure mode: $H$ is misspecified (wrong averaging kernel,
missing instrument bias, ignored representation error). The MAP
$x^*$ then converges to a wrong answer with a confident-looking
posterior. Diagnostics:

- **Innovation statistics** $d = y - H(x_b)$. Should be roughly mean-zero
  and consistent with $\sqrt{\mathrm{diag}(H B H^\top + R)}$. Large
  systematic offsets indicate bias.
- **Posterior vs prior log-evidence ratio.** If the data move the
  posterior much further than $\sigma_\text{post}$ from the prior, the
  $B$ or $R$ assumptions are wrong.
- **Cross-instrument disagreement.** Per-instrument residual statistics
  should be consistent; large disagreements flag uncorrected bias.

Epic 9 adds joint bias estimation for the cross-instrument case.

## See also

- Chapter 3: the dynamical model and its adjoint
- Chapter 4: `OptimalInterpolation` (the linear-$H$ case)
- Chapter 8: `IncrementalFourDVar` (uses `linearize()` for tangent-linear observations)
- Chapter 11: the observation operator family in detail
- Design doc: [`design/api/observation_operators.md`](design/api/observation_operators.md)
