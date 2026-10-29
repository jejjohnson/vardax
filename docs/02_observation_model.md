# Observation Model

The observation operator $H : \mathbb{R}^n \to \mathbb{R}^m$ maps the
state $x$ to predicted observations $\hat{y} = H(x)$. Real
instruments rarely measure the state directly — they smooth, average,
mask, or project. Getting $H$ right is what separates an inversion
that recovers truth from one that returns a biased fantasy.

## The likelihood model

vardax assumes Gaussian observation errors:

$$y = H(x) + \varepsilon, \text{ with } \varepsilon \sim \mathcal{N}(0, R),$$

so $p(y \mid x) = \mathcal{N}(y; H(x), R)$. The observation cost in the
variational problem is

$$J_\text{obs}(x) = \frac{1}{2} \|y - H(x)\|^2_{R^{-1}} = \frac{1}{2} (y - H(x))^\top R^{-1} (y - H(x)).$$

Non-Gaussian likelihoods (Student-$t$, censored, count data) require
either a non-Gaussian extension of $J_\text{obs}$ or amortized
inference with a flow / score-based head (chapter 10).

## The `ObservationOperator` protocol

Every vardax observation operator satisfies
`pipekit_cycle.ObservationOperator`, restated here with typed signatures:

```python
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

The simplest case — we observe the state where the mask is non-zero:

$$H(x) = m \odot x, \text{ with } m \in \{0, 1\}^n.$$

Tangent-linear: $H'(x) = \mathrm{diag}(m)$. Use case: SSH altimetry
along-track gaps, SST with cloud masks, in-situ samples on a grid.

The mask is passed per call, so the models apply it to $H'$ themselves; the examples on this page build on each other, starting from a toy state on $N = 6$ cells.

```python
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

Pre-computed projection from state to observation space:

$$H(x) = H_\text{mat} \cdot x.$$

Use case: Lagrangian footprint matrix from a particle simulator
(observation at receptor = footprint-weighted source field), spectral
filtering, spatial interpolation. The matrix is stored as an
`AbstractLinearOperator` so structure (sparse, low-rank, Kronecker) can
be exploited.

Here the footprint $F = H_\text{mat}$ maps the six cells to $M = 3$ receptors, each averaging two neighbouring cells. The block continues the one above.

```python
# Shapes: N = 6 state cells; M = 3 receptors

# F, each receptor averages two neighbouring cells
footprint = jnp.kron(jnp.eye(3), jnp.array([[0.5, 0.5]]))  # (M, N)
# today (lineax)
footprint_op = lx.MatrixLinearOperator(footprint)  # (M, N) op
# today (vardax)
footprint_obs = vdx.LinearObs(operator=footprint_op)
# H(x) = F x
receptor_obs = footprint_obs(state)  # (N,) → (M,)
print(receptor_obs)
```

It prints `[1.1 1.5 1.9]`.

### Averaging kernel — RTM L2 products

For satellite L2 products derived from optimal-estimation retrievals
(TROPOMI CH₄, EMIT CH₄, OCO CO₂, MOPITT CO, …), the L2 retrieval is
*not* a direct sample of the mixing-ratio profile $x$. The retrieval
applies a smoothing matrix $A$ (the averaging kernel) and falls back to
a retrieval prior $x_a$ where the signal is weak:

$$\hat{y} = A (h \odot x + (1 - h) \odot x_a).$$

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

The toy kernel $A$ below is tridiagonal, the prior $x_a$ is flat and the sensitivity $h$ falls from 1 to 0.2 with height. The block continues the one above and prints `True`.

```python
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

# today (vardax)
averaging_kernel = vdx.AveragingKernel(
    kernel_matrix=lx.MatrixLinearOperator(kernel),
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

Operational satellite work combines multiple instruments. Each
instrument has its own $H_i$, mask $m_i$, error covariance $R_i$, and
optionally averaging kernel $(A_i, x_{a,i}, h_i)$. The fused
observation cost composes at the likelihood level:

$$J_\text{obs}(x) = \sum_{i \in \mathcal{I}} \alpha_i \cdot \frac{1}{2} \|m_i \odot (y_i - H_i(x))\|^2_{R_i^{-1}}.$$

The per-instrument weight $\alpha_i$ defaults to uniform; tuning is for
hierarchical bias-correction (Epic 9). Each `InstrumentSpec` carries its own operator, mask and $R_i$. Quality masks zero-weight unreliable
pixels — they contribute zero log-likelihood, not dropped (this lets us
audit effective per-instrument observation count).

The registry below reuses the three operators above: TROPOMI through the averaging kernel, EMIT through the masked identity and GHGSat through the footprint. The block continues the one above.

```python
import jax

# Shapes: N = 6 state cells; M = 3 receptors

state_structure = jax.ShapeDtypeStruct((6,), jnp.float32)
receptor_structure = jax.ShapeDtypeStruct((3,), jnp.float32)
# R_TROPOMI = 0.04 I
tropomi_cov = 0.04 * lx.IdentityLinearOperator(state_structure)  # (N, N) op
# R_EMIT = 0.01 I
emit_cov = 0.01 * lx.IdentityLinearOperator(state_structure)  # (N, N) op
# R_GHGSat = 0.0025 I
ghgsat_cov = 0.0025 * lx.IdentityLinearOperator(receptor_structure)  # (M, M) op

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
# [H_EMIT(x); H_GHGSat(x); H_TROPOMI(x)], in sorted id order
stacked = fusion.to_observation_operator()(state)  # (N,) → (N + M + N,)
print({name: obs.shape for name, obs in predicted.items()}, stacked.shape)
```

It prints `{'TROPOMI': (6,), 'EMIT': (6,), 'GHGSat': (3,)} (15,)`.

`MultiInstrumentFusion.__call__` returns `dict[str, Array]` (per-instrument predicted observations); no vardax model consumes the dict yet. For strict `pipekit_cycle.ObservationOperator` contexts, the `.to_observation_operator()` adapter flattens to a single concatenated output, and its `linearize` is the Jacobian of that map, the per-instrument $H_i'$ stacked row-wise.

### Any differentiable model — parameter estimation

When the control is not the state but a source, a forcing or a parameter
vector $q$, the observation operator is a downstream model run
$\mathcal{G}(q)$: a transport model followed by its receptors, a
radiative-transfer forward, a neural emulator. The cost is then

$$J(q) = \frac{1}{2} (q - q_b)^\top B^{-1} (q - q_b) + \frac{1}{2} (y - \mathcal{G}(q))^\top R^{-1} (y - \mathcal{G}(q)),$$

which is exactly 3DVar (chapter 5) with $\mathcal{G}$ as $H$, or OI
(chapter 4) when $\mathcal{G}$ is linear. `CallableObs` wraps any
differentiable JAX function as that operator; `linearize(x)` is a
matrix-free `lineax.JacobianLinearOperator`, so the tangent-linear map is
a JVP and its adjoint a VJP.

In the example below the control is $P = 3$ unconstrained emission parameters $q$, and $\mathcal{G}(q) = G \operatorname{softplus}(q)$ maps them to $M = 6$ receptors through a transport matrix $G$ that decays with the distance between receptor and source. The observations are noise-free, $B = I$ and $R = 10^{-2} I$. The block continues the one above.

```python
import optimistix as optx

# Shapes: S = 1 sample; T = 1 time; P = 3 sources; M = 6 receptors

# G, receptor m sees source p with weight exp(−|m − 2p|)
receptor_index = jnp.arange(6.0)  # (M,)
source_index = 2.0 * jnp.arange(3.0)  # (P,)
offset = einx.subtract("m, p -> m p", receptor_index, source_index)  # (M, P)
transport = jnp.exp(-jnp.abs(offset))  # (M, P)


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

# today (lineax)
# B = I
source_cov = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float32))
# today (lineax)
# R = 10⁻² I; tagged PSD so the default CG solve accepts it
receptor_cov = lx.TaggedLinearOperator(  # (T M, T M) op
    1e-2 * lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 6), jnp.float32)),
    lx.positive_semidefinite_tag,
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

```python
import diffrax as dfx

# Shapes: T = 1 time; P = 3 sources


# user
def simulate(source: Float[Array, "t p"]) -> Float[Array, "t p"]:
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
exact_tangent = (1.0 - jnp.exp(-1.0)) * jnp.eye(3)  # (T P, T P)
print(bool(jnp.allclose(simulator_tangent.as_matrix(), exact_tangent, atol=1e-4)))
```

## Tangent-linear and adjoint contract

For incremental 4DVar (chapter 8), every observation operator must
expose `linearize`, whose operator applies $H'$ with `mv` and $H'^\top$ with `transpose().mv`. The block continues the one above and uses the averaging kernel:

```python
# Shapes: N = 6 levels

key_tangent, key_residual = jax.random.split(jax.random.PRNGKey(0))
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

The block ends with the adjoint test, which prints `True`; `tests/test_obs_operators.py` runs the same test for `AveragingKernel`:

$$\langle H' u, v \rangle = \langle u, (H')^\top v \rangle \text{ for all } u, v$$

up to floating-point tolerance. Failing this test means the gradient
flowing back through $H$ is wrong, and any downstream optimiser will
take wrong steps.

## Observation-error covariance $R$

$R$ is supplied as a `lineax.AbstractLinearOperator`. Common
parameterisations:

- **Diagonal** — `lineax.DiagonalLinearOperator(variances)`. Default
  for heteroscedastic retrievals (per-pixel σ from L2 metadata).
- **Block-diagonal** — `gaussx.BlockDiag(*instrument_covs)`.
  Per-instrument diagonals when fusing.
- **Structured** — `gaussx.ImplicitKernelOperator(matern, coords)` if obs errors are
  spatially correlated (rare but matters for high-resolution imaging
  spectrometers).

gaussx operators do not register `lineax.linearise`, so the models' solves reject them; wrap one as `lx.FunctionLinearOperator(op.mv, structure, lx.positive_semidefinite_tag)` first, as chapter 4 does for $B$.

The choice flows into $J_\text{obs}$ via $R^{-1}$, computed lazily —
the inverse is never materialised, only applied to vectors via
`lineax.CG`.

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

- Chapter 3 — dynamical model + its adjoint
- Chapter 4 — `OptimalInterpolation` (the linear-$H$ case)
- Chapter 8 — `IncrementalFourDVar` (uses `linearize()` for tangent-linear obs)
- Chapter 11 — observation operator family in detail
- Design doc: [`design/api/observation_operators.md`](design/api/observation_operators.md)
