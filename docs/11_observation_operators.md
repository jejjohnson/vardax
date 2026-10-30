---
numbering:
  enumerator: "11.%s"
---

# Observation Operators

Chapter 2 introduced the observation model [](#eq-obs-model), its cost [](#eq-obs-cost) and the `ObservationOperator` protocol, and it defines each operator family. This chapter is the catalogue of the operators vardax ships (masked identity, linear projection, averaging kernel, multi-instrument fusion and scattered point observations) and of their implementation details.

The first four satisfy `pipekit_cycle.ObservationOperator` (the first three directly; multi-instrument fusion via its `.to_observation_operator()` adapter, see chapter 2), which `tests/test_pipekit_protocols.py` checks. These four expose `__call__(x)` and `linearize(x)`; the adjoint test at the end of this chapter checks the tangent-linear contract.

The Python blocks on this page run in order, each continuing the ones above it.

## Masked identity

The masked identity [](#eq-masked-identity) observes the state where $m_i = 1$. The PRNG keys for the whole page are split here.

```{code-block} python
:caption: Applies the masked identity [](#eq-masked-identity) to a random state and reads its tangent H′(x).
:label: code-obsops-masked-identity
import jax
import jax.numpy as jnp
import vardax as vdx

# Shapes: N = 6 grid cells

key = jax.random.PRNGKey(0)
(
    key_state,
    key_mask,
    key_left,
    key_right,
    key_tropomi,
    key_emit,
    key_truth,
    key_noise,
    key_adjoint,
) = jax.random.split(key, 9)

# x ~ 𝒩(0, I)
state = jax.random.normal(key_state, (6,))  # (N,)
# m ~ Bernoulli(0.5)
is_observed = jax.random.bernoulli(key_mask, 0.5, (6,))  # (N,) bool
mask = is_observed.astype(jnp.float32)  # (N,)

# today (vardax)
masked_op = vdx.MaskedIdentity()
# today (vardax)
# H(x) = m ⊙ x
predicted = masked_op(state, mask=mask)  # (N,) → (N,)
# today (vardax)
# H′(x) = I; the mask enters at call time, not in the tangent
tangent = masked_op.linearize(state)  # (N, N) op
# H′ m = m
tangent_mask = tangent.mv(mask)  # (N,) → (N,)
print(type(tangent).__name__, bool(jnp.allclose(tangent_mask, mask)))
```

It prints `JacobianLinearOperator True`: the tangent is the identity, so masked residuals come from the cost, which multiplies by $m$.

The mask can be soft ($m_i \in [0, 1]$) for fuzzy observations. The operator still works, but the obs-error covariance interpretation changes (a soft mask implicitly inflates $R$ where $m$ is small).

## Linear projection

The linear projection [](#eq-linear-obs) applies a stored matrix $H_\text{mat}$. The block builds it from a footprint in which each of $M = 3$ receptors averages two neighbouring cells.

```{code-block} python
:caption: Builds a footprint matrix H_mat and applies the linear projection [](#eq-linear-obs) and its tangent.
:label: code-obsops-linear-projection
import lineax as lx

# Shapes: M = 3 receptors; N = 6 grid cells

# [½ ½], one receptor over two neighbouring cells
pair_average = jnp.array([[0.5, 0.5]])  # (1, 2)
# H_mat = I₃ ⊗ [½ ½]
footprint = jnp.kron(jnp.eye(3), pair_average)  # (M, N)
footprint_op = lx.MatrixLinearOperator(footprint)  # (M, N) op

# today (vardax)
linear_op = vdx.LinearObs(operator=footprint_op)
# today (vardax)
# H(x) = H_mat x
predicted = linear_op(state)  # (N,) → (M,)
# today (vardax)
# H′(x) = H_mat, the stored operator itself
tangent = linear_op.linearize(state)  # (M, N) op
print(predicted.shape, tangent is linear_op.operator)
```

It prints `(3,) True`.

Use cases:

- **Lagrangian footprint**: a particle simulator produces a footprint matrix mapping the source field to receptor observations. The matrix is often sparse (only cells with non-zero footprint contribute), and lineax has no sparse operator, so wrap the sparse product in a `lineax.FunctionLinearOperator`.
- **Spectral filtering**: observe specific frequency bands. Store $H_\text{mat} = F^{-1} S F$ as a composition of `lineax` operators.
- **Spatial interpolation**: bilinear or bicubic projection from the grid to observation points. The interpolation matrix is sparse.

## Averaging kernel

The averaging kernel [](#eq-averaging-kernel) is the right model for RTM-derived L2 satellite products; chapter 2 defines it and its symbols, and this section covers its construction and structure. For column CH₄ the weighting $h$ tends to 1 in the lower troposphere, where the signal comes from the atmosphere, and to 0 in the stratosphere, where the retrieval prior $x_a$ takes over. The tangent-linear map $H'(x) = A \cdot \mathrm{diag}(h)$ inherits the structure of $A$ (sparse, Kronecker, low-rank).

### Constructing from L2 metadata

L2 products typically include the averaging-kernel matrix, prior, and weighting as sidecar metadata in the netCDF / HDF5 file.

```{code-block} python
:caption: Builds the averaging kernel [](#eq-averaging-kernel) from a toy L2 sidecar and applies it to a flat column.
:label: code-obsops-averaging-kernel-l2
import xarray as xr

# Shapes: L = 4 retrieval levels


# user
def averaging_kernel_from_l2(dataset: xr.Dataset) -> vdx.AveragingKernel:
    # A, dense from the sidecar matrix
    kernel_values = jnp.asarray(dataset["averaging_kernel"].values)  # (L, L)
    kernel = lx.MatrixLinearOperator(kernel_values)  # (L, L) op
    # x_a, the retrieval prior
    prior_profile = jnp.asarray(dataset["retrieval_prior"].values)  # (L,)
    # h, the pressure weighting
    pressure_weights = jnp.asarray(dataset["weighting_vector"].values)  # (L,)
    # today (vardax)
    # H(x) = A (h ⊙ x + (1 − h) ⊙ x_a)
    return vdx.AveragingKernel(
        kernel_matrix=kernel,
        prior_profile=prior_profile,
        pressure_weights=pressure_weights,
    )


# a toy L2 sidecar: smoothing kernel, flat prior, weights falling with height
level_matrix = jnp.array(
    [
        [0.6, 0.3, 0.1, 0.0],
        [0.2, 0.5, 0.2, 0.1],
        [0.1, 0.2, 0.5, 0.2],
        [0.0, 0.1, 0.3, 0.6],
    ]
)  # (L, L)
l2_sidecar = xr.Dataset(
    {
        "averaging_kernel": (("level", "level_out"), level_matrix),
        "retrieval_prior": ("level", jnp.full(4, 1.9)),
        "weighting_vector": ("level", jnp.array([1.0, 0.9, 0.6, 0.2])),
    }
)
# user
tropomi = averaging_kernel_from_l2(l2_sidecar)
# x = 2 on every level
column = jnp.full(4, 2.0)  # (L,)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a)
predicted_column = tropomi(column)  # (L,) → (L,)
print(predicted_column)
```

It prints `[1.993     1.979     1.962     1.9390001]`: each row of $A$ sums to one, so the output lies between the state $x = 2$ and the retrieval prior $x_a = 1.9$, closer to the prior where $h$ is small.

### Structured kernels

For separable kernels (e.g. independent vertical and horizontal smoothing), use Kronecker structure, $A = A_v \otimes A_h$, here with $h = 0.8$ and $x_a = 0$.

```{code-block} python
:caption: Applies the averaging kernel [](#eq-averaging-kernel) with a Kronecker kernel A = A_v ⊗ A_h and checks it against the dense product.
:label: code-obsops-kronecker-kernel
import einx
import gaussx as gx

# Shapes: L = 2 levels; C = 3 columns; N = L C = 6 cells

# A_v, vertical smoothing
vertical_kernel = jnp.array([[0.7, 0.3], [0.3, 0.7]])  # (L, L)
# A_h, horizontal smoothing
horizontal_kernel = jnp.array(
    [[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]]
)  # (C, C)
vertical_op = lx.MatrixLinearOperator(vertical_kernel)  # (L, L) op
horizontal_op = lx.MatrixLinearOperator(horizontal_kernel)  # (C, C) op
# today (gaussx)
# A = A_v ⊗ A_h
kronecker_kernel = gx.Kronecker(vertical_op, horizontal_op)  # (N, N) op
# today (vardax)
# H(x) = A (h ⊙ x + (1 − h) ⊙ x_a)
separable_ak = vdx.AveragingKernel(
    kernel_matrix=kronecker_kernel,
    prior_profile=jnp.zeros(6),
    pressure_weights=jnp.full(6, 0.8),
)
# today (vardax)
# vec(A_v X A_hᵀ) without forming the (N, N) matrix
smoothed = separable_ak(state)  # (N,) → (N,)
# A = A_v ⊗ A_h, dense for the check
dense_kernel = jnp.kron(vertical_kernel, horizontal_kernel)  # (N, N)
# h ⊙ x with h = 0.8 and x_a = 0
weighted_state = 0.8 * state  # (N,)
# A (h ⊙ x)
dense_smoothed = einx.dot("i j, j -> i", dense_kernel, weighted_state)  # (N,)
print(bool(jnp.allclose(smoothed, dense_smoothed, atol=1e-6)))
```

It prints `True`.

With Kronecker structure $A$ is never materialised, and the mat-vec $\mathrm{vec}(A \cdot X)$ becomes two small mat-vecs instead of one giant one.

For low-rank kernels (information content $\le$ rank), use low-rank
operators:

```{code-block} python
:caption: Composes a lazy low-rank kernel A = U diag(s) Vᵀ for the averaging kernel [](#eq-averaging-kernel).
:label: code-obsops-low-rank-kernel
# Shapes: N = 6 cells; K = 2 rank

# U ~ 𝒩(0, I)
left_vectors = jax.random.normal(key_left, (6, 2))  # (N, K)
# s
singular_values = jnp.array([0.9, 0.4])  # (K,)
# V ~ 𝒩(0, I)
right_vectors = jax.random.normal(key_right, (6, 2))  # (N, K)
# Vᵀ
right_transposed = einx.id("n k -> k n", right_vectors)  # (N, K) → (K, N)
left_op = lx.MatrixLinearOperator(left_vectors)  # (N, K) op
singular_op = lx.DiagonalLinearOperator(singular_values)  # (K, K) op
right_op = lx.MatrixLinearOperator(right_transposed)  # (K, N) op
# A = U diag(s) Vᵀ, composed lazily
low_rank_kernel = left_op @ singular_op @ right_op  # (N, N) op
# A x
low_rank_smoothed = low_rank_kernel.mv(state)  # (N,) → (N,)
print(low_rank_smoothed.shape)
```

It prints `(6,)`.

Both save memory and respect the underlying retrieval physics (retrievals with low DOF should have low-rank $A$).

### Why skipping the AK is wrong

Chapter 2 explains why treating an L2 retrieval as $y = x + \varepsilon$ instead of [](#eq-averaging-kernel) biases the posterior when $x_a \ne x_b$. The retrieval folds in information from $x_a$ that we then attribute to the data, so the MAP is pulled toward $x_a$.

The bias is largest when:

- Information content is low (rank-deficient $A$, large $1-h$ fraction)
- $x_a$ differs from $x_b$ (e.g. inventory prior vs. climatology)
- Long inversion windows compound the bias

The fix is to always use `AveragingKernel`. The cost is one extra mat-vec per observation, which is negligible next to the forward-model cost.

## Multi-instrument fusion

For multi-satellite inversions, each instrument gets an `InstrumentSpec` with its own operator, mask and $R_i$. The toy below has a coarse TROPOMI-like instrument behind an averaging kernel and a fine EMIT-like one on the grid. `MultiInstrumentFusion` returns the per-instrument predictions, and its `cost` sums the terms of the fusion cost [](#eq-fusion-cost), here with $\alpha_i = 1$.

```{code-block} python
:caption: Fuses a TROPOMI-like and an EMIT-like instrument and sums the fusion cost [](#eq-fusion-cost) with αᵢ = 1.
:label: code-obsops-fusion
from jaxtyping import Array, Float

# Shapes: N = 6 grid cells; M_t = 3 TROPOMI pixels; M_e = 6 EMIT pixels

# today (vardax)
# TROPOMI: coarse pixels through an averaging kernel, H_t(x) = A_t (h ⊙ x)
tropomi_op = vdx.AveragingKernel(
    kernel_matrix=footprint_op,
    prior_profile=jnp.zeros(6),
    pressure_weights=jnp.full(6, 0.9),
)
# m_t: the third TROPOMI pixel fails quality control
tropomi_mask = jnp.array([1.0, 1.0, 0.0])  # (M_t,)
# R_t = 0.04 I
tropomi_obs_cov = lx.DiagonalLinearOperator(jnp.full(3, 0.04))  # (M_t, M_t) op
# today (vardax)
tropomi_spec = vdx.InstrumentSpec(
    obs_op=tropomi_op,
    mask=tropomi_mask,
    obs_cov_op=tropomi_obs_cov,
    instrument_id="TROPOMI",
)
# R_e = 0.01 I
emit_obs_cov = lx.DiagonalLinearOperator(jnp.full(6, 0.01))  # (M_e, M_e) op
# today (vardax)
# EMIT: fine pixels on the grid, H_e(x) = x
emit_spec = vdx.InstrumentSpec(
    obs_op=masked_op,
    mask=mask,
    obs_cov_op=emit_obs_cov,
    instrument_id="EMIT",
)
# today (vardax)
registry = vdx.InstrumentRegistry(entries={"TROPOMI": tropomi_spec, "EMIT": emit_spec})
# today (vardax)
fusion = vdx.MultiInstrumentFusion(registry=registry)
# ŷᵢ = Hᵢ(x), one entry per instrument
predictions = fusion(state)  # (N,) → {TROPOMI: (M_t,), EMIT: (M_e,)}
# ε_t ~ 𝒩(0, I)
tropomi_noise = jax.random.normal(key_tropomi, (3,))  # (M_t,)
# ε_e ~ 𝒩(0, I)
emit_noise = jax.random.normal(key_emit, (6,))  # (M_e,)
observations = {
    # y_t = ŷ_t + 0.2 ε_t
    "TROPOMI": predictions["TROPOMI"] + 0.2 * tropomi_noise,
    # y_e = ŷ_e + 0.1 ε_e
    "EMIT": predictions["EMIT"] + 0.1 * emit_noise,
}

# today (vardax)
# J_obs = Σᵢ αᵢ ½ ‖mᵢ ⊙ (yᵢ − Hᵢ(x))‖²_{Rᵢ⁻¹}, with αᵢ = 1 at the truth
obs_cost = fusion.cost(state, observations)  # → ()
print({name: value.shape for name, value in predictions.items()})
print(f"J_obs at the truth {float(obs_cost):.2f}")
```

It prints `{'TROPOMI': (3,), 'EMIT': (6,)}` and `J_obs at the truth 2.57`.

Because the fusion happens at the likelihood level [](#eq-fusion-cost), nothing is regridded to a common grid. Cross-instrument errors default to independent; correlated errors need a block-non-diagonal $R$. The quality mask $m_i$ lives in each `InstrumentSpec`, since `Batch*` has no per-pixel instrument index.

### Cross-instrument bias

Real instruments disagree systematically. TROPOMI and GHGSat differ by about 5 to 20 ppb on methane retrievals due to spectral calibration, retrieval algorithm differences, and footprint effects. Pretending they agree biases the joint analysis.

For joint inversion with bias estimation (Epic 9), introduce a per-instrument bias state $b_i$, which shifts each prediction to $H_i(x) + b_i$ and adds a Gaussian prior cost on $b_i$:

```{code-block} python
:caption: Adds a per-instrument bias bᵢ to the predictions of [](#eq-fusion-cost) and evaluates its Gaussian prior cost.
:label: code-obsops-biased-fusion
import equinox as eqx

# Shapes: N = 6 grid cells; one scalar bias bᵢ per instrument


# user
class BiasedFusion(eqx.Module):
    """Per-instrument additive bias on top of a fusion, with prior bᵢ ~ 𝒩(0, σ_b²)."""

    base: vdx.MultiInstrumentFusion
    bias_std: dict[str, float]

    def __call__(
        self, state: Float[Array, " n"], bias: dict[str, Float[Array, ""]]
    ) -> dict[str, Float[Array, " m"]]:
        # today (vardax)
        # Hᵢ(x), one entry per instrument
        predictions = self.base(state)
        # ŷᵢ = Hᵢ(x) + bᵢ
        return {name: predictions[name] + bias[name] for name in predictions}

    def bias_cost(self, bias: dict[str, Float[Array, ""]]) -> Float[Array, ""]:
        total = jnp.zeros(())
        for name, std in self.bias_std.items():
            # bᵢ / σ_b,ᵢ
            standardised = bias[name] / std  # → ()
            # J_b = Σᵢ ½ bᵢ² / σ_b,ᵢ²
            total = total + 0.5 * standardised**2
        return total  # → ()


# user
biased = BiasedFusion(base=fusion, bias_std={"TROPOMI": 0.02, "EMIT": 0.01})
bias = {"TROPOMI": jnp.float32(0.01), "EMIT": jnp.float32(0.0)}
# ŷᵢ = Hᵢ(x) + bᵢ
biased_predictions = biased(state, bias)  # (N,) → {TROPOMI: (M_t,), EMIT: (M_e,)}
# J_b = Σᵢ ½ bᵢ² / σ_b,ᵢ²
bias_cost = biased.bias_cost(bias)  # → ()
print(f"J_b = {float(bias_cost):.3f}")
```

It prints `J_b = 0.125`.

The bias $b_i$ becomes a joint state variable, and the analysis solves for $(x, \boldsymbol{b})$ together. The bias prior tames the underdeterminacy. Hierarchical priors per (instrument, basin, season)
are the operational pattern.

This is not yet shipped in vardax; the design is documented for Epic 9.

## Scattered point observations

Along-track altimetry, drifters and station networks deliver observations
at scattered coordinates and times, not on the state grid. Scattering them
into a mostly-`NaN` grid would discard their positions and force one $R$
for all times. A `PointObsBatch` carries them as they are, padded to a
fixed count $n_\text{obs}$ so shapes stay static under `jit`:

| Field | Shape | Meaning |
|---|---|---|
| `coords` | `(N_obs, D)` | $s_i$, e.g. `(x,)`, `(y, x)` or `(lon, lat, t)` |
| `values` | `(N_obs,)` | $y_i$ |
| `sigma` | `(N_obs,)` | per-observation error std $\sigma_i$ |
| `valid` | `(N_obs,)` | $m_i$, `False` on padding |
| `time_index` | `(N_obs,)` or `None` | window step $\tau_i$ for gridded-state models |
| `n_times` | static `int` | states in the window (rollout steps + 1) |

The observation cost is

$$J_o = \frac{1}{2} \sum_i m_i \frac{\left(y_i - \mathcal H(x)(s_i, t_i)\right)^2}{\sigma_i^2}$$ (eq-point-obs-cost)

where $\mathcal H(x)(s_i, t_i)$ is the state evaluated at the coordinates $s_i$ and time $t_i$ of observation $i$, and the other symbols are the fields in the table.

Padding rows contribute exactly zero value **and** zero gradient.
`point_obs_cost` masks the whitened residual with `jnp.where` *before*
squaring, and first replaces the padding's coordinates, values and
$\sigma$ with finite placeholders. Multiplying by $m_i$ after squaring
would not be enough: $0 \cdot \text{NaN} = \text{NaN}$ in the gradient of
a padding row whose value or $\sigma = 0$ is garbage.

Two operators evaluate $\mathcal H(x)$ at the points:

- **`GridPointObs(grid_axes)`**: for a gridded state. The state has a
  leading time axis, either a 4D-Var trajectory $(x_0, \dots, x_T)$ or a
  3D-Var state `(T, N)`. Observation $i$ reads $x_{\tau_i}$ and
  interpolates it linearly (bilinearly on a 2-D grid) at $s_i$, clamped at
  the grid edge. This is `InterpObs`'s `order=1` stencil, computed at call
  time so that the coordinates may be traced.
- **`FieldPointObs()`**: for a state that is a callable field
  $f(s)$, such as a neural field. It returns $\hat y_i = f(s_i)$ and
  passes all `D` columns to the field, a time column included.

`ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`, `ReducedBasisFourDVar` and
`IncrementalFourDVar` choose their path from the batch type. A
`PointObsBatch` needs `obs_op=GridPointObs(...)`, and the per-observation
$\sigma_i$ replaces `obs_cov_op`, which must be `None` (or an identity
operator; anything else raises `ValueError` rather than being silently
ignored). 4D-Var models compare observation $i$
with the trajectory state at step $\tau_i$, so $\mathcal H_t$ and $R_t$
can change from step to step. `ThreeDVar` also accepts `FieldPointObs`
together with a `NeuralFieldParameterization`, which evaluates the field
at the observation coordinates without a grid. A leading batch axis on
the batch's arrays is optional.

The toy below has a decaying forward model $M(x) = x / 2$ on $N = 8$ cells, four observations spread over three window steps and one padding row.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) with the point-observation term [](#eq-point-obs-cost) on a toy decay model.
:label: code-obsops-point-obs
# Shapes: N = 8 grid cells; M = 5 point observations, one of them padding;
# D = 1 coordinate; T + 1 = 3 states in the window

# the grid coordinates of the cells
cell_axis = jnp.arange(8.0)  # (N,)


# user
class Decay(eqx.Module):
    """M(x) = x / 2, a forward model with a unit step."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # x_{t+1} = x_t / 2
        return 0.5 * state


# x₀ ~ 𝒩(0, I)
initial_truth = jax.random.normal(key_truth, (8,))  # (N,)
# 2⁻ᵗ, t = 0 … T
decay_factors = 0.5 ** jnp.arange(3.0)  # (T+1,)
# x_t = M^t(x₀) = 2⁻ᵗ x₀
trajectory = einx.multiply("t, n -> t n", decay_factors, initial_truth)  # (T+1, N)
# sᵢ, the observation coordinates
coords = jnp.array([[0.5], [2.0], [4.25], [6.5], [0.0]])  # (M, D)
# mᵢ, False on the padding row
valid = jnp.array([True, True, True, True, False])  # (M,)
# τᵢ, the window step of each observation
time_index = jnp.array([0, 1, 1, 2, 0])  # (M,)
# today (vardax)
point_batch = vdx.PointObsBatch(
    coords=coords,
    values=jnp.zeros(5),
    sigma=jnp.full(5, 0.1),
    valid=valid,
    time_index=time_index,
    n_times=3,
)
# today (vardax)
point_op = vdx.GridPointObs((cell_axis,))
# today (vardax)
# 𝓗(x_{τᵢ})(sᵢ), linear interpolation in the frame τᵢ
truth_points = point_op(trajectory, point_batch)  # (T+1, N) → (M,)
# εᵢ ~ 𝒩(0, 1)
point_noise = jax.random.normal(key_noise, (5,))  # (M,)
# yᵢ = 𝓗(x_{τᵢ})(sᵢ) + σ εᵢ, σ = 0.1
ssh = truth_points + 0.1 * point_noise  # (M,)
point_batch = eqx.tree_at(lambda batch: batch.values, point_batch, ssh)

point_structure = jax.ShapeDtypeStruct((8,), jnp.float32)
# B = I
point_prior_cov = lx.IdentityLinearOperator(point_structure)  # (N, N) op
# today (vardax)
# J(x₀) = ½ ‖x₀ − x_b‖²_{B⁻¹} + ½ Σᵢ mᵢ (yᵢ − 𝓗(x_{τᵢ})(sᵢ))² / σᵢ²
model = vdx.StrongFourDVar(
    forward=Decay(),
    obs_op=point_op,
    prior_mean=jnp.zeros(8),
    prior_cov_op=point_prior_cov,
    obs_cov_op=None,
)
# x₀* = argmin J(x₀)
initial_analysis = model(point_batch)  # → (N,)
print(initial_analysis.shape)
```

It prints `(8,)`.

When the points cover every grid cell, the point cost equals the masked
grid cost with $R = \mathrm{diag}(\sigma^2)$. The analyses then agree, and
the linear-Gaussian agreement with `OptimalInterpolation` holds for point
batches too (`tests/test_point_observations.py`).

## Adjoint test

Every observation operator's `linearize()` must pass the adjoint test [](#eq-adjoint-test) of chapter 2. The block checks it on ten random probe pairs $(u, v)$ for each operator on this page:

```{code-block} python
:caption: Runs the adjoint test [](#eq-adjoint-test) on every observation operator of the page.
:label: code-obsops-adjoint-test
from jaxtyping import PRNGKeyArray

# Shapes: N = 6 grid cells; M observations of each operator; P = 10 probe pairs


# user
def adjoint_test(
    obs_op: vdx.ObservationOperator,
    state: Float[Array, " n"],
    *,
    n_probes: int = 10,
    key: PRNGKeyArray,
) -> bool:
    # today (vardax)
    # H′(x)
    tangent = obs_op.linearize(state)  # (M, N) op
    state_keys, obs_keys = jax.random.split(key, (2, n_probes))  # (2, P)
    for state_key, obs_key in zip(state_keys, obs_keys, strict=True):
        # u ~ 𝒩(0, I) in state space
        state_probe = jax.random.normal(state_key, state.shape)  # (N,)
        # H′ u
        tangent_probe = tangent.mv(state_probe)  # (N,) → (M,)
        # v ~ 𝒩(0, I) in observation space
        obs_probe = jax.random.normal(obs_key, tangent_probe.shape)  # (M,)
        # (H′)ᵀ v
        adjoint_probe = tangent.transpose().mv(obs_probe)  # (M,) → (N,)
        # ⟨H′ u, v⟩
        forward_inner = einx.dot("m, m ->", tangent_probe, obs_probe)  # → ()
        # ⟨u, (H′)ᵀ v⟩
        backward_inner = einx.dot("n, n ->", state_probe, adjoint_probe)  # → ()
        if not jnp.allclose(forward_inner, backward_inner, rtol=1e-4, atol=1e-5):
            return False
    return True


operators = {
    "MaskedIdentity": masked_op,
    "LinearObs": linear_op,
    "AveragingKernel": separable_ak,
    "TROPOMI": tropomi_op,
}
passed = {
    name: adjoint_test(op, state, key=key_adjoint) for name, op in operators.items()
}
print(passed)
```

It prints `{'MaskedIdentity': True, 'LinearObs': True, 'AveragingKernel': True, 'TROPOMI': True}`.

The repository runs it in `tests/test_obs_operators.py` (averaging kernel), `tests/test_interp_obs.py` and `tests/test_callable_obs.py`.

## See also

- Chapter 2: observation model and the `ObservationOperator` protocol
- Chapter 8: incremental 4DVar uses `linearize()` for tangent-linear obs
- Chapter 17: the methane example exercises multi-instrument fusion
- Design doc: [`design/api/observation_operators.md`](design/api/observation_operators.md)
- Design doc: [`design/decisions.md#d9`](design/decisions.md#d9-averaging-kernel-multi-instrument-as-first-class)
