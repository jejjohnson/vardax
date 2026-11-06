---
numbering:
  enumerator: "11.%s"
---

# Observation Operators

Chapter 2 introduced the observation model [](#eq-obs-model), its cost [](#eq-obs-cost) and the `ObservationOperator` protocol, and it defines each operator family. This chapter is the catalogue of the operators vardax ships (masked identity, linear projection, averaging kernel, multi-instrument fusion and scattered point observations) and of their implementation details.

The first four satisfy `pipekit_cycle.ObservationOperator` (the first three directly; multi-instrument fusion via its `.to_observation_operator()` adapter, see chapter 2), which `tests/test_pipekit_protocols.py` checks. These four expose `__call__(x)` and `linearize(x)`; the adjoint test at the end of this chapter checks the tangent-linear contract.

The Python blocks on this page run in order, each continuing the ones above it.

## Masked identity

The masked identity [](#eq-masked-identity) observes the state $\vect{x} \in \mathbb{R}^{D_x}$ where the mask $\vect{m} \in \{0, 1\}^{D_y}$ has $m_i = 1$, with $D_y = D_x$. The PRNG keys for the whole page are split here.

```{code-block} python
:caption: Applies the masked identity [](#eq-masked-identity) to a random state and reads its tangent H.
:label: code-obsops-masked-identity
import jax
import jax.numpy as jnp
import vardax as vdx

# Shapes: D_x = 6 grid cells; D_y = D_x

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
state = jax.random.normal(key_state, (6,))  # (D_x,)
# m ~ Bernoulli(0.5)
is_observed = jax.random.bernoulli(key_mask, 0.5, (6,))  # (D_y,) bool
mask = is_observed.astype(jnp.float32)  # (D_y,)

# today (vardax)
masked_op = vdx.MaskedIdentity()
# today (vardax)
# h(x) = m ⊙ x
predicted = masked_op(state, mask=mask)  # (D_x,) → (D_y,)
# today (vardax)
# H = I; the mask enters at call time, not in the tangent
tangent = masked_op.linearize(state)  # (D_y, D_x) op
# H m = m
tangent_mask = tangent.mv(mask)  # (D_x,) → (D_y,)
print(type(tangent).__name__, bool(jnp.allclose(tangent_mask, mask)))
```

It prints `IdentityLinearOperator True`: the tangent is the identity, so masked residuals come from the cost, which multiplies by $\vect{m}$.

The mask can be soft ($m_i \in [0, 1]$) for fuzzy observations. The operator still works, but the obs-error covariance interpretation changes (a soft mask implicitly inflates $\gmat{\Sigma}_\varepsilon$ where $m_i$ is small).

## Linear projection

The linear projection [](#eq-linear-obs) applies a stored matrix $\mat{H} \in \mathbb{R}^{D_y \times D_x}$. The block builds it from a footprint in which each of $D_y = 3$ receptors averages two neighbouring cells of the $D_x = 6$.

```{code-block} python
:caption: Builds a footprint matrix H and applies the linear projection [](#eq-linear-obs) and its tangent.
:label: code-obsops-linear-projection
import lineax as lx

# Shapes: D_x = 6 grid cells; D_y = 3 receptors

# (½ ½), one receptor over two neighbouring cells
pair_average = jnp.array([[0.5, 0.5]])  # (1, 2)
# H = I₃ ⊗ (½ ½)
footprint = jnp.kron(jnp.eye(3), pair_average)  # (D_y, D_x)
footprint_op = lx.MatrixLinearOperator(footprint)  # (D_y, D_x) op

# today (vardax)
linear_op = vdx.LinearObs(operator=footprint_op)
# today (vardax)
# h(x) = H x
predicted = linear_op(state)  # (D_x,) → (D_y,)
# today (vardax)
# H = J_h(x) = H, the stored operator itself
tangent = linear_op.linearize(state)  # (D_y, D_x) op
print(predicted.shape, tangent is linear_op.operator)
```

It prints `(3,) True`.

Use cases:

- **Lagrangian footprint**: a particle simulator produces a footprint matrix mapping the source field to receptor observations. The matrix is often sparse (only cells with non-zero footprint contribute), and lineax has no sparse operator, so wrap the sparse product in a `lineax.FunctionLinearOperator`.
- **Spectral filtering**: observe specific frequency bands. Store $\mat{H} = \mathfrak{F}^{-1} \mat{M} \mathfrak{F}$, with $\mathfrak{F}$ the Fourier transform and $\mat{M}$ the diagonal mask of the retained bands, as a composition of `lineax` operators.
- **Spatial interpolation**: bilinear or bicubic projection from the grid to observation points. The interpolation matrix is sparse.

## Averaging kernel

The averaging kernel [](#eq-averaging-kernel) is the right model for RTM-derived L2 satellite products; chapter 2 defines it and its symbols, and this section covers its construction and structure. For column CH₄ the weighting $\vect{w}$ tends to 1 in the lower troposphere, where the signal comes from the atmosphere, and to 0 in the stratosphere, where the retrieval prior $\gvect{\mu}_\text{ret}$ takes over. The tangent-linear map $\mat{H} = \mat{J}_{\hat{x}} \op{diag}(\vect{w})$ inherits the structure of $\mat{J}_{\hat{x}}$ (sparse, Kronecker, low-rank).

### Constructing from L2 metadata

L2 products typically include the averaging-kernel matrix, prior, and weighting as sidecar metadata in the netCDF / HDF5 file.

```{code-block} python
:caption: Builds the averaging kernel [](#eq-averaging-kernel) from a toy L2 sidecar and applies it to a flat column.
:label: code-obsops-averaging-kernel-l2
import xarray as xr

# Shapes: D_x = 4 retrieval levels; D_y = D_x


# user
def averaging_kernel_from_l2(dataset: xr.Dataset) -> vdx.AveragingKernel:
    # J_x̂, dense from the sidecar matrix
    kernel_values = jnp.asarray(dataset["averaging_kernel"].values)  # (D_x, D_x)
    kernel = lx.MatrixLinearOperator(kernel_values)  # (D_x, D_x) op
    # μ_ret, the retrieval prior
    prior_profile = jnp.asarray(dataset["retrieval_prior"].values)  # (D_x,)
    # w, the pressure weighting
    pressure_weights = jnp.asarray(dataset["weighting_vector"].values)  # (D_x,)
    # today (vardax)
    # h(x) = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_ret)
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
)  # (D_x, D_x)
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
column = jnp.full(4, 2.0)  # (D_x,)
# ŷ = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_ret)
predicted_column = tropomi(column)  # (D_x,) → (D_y,)
print(predicted_column)
```

It prints `[1.993     1.979     1.962     1.9390001]`: each row of $\mat{J}_{\hat{x}}$ sums to one, so the output lies between the state $\vect{x} = 2 \cdot \mathbf{1}$ and the retrieval prior $\gvect{\mu}_\text{ret} = 1.9 \cdot \mathbf{1}$, closer to the prior where $\vect{w}$ is small.

### Structured kernels

For separable kernels (e.g. independent vertical and horizontal smoothing), use Kronecker structure, $\mat{J}_{\hat{x}} = \mat{J}_\text{lev} \otimes \mat{J}_\text{col}$ with $\mat{J}_\text{lev} \in \mathbb{R}^{N_\text{lev} \times N_\text{lev}}$ and $\mat{J}_\text{col} \in \mathbb{R}^{N_\text{col} \times N_\text{col}}$, here with $\vect{w} = 0.8 \cdot \mathbf{1}$ and $\gvect{\mu}_\text{ret} = \mathbf{0}$. The state has $N_\text{lev} = 2$ levels and $N_\text{col} = 3$ columns, so $D_x = N_\text{lev} N_\text{col} = 6$.

```{code-block} python
:caption: Applies the averaging kernel [](#eq-averaging-kernel) with a Kronecker kernel J_x̂ = J_lev ⊗ J_col and checks it against the dense product.
:label: code-obsops-kronecker-kernel
import einx
import gaussx as gx

# Shapes: N_lev = 2 levels; N_col = 3 columns; D_x = D_y = N_lev N_col = 6 cells

# J_lev, vertical smoothing
vertical_kernel = jnp.array([[0.7, 0.3], [0.3, 0.7]])  # (N_lev, N_lev)
# J_col, horizontal smoothing
horizontal_kernel = jnp.array(
    [[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]]
)  # (N_col, N_col)
vertical_op = lx.MatrixLinearOperator(vertical_kernel)  # (N_lev, N_lev) op
horizontal_op = lx.MatrixLinearOperator(horizontal_kernel)  # (N_col, N_col) op
# today (gaussx)
# J_x̂ = J_lev ⊗ J_col
kronecker_kernel = gx.Kronecker(vertical_op, horizontal_op)  # (D_x, D_x) op
# today (vardax)
# h(x) = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_ret)
separable_ak = vdx.AveragingKernel(
    kernel_matrix=kronecker_kernel,
    prior_profile=jnp.zeros(6),
    pressure_weights=jnp.full(6, 0.8),
)
# today (vardax)
# vec(J_lev X J_colᵀ) without forming the (D_x, D_x) matrix
smoothed = separable_ak(state)  # (D_x,) → (D_y,)
# J_x̂ = J_lev ⊗ J_col, dense for the check
dense_kernel = jnp.kron(vertical_kernel, horizontal_kernel)  # (D_x, D_x)
# w ⊙ x with w = 0.8 and μ_ret = 0
weighted_state = 0.8 * state  # (D_x,)
# J_x̂ (w ⊙ x)
dense_smoothed = einx.dot("i j, j -> i", dense_kernel, weighted_state)  # (D_y,)
print(bool(jnp.allclose(smoothed, dense_smoothed, atol=1e-6)))
```

It prints `True`.

With Kronecker structure $\mat{J}_{\hat{x}}$ is never materialised, and the mat-vec $\mat{J}_{\hat{x}} \op{vec}(\mat{X})$, with the state laid out as $\mat{X} \in \mathbb{R}^{N_\text{lev} \times N_\text{col}}$, becomes two small mat-vecs instead of one giant one.

For low-rank kernels (information content $\le$ rank), use low-rank
operators, $\mat{J}_{\hat{x}} = \mat{U} \op{diag}(\gvect{\sigma}) \mat{V}^\top$ with $\mat{U}, \mat{V} \in \mathbb{R}^{D_x \times r}$ and singular values $\gvect{\sigma} \in \mathbb{R}^r$:

```{code-block} python
:caption: Composes a lazy low-rank kernel J_x̂ = U diag(σ) Vᵀ for the averaging kernel [](#eq-averaging-kernel).
:label: code-obsops-low-rank-kernel
# Shapes: D_x = D_y = 6 cells; r = 2 rank

# U ~ 𝒩(0, I)
left_vectors = jax.random.normal(key_left, (6, 2))  # (D_x, r)
# σ
singular_values = jnp.array([0.9, 0.4])  # (r,)
# V ~ 𝒩(0, I)
right_vectors = jax.random.normal(key_right, (6, 2))  # (D_x, r)
# Vᵀ
right_transposed = einx.id("n k -> k n", right_vectors)  # (D_x, r) → (r, D_x)
left_op = lx.MatrixLinearOperator(left_vectors)  # (D_x, r) op
singular_op = lx.DiagonalLinearOperator(singular_values)  # (r, r) op
right_op = lx.MatrixLinearOperator(right_transposed)  # (r, D_x) op
# J_x̂ = U diag(σ) Vᵀ, composed lazily
low_rank_kernel = left_op @ singular_op @ right_op  # (D_x, D_x) op
# J_x̂ x
low_rank_smoothed = low_rank_kernel.mv(state)  # (D_x,) → (D_y,)
print(low_rank_smoothed.shape)
```

It prints `(6,)`.

Both save memory and respect the underlying retrieval physics (retrievals with low DOF should have low-rank $\mat{J}_{\hat{x}}$).

### Why skipping the AK is wrong

Chapter 2 explains why treating an L2 retrieval as $\vect{y} = \vect{x} + \gvect{\varepsilon}$ instead of [](#eq-averaging-kernel) biases the posterior when $\gvect{\mu}_\text{ret} \ne \gvect{\mu}_x$. The retrieval folds in information from $\gvect{\mu}_\text{ret}$ that we then attribute to the data, so the MAP is pulled toward $\gvect{\mu}_\text{ret}$.

The bias is largest when:

- Information content is low (rank-deficient $\mat{J}_{\hat{x}}$, large $\mathbf{1} - \vect{w}$ fraction)
- $\gvect{\mu}_\text{ret}$ differs from $\gvect{\mu}_x$ (e.g. inventory prior vs. climatology)
- Long inversion windows compound the bias

The fix is to always use `AveragingKernel`. The cost is one extra mat-vec per observation, which is negligible next to the forward-model cost.

## Multi-instrument fusion

For multi-satellite inversions, each instrument gets an `InstrumentSpec` with its own operator, mask $\vect{m}_i$ and error covariance $\gmat{\Sigma}_{\varepsilon,i}$. The toy below has a coarse TROPOMI-like instrument behind an averaging kernel and a fine EMIT-like one on the grid, on $D_x = 6$ cells. `MultiInstrumentFusion` returns the per-instrument predictions, and its `cost` sums the terms of the fusion cost [](#eq-fusion-cost), here with $\alpha_i = 1$.

```{code-block} python
:caption: Fuses a TROPOMI-like and an EMIT-like instrument and sums the fusion cost [](#eq-fusion-cost) with αᵢ = 1.
:label: code-obsops-fusion
from jaxtyping import Array, Float

# Shapes: D_x = 6 grid cells; D_y,TROPOMI = 3 TROPOMI pixels; D_y,EMIT = 6 EMIT pixels

# today (vardax)
# TROPOMI: coarse pixels through an averaging kernel, h_TROPOMI(x) = J_x̂ (w ⊙ x)
tropomi_op = vdx.AveragingKernel(
    kernel_matrix=footprint_op,
    prior_profile=jnp.zeros(6),
    pressure_weights=jnp.full(6, 0.9),
)
# m_TROPOMI: the third TROPOMI pixel fails quality control
tropomi_mask = jnp.array([1.0, 1.0, 0.0])  # (D_y,TROPOMI,)
# Σ_ε,TROPOMI = 0.04 I
tropomi_obs_cov = lx.DiagonalLinearOperator(  # (D_y,TROPOMI, D_y,TROPOMI) op
    jnp.full(3, 0.04)
)
# today (vardax)
tropomi_spec = vdx.InstrumentSpec(
    obs_op=tropomi_op,
    mask=tropomi_mask,
    obs_cov_op=tropomi_obs_cov,
    instrument_id="TROPOMI",
)
# Σ_ε,EMIT = 0.01 I
emit_obs_cov = lx.DiagonalLinearOperator(  # (D_y,EMIT, D_y,EMIT) op
    jnp.full(6, 0.01)
)
# today (vardax)
# EMIT: fine pixels on the grid, h_EMIT(x) = x
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
# ŷᵢ = hᵢ(x), one entry per instrument
predictions = fusion(state)  # (D_x,) → {TROPOMI: (D_y,TROPOMI,), EMIT: (D_y,EMIT,)}
# ε_TROPOMI ~ 𝒩(0, I)
tropomi_noise = jax.random.normal(key_tropomi, (3,))  # (D_y,TROPOMI,)
# ε_EMIT ~ 𝒩(0, I)
emit_noise = jax.random.normal(key_emit, (6,))  # (D_y,EMIT,)
observations = {
    # y_TROPOMI = ŷ_TROPOMI + 0.2 ε_TROPOMI
    "TROPOMI": predictions["TROPOMI"] + 0.2 * tropomi_noise,
    # y_EMIT = ŷ_EMIT + 0.1 ε_EMIT
    "EMIT": predictions["EMIT"] + 0.1 * emit_noise,
}

# today (vardax)
# 𝒥_o = ∑ᵢ αᵢ ½ ‖mᵢ ⊙ (yᵢ − hᵢ(x))‖²_Σ_ε,i⁻¹, with αᵢ = 1 at the truth
obs_cost = fusion.cost(state, observations)  # → ()
print({name: value.shape for name, value in predictions.items()})
print(f"observation cost at the truth {float(obs_cost):.2f}")
```

It prints `{'TROPOMI': (3,), 'EMIT': (6,)}` and `observation cost at the truth 2.57`.

Because the fusion happens at the likelihood level [](#eq-fusion-cost), nothing is regridded to a common grid. Cross-instrument errors default to independent; correlated errors need a block-non-diagonal $\gmat{\Sigma}_\varepsilon$. The quality mask $\vect{m}_i$ lives in each `InstrumentSpec`, since `Batch*` has no per-pixel instrument index.

### Cross-instrument bias

Real instruments disagree systematically. TROPOMI and GHGSat differ by about 5 to 20 ppb on methane retrievals due to spectral calibration, retrieval algorithm differences, and footprint effects. Pretending they agree biases the joint analysis.

For joint inversion with bias estimation (Epic 9), introduce a per-instrument bias state $b_i$, which shifts each prediction to $h_i(\vect{x}) + b_i$ and adds a Gaussian prior cost on $b_i$, with prior standard deviation $\sigma_{b,i}$:

```{code-block} python
:caption: Adds a per-instrument bias bᵢ to the predictions of [](#eq-fusion-cost) and evaluates its Gaussian prior cost.
:label: code-obsops-biased-fusion
import equinox as eqx

# Shapes: D_x = 6 grid cells; D_y,TROPOMI = 3, D_y,EMIT = 6;
# one scalar bias bᵢ per instrument


# user
class BiasedFusion(eqx.Module):
    """Per-instrument additive bias on top of a fusion, with prior bᵢ ~ 𝒩(0, σ_b,ᵢ²)."""

    base: vdx.MultiInstrumentFusion
    bias_std: dict[str, float]

    def __call__(
        self, state: Float[Array, " n"], bias: dict[str, Float[Array, ""]]
    ) -> dict[str, Float[Array, " m"]]:
        # today (vardax)
        # hᵢ(x), one entry per instrument
        predictions = self.base(state)
        # ŷᵢ = hᵢ(x) + bᵢ
        return {name: predictions[name] + bias[name] for name in predictions}

    def bias_cost(self, bias: dict[str, Float[Array, ""]]) -> Float[Array, ""]:
        total = jnp.zeros(())
        for name, std in self.bias_std.items():
            # bᵢ / σ_b,ᵢ
            standardised = bias[name] / std  # → ()
            # 𝒥_bias = ∑ᵢ ½ bᵢ² / σ_b,ᵢ²
            total = total + 0.5 * standardised**2
        return total  # → ()


# user
biased = BiasedFusion(base=fusion, bias_std={"TROPOMI": 0.02, "EMIT": 0.01})
bias = {"TROPOMI": jnp.float32(0.01), "EMIT": jnp.float32(0.0)}
# ŷᵢ = hᵢ(x) + bᵢ
biased_predictions = biased(state, bias)  # (D_x,) → {TROPOMI: (D_y,TROPOMI,), ...}
# 𝒥_bias = ∑ᵢ ½ bᵢ² / σ_b,ᵢ²
bias_cost = biased.bias_cost(bias)  # → ()
print(f"bias cost = {float(bias_cost):.3f}")
```

It prints `bias cost = 0.125`.

The bias $b_i$ becomes a joint state variable, and the analysis solves for $(\vect{x}, \vect{b})$ together, with $\vect{b} \in \mathbb{R}^{N_\text{inst}}$, one bias per instrument. The bias prior tames the underdeterminacy. Hierarchical priors per (instrument, basin, season)
are the operational pattern.

This is not yet shipped in vardax; the design is documented for Epic 9.

## Scattered point observations

Along-track altimetry, drifters and station networks deliver observations
at scattered coordinates and times, not on the state grid. Scattering them
into a mostly-`NaN` grid would discard their positions and force one $\gmat{\Sigma}_\varepsilon$
for all times. A `PointObsBatch` carries them as they are, padded to a
fixed count $N_\text{obs}$ so shapes stay static under `jit`:

| Field | Shape | Meaning |
|---|---|---|
| `coords` | `(N_obs, D_s)` | $\vect{s}_i \in \mathbb{R}^{D_s}$, e.g. `(x,)`, `(y, x)` or `(lon, lat, t)` |
| `values` | `(N_obs,)` | $y_i$ |
| `sigma` | `(N_obs,)` | per-observation error std $\sigma_i$ |
| `valid` | `(N_obs,)` | $m_i$, `False` on padding |
| `time_index` | `(N_obs,)` or `None` | window step $\tau_i$ for gridded-state models |
| `n_times` | static `int` | states in the window, $T + 1$ for $T$ rollout steps |

The observation cost is

$$\mathcal{J}_o = \frac{1}{2} \sum_i m_i \frac{\left(y_i - h(\vect{x})(\vect{s}_i, t_i)\right)^2}{\sigma_i^2}$$ (eq-point-obs-cost)

where $h(\vect{x})(\vect{s}_i, t_i)$ is the state evaluated at the coordinates $\vect{s}_i$ and time $t_i$ of observation $i$, and the other symbols are the fields in the table.

Padding rows contribute exactly zero value **and** zero gradient.
`point_obs_cost` masks the whitened residual with `jnp.where` *before*
squaring, and first replaces the padding's coordinates, values and
$\sigma_i$ with finite placeholders. Multiplying by $m_i$ after squaring
would not be enough: $0 \cdot \text{NaN} = \text{NaN}$ in the gradient of
a padding row whose value or $\sigma_i = 0$ is garbage.

Two operators evaluate $h(\vect{x})$ at the points:

- **`GridPointObs(grid_axes)`**: for a gridded state. The state has a
  leading time axis, either a 4D-Var trajectory $(\vect{x}_0, \ldots, \vect{x}_T)$ or a
  3D-Var state of shape $(T + 1, D_x)$. Observation $i$ reads $\vect{x}_{\tau_i}$ and
  interpolates it linearly (bilinearly on a 2-D grid) at $\vect{s}_i$, clamped at
  the grid edge. This is `InterpObs`'s `order=1` stencil, computed at call
  time so that the coordinates may be traced.
- **`FieldPointObs()`**: for a state that is a callable field
  $x(\vect{s})$, such as a neural field. It returns $\hat{y}_i = x(\vect{s}_i)$ and
  passes all $D_s$ columns to the field, a time column included.

`ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`, `ReducedBasisFourDVar` and
`IncrementalFourDVar` choose their path from the batch type. A
`PointObsBatch` needs `obs_op=GridPointObs(...)`, and the per-observation
$\sigma_i$ replaces `obs_cov_op`, which must be `None` (or an identity
operator; anything else raises `ValueError` rather than being silently
ignored). 4D-Var models compare observation $i$
with the trajectory state at step $\tau_i$, so $h_t$ and $\gmat{\Sigma}_{\varepsilon,t}$
can change from step to step. `ThreeDVar` also accepts `FieldPointObs`
together with a `NeuralFieldParameterization`, which evaluates the field
at the observation coordinates without a grid. A leading batch axis on
the batch's arrays is optional.

The toy below has a decaying forward model $f(\vect{x}) = \vect{x} / 2$ on $D_x = 8$ cells, four observations spread over a window of $T = 2$ steps, which holds $T + 1 = 3$ times, and one padding row.

```{code-block} python
:caption: Minimises the strong-constraint cost [](#eq-strong4dvar-cost) with the point-observation term [](#eq-point-obs-cost) on a toy decay model.
:label: code-obsops-point-obs
# Shapes: D_x = 8 grid cells; N_obs = 5 point observations, one of them padding;
# D_s = 1 coordinate; T = 2 steps, so T + 1 = 3 times in the window

# the grid coordinates of the cells
cell_axis = jnp.arange(8.0)  # (D_x,)


# user
class Decay(eqx.Module):
    """f(x) = x / 2, a forward model with a unit step."""

    dt: float = 1.0

    def step(self, state: Float[Array, " n"], dt: float) -> Float[Array, " n"]:
        # x_t = x_{t−1} / 2
        return 0.5 * state


# x₀ ~ 𝒩(0, I)
initial_truth = jax.random.normal(key_truth, (8,))  # (D_x,)
# 2⁻ᵗ, t = 0 … T
decay_factors = 0.5 ** jnp.arange(3.0)  # (T+1,)
# x_t = f_t(x₀) = 2⁻ᵗ x₀
trajectory = einx.multiply(  # (T+1, D_x)
    "t, n -> t n", decay_factors, initial_truth
)
# sᵢ, the observation coordinates
coords = jnp.array([[0.5], [2.0], [4.25], [6.5], [0.0]])  # (N_obs, D_s)
# mᵢ, False on the padding row
valid = jnp.array([True, True, True, True, False])  # (N_obs,)
# τᵢ, the window step of each observation
time_index = jnp.array([0, 1, 1, 2, 0])  # (N_obs,)
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
# h(x_τᵢ)(sᵢ), linear interpolation in the frame τᵢ
truth_points = point_op(trajectory, point_batch)  # (T+1, D_x) → (N_obs,)
# εᵢ ~ 𝒩(0, 1)
point_noise = jax.random.normal(key_noise, (5,))  # (N_obs,)
# yᵢ = h(x_τᵢ)(sᵢ) + σ εᵢ, σ = 0.1
ssh = truth_points + 0.1 * point_noise  # (N_obs,)
point_batch = eqx.tree_at(lambda batch: batch.values, point_batch, ssh)

point_structure = jax.ShapeDtypeStruct((8,), jnp.float32)
# Σ_x = I
point_prior_cov = lx.IdentityLinearOperator(point_structure)  # (D_x, D_x) op
# today (vardax)
# 𝒥(x₀) = ½ ‖x₀ − μ_x‖²_Σ_x⁻¹ + ½ ∑ᵢ mᵢ (yᵢ − h(x_τᵢ)(sᵢ))² / σᵢ²
model = vdx.StrongFourDVar(
    forward=Decay(),
    obs_op=point_op,
    prior_mean=jnp.zeros(8),
    prior_cov_op=point_prior_cov,
    obs_cov_op=None,
)
# x₀⋆ = argmin 𝒥(x₀)
initial_analysis = model(point_batch)  # → (D_x,)
print(initial_analysis.shape)
```

It prints `(8,)`.

When the points cover every grid cell, the point cost equals the masked
grid cost with $\gmat{\Sigma}_\varepsilon = \op{diag}(\gvect{\sigma}^2)$. The analyses then agree, and
the linear-Gaussian agreement with `OptimalInterpolation` holds for point
batches too (`tests/test_point_observations.py`).

## Adjoint test

Every observation operator's `linearize()` must pass the adjoint test [](#eq-adjoint-test) of chapter 2. The block checks it on ten random probe pairs $(\vect{u}, \vect{v})$, with $\vect{u} \in \mathbb{R}^{D_x}$ and $\vect{v} \in \mathbb{R}^{D_y}$, for each operator on this page:

```{code-block} python
:caption: Runs the adjoint test [](#eq-adjoint-test) on every observation operator of the page.
:label: code-obsops-adjoint-test
from jaxtyping import PRNGKeyArray

# Shapes: D_x = 6 grid cells; D_y observations of each operator;
# N_probe = 10 probe pairs


# user
def adjoint_test(
    obs_op: vdx.ObservationOperator,
    state: Float[Array, " n"],
    *,
    n_probes: int = 10,
    key: PRNGKeyArray,
) -> bool:
    # today (vardax)
    # H = J_h(x)
    tangent = obs_op.linearize(state)  # (D_y, D_x) op
    state_keys, obs_keys = jax.random.split(key, (2, n_probes))  # (2, N_probe)
    for state_key, obs_key in zip(state_keys, obs_keys, strict=True):
        # u ~ 𝒩(0, I) in state space
        state_probe = jax.random.normal(state_key, state.shape)  # (D_x,)
        # H u
        tangent_probe = tangent.mv(state_probe)  # (D_x,) → (D_y,)
        # v ~ 𝒩(0, I) in observation space
        obs_probe = jax.random.normal(obs_key, tangent_probe.shape)  # (D_y,)
        # Hᵀ v
        adjoint_probe = tangent.transpose().mv(obs_probe)  # (D_y,) → (D_x,)
        # ⟨H u, v⟩
        forward_inner = einx.dot("m, m ->", tangent_probe, obs_probe)  # → ()
        # ⟨u, Hᵀ v⟩
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
