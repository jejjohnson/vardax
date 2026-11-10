---
status: draft
version: 0.4.0
---

# Layer 1 — Observation Operators

Per Decision D9, averaging kernel and multi-instrument fusion are day-one
operators, not deferred features. Every class here satisfies
`pipekit_cycle.ObservationOperator` directly (Decision D8): `__call__(state)`
returns predicted observations, and `linearize(state)` returns an
`AbstractLinearOperator` representing the tangent-linear operator
$\mat{H} = \partial h / \partial \vect{x}$.

---

## Protocol satisfaction

```{code-block} python
:caption: Declares the observation-operator protocol, the map h of [](#eq-obs-model) plus its tangent-linear H of [](#eq-tlm-obs), which every vardax operator satisfies.
:label: code-design-api-obsop-protocol
import lineax as lx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N = state size; M = observation size

# today (pipekit_cycle)
ObservationOperator = pc.ObservationOperator

# Every vardax obs operator implements:
#
#   def __call__(self, state: Float[Array, " N"]) -> Float[Array, " M"]: ...
#   def linearize(self, state: Float[Array, " N"]) -> lx.AbstractLinearOperator: ...
#
# The default `linearize` uses `lineax.JacobianLinearOperator` (autodiff
# Jacobian H = ∂h/∂x). Operators with structure may override with a
# structured op (gaussx Kronecker, LowRank, BlockDiag) for efficiency.
```

---

## `MaskedIdentity`

Default operator for any "we observe the state with gaps" use case.

```{code-block} python
:caption: Declares `MaskedIdentity`, the masked-identity operator of [](#eq-masked-identity).
:label: code-design-api-obsop-masked-identity
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size


# today (vardax)
class MaskedIdentity(eqx.Module):
    """h(x) = m ⊙ x."""

    def __call__(
        self, x: Float[Array, " N"], mask: Float[Array, " N"] | None = None
    ) -> Float[Array, " N"]:
        if mask is None:
            return x
        # h(x) = m ⊙ x
        return x * mask

    def linearize(self, x: Float[Array, " N"]) -> lx.AbstractLinearOperator:
        # today (lineax)
        # H = ∂h/∂x by autodiff
        return lx.JacobianLinearOperator(self.__call__, x)
```

Use case: SSH altimetry with along-track gaps, SST with cloud masks.

---

## `LinearObs`

```{code-block} python
:caption: Declares `LinearObs`, the linear operator h(x) = H x of [](#eq-linear-obs), whose tangent-linear is the operator itself.
:label: code-design-api-obsop-linear
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size; M = observation size


# today (vardax)
class LinearObs(eqx.Module):
    """h(x) = operator @ x."""

    # gaussx / lineax operator
    operator: lx.AbstractLinearOperator

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " M"]:
        # today (lineax)
        # h(x) = H x
        return self.operator.mv(x)

    def linearize(self, x: Float[Array, " N"]) -> lx.AbstractLinearOperator:
        # linear → Jacobian is the operator itself: H = H
        return self.operator
```

Use case: any pre-computed linear projection (interpolation matrix, footprint
matrix from Lagrangian transport).

---

## `AveragingKernel`

Decision D9: first-class day-one operator for any satellite L2 product.

### Mathematical formulation

The averaging kernel maps a model state $\vect{x}$ (mixing-ratio profile, surface
field) to the L2 retrieval-equivalent observation:

$$h(\vect{x}) = \mat{J}_{\hat{x}} (\vect{w} \odot \vect{x} + (1 - \vect{w}) \odot \gvect{\mu}_\text{ret})$$

where:

- $\vect{x} \in \mathbb{R}^{D_x}$ — model state (gas profile, surface field)
- $\gvect{\mu}_\text{ret} \in \mathbb{R}^{D_x}$ — retrieval prior (a priori state from L2 metadata)
- $\vect{w} \in \mathbb{R}^{D_x}$ — weighting vector (often pressure-weighted; `w = 1`
  for surface-only obs)
- $\mat{J}_{\hat{x}} \in \mathbb{R}^{D_x \times D_x}$ — averaging kernel matrix
- $h(\vect{x}) \in \mathbb{R}^{D_y}$ — predicted L2 observation

This operator is **mandatory** for any RTM-derived satellite product
(TROPOMI CH₄, EMIT CH₄, OCO CO₂, MOPITT CO, …). Skipping it is the most
common cause of bias in operational inversions.

### Class contract

```{code-block} python
:caption: Declares `AveragingKernel`, the L2 retrieval operator of [](#eq-averaging-kernel), with tangent-linear A diag(h).
:label: code-design-api-obsop-averaging-kernel
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

# Shapes: N = state size (profile levels or surface cells)


# today (vardax)
class AveragingKernel(eqx.Module):
    # gaussx: may be Kronecker / LowRank / dense
    kernel_matrix: lx.AbstractLinearOperator
    # retrieval prior μ_ret
    prior_profile: Float[Array, " N"]
    # weighting vector w
    pressure_weights: Float[Array, " N"]

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        weights = self.pressure_weights
        # w ⊙ x + (1 − w) ⊙ μ_ret
        blended = weights * x + (1.0 - weights) * self.prior_profile  # (N,)
        # today (lineax)
        # h(x) = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_ret)
        return self.kernel_matrix.mv(blended)

    def linearize(self, x: Float[Array, " N"]) -> lx.AbstractLinearOperator:
        # today (lineax)
        # diag(h)
        weight_op = lx.DiagonalLinearOperator(self.pressure_weights)  # (N, N) op
        # h(x) = J_x̂ diag(w), linear in x
        return self.kernel_matrix @ weight_op
```

`kernel_matrix` is stored as an `AbstractLinearOperator` so structure (e.g. Kronecker
for separable kernels) can be exploited by `gaussx.ops.solve`.

### Construction from L2 metadata

```{code-block} python
:caption: Builds the `AveragingKernel` of [](#eq-averaging-kernel) from the J_x̂, μ_ret and w variables of an L2 dataset.
:label: code-design-api-obsop-ak-from-l2
import jax.numpy as jnp
import lineax as lx
import vardax as vdx
import xarray as xr

# Shapes: N = state size


# today* (vardax._src.obs_operators): ships with dense-array arguments, not a Dataset
def averaging_kernel_from_l2(
    ds: xr.Dataset,
    *,
    ak_var: str = "averaging_kernel",
    prior_var: str = "x_a",
    weight_var: str = "h",
) -> vdx.AveragingKernel:
    """Build AveragingKernel from L2 sidecar metadata."""
    # A
    kernel_dense = jnp.asarray(ds[ak_var].values)  # (N, N)
    # x_a
    prior_profile = jnp.asarray(ds[prior_var].values)  # (N,)
    # h
    pressure_weights = jnp.asarray(ds[weight_var].values)  # (N,)
    # today (vardax)
    return vdx.AveragingKernel(
        # today (lineax)
        kernel_matrix=lx.MatrixLinearOperator(kernel_dense),
        prior_profile=prior_profile,
        pressure_weights=pressure_weights,
    )
```

---

## `MultiInstrumentFusion`

Decision D9: compose per-instrument operators at the likelihood level. No
pre-regridding, no shared coordinate system imposed.

### Mathematical formulation

Each instrument $i \in \{TROPOMI, EMIT, GHGSat, \ldots\}$ has its own
operator $h_i$, mask $\vect{m}_i$, error covariance $\gmat{\Sigma}_{\varepsilon,i}$. The fused observation
cost:

$$\mathcal{J}_o(\vect{x}) = \sum_i \alpha_i \frac{1}{\vert \Omega_i \vert} \Vert \vect{m}_i \odot (h_i(\vect{x}) - \vect{y}_i) \Vert^2_{\gmat{\Sigma}_{\varepsilon,i}^{-1}}$$

Per-pixel `instrument` index on `Batch*` selects the operator. Quality
masks zero-weight unreliable pixels (no contribution to the likelihood, not
dropped).

### Class contract

```{code-block} python
:caption: Declares `MultiInstrumentFusion`, which returns the per-instrument predictions hᵢ(x) that the fused cost [](#eq-fusion-cost) combines.
:label: code-design-api-obsop-fusion
import equinox as eqx
import lineax as lx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size; Mᵢ = observation size of instrument i

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# today* (vardax): shipped __call__ takes only x, no batch
class MultiInstrumentFusion(eqx.Module):
    """Compose per-instrument ObservationOperators at the likelihood level."""

    # today (vardax)
    registry: vdx.InstrumentRegistry
    # per-instrument αᵢ; default uniform
    weights: dict[str, float] | None = None

    def __call__(
        self, x: Float[Array, " N"], batch: Batch
    ) -> dict[str, Float[Array, " Mi"]]:
        """Return per-instrument predicted observations."""
        # {i: hᵢ(x)}
        return {
            inst_id: spec.obs_op(x) for inst_id, spec in self.registry.entries.items()
        }

    def linearize(self, x: Float[Array, " N"]) -> dict[str, lx.AbstractLinearOperator]:
        """Per-instrument tangent-linear operators.

        Typically block-diagonal across instruments.
        """
        # {i: Hᵢ(x)}
        return {
            inst_id: spec.obs_op.linearize(x)
            for inst_id, spec in self.registry.entries.items()
        }
```

Note: `MultiInstrumentFusion.__call__` returns a `dict[str, Array]`, not
a single `Array` — the cost function consumes the dict and combines at the
likelihood level. This is a slight departure from the strict
`ObservationOperator.__call__(state) → obs` signature, so vardax exposes
a `to_observation_operator()` adapter that flattens to a single output for
strict-protocol contexts.

### `InstrumentRegistry`

```{code-block} python
:caption: Declares the per-instrument (hᵢ, mᵢ, Σ_ε,ᵢ) bundle and the registry that feed the fused cost [](#eq-fusion-cost).
:label: code-design-api-obsop-registry
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import pipekit_cycle as pc
import xarray as xr
from jaxtyping import Array, Float

# Shapes: Mᵢ = observation size of instrument i


# today (vardax)
class InstrumentSpec(eqx.Module):
    """Per-instrument (J_x̂, μ_ret, w, m, Σ_ε) tuple."""

    # today (pipekit_cycle)
    # hᵢ, often AveragingKernel
    obs_op: pc.ObservationOperator
    # quality mask mᵢ
    mask: Float[Array, " Mi"]
    # obs-err covariance Σ_ε,ᵢ
    obs_cov_op: lx.AbstractLinearOperator
    instrument_id: str = eqx.field(static=True)


# today (vardax)
class InstrumentRegistry(eqx.Module):
    """Keyed lookup of InstrumentSpec by instrument_id."""

    entries: dict[str, InstrumentSpec]

    # proposed (D9)
    @classmethod
    def from_l2_dict(cls, l2_datasets: dict[str, xr.Dataset]) -> "InstrumentRegistry":
        """Build registry from per-instrument L2 datasets."""
        entries = {}
        for inst_id, ds in l2_datasets.items():
            # mᵢ
            quality_mask = jnp.asarray(ds["qa_flag"].values)  # (Mi,)
            # diag(Σ_ε,ᵢ)
            uncertainty = jnp.asarray(ds["xch4_uncertainty"].values)  # (Mi,)
            entries[inst_id] = InstrumentSpec(
                # today* (vardax._src.obs_operators): Dataset form sketched above
                obs_op=averaging_kernel_from_l2(ds),
                mask=quality_mask,
                # today (lineax)
                # Σ_ε,ᵢ = diag(σᵢ)
                obs_cov_op=lx.DiagonalLinearOperator(uncertainty),
                instrument_id=inst_id,
            )
        return cls(entries=entries)
```

---

## Bias-aware fusion (Epic 9, planned)

For multi-instrument joint inversions where instruments may disagree
systematically:

```{code-block} python
:caption: Declares the planned bias-aware fusion, which adds a per-instrument bias bᵢ to each prediction hᵢ(x) in [](#eq-fusion-cost).
:label: code-design-api-obsop-bias-aware
import equinox as eqx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: N = state size; Mᵢ = observation size of instrument i

# today (vardax)
Batch = vdx.Batch1D | vdx.Batch2D | vdx.Batch2DMultivar


# proposed (Epic 9)
class BiasAwareFusion(eqx.Module):
    """MultiInstrumentFusion with per-instrument bias as joint state."""

    # today (vardax)
    base: vdx.MultiInstrumentFusion
    # proposed (Epic 9)
    # bᵢ ~ 𝒩(0, σ_b²) per instrument
    bias_prior: dict[str, GaussianPrior]

    def __call__(
        self, x: Float[Array, " N"], bias: dict[str, Array], batch: Batch
    ) -> dict[str, Float[Array, " Mi"]]:
        # {i: hᵢ(x) + bᵢ}
        return {
            inst_id: self.base.registry.entries[inst_id].obs_op(x) + bias[inst_id]
            for inst_id in self.base.registry.entries
        }
```

Per-instrument bias becomes a joint state element in `IncrementalFourDVar` —
`Posterior.mean` carries both state and bias estimates.

---

## Spectral / Fourier observation operators (planned)

For frequency-domain observations (along-track altimetry spectra, spectral
SST products):

```{code-block} python
:caption: Declares the planned spectral operator h(x) = F⁻¹ S F x, a linear operator in the sense of [](#eq-linear-obs).
:label: code-design-api-obsop-spectral
import equinox as eqx
import lineax as lx

# Shapes: N = state size


# proposed (Epic 4)
class SpectralObs(eqx.Module):
    """h(x) = F⁻¹ · S · F · x, selective frequency observation."""

    # FFT operator F
    F: lx.AbstractLinearOperator
    # frequency selection / weighting S
    S: lx.AbstractLinearOperator
    ...
```

Reserved for Epic 4 follow-up; not in the v0.3 surface.

---

## Test conformance

Every observation operator passes `tests/test_pipekit_protocols.py`:

```{code-block} python
:caption: Tests that an operator satisfies the protocol and passes the adjoint test [](#eq-adjoint-test).
:label: code-design-api-obsop-conformance-test
import einx
import jax.numpy as jnp
import lineax as lx
import pipekit_cycle as pc

# Shapes: N = state size; M = observation size


# user
def test_obs_op_satisfies_protocol(obs_op: pc.ObservationOperator) -> None:
    # today (pipekit_cycle)
    assert isinstance(obs_op, pc.ObservationOperator)
    # user (fixture, not shown)
    state = sample_state()  # → (N,)
    # y = h(x)
    predicted = obs_op(state)  # (N,) → (M,)
    assert predicted.shape == expected_obs_shape

    # H at x
    tangent_linear = obs_op.linearize(state)  # (M, N) op
    assert isinstance(tangent_linear, lx.AbstractLinearOperator)

    # Adjoint test:
    # user (fixture, not shown)
    u, v = sample_random_arrays()  # → (N,), (M,)
    # today (lineax)
    # Hu
    forward = tangent_linear.mv(u)  # (N,) → (M,)
    # today (lineax)
    # Hᵀv
    backward = tangent_linear.transpose().mv(v)  # (M,) → (N,)
    # ⟨Hu, v⟩
    forward_inner = einx.dot("m, m ->", forward, v)  # → ()
    # ⟨u, Hᵀv⟩
    backward_inner = einx.dot("n, n ->", u, backward)  # → ()
    # ⟨Hu, v⟩ = ⟨u, Hᵀv⟩
    assert jnp.allclose(forward_inner, backward_inner, atol=1e-5)
```
