---
status: draft
version: 0.3.0
---

# Layer 1 — Observation Operators

Per Decision D9, averaging kernel and multi-instrument fusion are day-one
operators, not deferred features. Every class here satisfies
`pipekit_cycle.ObservationOperator` directly (Decision D8): `__call__(state)`
returns predicted observations, and `linearize(state)` returns an
`AbstractLinearOperator` representing the tangent-linear operator
$H' = \partial H / \partial x$.

---

## Protocol satisfaction

```python
from pipekit_cycle import ObservationOperator
from lineax import JacobianLinearOperator, AbstractLinearOperator

# Every vardax obs operator implements:
#
#   def __call__(self, state: Array) -> Array: ...
#   def linearize(self, state: Array) -> AbstractLinearOperator: ...
#
# The default `linearize` uses `lineax.JacobianLinearOperator` (autodiff
# Jacobian). Operators with structure may override with a structured op
# (gaussx Kronecker, LowRank, BlockDiag) for efficiency.
```

---

## `MaskedIdentity`

Default operator for any "we observe the state with gaps" use case.

```python
class MaskedIdentity(eqx.Module):
    """H(x) = mask ⊙ x."""

    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        return x * mask if mask is not None else x

    def linearize(self, x: Array) -> AbstractLinearOperator:
        return JacobianLinearOperator(self.__call__, x)
```

Use case: SSH altimetry with along-track gaps, SST with cloud masks.

---

## `LinearObs`

```python
class LinearObs(eqx.Module):
    """H(x) = H_mat @ x."""

    H_mat: AbstractLinearOperator   # gaussx / lineax operator

    def __call__(self, x: Array) -> Array:
        return self.H_mat @ x

    def linearize(self, x: Array) -> AbstractLinearOperator:
        return self.H_mat  # linear → Jacobian is the operator itself
```

Use case: any pre-computed linear projection (interpolation matrix, footprint
matrix from Lagrangian transport).

---

## `AveragingKernel`

Decision D9: first-class day-one operator for any satellite L2 product.

### Mathematical formulation

The averaging kernel maps a model state $x$ (mixing-ratio profile, surface
field) to the L2 retrieval-equivalent observation:

$$\hat{y} = A\,(h \odot x + (1 - h) \odot x_a)$$

where:

- $x \in \mathbb{R}^N$ — model state (gas profile, surface field)
- $x_a \in \mathbb{R}^N$ — retrieval prior (a priori state from L2 metadata)
- $h \in \mathbb{R}^N$ — weighting vector (often pressure-weighted; `h = 1`
  for surface-only obs)
- $A \in \mathbb{R}^{N \times N}$ — averaging kernel matrix
- $\hat{y}$ — predicted L2 observation

This operator is **mandatory** for any RTM-derived satellite product
(TROPOMI CH₄, EMIT CH₄, OCO CO₂, MOPITT CO, …). Skipping it is the most
common cause of bias in operational inversions.

### Class contract

```python
class AveragingKernel(eqx.Module):
    A: AbstractLinearOperator  # gaussx — may be Kronecker / LowRank / dense
    x_a: Float[Array, "N"]      # retrieval prior
    h: Float[Array, "N"]        # weighting vector

    def __call__(self, x: Array) -> Array:
        return self.A @ (self.h * x + (1.0 - self.h) * self.x_a)

    def linearize(self, x: Array) -> AbstractLinearOperator:
        # H'(x) = A · diag(h) — linear in x
        return self.A @ DiagonalLinearOperator(self.h)
```

`A` is stored as an `AbstractLinearOperator` so structure (e.g. Kronecker
for separable kernels) can be exploited by `gaussx.ops.solve`.

### Construction from L2 metadata

```python
def averaging_kernel_from_l2(ds: xr.Dataset, *,
                              ak_var: str = "averaging_kernel",
                              prior_var: str = "x_a",
                              weight_var: str = "h",
                              ) -> AveragingKernel:
    """Build AveragingKernel from L2 sidecar metadata."""
    A_dense = jnp.asarray(ds[ak_var].values)
    return AveragingKernel(
        A=lineax.MatrixLinearOperator(A_dense),
        x_a=jnp.asarray(ds[prior_var].values),
        h=jnp.asarray(ds[weight_var].values),
    )
```

---

## `MultiInstrumentFusion`

Decision D9: compose per-instrument operators at the likelihood level. No
pre-regridding, no shared coordinate system imposed.

### Mathematical formulation

Each instrument $i \in \{TROPOMI, EMIT, GHGSat, \ldots\}$ has its own
operator $H_i$, mask $m_i$, error covariance $R_i$. The fused observation
cost:

$$J_\text{obs}(x) = \sum_i \alpha_i \cdot \frac{1}{|\Omega_i|}\,\|m_i \odot (H_i(x) - y_i)\|^2_{R_i^{-1}}$$

Per-pixel `instrument` index on `Batch*` selects the operator. Quality
masks zero-weight unreliable pixels (no contribution to the likelihood, not
dropped).

### Class contract

```python
class MultiInstrumentFusion(eqx.Module):
    """Compose per-instrument ObservationOperators at the likelihood level."""

    registry: InstrumentRegistry
    weights: dict[str, float] | None = None  # per-instrument α_i; default uniform

    def __call__(self, x: Array, batch: Batch) -> dict[str, Array]:
        """Return per-instrument predicted observations."""
        return {
            inst_id: spec.obs_op(x)
            for inst_id, spec in self.registry.entries.items()
        }

    def linearize(self, x: Array) -> dict[str, AbstractLinearOperator]:
        """Per-instrument tangent-linear operators (typically block-diagonal across instruments)."""
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

```python
class InstrumentSpec(eqx.Module):
    """Per-instrument (A, x_a, h, mask, R) tuple."""
    obs_op: ObservationOperator              # often AveragingKernel
    mask: Float[Array, "..."]                # quality mask
    R_op: AbstractLinearOperator             # obs-err covariance
    instrument_id: str = eqx.field(static=True)


class InstrumentRegistry(eqx.Module):
    """Keyed lookup of InstrumentSpec by instrument_id."""
    entries: dict[str, InstrumentSpec]

    @classmethod
    def from_l2_dict(cls, l2_datasets: dict[str, xr.Dataset]) -> "InstrumentRegistry":
        """Build registry from per-instrument L2 datasets."""
        entries = {}
        for inst_id, ds in l2_datasets.items():
            entries[inst_id] = InstrumentSpec(
                obs_op=averaging_kernel_from_l2(ds),
                mask=jnp.asarray(ds["qa_flag"].values),
                R_op=lineax.DiagonalLinearOperator(jnp.asarray(ds["xch4_uncertainty"].values)),
                instrument_id=inst_id,
            )
        return cls(entries=entries)
```

---

## Bias-aware fusion (Epic 9, planned)

For multi-instrument joint inversions where instruments may disagree
systematically:

```python
class BiasAwareFusion(eqx.Module):
    """MultiInstrumentFusion with per-instrument bias as joint state."""

    base: MultiInstrumentFusion
    bias_prior: dict[str, GaussianPrior]   # b_i ~ N(0, σ_b²) per instrument

    def __call__(self, x: Array, bias: dict[str, Array], batch: Batch) -> dict[str, Array]:
        return {
            inst_id: self.base.registry.entries[inst_id].obs_op(x) + bias[inst_id]
            for inst_id in self.base.registry.entries
        }
```

Per-instrument bias becomes a joint state element in `IncrementalVarDA*` —
`Posterior.mean` carries both state and bias estimates.

---

## Spectral / Fourier observation operators (planned)

For frequency-domain observations (along-track altimetry spectra, spectral
SST products):

```python
class SpectralObs(eqx.Module):
    """H(x) = F^{-1} · S · F · x — selective frequency observation."""
    F: AbstractLinearOperator       # FFT operator
    S: AbstractLinearOperator       # frequency selection / weighting
    ...
```

Reserved for Epic 4 follow-up; not in the v0.3 surface.

---

## Test conformance

Every observation operator passes `tests/test_pipekit_protocols.py`:

```python
def test_obs_op_satisfies_protocol(obs_op: ObservationOperator):
    assert isinstance(obs_op, ObservationOperator)
    x = sample_state()
    y = obs_op(x)
    assert y.shape == expected_obs_shape

    H_lin = obs_op.linearize(x)
    assert isinstance(H_lin, AbstractLinearOperator)

    # Adjoint test:
    u, v = sample_random_arrays()
    forward = H_lin @ u
    backward = H_lin.T @ v
    assert jnp.allclose(jnp.dot(forward, v), jnp.dot(u, backward), atol=1e-5)
```
