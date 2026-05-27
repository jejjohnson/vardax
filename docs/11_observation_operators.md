# Observation Operators

Chapter 2 introduced the observation model — the likelihood
$p(y \mid x) = \mathcal{N}(H(x), R)$ and the `ObservationOperator`
protocol. This chapter goes deeper into the operator family vardax
ships: masked identity, linear projection, averaging kernel, and
multi-instrument fusion.

All four satisfy `pipekit_cycle.ObservationOperator` (the first three
directly; multi-instrument fusion via its `.to_observation_operator()`
adapter — chapter 2). All four expose `__call__(x)` and `linearize(x)`,
with the tangent-linear / adjoint contract enforced by the planned
`tests/test_pipekit_protocols.py` conformance suite (Epic 1; not yet
present in the v0.1.6 codebase).

## Masked identity

$$
H(x) = m \odot x, \quad m \in \{0, 1\}^n.
$$

```python
from vardax.obs_operators import MaskedIdentity

obs_op = MaskedIdentity()
y_pred = obs_op(x, mask=batch.mask)
H_lin = obs_op.linearize(x)            # diag(m), via lineax.JacobianLinearOperator
```

Use case: SSH altimetry along-track gaps, SST with cloud masks,
sparse in-situ samples on a grid.

The mask can be soft ($m_i \in [0, 1]$) for fuzzy observations — the
operator still works, but the obs-error covariance interpretation
changes (a soft mask implicitly inflates $R$ where $m$ is small).

## Linear projection

$$
H(x) = H_\text{mat} \cdot x.
$$

```python
from vardax.obs_operators import LinearObs

obs_op = LinearObs(H_mat=footprint_op)         # AbstractLinearOperator
y_pred = obs_op(x)
H_lin = obs_op.linearize(x)                    # H_mat itself
```

Use cases:

- **Lagrangian footprint** — particle simulator produces a footprint
  matrix mapping source field to receptor observation. The matrix is
  often sparse (only cells with non-zero footprint contribute) — wrap
  as a `lineax.SparseLinearOperator`.
- **Spectral filtering** — observe specific frequency bands. Store
  $H_\text{mat} = F^{-1} S F$ as a composition of `lineax`
  operators.
- **Spatial interpolation** — bilinear / bicubic projection from grid
  to observation points. The interpolation matrix is sparse.

The matrix is stored as `AbstractLinearOperator` so structure
(sparsity, low-rank, Kronecker) can be exploited without dense
materialisation.

## Averaging kernel

The averaging kernel is the right model for RTM-derived L2 satellite
products. Chapter 2 introduced it; here we go deeper into structure
and validation.

### The math, restated

$$
\hat{y} = A\, (h \odot x + (1 - h) \odot x_a).
$$

The kernel matrix $A$ smooths the model state through the retrieval's
sensitivity profile. The weighting vector $h$ tells you how much of
the L2 signal comes from the atmosphere ($h \to 1$ in the lower
troposphere for column CH₄) versus the retrieval prior ($h \to 0$ in
the stratosphere). The fallback prior $x_a$ is what the retrieval
assumes outside the observed information content.

Tangent-linear: $H'(x) = A \cdot \mathrm{diag}(h)$. The structure of
$H'$ is the structure of $A$ — sparse, Kronecker, low-rank, etc. —
inherited cleanly.

### Constructing from L2 metadata

L2 products typically include the averaging-kernel matrix, prior, and
weighting as sidecar metadata in the netCDF / HDF5 file:

```python
import xarray as xr
import lineax as lx
from vardax.obs_operators import AveragingKernel

def averaging_kernel_from_l2(ds: xr.Dataset) -> AveragingKernel:
    return AveragingKernel(
        A=lx.MatrixLinearOperator(jnp.asarray(ds.averaging_kernel.values)),
        x_a=jnp.asarray(ds.retrieval_prior.values),
        h=jnp.asarray(ds.weighting_vector.values),
    )

tropomi = averaging_kernel_from_l2(tropomi_ds)
```

### Structured kernels

For separable kernels (e.g. independent vertical and horizontal
smoothing), use Kronecker structure:

```python
import gaussx as gx

A_kron = gx.KroneckerLinearOperator([
    gx.MatrixLinearOperator(vertical_kernel),
    gx.MatrixLinearOperator(horizontal_kernel),
])
ak = AveragingKernel(A=A_kron, x_a=xa, h=h)
```

Kronecker structure means $A$ never materialised; mat-vec
$\mathrm{vec}(A \cdot X)$ becomes two small mat-vecs instead of one
giant one.

For low-rank kernels (information content $\le$ rank), use low-rank
operators:

```python
A_lr = gx.LowRankLinearOperator(U=u, S=s, V=v)
```

Both saves memory and respects the underlying retrieval physics
(retrievals with low DOF should have low-rank $A$).

### Why skipping the AK is wrong

Treating an L2 retrieval as $y = x + \varepsilon$ (instead of
$y = A(h \cdot x + (1-h) x_a) + \varepsilon$) gives systematically
biased posteriors when $x_a \ne x_b$. The retrieval folds in
information from $x_a$ that we then attribute to the data. The MAP is
pulled toward $x_a$.

The bias is largest when:

- Information content is low (rank-deficient $A$, large $1-h$ fraction)
- $x_a$ differs from $x_b$ (e.g. inventory prior vs. climatology)
- Long inversion windows compound the bias

The fix is to always use `AveragingKernel`. The cost is one extra
matrix mat-vec per observation — negligible compared to forward-model
cost.

## Multi-instrument fusion

For multi-satellite inversions:

```python
from vardax.obs_operators import (
    MultiInstrumentFusion, InstrumentRegistry, InstrumentSpec,
)

tropomi_spec = InstrumentSpec(
    obs_op=AveragingKernel(A=A_t, x_a=xa_t, h=h_t),
    mask=tropomi_qa,
    R_op=lx.DiagonalLinearOperator(tropomi_uncertainty),
    instrument_id="TROPOMI",
)
emit_spec = InstrumentSpec(
    obs_op=AveragingKernel(A=A_e, x_a=xa_e, h=h_e),
    mask=emit_qa,
    R_op=lx.DiagonalLinearOperator(emit_uncertainty),
    instrument_id="EMIT",
)

fusion = MultiInstrumentFusion(
    registry=InstrumentRegistry(entries={
        "TROPOMI": tropomi_spec,
        "EMIT": emit_spec,
    }),
)

# Returns dict[str, Array] — per-instrument predicted observations
predictions = fusion(x, batch)
```

The fusion happens at the **likelihood level**: each instrument
contributes a term to $J_\text{obs}$, with its own $H_i$, $m_i$, $R_i$.
No pre-regridding to a common grid. No assumption of independent vs.
correlated cross-instrument errors (the default is independent;
correlated errors require block-non-diagonal $R$).

Per-pixel `instrument` index on `Batch*` selects the operator. Quality
masks zero-weight bad pixels — they contribute zero log-likelihood but
remain in the audit count.

### Cross-instrument bias

Real instruments disagree systematically. TROPOMI and GHGSat differ by
$\sim 5$–$20$ ppb on methane retrievals due to spectral calibration,
retrieval algorithm differences, and footprint effects. Pretending
they agree biases the joint analysis.

For joint inversion with bias estimation (Epic 9), introduce a
per-instrument bias state $b_i$:

```python
class BiasedFusion(eqx.Module):
    base: MultiInstrumentFusion
    bias_prior: dict[str, GaussianPrior]    # b_i ~ N(0, σ_b²)

    def __call__(self, x, bias_dict, batch):
        return {
            inst_id: self.base.registry.entries[inst_id].obs_op(x) + bias_dict[inst_id]
            for inst_id in self.base.registry.entries
        }
```

`b_i` becomes a joint state variable; the analysis solves for $(x,
\boldsymbol{b})$ together. The bias prior tames the
underdeterminacy. Hierarchical priors per (instrument, basin, season)
are the operational pattern.

This is not yet shipped in vardax; the design is documented for Epic 9.

## Adjoint test

Every observation operator's `linearize()` must pass the adjoint test:

$$
\langle H' u, v \rangle = \langle u, (H')^\top v \rangle, \quad \forall u, v.
$$

Implementation:

```python
def adjoint_test(obs_op, x, key, n_probes=10):
    H_lin = obs_op.linearize(x)
    for _ in range(n_probes):
        u = jax.random.normal(key, x.shape)
        v = jax.random.normal(key, H_lin(u).shape)
        forward_inner = jnp.dot(H_lin @ u, v)
        backward_inner = jnp.dot(u, H_lin.T @ v)
        assert jnp.allclose(forward_inner, backward_inner, atol=1e-5)
```

The test is in `tests/test_pipekit_protocols.py` and runs against every
operator. Failure means the gradient computation through that operator
is wrong; any 4DVar built on it will take wrong steps.

## See also

- Chapter 2 — observation model + ObservationOperator protocol
- Chapter 8 — incremental 4DVar uses `linearize()` for tangent-linear obs
- Chapter 17 — methane example exercises multi-instrument fusion
- Design doc: [`design/api/observation_operators.md`](design/api/observation_operators.md)
- Design doc: [`design/decisions.md#d9`](design/decisions.md#d9-averaging-kernel-multi-instrument-as-first-class)
