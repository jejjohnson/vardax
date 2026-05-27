# Observation Operators

The observation operator $H$ maps a model state $\mathbf{x}$ to predicted
observations $\hat{\mathbf{y}} = H(\mathbf{x})$. The variational cost

$$J_\text{obs}(\mathbf{x}) = \|H(\mathbf{x}) - \mathbf{y}\|^2_{\mathbf{R}^{-1}}$$

is meaningful only after $H$ matches the structure of the actual observation
process.

vardax exposes three operator families: **masked identity**, **averaging
kernel**, and **multi-instrument fusion**. All satisfy
`pipekit_cycle.ObservationOperator` — they implement `__call__(x) → ŷ` and
`linearize(x) → AbstractLinearOperator`.

## Masked Identity

The simplest case — observations are samples of the state at known
locations:

$$H(\mathbf{x}) = \mathbf{m} \odot \mathbf{x}$$

with mask $\mathbf{m} \in \{0, 1\}^N$. Tangent linear:
$H'(\mathbf{x}) = \mathrm{diag}(\mathbf{m})$.

Use case: SSH altimetry along-track, SST with cloud masks, sparse in-situ.

```python
from vardax.obs_operators import MaskedIdentity

obs_op = MaskedIdentity()
y_pred = obs_op(x, mask=batch.mask)
H_lin = obs_op.linearize(x)         # JacobianLinearOperator via autodiff
```

## Averaging Kernel

For RTM-derived L2 satellite products (TROPOMI CH₄, EMIT CH₄, OCO CO₂,
MOPITT CO, …), the L2 retrieval is not a direct measurement of the
mixing-ratio profile $\mathbf{x}$ — it's a smoothed projection through an
averaging kernel:

$$\hat{\mathbf{y}} = \mathbf{A}\,\big(\mathbf{h} \odot \mathbf{x} + (\mathbf{1} - \mathbf{h}) \odot \mathbf{x}_a\big)$$

where:

| Symbol | Description |
|---|---|
| $\mathbf{x} \in \mathbb{R}^N$ | Model state (profile, surface field) |
| $\mathbf{x}_a \in \mathbb{R}^N$ | Retrieval prior from L2 metadata |
| $\mathbf{h} \in \mathbb{R}^N$ | Weighting vector (often pressure-weighted) |
| $\mathbf{A} \in \mathbb{R}^{N \times N}$ | Averaging kernel matrix |

Tangent linear: $H'(\mathbf{x}) = \mathbf{A} \cdot \mathrm{diag}(\mathbf{h})$.

**Skipping the averaging kernel is the most common cause of bias in
operational satellite inversions** — vardax exposes it as a first-class
operator (Decision D9).

```python
from vardax.obs_operators import AveragingKernel

ak = AveragingKernel(A=A_op, x_a=retrieval_prior, h=weighting)
y_pred = ak(x)
H_lin = ak.linearize(x)             # A @ diag(h) — structured op
```

### Limiting cases

- $\mathbf{h} = \mathbf{1}$, $\mathbf{A} = \mathbf{I}$ ⇒ identity (no smoothing).
- $\mathbf{A} = 0$, $\mathbf{h} = 0$ ⇒ pure prior (no information from the obs).
- Pressure-weighted column average: $\mathbf{A}$ has one non-zero row;
  $\hat{y}$ is a scalar.

## Multi-Instrument Fusion

Operational methane inversion combines multiple instruments with different
spatial / spectral characteristics. Per-instrument observation operators
are composed at the **likelihood level** — no pre-regridding, no shared
coordinate system imposed:

$$J_\text{obs}(\mathbf{x}) = \sum_{i \in \mathcal{I}} \alpha_i \cdot \frac{1}{|\Omega_i|}\,\|\mathbf{m}_i \odot (H_i(\mathbf{x}) - \mathbf{y}_i)\|^2_{\mathbf{R}_i^{-1}}$$

with per-instrument $(\mathbf{A}_i, \mathbf{x}_{a,i}, \mathbf{h}_i,
\mathbf{m}_i, \mathbf{R}_i)$ and instrument-specific weight $\alpha_i$
(default uniform).

Per-pixel `instrument` index on `Batch*` selects the operator. Quality
masks zero-weight unreliable pixels — they contribute zero log-likelihood
rather than being dropped (allows audit of effective per-instrument
observation count).

```python
from vardax.obs_operators import (
    AveragingKernel, MultiInstrumentFusion, InstrumentRegistry, InstrumentSpec,
)

fusion = MultiInstrumentFusion(
    registry=InstrumentRegistry(entries={
        "TROPOMI": InstrumentSpec(
            obs_op=AveragingKernel(A=A_t, x_a=xa_t, h=h_t),
            mask=tropomi_qa, R_op=lx.DiagonalLinearOperator(tropomi_unc),
            instrument_id="TROPOMI",
        ),
        "EMIT": InstrumentSpec(...),
        "GHGSat": InstrumentSpec(...),
    }),
)

# Per-instrument predicted obs
predictions = fusion(x, batch)        # dict[str, Array]
```

### Per-instrument bias (planned)

For joint inversion across systematically-disagreeing instruments, bias
becomes a state element:

$$\hat{\mathbf{y}}_i = H_i(\mathbf{x}) + b_i, \quad b_i \sim \mathcal{N}(0, \sigma_b^2)$$

with hierarchical priors per (instrument, basin, season). Reserved for
Epic 9 (multi-instrument bias estimation).

## Tangent-linear / adjoint contract

For incremental 4DVar (chapter 13), every observation operator must expose

```python
H_lin = obs_op.linearize(x)           # returns AbstractLinearOperator
y = H_lin @ dx                        # forward TLM
adj = H_lin.T @ residual              # adjoint
```

The adjoint test is part of `test_pipekit_protocols.py`:

$$\langle H' \mathbf{u}, \mathbf{v}\rangle = \langle \mathbf{u}, (H')^\top \mathbf{v}\rangle \;\;\forall\, \mathbf{u}, \mathbf{v}$$

up to numerical tolerance.

## See also

- Chapter 13: Incremental 4DVar — uses `linearize()` for tangent-linear obs
- Chapter 14: Posterior covariance — uses `linearize()` for Hessian
  assembly
- Design doc: [`design/api/observation_operators.md`](design/api/observation_operators.md)
  for the full class hierarchy
- Decision [D9](design/decisions.md#d9-averaging-kernel--multi-instrument-as-first-class)
  — averaging kernel + multi-instrument as first-class
