# Observation Model

The observation operator $H : \mathbb{R}^n \to \mathbb{R}^m$ maps the
state $x$ to predicted observations $\hat{y} = H(x)$. Real
instruments rarely measure the state directly — they smooth, average,
mask, or project. Getting $H$ right is what separates an inversion
that recovers truth from one that returns a biased fantasy.

## The likelihood model

vardax assumes Gaussian observation errors:

$$
y = H(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, R),
$$

so $p(y \mid x) = \mathcal{N}(y; H(x), R)$. The observation cost in the
variational problem is

$$
J_\text{obs}(x) = \tfrac{1}{2} \|y - H(x)\|^2_{R^{-1}}
                = \tfrac{1}{2} (y - H(x))^\top R^{-1} (y - H(x)).
$$

Non-Gaussian likelihoods (Student-$t$, censored, count data) require
either a non-Gaussian extension of $J_\text{obs}$ or amortized
inference with a flow / score-based head (chapter 10).

## The `ObservationOperator` protocol

Every vardax observation operator satisfies
`pipekit_cycle.ObservationOperator`:

```python
def __call__(self, x: Array, ...) -> Array:
    """Forward: H(x)."""

def linearize(self, x: Array) -> AbstractLinearOperator:
    """Tangent-linear operator H'(x) for use in 4DVar inner loops."""
```

The `linearize()` method returns a `lineax.AbstractLinearOperator`
(typically `JacobianLinearOperator` via autodiff, but structured
overrides are common). The transpose / adjoint comes for free from
`lineax`.

This contract is the same regardless of whether the operator is a
masked identity, an averaging kernel, or a multi-instrument fusion.

## Four operator families

### Masked identity

The simplest case — we observe the state where the mask is non-zero:

$$
H(x) = m \odot x, \quad m \in \{0, 1\}^n.
$$

Tangent-linear: $H'(x) = \mathrm{diag}(m)$. Use case: SSH altimetry
along-track gaps, SST with cloud masks, in-situ samples on a grid.

```python
from vardax.obs_operators import MaskedIdentity
obs_op = MaskedIdentity()
```

### Linear projection

Pre-computed projection from state to observation space:

$$
H(x) = H_\text{mat} \cdot x.
$$

Use case: Lagrangian footprint matrix from a particle simulator
(observation at receptor = footprint-weighted source field), spectral
filtering, spatial interpolation. The matrix is stored as an
`AbstractLinearOperator` so structure (sparse, low-rank, Kronecker) can
be exploited.

```python
from vardax.obs_operators import LinearObs
obs_op = LinearObs(H_mat=footprint_op)
```

### Averaging kernel — RTM L2 products

For satellite L2 products derived from optimal-estimation retrievals
(TROPOMI CH₄, EMIT CH₄, OCO CO₂, MOPITT CO, …), the L2 retrieval is
*not* a direct sample of the mixing-ratio profile $x$. The retrieval
applies a smoothing matrix $A$ (the averaging kernel) and falls back to
a retrieval prior $x_a$ where the signal is weak:

$$
\hat{y} = A\,(h \odot x + (1 - h) \odot x_a).
$$

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

```python
from vardax.obs_operators import AveragingKernel
ak = AveragingKernel(A=A_op, x_a=retrieval_prior, h=weighting_vector)
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

$$
J_\text{obs}(x) = \sum_{i \in \mathcal{I}} \alpha_i \cdot
                  \tfrac{1}{2} \|m_i \odot (y_i - H_i(x))\|^2_{R_i^{-1}}.
$$

The per-instrument weight $\alpha_i$ defaults to uniform; tuning is for
hierarchical bias-correction (Epic 9). The per-pixel `instrument` index
on `Batch*` selects the operator. Quality masks zero-weight unreliable
pixels — they contribute zero log-likelihood, not dropped (this lets us
audit effective per-instrument observation count).

```python
from vardax.obs_operators import (
    MultiInstrumentFusion, InstrumentRegistry, InstrumentSpec,
)

fusion = MultiInstrumentFusion(
    registry=InstrumentRegistry(entries={
        "TROPOMI": InstrumentSpec(
            obs_op=AveragingKernel(A=A_t, x_a=xa_t, h=h_t),
            mask=tropomi_qa, R_op=tropomi_R, instrument_id="TROPOMI",
        ),
        "EMIT": InstrumentSpec(...),
        "GHGSat": InstrumentSpec(...),
    }),
)
```

`MultiInstrumentFusion.__call__` returns `dict[str, Array]` (per-instrument
predicted observations); the cost function consumes the dict. For
strict `pipekit_cycle.ObservationOperator` contexts, the
`.to_observation_operator()` adapter flattens to a single concatenated
output with a block-diagonal linear operator.

## Tangent-linear and adjoint contract

For incremental 4DVar (chapter 8), every observation operator must
expose

```python
H_lin = obs_op.linearize(x)            # AbstractLinearOperator
y_inc = H_lin @ dx                     # forward TLM
adj   = H_lin.T @ residual             # adjoint
```

The adjoint test is part of `test_pipekit_protocols.py`:

$$
\langle H' u, v \rangle = \langle u, (H')^\top v \rangle
\quad\forall\, u, v
$$

up to floating-point tolerance. Failing this test means the gradient
flowing back through $H$ is wrong, and any downstream optimiser will
take wrong steps.

## Observation-error covariance $R$

$R$ is supplied as a `lineax.AbstractLinearOperator`. Common
parameterisations:

- **Diagonal** — `lineax.DiagonalLinearOperator(variances)`. Default
  for heteroscedastic retrievals (per-pixel σ from L2 metadata).
- **Block-diagonal** — `lineax.BlockDiagonalLinearOperator([R_i for i in instruments])`.
  Per-instrument diagonals when fusing.
- **Structured** — `gaussx.MaternLinearOperator(...)` if obs errors are
  spatially correlated (rare but matters for high-resolution imaging
  spectrometers).

The choice flows into `J_\text{obs}$ via $R^{-1}$, computed lazily —
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
