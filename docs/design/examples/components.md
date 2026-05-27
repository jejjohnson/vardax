---
status: draft
version: 0.3.0
---

# Layer 1 — Component Examples

Implementing protocols and composing `eqx.Module` operators.

---

## Implementing `Prior` — Autoencoder

```python
import equinox as eqx
from vardax.protocols import Prior  # runtime-checkable Protocol

class ConvAEPrior(eqx.Module):
    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    def __call__(self, x):
        """φ(x) → x_prior: encode then decode."""
        return self.decoder(self.encoder(x))

# Structural conformance — no inheritance needed
assert isinstance(ConvAEPrior(enc, dec), Prior)
```

---

## Implementing `ObservationOperator` — Masked identity

Per Decision D8, vardax obs operators satisfy `pipekit_cycle.ObservationOperator`
directly. Implement both `__call__` and `linearize`:

```python
from pipekit_cycle import ObservationOperator
from lineax import JacobianLinearOperator

class MaskedIdentity(eqx.Module):
    """H(x) = mask ⊙ x — observe only at mask locations."""

    def __call__(self, x, mask=None):
        return x * mask if mask is not None else x

    def linearize(self, x):
        return JacobianLinearOperator(self.__call__, x)

assert isinstance(MaskedIdentity(), ObservationOperator)
```

---

## Implementing `ObservationOperator` — Averaging kernel (Decision D9)

```python
import lineax as lx
import gaussx as gx
from vardax.obs_operators import AveragingKernel

# Built-in AveragingKernel implements the full pattern
ak = AveragingKernel(
    A=lx.MatrixLinearOperator(A_matrix),       # or gaussx structured op
    x_a=retrieval_prior,
    h=weighting_vector,
)

# ŷ = A · (h ⊙ x + (1-h) ⊙ x_a)
y_pred = ak(x)

# Tangent-linear operator for incremental 4DVar
H_lin = ak.linearize(x)  # AbstractLinearOperator
y_adjoint = H_lin.T @ residual
```

---

## Implementing `ObservationOperator` — Custom multi-instrument

```python
from vardax.obs_operators import (
    AveragingKernel, MultiInstrumentFusion, InstrumentRegistry, InstrumentSpec
)

# Per-instrument operator + quality mask + error cov
tropomi_spec = InstrumentSpec(
    obs_op=AveragingKernel(A=tropomi_A, x_a=tropomi_xa, h=tropomi_h),
    mask=tropomi_qa_flag,
    R_op=lx.DiagonalLinearOperator(tropomi_uncertainty),
    instrument_id="TROPOMI",
)
emit_spec = InstrumentSpec(...instrument_id="EMIT", ...)
ghgsat_spec = InstrumentSpec(...instrument_id="GHGSat", ...)

# Compose at the likelihood level
fusion = MultiInstrumentFusion(
    registry=InstrumentRegistry(entries={
        "TROPOMI": tropomi_spec,
        "EMIT": emit_spec,
        "GHGSat": ghgsat_spec,
    }),
)

# Returns dict[instrument_id, predicted_obs]
predictions = fusion(x, batch)
```

---

## Implementing `GradModulator` — ConvLSTM

```python
from vardax.protocols import GradModulator
import equinox as eqx

class ConvLSTMGradMod(eqx.Module):
    conv_lstm: eqx.Module
    output_proj: eqx.nn.Conv2d

    def __call__(self, grad, carry):
        h, c = carry
        h, c = self.conv_lstm(grad, (h, c))
        return self.output_proj(h), (h, c)

assert isinstance(ConvLSTMGradMod(lstm, proj), GradModulator)
```

---

## Implementing `Prior` — Wrap a somax forward as a `DynamicalPrior`

```python
import somax
from vardax.priors import DynamicalPrior

# somax already satisfies pipekit_cycle.ForwardModel
swm = somax.ShallowWaterModel(grid=grid, params=params)

# Wrap as a vardax Prior (integrates n_steps forward)
prior = DynamicalPrior(forward=swm, n_steps=10)
```

For methane / plumax:

```python
import plumax

plume = plumax.GaussianPlumeForward(met=met_field, dispersion="MO")
prior = DynamicalPrior(forward=plume, n_steps=1)  # single-shot for Tier I
```

---

## Composing components into a `VarDANet2D`

```python
from vardax.models import VarDANet2D
from vardax import SolverConfig

model = VarDANet2D(
    prior=ConvAEPrior(encoder=enc, decoder=dec),
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod(lstm, proj),
    config=SolverConfig(
        n_steps=15,
        alpha=0.2,
        prior_weight=1.0,
        grad_mode="one_step",
    ),
)
```

---

## Composing components into an `IncrementalVarDA2D`

```python
import gaussx as gx
import lineax as lx
from vardax.models import IncrementalVarDA2D
from vardax import IncrementalConfig

# Background covariance via gaussx Matérn factorisation (D11)
B_op = gx.MaternLinearOperator(grid_coords=coords, length_scale=10.0, nu=1.5, sigma=1.0)

# Diagonal obs-error covariance
R_op = lx.DiagonalLinearOperator(obs_err_variances)

model = IncrementalVarDA2D(
    forward=somax_model,                  # pipekit_cycle.ForwardModel
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),
    prior_mean=x_b,
    prior_cov_op=B_op,
    obs_cov_op=R_op,
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
)

x_star = model(batch)  # operational analysis
```

---

## Posterior adapters

```python
from vardax.posterior import LaplaceCovariance, GaussNewtonHessian, GaussianMarkLikelihood

# At MAP, build posterior
x_star = model(batch)
posterior = LaplaceCovariance()(x_star, model.as_analysis_step(), batch)

# Posterior(mean=x_star, cov=AbstractLinearOperator, samples=None, provenance={...})

# Export to population model
mark = GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={"event_id": "ev_001", "time": ..., "geometry": ...},
).to_dict()

# mark is JSON-friendly — write to GeoCatalog, database, etc.
```
