---
status: draft
version: 0.4.0
---

# Layer 1 — Component Examples

Implementing protocols and composing `eqx.Module` operators.

---

## Implementing `Prior` — Autoencoder

```python
import equinox as eqx
from vardax.protocols import Prior   # runtime-checkable Protocol

class ConvAEPrior(eqx.Module):
    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    def __call__(self, x):
        return self.decoder(self.encoder(x))

# Structural conformance — no inheritance needed
assert isinstance(ConvAEPrior(enc, dec), Prior)
```

---

## Implementing `ObservationOperator` — Masked identity

Vardax obs operators satisfy `pipekit_cycle.ObservationOperator`
directly (Decision D8). Implement both `__call__` and `linearize`:

```python
from pipekit_cycle import ObservationOperator
from lineax import JacobianLinearOperator

class MaskedIdentity(eqx.Module):
    """H(x) = mask ⊙ x."""

    def __call__(self, x, mask=None):
        return x * mask if mask is not None else x

    def linearize(self, x):
        return JacobianLinearOperator(self.__call__, x)

assert isinstance(MaskedIdentity(), ObservationOperator)
```

---

## Implementing `ObservationOperator` — Averaging kernel (D9)

```python
import lineax as lx
from vardax.obs_operators import AveragingKernel

ak = AveragingKernel(
    A=lx.MatrixLinearOperator(A_matrix),       # or gaussx structured op
    x_a=retrieval_prior,
    h=weighting_vector,
)

# ŷ = A · (h ⊙ x + (1-h) ⊙ x_a)
y_pred = ak(x)

# Tangent-linear operator for incremental 4DVar
H_lin = ak.linearize(x)
y_adjoint = H_lin.T @ residual
```

---

## Implementing `ObservationOperator` — Multi-instrument fusion

```python
from vardax.obs_operators import (
    AveragingKernel, MultiInstrumentFusion, InstrumentRegistry, InstrumentSpec,
)

tropomi_spec = InstrumentSpec(
    obs_op=AveragingKernel(A=tropomi_A, x_a=tropomi_xa, h=tropomi_h),
    mask=tropomi_qa_flag,
    R_op=lx.DiagonalLinearOperator(tropomi_uncertainty),
    instrument_id="TROPOMI",
)
emit_spec = InstrumentSpec(...)
ghgsat_spec = InstrumentSpec(...)

fusion = MultiInstrumentFusion(
    registry=InstrumentRegistry(entries={
        "TROPOMI": tropomi_spec,
        "EMIT": emit_spec,
        "GHGSat": ghgsat_spec,
    }),
)

# Returns dict[instrument_id, predicted_obs]
predictions = fusion(x, batch)

# For strict pipekit_cycle.ObservationOperator contexts, use the adapter:
fusion_as_op = fusion.to_observation_operator()
assert isinstance(fusion_as_op, ObservationOperator)
```

---

## Implementing `GradModulator` — ConvLSTM (FourDVarNet only)

```python
from vardax.protocols import GradModulator

class ConvLSTMGradMod(eqx.Module):
    conv_lstm: eqx.Module
    output_proj: eqx.nn.Conv2d

    def __call__(self, grad, carry):
        h, c = carry
        h, c = self.conv_lstm(grad, (h, c))
        return self.output_proj(h), (h, c)

assert isinstance(ConvLSTMGradMod(lstm, proj), GradModulator)
```

The grad modulator family is `FourDVarNet`-specific. Classical methods
use `optimistix.AbstractMinimiser` for the inner solver instead.

---

## Wrapping a `Minimiser` (classical methods)

```python
import optimistix as optx
from vardax.minimisers import Minimiser
from vardax.models import ThreeDVar

# Pick any optimistix minimiser
gn = Minimiser(optx.GaussNewton(rtol=1e-5, atol=1e-5),
               adjoint=optx.ImplicitAdjoint())
bfgs = Minimiser(optx.BFGS(rtol=1e-5, atol=1e-5))
ncg = Minimiser(optx.NonlinearCG(rtol=1e-5, atol=1e-5))

model = ThreeDVar(
    obs_op=obs_op,
    prior_mean=x_b, prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=gn.minimiser,         # the underlying optimistix solver
    minimiser_adjoint=gn.adjoint,
)
```

The `Minimiser` wrapper is convenience; you can also pass the
`optimistix.AbstractMinimiser` directly into the model constructor.

---

## Wrapping a somax / plumax forward as a `DynamicalPrior`

```python
import somax
from vardax.priors import DynamicalPrior
import diffrax as dfx

# somax already satisfies pipekit_cycle.ForwardModel
swm = somax.ShallowWaterModel(grid=grid, params=params)

# Wrap as a Prior (integrates n_steps forward)
prior = DynamicalPrior(
    forward=swm, n_steps=10,
    forward_adjoint=dfx.BacksolveAdjoint(),   # memory-efficient
)
```

For methane / plumax:

```python
import plumax
plume = plumax.GaussianPlumeForward(met=met_field, dispersion="MO")
prior = DynamicalPrior(forward=plume, n_steps=1)   # single-shot for Tier I
```

---

## Composing components into an `OptimalInterpolation`

```python
import gaussx as gx
from vardax.models import OptimalInterpolation
from vardax.obs_operators import LinearObs

model = OptimalInterpolation(
    obs_op=LinearObs(H_mat=H_lin_op),   # must be linear
    prior_mean=x_b,
    prior_cov_op=gx.MaternLinearOperator(coords, length_scale=10.0, sigma=0.1),
    obs_cov_op=lx.DiagonalLinearOperator(obs_uncertainty),
)

x_star = model(batch)                   # closed-form
posterior = model.posterior(batch)      # closed-form (mean, cov, provenance)
```

---

## Composing components into a `ThreeDVar`

```python
import optimistix as optx
from vardax.models import ThreeDVar

model = ThreeDVar(
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),   # nonlinear via h ⊙ x term
    prior_mean=x_b,
    prior_cov_op=B_op, obs_cov_op=R_op,
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
)
x_star = model(batch)
```

---

## Composing components into an `IncrementalFourDVar`

```python
from vardax.models import IncrementalFourDVar
from vardax import IncrementalConfig
import diffrax as dfx

model = IncrementalFourDVar(
    forward=somax_model,
    obs_op=AveragingKernel(A=A, x_a=xa, h=h),
    prior_mean=x_b,
    prior_cov_op=B_op, obs_cov_op=R_op,
    config=IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
    forward_adjoint=dfx.BacksolveAdjoint(),
)

x_star = model(batch)
posterior = model.posterior(batch)   # reuses GN Hessian from last outer iter
```

---

## Composing components into a `FourDVarNet`

```python
from vardax.models import FourDVarNet
from vardax import SolverConfig
from vardax.adjoints import OneStepAdjoint

model = FourDVarNet(
    prior=ConvAEPrior(encoder=enc, decoder=dec),
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod(lstm, proj),
    config=SolverConfig(n_steps=15, alpha=0.2),
    solver_adjoint=OneStepAdjoint(),    # O(1) memory training
)
```

---

## Posterior adapters

```python
from vardax.posterior import LaplaceCovariance, GaussianMarkLikelihood

# Classical: pair with explicit adapter
x_star = strong_4dvar(batch)
posterior = LaplaceCovariance()(x_star, strong_4dvar.as_analysis_step(), batch)

# OI / Incremental: direct call, no adapter
posterior = oi.posterior(batch)
posterior = incremental.posterior(batch)

# Export to population model
mark = GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={"event_id": "ev_001", "time": ..., "geometry": ...},
).to_dict()
```
