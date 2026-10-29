---
status: draft
version: 0.4.0
---

# Layer 1 — Component Examples

Implementing protocols and composing `eqx.Module` operators.

---

## Implementing `Prior` — Autoencoder

```{code-block} python
:caption: Implements an autoencoder prior φ for the learned prior term of the 4DVarNet cost [](#eq-4dvarnet-cost) and checks it against the Prior protocol.
:label: code-design-components-prior
import equinox as eqx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: T times; N state cells
# Given: enc, dec (eqx.nn.Sequential encoder and decoder)


# user
class ConvAEPrior(eqx.Module):
    """Autoencoder prior φ(x) = dec(enc(x))."""

    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    def __call__(self, x: Float[Array, "T N"]) -> Float[Array, "T N"]:
        # z = enc(x)
        latent = self.encoder(x)
        # φ(x) = dec(z)
        return self.decoder(latent)


# today (vardax)
# Structural conformance, no inheritance needed (runtime-checkable Protocol)
assert isinstance(ConvAEPrior(enc, dec), vdx.Prior)
```

---

## Implementing `ObservationOperator` — Masked identity

Vardax obs operators satisfy `pipekit_cycle.ObservationOperator`
directly (Decision D8). Implement both `__call__` and `linearize`:

```{code-block} python
:caption: Implements the masked-identity operator [](#eq-masked-identity) with a Jacobian linearisation and checks it against the pipekit_cycle protocol.
:label: code-design-components-masked-identity
import equinox as eqx
import lineax as lx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: N state cells


# user
class MaskedIdentity(eqx.Module):
    """H(x) = mask ⊙ x."""

    def __call__(
        self, x: Float[Array, " N"], mask: Float[Array, " N"] | None = None
    ) -> Float[Array, " N"]:
        # H(x) = m ⊙ x, or x when no mask is given
        return x * mask if mask is not None else x

    def linearize(self, x: Float[Array, " N"]) -> lx.AbstractLinearOperator:
        # today (lineax)
        # H′(x) = ∂H/∂x
        return lx.JacobianLinearOperator(self.__call__, x)  # (N, N) op


# today (pipekit_cycle)
assert isinstance(MaskedIdentity(), pc.ObservationOperator)
```

---

## Implementing `ObservationOperator` — Averaging kernel (D9)

```{code-block} python
:caption: Applies the averaging-kernel operator [](#eq-averaging-kernel) and its tangent-linear adjoint, the pair checked by [](#eq-adjoint-test).
:label: code-design-components-averaging-kernel
import lineax as lx
import vardax as vdx

# Shapes: N levels
# Given: A_matrix (N, N), retrieval_prior x_a (N,), weighting_vector h (N,),
#        state x (N,), residual r (N,)

# today (lineax)
# A, or a gaussx structured operator
kernel_op = lx.MatrixLinearOperator(A_matrix)  # (N, N) op
# today (vardax)
ak = vdx.AveragingKernel(
    kernel_matrix=kernel_op,
    prior_profile=retrieval_prior,
    pressure_weights=weighting_vector,
)

# today (vardax)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a)
y_pred = ak(x)  # (N,) → (N,)

# Tangent-linear operator for incremental 4DVar
# today (vardax)
# H′ = A diag(h)
obs_jacobian = ak.linearize(x)  # (N,) → (N, N) op
# today (lineax)
# H′ᵀ
obs_adjoint = obs_jacobian.transpose()  # (N, N) op
# today (lineax)
# H′ᵀ r
y_adjoint = obs_adjoint.mv(residual)  # (N,) → (N,)
```

---

## Implementing `ObservationOperator` — Multi-instrument fusion

```{code-block} python
:caption: Builds the multi-instrument operator behind the fusion cost [](#eq-fusion-cost) from per-instrument averaging kernels [](#eq-averaging-kernel).
:label: code-design-components-fusion
import lineax as lx
import pipekit_cycle as pc
import vardax as vdx

# Shapes: N state cells; Mᵢ observations of instrument i
# Given: tropomi_A, tropomi_xa, tropomi_h, tropomi_qa_flag,
#        tropomi_uncertainty (Mᵢ,), state x (N,), batch

# today (vardax)
# Hᵢ(x) = Aᵢ (hᵢ ⊙ x + (1 − hᵢ) ⊙ x_a,i)
tropomi_ak = vdx.AveragingKernel(
    kernel_matrix=tropomi_A, prior_profile=tropomi_xa, pressure_weights=tropomi_h
)
# today (lineax)
# Rᵢ = diag(σᵢ²)
tropomi_cov = lx.DiagonalLinearOperator(tropomi_uncertainty)  # (Mᵢ, Mᵢ) op
# today (vardax)
# (Hᵢ, mᵢ, Rᵢ) for i = TROPOMI
tropomi_spec = vdx.InstrumentSpec(
    obs_op=tropomi_ak,
    mask=tropomi_qa_flag,
    obs_cov_op=tropomi_cov,
    instrument_id="TROPOMI",
)
# today* (vardax)
emit_spec = vdx.InstrumentSpec(...)
# today* (vardax)
ghgsat_spec = vdx.InstrumentSpec(...)

# today (vardax)
registry = vdx.InstrumentRegistry(
    entries={
        "TROPOMI": tropomi_spec,
        "EMIT": emit_spec,
        "GHGSat": ghgsat_spec,
    }
)
# today (vardax)
fusion = vdx.MultiInstrumentFusion(registry=registry)

# Returns dict[instrument_id, predicted_obs]
# today* (vardax; __call__ takes the state only)
# {i: Hᵢ(x)}
predictions = fusion(x, batch)  # (N,) → {i: (Mᵢ,)}

# For strict pipekit_cycle.ObservationOperator contexts, use the adapter:
# today (vardax)
# H(x) = [H_TROPOMI(x); H_EMIT(x); H_GHGSat(x)]
fusion_as_op = fusion.to_observation_operator()
# today (pipekit_cycle)
assert isinstance(fusion_as_op, pc.ObservationOperator)
```

---

## Implementing `GradModulator` — ConvLSTM (FourDVarNet only)

```{code-block} python
:caption: Implements the learned gradient modulator Φ of the 4DVarNet update [](#eq-4dvarnet-update) as a ConvLSTM.
:label: code-design-components-grad-mod
import equinox as eqx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: C channels; G hidden channels; H × W grid
# Given: lstm (a ConvLSTM cell), proj (an eqx.nn.Conv2d from G to C channels)

Carry = tuple[Float[Array, "G H W"], Float[Array, "G H W"]]


# user
class ConvLSTMGradMod(eqx.Module):
    """Φ(∇ₓJ, h) → (update, h′) with a ConvLSTM cell and a projection."""

    conv_lstm: eqx.Module
    output_proj: eqx.nn.Conv2d

    def __call__(
        self, grad: Float[Array, "C H W"], carry: Carry
    ) -> tuple[Float[Array, "C H W"], Carry]:
        hidden, cell = carry
        # (h′, c′) = ConvLSTM(∇ₓJ, (h, c))
        hidden, cell = self.conv_lstm(grad, (hidden, cell))
        # Φ(∇ₓJ, h) = W h′
        update = self.output_proj(hidden)
        return update, (hidden, cell)


# today* (vardax; the shipped protocol is __call__(grad, state, carry))
assert isinstance(ConvLSTMGradMod(lstm, proj), vdx.GradModulator)
```

The grad modulator family is `FourDVarNet`-specific. Classical methods
use `optimistix.AbstractMinimiser` for the inner solver instead.

---

## Wrapping a `Minimiser` (classical methods)

```{code-block} python
:caption: Wraps optimistix minimisers for the 3DVar cost [](#eq-3dvar-cost) and passes one to ThreeDVar.
:label: code-design-components-minimiser
import optimistix as optx
import vardax as vdx

# Given: obs_op, background x_b, prior_cov B op, obs_cov R op

# Pick any optimistix minimiser
# proposed (D6; vdx.Minimiser ships as a Protocol, the wrapper is
#   vdx.OptimistixMinimiser(solver), with no adjoint field)
gn = vdx.Minimiser(
    optx.GaussNewton(rtol=1e-5, atol=1e-5), adjoint=optx.ImplicitAdjoint()
)
# proposed (D6)
bfgs = vdx.Minimiser(optx.BFGS(rtol=1e-5, atol=1e-5))
# proposed (D6)
ncg = vdx.Minimiser(optx.NonlinearCG(rtol=1e-5, atol=1e-5))

# today* (vardax; gn.minimiser and gn.adjoint are proposed)
# x* = argmin J(x)
model = vdx.ThreeDVar(
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    # the underlying optimistix solver
    minimiser=gn.minimiser,
    minimiser_adjoint=gn.adjoint,
)
```

The `Minimiser` wrapper is convenience; you can also pass the
`optimistix.AbstractMinimiser` directly into the model constructor.

---

## Wrapping a somax / plumax forward as a `DynamicalPrior`

```{code-block} python
:caption: Wraps a somax shallow-water forward [](#eq-shallow-water) as a dynamical prior that rolls the state forward [](#eq-rollout).
:label: code-design-components-dynamical-prior
import diffrax as dfx
import somax
import vardax as vdx

# Given: grid, params

# not pinned (somax)
# somax already satisfies pipekit_cycle.ForwardModel
swm = somax.ShallowWaterModel(grid=grid, params=params)

# Wrap as a Prior (integrates n_steps forward)
# today* (vardax; ships as DynamicalPrior(model, params, solver, stepsize, adjoint))
# φ(x₀) = M₁₀(x₀)
prior = vdx.DynamicalPrior(
    forward=swm,
    n_steps=10,
    # memory-efficient
    forward_adjoint=dfx.BacksolveAdjoint(),
)
```

For methane / plumax:

```{code-block} python
:caption: Wraps a single-shot plumax Gaussian-plume forward as a dynamical prior, one step of [](#eq-rollout).
:label: code-design-components-plume-prior
import plumax
import vardax as vdx

# Given: met_field

# not pinned (plumax)
plume = plumax.GaussianPlumeForward(met=met_field, dispersion="MO")
# today* (vardax; ships as DynamicalPrior(model, params, solver, stepsize, adjoint))
# single-shot for Tier I: φ(x) = M₁(x)
prior = vdx.DynamicalPrior(forward=plume, n_steps=1)
```

---

## Composing components into an `OptimalInterpolation`

```{code-block} python
:caption: Composes a linear operator and a Matérn prior into OptimalInterpolation for the closed-form mean [](#eq-blue-mean) and covariance [](#eq-blue-cov).
:label: code-design-components-oi
import gaussx as gx
import lineax as lx
import vardax as vdx

# Shapes: S windows; N state cells; M observations; D = 2 coordinates
# Given: H_lin_op (M, N) lineax op, background x_b (N,), coords (N, D),
#        obs_uncertainty (M,), batch

# today (vardax)
# H(x) = H x, must be linear
obs_op = vdx.LinearObs(operator=H_lin_op)
# proposed (gaussx)
# Bᵢⱼ = k_Matérn(sᵢ, sⱼ; ℓ = 10, σ = 0.1)
prior_cov = gx.MaternLinearOperator(  # (N, D) → (N, N) PSD op
    coords, length_scale=10.0, sigma=0.1
)
# today (lineax)
# R = diag(σ²)
obs_cov = lx.DiagonalLinearOperator(obs_uncertainty)  # (M,) → (M, M) op

# today (vardax)
model = vdx.OptimalInterpolation(
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
)

# today (vardax)
# closed form: x* = x_b + K (y − H x_b)
analysis = model(batch)  # → (S, N)
# today (vardax)
# closed form: P* = (I − K H) B, with mean and provenance
posterior = model.posterior(batch)  # Posterior(mean (N,), cov (N, N) op)
```

---

## Composing components into a `ThreeDVar`

```{code-block} python
:caption: Composes an averaging kernel [](#eq-averaging-kernel) and Gauss-Newton into ThreeDVar for the cost [](#eq-3dvar-cost).
:label: code-design-components-threedvar
import optimistix as optx
import vardax as vdx

# Shapes: S windows; N levels
# Given: A (N, N) lineax op, xa x_a (N,), h (N,), background x_b (N,),
#        prior_cov B op, obs_cov R op, batch

# today (vardax)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a), nonlinear via the h ⊙ x term
obs_op = vdx.AveragingKernel(kernel_matrix=A, prior_profile=xa, pressure_weights=h)

# today* (vardax; optx.GaussNewton is a least-squares solver, not an
#   AbstractMinimiser, so the model rejects it: use optx.BFGS or NonlinearCG)
# J(x) = ½ ‖x − x_b‖²_{B⁻¹} + ½ ‖y − H(x)‖²_{R⁻¹}
model = vdx.ThreeDVar(
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    minimiser=optx.GaussNewton(rtol=1e-5, atol=1e-5),
    minimiser_adjoint=optx.ImplicitAdjoint(),
)
# today (vardax)
# x* = argmin J(x)
analysis = model(batch)  # → (S, N)
```

---

## Composing components into an `IncrementalFourDVar`

```{code-block} python
:caption: Composes IncrementalFourDVar for the incremental cost [](#eq-incremental-cost) in control variables [](#eq-cvt) and reads the Gauss-Newton posterior [](#eq-gn-cov).
:label: code-design-components-incremental
import diffrax as dfx
import vardax as vdx

# Shapes: S windows; T times; N state cells
# Given: somax_model, A (N, N) lineax op, xa x_a (N,), h (N,),
#        background x_b (N,), prior_cov B op, obs_cov R op, batch

# today (vardax)
# ŷ = A (h ⊙ x + (1 − h) ⊙ x_a)
obs_op = vdx.AveragingKernel(kernel_matrix=A, prior_profile=xa, pressure_weights=h)

# today (vardax)
# 3 outer relinearisations, 20 CG steps each, in χ with x₀ − x_b = S χ
model = vdx.IncrementalFourDVar(
    forward=somax_model,
    obs_op=obs_op,
    prior_mean=background,
    prior_cov_op=prior_cov,
    obs_cov_op=obs_cov,
    config=vdx.IncrementalConfig(n_outer=3, n_inner=20, cvt=True),
    forward_adjoint=dfx.BacksolveAdjoint(),
)

# today (vardax)
# x₀* after the last outer iteration
analysis = model(batch)  # → (S, N)
# today (vardax)
# P*_GN = 𝐇_GN⁻¹, reusing the Hessian of the last outer iteration
posterior = model.posterior(batch)  # Posterior(mean (N,), cov (N, N) op)
```

---

## Composing components into a `FourDVarNet`

```{code-block} python
:caption: Composes the page's prior and gradient modulator into a FourDVarNet that minimises [](#eq-4dvarnet-cost) by the learned update [](#eq-4dvarnet-update).
:label: code-design-components-fourdvarnet
import vardax as vdx

# Shapes: K = 15 inner steps
# Given: enc, dec, lstm, proj, and the page's ConvAEPrior, MaskedIdentity and
#        ConvLSTMGradMod

# proposed (D14; ships today as vdx.FourDVarNet1D / vdx.FourDVarNet2D)
# x_{k+1} = x_k − Φ(∇ₓJ(x_k), h_k), k = 0 … K − 1
model = vdx.FourDVarNet(
    prior=ConvAEPrior(encoder=enc, decoder=dec),
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod(lstm, proj),
    # proposed (D4)
    config=vdx.SolverConfig(n_steps=15, alpha=0.2),
    # O(1) memory training
    solver_adjoint=vdx.adjoints.OneStepAdjoint(),
)
```

---

## Posterior adapters

```{code-block} python
:caption: Builds the Laplace posterior [](#eq-laplace-cov) with an adapter, reads the closed-form posteriors [](#eq-blue-cov) and [](#eq-gn-cov) directly, and exports a Gaussian mark likelihood.
:label: code-design-components-posterior-adapters
import vardax as vdx

# Shapes: N state cells
# Given: strong_4dvar, oi, incremental (built as above), batch

# Classical: pair with explicit adapter
# today (vardax)
# x₀* = argmin J(x₀)
analysis = strong_4dvar(batch)  # → (S, N)
# today* (vardax; the adapter takes one window's analysis, e.g. analysis[0])
# P*_Laplace = (B⁻¹ + H′ᵀ R⁻¹ H′)⁻¹ at x₀*
posterior = vdx.LaplaceCovariance()(analysis, strong_4dvar.as_analysis_step(), batch)

# OI / Incremental: direct call, no adapter
# today (vardax)
# P* = (I − K H) B
posterior = oi.posterior(batch)
# today (vardax)
# P*_GN = 𝐇_GN⁻¹
posterior = incremental.posterior(batch)

# Export to population model
# today (vardax)
mark_likelihood = vdx.GaussianMarkLikelihood(
    posterior=posterior,
    event_metadata={"event_id": "ev_001", "time": ..., "geometry": ...},
)
# today (vardax)
mark = mark_likelihood.to_dict()
```
