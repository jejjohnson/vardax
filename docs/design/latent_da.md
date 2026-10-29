---
status: draft
version: 0.1.0
---

# vardax × Latent Data Assimilation

**Subject:** First-class variational data assimilation in a learned
low-dimensional latent space — `LatentThreeDVar`,
`LatentStrongFourDVar`, `LatentHybridFourDVar` — built on the new
`pipekit_cycle.LatentMap` / `LatentForwardModel` /
`LiftedObservationOperator` protocols.

**Date:** 2026-05-28

**Decision anchor:** [D17 — Latent DA as a first-class peer family](decisions.md#d17-latent-da-as-a-first-class-peer-family).
Foundation in pipekit-cycle: `packages/pipekit-cycle/docs/design/latent.md`.

**Math reference:** [Chapter 18 — Latent Variational DA](../18_latent_da.md).

---

## 1  Motivation

Vardax today exposes three points where a learned subspace already
helps:

* `BilinAEPrior1D/2D`, `MLPAEPrior1D`, `ConvAEPrior1D` — autoencoder
  priors used **inside the cost** of `FourDVarNet*` to regularise the
  reconstruction.
* `AmortizedPosterior.MLPObsEncoder` — encodes `(y, mask)` into a
  context vector before the head produces $q_\phi(x \mid y)$.
* `RegressionHead`, `ConditionalFlowHead`, `ScoreDiffusionHead` — heads
  whose internal state lives implicitly in a low-dim representation.

What is **missing** is the natural variational counterpart of these:
solving the variational problem *itself* in latent space. The
benchmark literature (Peyron et al. 2021, Cheng et al. 2023, Fablet
et al. 2021) consistently reports order-of-magnitude wall-clock wins on
this exact reformulation. Vardax should offer it as a peer family of
`AnalysisStep` classes, not as a per-method retrofit.

Three new Layer-2 models fall out:

| Model | Control vector | Forecast in | Use when |
|---|---|---|---|
| `LatentThreeDVar` | $z$ | — | Single-time snapshot inversion in latent space. |
| `LatentStrongFourDVar` | $z_0$ | $\mathcal{Z}$ | Multi-time, latent dynamics $M_z$ available (learned or `EncodedForwardModel`). |
| `LatentHybridFourDVar` | $z_0$ | $\mathcal{X}$ | Multi-time, physics forecast in $\mathcal{X}$, update in $\mathcal{Z}$. |

A fourth deliverable is an `as_latent_map()` adapter on the existing AE
priors so they satisfy `pipekit_cycle.LatentMap` without rewriting the
internals.

---

## 2  Updated three-layer stack

The new components slot into the existing stack without disturbing the
seven peer `AnalysisStep` classes from v0.4 (D14):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Models  (each satisfies pipekit_cycle.AnalysisStep)              │
│                                                                             │
│  Classical:                                                                 │
│    OptimalInterpolation, ThreeDVar, StrongFourDVar, WeakFourDVar,           │
│    IncrementalFourDVar                                                      │
│                                                                             │
│  Learned:                                                                   │
│    FourDVarNet, AmortizedPosterior                                          │
│                                                                             │
│  Latent (NEW):                                                              │
│    LatentThreeDVar                                                          │
│    LatentStrongFourDVar                                                     │
│    LatentHybridFourDVar                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Components                                                       │
│                                                                             │
│  Priors:           BilinAE, ConvAE, MLPAE  (gain encode/decode +            │
│                    latent_dim/state_signature → satisfy LatentMap),         │
│                    DynamicalPrior, Diffusion                                │
│  LatentMap (NEW):  LatentPrior wraps (LatentMap, B_z_op) for background     │
│                    error covariance in z                                    │
│  Forward (NEW):    NeuralLatentForwardModel adapter (eqx.Module → protocol) │
│  ObservationOperator: MaskedIdentity, LinearObs, AveragingKernel,           │
│                    MultiInstrumentFusion, + LiftedObservationOperator (pkc) │
│  GradModulator:    (unchanged)                                              │
│  CostFunction:     + variational_cost_latent, latent_incremental_cost       │
│  Minimiser:        (unchanged — optimistix wrapper)                         │
│  PosteriorAdapter: + LatentLaplaceCovariance (Laplace in z, decoded to x)   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives  (pure JAX)                                           │
│                                                                             │
│  + variational_cost_latent, latent_incremental_cost                         │
│  + decode_jacobian helper for chain-rule linearisation                      │
│  + identity_latent_map for testing (phi = psi = id)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3  The math, in one screen

For state $x \in \mathbb{R}^{N_x}$, observations $y$, autoencoder
$(\varphi, \psi)$, latent dynamics $M_z$, and lifted operator
$\tilde{H} = H \circ \psi$:

**Latent 3DVar** — find $z^* = \arg\min_z J_{3D}(z)$ with

$$
J_{3D}(z) = \tfrac{1}{2}\|z - z_b\|^2_{\mathbf{B}_z^{-1}}
          + \tfrac{1}{2}\|y - H(\psi(z))\|^2_{\mathbf{R}^{-1}}.
$$

Analysis: $x^* = \psi(z^*)$.

**Latent Strong-4DVar** — control $z_0$, rollout $z_{k+1} = M_z(z_k)$:

$$
J_{4D}(z_0) = \tfrac{1}{2}\|z_0 - z_b\|^2_{\mathbf{B}_z^{-1}}
            + \tfrac{1}{2}\sum_{k=0}^{K}\|y_k - H(\psi(z_k))\|^2_{\mathbf{R}^{-1}}.
$$

**Latent Hybrid 4DVar** — control $z_0$, decode once to seed the
physical state, then roll out $x_{k+1} = M_x(x_k)$ entirely in
$\mathcal{X}$; observation residual evaluated on $x_k$:

$$
J_{H}(z_0) = \tfrac{1}{2}\|z_0 - z_b\|^2_{\mathbf{B}_z^{-1}}
           + \tfrac{1}{2}\sum_{k=0}^{K}\|y_k - H(x_k)\|^2_{\mathbf{R}^{-1}},
\quad
x_0 = \psi(z_0),\;\; x_{k+1} = M_x(x_k).
$$

In all three, the cost is **smaller-dim** ($N_z \ll N_x$), the
minimiser converges faster, and the Hessian / Laplace approximation
lives in $\mathcal{Z}$. Gradients flow through $\psi$ via JAX autodiff
for `eqx.Module` decoders — no hand-coded adjoint.

Full derivations, tangent-linear forms, and the Sherman–Morrison–
Woodbury identity used to switch between $\mathbf{B}_z$-space and
$\mathbf{R}$-space gain formulas live in the math reference
[Chapter 18](../18_latent_da.md).

---

## 4  Protocol composition

Vardax does not redefine any protocols — it consumes the three new ones
shipped by `pipekit-cycle/latent`:

```{code-block} python
:caption: Imports the four pipekit-cycle latent protocols that the latent costs [](#eq-latent-3dvar-cost), [](#eq-latent-strong4dvar-cost) and [](#eq-latent-hybrid-cost) are built from.
:label: code-design-latent-protocol-imports
# today (pipekit_cycle)
from pipekit_cycle import LatentMap

# proposed (pipekit-cycle latent.md)
from pipekit_cycle import (
    EncodedForwardModel,
    LatentForwardModel,
    LiftedObservationOperator,
)
```

Three vardax adapters wire existing components to the new protocols:

### 4.1  `LatentPrior`

```{code-block} python
:caption: Declares `LatentPrior`, which pairs a `LatentMap` with B_z and computes the background term of [](#eq-latent-3dvar-cost).
:label: code-design-latent-prior
import einx
import equinox as eqx
import lineax as lx
import pipekit_cycle as pc
from jaxtyping import Array, Float

# Shapes: Nx = state size; Nz = latent size, Nz ≪ Nx


# proposed (D17)
class LatentPrior(eqx.Module):
    """Variational prior in latent space; wraps a LatentMap with B_z.

    Used as the prior term in latent variational costs:

        J_prior(z) = 1/2 (z - z_b)^T B_z^{-1} (z - z_b).
    """

    # today (pipekit_cycle)
    latent_map: pc.LatentMap
    z_b: Float[Array, " Nz"]
    # B_z itself, NOT its inverse
    B_z_op: lx.AbstractLinearOperator

    def cost(self, z: Float[Array, " Nz"]) -> Float[Array, ""]:
        # B_z_op is the covariance operator, so a solve applies B_z⁻¹; a
        # precision operator should be wrapped in
        # `lineax.TaggedLinearOperator(B_z_inv, lx.tags.symmetric)` and
        # adapted to a `lineax.matmul`-based cost instead.
        # d = z − z_b
        residual = z - self.z_b  # (Nz,) → (Nz,)
        # B_z⁻¹ d
        whitened = lx.linear_solve(self.B_z_op, residual).value  # (Nz,) → (Nz,)
        # dᵀ B_z⁻¹ d
        quadratic = einx.dot("z, z ->", residual, whitened)  # → ()
        # J_prior(z) = ½ dᵀ B_z⁻¹ d
        return 0.5 * quadratic

    def encode(self, x: Float[Array, " Nx"]) -> Float[Array, " Nz"]:
        # today (pipekit_cycle)
        # z = φ(x)
        return self.latent_map.encode(x)

    def decode(self, z: Float[Array, " Nz"]) -> Float[Array, " Nx"]:
        # today (pipekit_cycle)
        # x = ψ(z)
        return self.latent_map.decode(z)
```

### 4.2  Making existing AE priors satisfy `LatentMap`

Of today's AE priors only `BilinAEPrior1D` exposes both `.encode` and
`.decode`; `MLPAEPrior1D`, `BilinAEPrior2D`, `BilinAEPrior2DMultivar`,
and `ConvAEPrior1D` currently only have `__call__` (the
encode-then-decode round-trip used by the FourDVarNet prior cost).
v0.5 extracts the two halves so each prior satisfies
`pipekit_cycle.LatentMap`. The work is mechanical — the existing
`__call__` is already implemented as encode-then-decode internally:

| Prior | Today | v0.5 |
|---|---|---|
| `BilinAEPrior1D` | `__call__`, `encode`, `decode` | + `latent_dim`, `state_signature` properties |
| `MLPAEPrior1D` | `__call__` only | split into `encode` + `decode`, add properties |
| `BilinAEPrior2D` | `__call__` only | split, add properties |
| `BilinAEPrior2DMultivar` | `__call__` only | split, add properties |
| `ConvAEPrior1D` | `__call__` only | split, add properties |

Pattern (illustrated on `MLPAEPrior1D`):

```{code-block} python
:caption: Splits `MLPAEPrior1D` into the encoder φ and decoder ψ of [](#eq-latent-3dvar-cost) so it satisfies `pipekit_cycle.LatentMap`, keeping the round trip ψ(φ(x)) bit-for-bit.
:label: code-design-latent-ae-split
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float

# Shapes: Nx = state size; Nz = latent size


# today* (vardax); ships __call__ only, with no encode / decode / properties
class MLPAEPrior1D(eqx.Module):
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    _latent_dim: int = eqx.field(static=True)
    _state_signature: Any = eqx.field(static=True, default=None)

    # proposed (D17): extracted halves of the existing __call__.
    # z = φ(x)
    def encode(self, x: Float[Array, " Nx"]) -> Float[Array, " Nz"]:
        return self.encoder(x)

    # proposed (D17)
    # x = ψ(z)
    def decode(self, z: Float[Array, " Nz"]) -> Float[Array, " Nx"]:
        return self.decoder(z)

    # today (vardax): existing, preserved bit-for-bit.
    # x̂ = ψ(φ(x))
    def __call__(self, x: Float[Array, " Nx"]) -> Float[Array, " Nx"]:
        latent = self.encode(x)  # (Nx,) → (Nz,)
        return self.decode(latent)

    # proposed (D17): make the AE satisfy pipekit_cycle.LatentMap structurally.
    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    # proposed (D17)
    @property
    def state_signature(self) -> Any:
        return self._state_signature
```

After this change, `isinstance(prior, pipekit_cycle.LatentMap)` is
true at runtime for all five AE priors. `FourDVarNet*` behaviour is
unchanged (the existing `__call__` reconstruction path is preserved
bit-for-bit).

### 4.3  `NeuralLatentForwardModel`

Wraps an `eqx.Module` that produces $z_{k+1} = M_z(z_k)$. The vast
majority of latent dynamics in the literature is one of three things:

1. A residual MLP $M_z(z) = z + f_\theta(z)$,
2. A neural ODE integrated by `diffrax`,
3. A learned linear operator (for short horizons).

Rather than ship any one of these (per the user's earlier decision —
"no, leave to users"), we ship the adapter:

```{code-block} python
:caption: Declares `NeuralLatentForwardModel`, the adapter that turns any learned z → z module into the latent dynamics M_z of [](#eq-latent-strong4dvar-cost).
:label: code-design-latent-forward-model
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float

# Shapes: Nz = latent size


# proposed (D17)
class NeuralLatentForwardModel(eqx.Module):
    """Wraps an eqx.Module mapping z_k → z_{k+1}.

    Satisfies pipekit_cycle.LatentForwardModel.  Users supply the
    learned dynamics module via composition.
    """

    # any z → z module
    net: eqx.Module
    dt: float = 1.0
    latent_signature: Any = eqx.field(static=True, default=None)

    # zₖ₊₁ = M_z(zₖ)
    def step(self, z: Float[Array, " Nz"], dt: float) -> Float[Array, " Nz"]:
        return self.net(z)
```

For users who only have an x-space `ForwardModel`, the
`pipekit_cycle.EncodedForwardModel` helper provides the AE round-trip
without learning a separate $M_z$.

---

## 5  Layer-2 models

Three new sibling classes live in `vardax/_src/models/`:

### 5.1  `LatentThreeDVar`

```{code-block} python
:caption: Declares `LatentThreeDVar`, which minimises the latent 3DVar cost [](#eq-latent-3dvar-cost) over z and decodes the analysis x* = ψ(z*).
:label: code-design-latent-3dvar
import equinox as eqx
import lineax as lx
import optimistix as optx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: Nx = state size; Nz = latent size; D = observations


# proposed (D17)
class LatentThreeDVar(eqx.Module):
    """Latent 3DVar: minimise J_3D(z) over z, decode to x."""

    # today (pipekit_cycle)
    latent_map: pc.LatentMap
    # today (pipekit_cycle); x-space
    obs_op: pc.ObservationOperator
    # proposed (D17)
    prior: LatentPrior
    obs_cov_op: lx.AbstractLinearOperator
    # today (optimistix)
    minimiser: optx.AbstractMinimiser = optx.BFGS(rtol=1e-6, atol=1e-6)
    # today (optimistix)
    minimiser_adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()

    # today (vardax): Batch1D, Batch2D
    def __call__(self, batch: vdx.Batch1D | vdx.Batch2D) -> Float[Array, " Nx"]:
        obs = batch.input
        mask = batch.mask
        # proposed (pipekit-cycle latent.md)
        # H̃ = H ∘ ψ
        lifted_obs_op = pc.LiftedObservationOperator(
            decoder=self.latent_map,
            inner=self.obs_op,
        )

        def cost_fn(z: Float[Array, " Nz"], args: None) -> Float[Array, ""]:
            # H̃(z) = H(ψ(z))
            obs_pred = lifted_obs_op(z)  # (Nz,) → (D,)
            # proposed (D17)
            # J_3D(z) = ½‖z − z_b‖²_{B_z⁻¹} + ½‖y − H(ψ(z))‖²_{R⁻¹}
            return variational_cost_latent(  # → ()
                z=z,
                z_b=self.prior.z_b,
                B_z_op=self.prior.B_z_op,
                obs_pred=obs_pred,
                y=obs,
                mask=mask,
                R_op=self.obs_cov_op,
            )

        # today (optimistix)
        # z* = argmin J_3D(z), started from z_b
        solution = optx.minimise(
            cost_fn,
            self.minimiser,
            self.prior.z_b,
            adjoint=self.minimiser_adjoint,
        )
        # today (pipekit_cycle)
        # x* = ψ(z*)
        return self.latent_map.decode(solution.value)  # (Nz,) → (Nx,)

    # proposed (D17)
    def as_analysis_step(self) -> pc.AnalysisStep:
        return _LatentThreeDVarAnalysisStep(self)
```

`_LatentThreeDVarAnalysisStep` is the same five-line adapter used by
the existing seven peers; it adds the canonical
`(forecast, obs, *, obs_op, obs_err_cov) → analysis` signature so
`pipekit_cycle.LatentDACycle` can drive it.

### 5.2  `LatentStrongFourDVar`

```{code-block} python
:caption: Declares `LatentStrongFourDVar`, which rolls z₀ forward with M_z and minimises the latent strong-constraint cost [](#eq-latent-strong4dvar-cost) over z₀.
:label: code-design-latent-strong4dvar
from typing import Any

import einx
import equinox as eqx
import jax
import lineax as lx
import optimistix as optx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: Nx = state size; Nz = latent size; D = observations per time;
#   K = rollout steps, so T = K + 1 times in the window


# proposed (D17)
class LatentStrongFourDVar(eqx.Module):
    """Latent strong-constraint 4DVar: minimise J_4D(z_0) over z_0."""

    # today (pipekit_cycle)
    latent_map: pc.LatentMap
    # proposed (pipekit-cycle latent.md); M_z, or EncodedForwardModel
    forward: pc.LatentForwardModel
    # today (pipekit_cycle); x-space
    obs_op: pc.ObservationOperator
    # proposed (D17)
    prior: LatentPrior
    obs_cov_op: lx.AbstractLinearOperator
    n_steps: int = eqx.field(static=True, default=10)

    # today (optimistix)
    minimiser: optx.AbstractMinimiser = optx.BFGS(...)
    # today (optimistix)
    minimiser_adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()
    # if M_z uses diffrax
    forward_adjoint: Any = None

    # today (vardax): Batch1D
    def __call__(self, batch: vdx.Batch1D) -> Float[Array, " Nx"]:
        obs_seq = batch.input
        mask_seq = batch.mask
        # proposed (pipekit-cycle latent.md)
        # H̃ = H ∘ ψ
        lifted_obs_op = pc.LiftedObservationOperator(
            decoder=self.latent_map,
            inner=self.obs_op,
        )
        observe_seq = jax.vmap(lifted_obs_op)

        def rollout(z0: Float[Array, " Nz"]) -> Float[Array, "T Nz"]:
            def step(
                z: Float[Array, " Nz"], _: None
            ) -> tuple[Float[Array, " Nz"], Float[Array, " Nz"]]:
                # proposed (pipekit-cycle latent.md)
                # zₖ₊₁ = M_z(zₖ)
                z_next = self.forward.step(z, self.forward.dt)  # (Nz,) → (Nz,)
                return z_next, z_next

            # z₁, …, z_K
            _, trajectory = jax.lax.scan(step, z0, None, length=self.n_steps)
            # (z₀, z₁, …, z_K)
            return einx.rearrange("z, k z -> (1 + k) z", z0, trajectory)

        def cost_fn(z0: Float[Array, " Nz"], args: None) -> Float[Array, ""]:
            latent_seq = rollout(z0)  # (Nz,) → (T, Nz)
            # H(ψ(zₖ)), k = 0, …, K
            obs_pred = observe_seq(latent_seq)  # (T, Nz) → (T, D)
            # proposed (D17)
            # ½‖z₀ − z_b‖²_{B_z⁻¹}
            background_term = self.prior.cost(z0)  # → ()
            # proposed (D17)
            # ½ Σₖ ‖yₖ − H(ψ(zₖ))‖²_{R⁻¹}
            obs_term = obs_misfit_latent_seq(  # → ()
                obs_pred=obs_pred,
                y_seq=obs_seq,
                mask_seq=mask_seq,
                R_op=self.obs_cov_op,
            )
            # J_4D(z₀)
            return background_term + obs_term

        # today (optimistix)
        # z₀* = argmin J_4D(z₀), started from z_b
        solution = optx.minimise(
            cost_fn,
            self.minimiser,
            self.prior.z_b,
            adjoint=self.minimiser_adjoint,
        )
        # today (pipekit_cycle)
        # x₀* = ψ(z₀*)
        return self.latent_map.decode(solution.value)  # (Nz,) → (Nx,)
```

### 5.3  `LatentHybridFourDVar`

```{code-block} python
:caption: Declares `LatentHybridFourDVar`, which decodes x₀ = ψ(z₀), rolls the physics forward in x and minimises the hybrid cost [](#eq-latent-hybrid-cost) over z₀.
:label: code-design-latent-hybrid4dvar
import einx
import equinox as eqx
import jax
import lineax as lx
import optimistix as optx
import pipekit_cycle as pc
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: Nx = state size; Nz = latent size; D = observations per time;
#   K = rollout steps, so T = K + 1 times in the window


# proposed (D17)
class LatentHybridFourDVar(eqx.Module):
    """Hybrid latent 4DVar: physics forecast in x, control in z."""

    # today (pipekit_cycle)
    latent_map: pc.LatentMap
    # today (pipekit_cycle); M_x (physics)
    forward: pc.ForwardModel
    # today (pipekit_cycle)
    obs_op: pc.ObservationOperator
    # proposed (D17)
    prior: LatentPrior
    obs_cov_op: lx.AbstractLinearOperator
    n_steps: int = eqx.field(static=True, default=10)
    # today (optimistix)
    minimiser: optx.AbstractMinimiser = optx.BFGS(...)
    # today (optimistix)
    minimiser_adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()

    # today (vardax): Batch1D
    def __call__(self, batch: vdx.Batch1D) -> Float[Array, " Nx"]:
        obs_seq = batch.input
        mask_seq = batch.mask
        observe_seq = jax.vmap(self.obs_op)

        def rollout_x(x0: Float[Array, " Nx"]) -> Float[Array, "T Nx"]:
            def step(
                x: Float[Array, " Nx"], _: None
            ) -> tuple[Float[Array, " Nx"], Float[Array, " Nx"]]:
                # today (pipekit_cycle)
                # xₖ₊₁ = M_x(xₖ)
                x_next = self.forward.step(x, self.forward.dt)  # (Nx,) → (Nx,)
                return x_next, x_next

            # x₁, …, x_K
            _, trajectory = jax.lax.scan(step, x0, None, length=self.n_steps)
            # (x₀, x₁, …, x_K)
            return einx.rearrange("x, k x -> (1 + k) x", x0, trajectory)

        def cost_fn(z0: Float[Array, " Nz"], args: None) -> Float[Array, ""]:
            # today (pipekit_cycle)
            # x₀ = ψ(z₀)
            x0 = self.latent_map.decode(z0)  # (Nz,) → (Nx,)
            # x-space rollout
            state_seq = rollout_x(x0)  # (Nx,) → (T, Nx)
            # H(xₖ), k = 0, …, K
            obs_pred = observe_seq(state_seq)  # (T, Nx) → (T, D)
            # Decoder Jacobian appears only at x0 = ψ(z_0); the rest
            # of the chain is the x-space adjoint (see §18.4.3).
            # proposed (D17)
            # ½‖z₀ − z_b‖²_{B_z⁻¹}
            background_term = self.prior.cost(z0)  # → ()
            # proposed (D17)
            # ½ Σₖ ‖yₖ − H(xₖ)‖²_{R⁻¹}
            obs_term = obs_misfit_latent_seq(  # → ()
                obs_pred=obs_pred,
                y_seq=obs_seq,
                mask_seq=mask_seq,
                R_op=self.obs_cov_op,
            )
            # J_H(z₀)
            return background_term + obs_term

        # today (optimistix)
        # z₀* = argmin J_H(z₀), started from z_b
        solution = optx.minimise(
            cost_fn,
            self.minimiser,
            self.prior.z_b,
            adjoint=self.minimiser_adjoint,
        )
        # today (pipekit_cycle)
        # x₀* = ψ(z₀*)
        analysis_x0 = self.latent_map.decode(solution.value)  # (Nz,) → (Nx,)
        analysis_seq = rollout_x(analysis_x0)  # (Nx,) → (T, Nx)
        # x₀*
        return analysis_seq[0]  # (T, Nx) → (Nx,)
```

Note: the hybrid form differentiates through both $\psi$ (once, at
$z_0 \to x_0$) **and** $M_x$ (over the rollout). The
`forward_adjoint` slot from D15 (e.g.
`diffrax.BacksolveAdjoint()`) governs the rollout adjoint;
`minimiser_adjoint` governs the inner solve.

---

## 6  Layer-0 cost primitives

Three new pure functions in `vardax/_src/costs.py`. Names match the
public-API re-exports and the call sites in §5 exactly:

```{code-block} python
:caption: Declares the three latent cost primitives, the single-time cost [](#eq-latent-3dvar-cost), the time-summed misfit of [](#eq-latent-strong4dvar-cost) and the incremental form [](#eq-latent-incremental-cost).
:label: code-design-latent-cost-primitives
import lineax as lx
from jaxtyping import Array, Float

# Shapes: Nz = latent size; D = observations per time;
#   T = K + 1 times in the window


# proposed (D17)
# J(z) = ½‖z − z_b‖²_{B_z⁻¹} + ½‖y − ŷ‖²_{R⁻¹}
def variational_cost_latent(
    z: Float[Array, " Nz"],
    z_b: Float[Array, " Nz"],
    B_z_op: lx.AbstractLinearOperator,
    obs_pred: Float[Array, " D"],
    y: Float[Array, " D"],
    mask: Float[Array, " D"],
    R_op: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    """Single-time latent variational cost.

        J(z) = 1/2 (z - z_b)^T B_z^{-1} (z - z_b)
             + 1/2 (y - obs_pred)^T R^{-1} (y - obs_pred)

    B_z^{-1} is applied via lineax.linear_solve(B_z_op, ·); R^{-1}
    likewise.  obs_pred is whatever the lifted operator produces
    (a y-space array).
    """
    ...


# proposed (D17)
# Σₖ ½‖yₖ − ŷₖ‖²_{R⁻¹} over observed entries
def obs_misfit_latent_seq(
    obs_pred: Float[Array, "T D"],
    y_seq: Float[Array, "T D"],
    mask_seq: Float[Array, "T D"],
    R_op: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    """Time-summed, R-weighted, masked obs misfit.

        sum_k 1/2 (y_k - obs_pred_k)^T R^{-1} (y_k - obs_pred_k)

    Used by the 4DVar latent variants where the prior term is
    accumulated separately via `LatentPrior.cost`.
    """
    ...


# proposed (D17)
# δJ(δz) = ½‖z + δz − z_b‖²_{B_z⁻¹} + ½‖d − H̃′ δz‖²_{R⁻¹}
def latent_incremental_cost(
    dz: Float[Array, " Nz"],
    z_b: Float[Array, " Nz"],
    B_z_op: lx.AbstractLinearOperator,
    innovation: Float[Array, " D"],
    R_op: lx.AbstractLinearOperator,
    tangent_H: Float[Array, "D Nz"],
) -> Float[Array, ""]:
    """Incremental form for Gauss-Newton outer + CG inner."""
    ...
```

Plus a test fixture that's exported from the top-level namespace
(see §9):

```{code-block} python
:caption: Declares the identity `LatentMap` test fixture, with φ = ψ = id, that reduces [](#eq-latent-strong4dvar-cost) to the x-space cost [](#eq-strong4dvar-cost).
:label: code-design-latent-identity-map
import pipekit_cycle as pc


# proposed (D17); returns a today (pipekit_cycle) LatentMap
def identity_latent_map(dim: int) -> pc.LatentMap:
    """phi = psi = identity.  Reduces latent methods to the x-space baseline."""
    ...
```

---

## 7  Posterior — `LatentLaplaceCovariance`

The existing `LaplaceCovariance` lives in `vardax/_src/posterior/`. A
new sibling computes Laplace at the optimum in $\mathcal{Z}$ and
returns either the $z$-space covariance or its pushed-forward image
via $\psi$:

```{code-block} python
:caption: Declares `LatentLaplaceCovariance`, which inverts the latent Hessian at z* and optionally pushes it through ψ′ to x-space, as in [](#eq-latent-laplace-cov).
:label: code-design-latent-laplace
import equinox as eqx
import jax
import lineax as lx
import vardax as vdx
from jaxtyping import Array, Float

# Shapes: Nx = state size; Nz = latent size


# proposed (D17)
class LatentLaplaceCovariance(eqx.Module):
    """Laplace approximation at z*.

    cov_z = (∇^2 J(z*))^{-1}.
    cov_x = ψ'(z*) · cov_z · ψ'(z*)^T  (when ``project=True``).
    """

    project: bool = False

    # today (vardax): Batch1D
    def __call__(
        self,
        model: LatentStrongFourDVar,
        z_star: Float[Array, " Nz"],
        batch: vdx.Batch1D,
    ) -> "_LatentPosterior | _XPosterior":
        def cost_at(z: Float[Array, " Nz"]) -> Float[Array, ""]:
            # proposed (D17)
            # J(z)
            return _cost(model, z, batch)

        # 𝐇_z(z*) = ∇²J(z*)
        hessian_matrix = jax.hessian(cost_at)(z_star)  # (Nz,) → (Nz, Nz)
        hessian = lx.MatrixLinearOperator(  # (Nz, Nz) PSD op
            hessian_matrix, lx.positive_semidefinite_tag
        )

        def solve_hessian(v: Float[Array, " Nz"]) -> Float[Array, " Nz"]:
            # 𝐇_z⁻¹ v, by a solve (the original sketch used jnp.linalg.pinv)
            return lx.linear_solve(hessian, v).value

        # cov_z = 𝐇_z(z*)⁻¹
        cov_z = lx.FunctionLinearOperator(  # (Nz, Nz) PSD op
            solve_hessian, jax.eval_shape(lambda: z_star)
        )
        if not self.project:
            # proposed (D17)
            return _LatentPosterior(
                mean_z=z_star, cov_z=cov_z, decoder=model.latent_map
            )
        # today (pipekit_cycle)
        # ψ′(z*) = ∂ψ/∂z at z*
        decoder_jacobian = lx.JacobianLinearOperator(  # (Nx, Nz) op
            lambda z, args: model.latent_map.decode(z), z_star
        )
        # cov_x = ψ′(z*) cov_z ψ′(z*)ᵀ
        cov_x = decoder_jacobian @ cov_z @ decoder_jacobian.transpose()  # (Nx, Nx) op
        # today (pipekit_cycle)
        # x* = ψ(z*)
        x_star = model.latent_map.decode(z_star)  # (Nz,) → (Nx,)
        # proposed (D17)
        return _XPosterior(mean=x_star, cov=cov_x)
```

The pushed-forward covariance is rank $\le N_z$ — a low-rank object
that `lineax.LowRankUpdate` represents natively.

---

## 8  Training story

Latent variational DA opens three training modes; vardax exposes the
correct loss / step composition for each via `pipekit-train` (per D5,
training loops are example code; vardax ships the steps).

| Train | Frozen | Loss | Notes |
|---|---|---|---|
| **AE only** | — | `prior.cost(phi(x)) + ‖x − psi(phi(x))‖²` | Standard AE pretraining; produces a `LatentMap`. |
| **AE + analysis (end-to-end)** | — | reconstruction loss on `model(batch)` | Backprop through `LatentThreeDVar` / `LatentStrongFourDVar`; the inner solve uses `optimistix.ImplicitAdjoint` so memory is bounded. |
| **Latent dynamics $M_z$** | $\varphi, \psi$ frozen | `Σ_k ‖z_k − M_z^k(z_0)‖²` (one-step or rollout) | Trains the latent dynamics on encoded trajectories. |

A new `VardaxLatentReconLoss` adapter wraps `pipekit_train.Loss` for
the AE + analysis end-to-end case. It is identical to the existing
`VardaxReconLoss` except that it asserts the model exposes `latent_map`
(so we can log AE reconstruction error as a diagnostic).

---

## 9  Updated public API

Additions to `vardax/__init__.py`:

```{code-block} python
:caption: Lists the v0.5 additions to `vardax/__init__.py` for the latent family of [](#eq-latent-3dvar-cost), [](#eq-latent-strong4dvar-cost) and [](#eq-latent-hybrid-cost).
:label: code-design-latent-public-api
# Layer 2: latent models
# proposed (D17)
from vardax._src.models.latent import (
    LatentHybridFourDVar,
    LatentStrongFourDVar,
    LatentThreeDVar,
)

# Layer 1: latent components
# proposed (D17)
from vardax._src.latent import LatentPrior, NeuralLatentForwardModel

# Layer 1: posterior
# proposed (D17)
from vardax._src.posterior.latent import LatentLaplaceCovariance

# Layer 0: cost primitives
# proposed (D17)
from vardax._src.costs import (
    latent_incremental_cost,
    obs_misfit_latent_seq,
    variational_cost_latent,
)

# Layer 0: test fixture (also handy as a baseline check in user code)
# proposed (D17)
from vardax._src.latent import identity_latent_map
```

Re-exports from pipekit-cycle (for the user's convenience):

```{code-block} python
:caption: Re-exports the pipekit-cycle latent protocols and cycle from vardax for convenience, as used in [](#code-design-latent-l96-example).
:label: code-design-latent-reexports
# today (pipekit_cycle)
from pipekit_cycle import LatentMap

# proposed (pipekit-cycle latent.md)
from pipekit_cycle import (
    EncodedForwardModel,
    LatentDACycle,
    LatentDAState,
    LatentForwardModel,
    LiftedObservationOperator,
)
```

No existing exports are removed or renamed (per the v0.4 stability
contract). `BilinAEPrior*`, `MLPAEPrior*`, `ConvAEPrior*` gain the two
new properties (`latent_dim`, `state_signature`) but keep their current
signatures.

---

## 10  Worked example — Lorenz-96, latent 4DVar

```{code-block} python
:caption: Runs latent strong-constraint 4DVar [](#eq-latent-strong4dvar-cost) on Lorenz-96 [](#eq-lorenz96) through `pipekit_cycle.LatentDACycle`, from a pretrained autoencoder and a Matérn B_z.
:label: code-design-latent-l96-example
import gaussx
import jax
import lineax as lx
import pipekit_cycle as pc
import vardax as vdx

# Shapes: Nx = 40 Lorenz-96 cells; Nz = 8 latent dims; D = 40 observations

# 1.  Pretrained AE on L96 trajectories.
# today* (vardax); satisfies pc.LatentMap today, on (B, T, Nx) arrays
# z = φ(x), x = ψ(z)
ae = vdx.BilinAEPrior1D(state_dim=40, latent_dim=8, key=...)

# 2.  Either learn a latent M_z, or wrap the physics one.
# proposed (pipekit-cycle latent.md)
# M_z = φ ∘ M_x ∘ ψ
latent_forward = pc.EncodedForwardModel(latent_map=ae, inner=l96_diffrax_model)

# 3.  Background error covariance in z (Matérn with short scale).
# proposed (gaussx); gaussx 0.0.18 has no gaussx.matern
kernel = gaussx.matern.MaternKernel(nu=1.5, length_scale=0.5)
# proposed (gaussx)
# B_z = [k(zᵢ, zⱼ)]ᵢⱼ
latent_cov = kernel.to_lineax_op(dim=8)  # → (Nz, Nz) PSD op

# proposed (D17)
# J_prior(z₀) = ½‖z₀ − z_b‖²_{B_z⁻¹}
prior = vdx.LatentPrior(latent_map=ae, z_b=z_climatology, B_z_op=latent_cov)

# 4.  Latent strong-4DVar model.
# today (vardax)
# H(x) = m ⊙ x
obs_op = vdx.MaskedIdentity()  # (Nx,) → (D,)
# today* (lineax); IdentityLinearOperator takes a structure, not a size
obs_identity = lx.IdentityLinearOperator(40)
# R = σ² I
obs_cov = obs_identity * sigma_obs**2  # → (D, D) PSD op
# proposed (D17)
# z₀* = argmin J_4D(z₀)
model = vdx.LatentStrongFourDVar(
    latent_map=ae,
    forward=latent_forward,
    obs_op=obs_op,
    prior=prior,
    obs_cov_op=obs_cov,
    n_steps=20,
)

# 5.  Drive it through pipekit-cycle.
# proposed (D17)
analysis_step = model.as_analysis_step()
# proposed (pipekit-cycle latent.md)
cycle = pc.LatentDACycle(
    forward_model=latent_forward,
    latent_map=ae,
    obs_op=obs_op,
    analysis_step=analysis_step,
    obs_source=satellite_iter,
    forecast_space="z",
    update_space="z",
    re_encode_every=10**9,
    n_steps=24,
)

# today* (vardax); encode expects (B, T, Nx)
# z₀ = φ(x₀)
initial_latent = ae.encode(x0)  # (Nx,) → (Nz,)
# proposed (pipekit-cycle latent.md)
cycle_state = pc.LatentDAState(
    t=0.0, cycle_count=0, obs_err_cov=obs_cov, latent_state=initial_latent
)
# proposed (pipekit-cycle latent.md)
analyses, _ = cycle(x0, cycle_state)
```

The identity-AE smoke test (Section 12 below) substitutes
`ae = vdx.identity_latent_map(40)` and confirms that the output
matches the baseline `StrongFourDVar` on the same Lorenz-96 fixture.

---

## 11  Decision D17 — Latent DA as a first-class peer family

(Filed in `decisions.md`; summarised here for completeness.)

> Latent DA is *not* a configuration of `StrongFourDVar` / `ThreeDVar`,
> nor a wrapper around `FourDVarNet`. It is its own family of three
> peer classes — `LatentThreeDVar`, `LatentStrongFourDVar`,
> `LatentHybridFourDVar` — coexisting with the seven peers established
> in D14.
>
> Rationale: the control vector ($z$ vs $x$), the cost dimensionality,
> the Hessian object, and the posterior covariance all live in
> different spaces. Burying that under a `space: Literal["x", "z"]`
> flag would obscure the structural difference exactly the way
> `mode: Literal["strong", "weak"]` obscured strong vs weak 4DVar in
> the pre-v0.4 design.
>
> The peer family pattern from D14 already supports the addition;
> nothing in v0.4 changes.

---

## 12  Acceptance criteria for v0.5

* `LatentThreeDVar`, `LatentStrongFourDVar`, `LatentHybridFourDVar`
  importable from `vardax`; each implements `.as_analysis_step()` and
  passes `isinstance(., pipekit_cycle.AnalysisStep)`.
* Existing `BilinAEPrior*`, `MLPAEPrior*`, `ConvAEPrior*` satisfy
  `pipekit_cycle.LatentMap` (runtime-checkable test).
* `identity_latent_map(N)` smoke test: latent methods reduce to their
  x-space siblings within $1\mathrm{e}{-5}$ on Lorenz-96 with $N_x =
  N_z = 40$.
* Lorenz-96 end-to-end notebook with a real $N_z = 8$ AE; reports
  cost-per-cycle and analysis RMSE vs. the x-space baseline.
* Math reference chapter 18 lands together with the code; references
  the same notation as chapters 5–8.
* No regression in existing `FourDVarNet*` tests; the prior protocol
  changes are additive only.

---

## 13  Out of scope (deferred)

* **Latent WeakFourDVar** — the model-error control vector is harder to
  parameterise in $\mathcal{Z}$; we revisit once the strong-constraint
  version is in users' hands.
* **Latent AmortizedPosterior** — the existing amortised posterior
  already operates in a learned context space; the relationship to
  `LatentMap` is closer to *identification* than *addition*. We will
  refactor in v0.6 once `LatentMap` adoption is settled.
* **Variational autoencoders.** `LatentMap` currently expects a
  deterministic encoder; VAE support requires a `sample_encode`
  protocol method which we defer to v0.2 of the pipekit-cycle latent
  module.
* **Posterior in x via full pushforward.** Computing
  $\mathrm{cov}_x = \psi'\,\mathrm{cov}_z\,{\psi'}^\top$ exactly is
  $O(N_x^2)$; we ship the low-rank representation by default and
  expose `.densify()` only for small problems.

---

## 14  References

1. Peyron, M. et al. (2021). *Latent space data assimilation by using
   deep learning.* QJRMS.
2. Cheng, S. et al. (2023). *Generalised latent assimilation in
   heterogeneous reduced spaces with machine learning surrogates.*
   J. Sci. Comput.
3. Fablet, R. et al. (2021). *Learning variational data assimilation
   models and solvers (4DVarNet).* JAMES.
4. Bolte, J. et al. (2023). *One-step differentiation of iterative
   algorithms.* NeurIPS.
5. vardax math reference: [chapter 18 — Latent Variational DA](../18_latent_da.md).
6. pipekit-cycle foundation:
   `packages/pipekit-cycle/docs/design/latent.md`.
