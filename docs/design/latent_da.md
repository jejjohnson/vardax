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

```python
from pipekit_cycle import (
    LatentMap,
    LatentForwardModel,
    LiftedObservationOperator,
    EncodedForwardModel,
)
```

Three vardax adapters wire existing components to the new protocols:

### 4.1  `LatentPrior`

```python
class LatentPrior(eqx.Module):
    """Variational prior in latent space — wraps a LatentMap with B_z.

    Used as the prior term in latent variational costs:

        J_prior(z) = 1/2 (z - z_b)^T B_z^{-1} (z - z_b).
    """

    latent_map: LatentMap
    z_b: Float[Array, " Nz"]
    B_z_op: lineax.AbstractLinearOperator    # B_z itself, NOT its inverse

    def cost(self, z):
        # 1/2 (z - z_b)^T B_z^{-1} (z - z_b) — the linear_solve applies
        # B_z^{-1} to the residual.  B_z_op is the covariance operator;
        # a precision operator should be wrapped in
        # `lineax.TaggedLinearOperator(B_z_inv, lx.tags.symmetric)` and
        # adapted to a `lineax.matmul`-based cost instead.
        d = z - self.z_b
        return 0.5 * jnp.dot(d, lx.linear_solve(self.B_z_op, d).value)

    def encode(self, x): return self.latent_map.encode(x)
    def decode(self, z): return self.latent_map.decode(z)
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

```python
class MLPAEPrior1D(eqx.Module):
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    _latent_dim: int = eqx.field(static=True)
    _state_signature: Any = eqx.field(static=True, default=None)

    # NEW — extracted halves of the existing __call__.
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)

    # Existing — preserved bit-for-bit.
    def __call__(self, x): return self.decode(self.encode(x))

    # NEW — make the AE satisfy pipekit_cycle.LatentMap structurally.
    @property
    def latent_dim(self): return self._latent_dim
    @property
    def state_signature(self): return self._state_signature
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

```python
class NeuralLatentForwardModel(eqx.Module):
    """Wraps an eqx.Module mapping z_k → z_{k+1}.

    Satisfies pipekit_cycle.LatentForwardModel.  Users supply the
    learned dynamics module via composition.
    """

    net: eqx.Module                       # any z → z module
    dt: float = 1.0
    latent_signature: Any = eqx.field(static=True, default=None)

    def step(self, z, dt):
        return self.net(z)
```

For users who only have an x-space `ForwardModel`, the
`pipekit_cycle.EncodedForwardModel` helper provides the AE round-trip
without learning a separate $M_z$.

---

## 5  Layer-2 models

Three new sibling classes live in `vardax/_src/models/`:

### 5.1  `LatentThreeDVar`

```python
class LatentThreeDVar(eqx.Module):
    """Latent 3DVar — minimise J_3D(z) over z, decode to x."""

    latent_map: LatentMap
    obs_op: ObservationOperator                       # x-space
    prior: LatentPrior
    obs_cov_op: lineax.AbstractLinearOperator
    minimiser: optimistix.AbstractMinimiser = BFGS(rtol=1e-6, atol=1e-6)
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()

    def __call__(self, batch: Batch1D | Batch2D) -> Array:
        y, mask = batch.input, batch.mask
        lifted_H = LiftedObservationOperator(
            decoder=self.latent_map, inner=self.obs_op,
        )

        def cost_fn(z, _):
            obs_pred = lifted_H(z)
            return variational_cost_latent(
                z=z, z_b=self.prior.z_b, B_z_op=self.prior.B_z_op,
                obs_pred=obs_pred, y=y, mask=mask, R_op=self.obs_cov_op,
            )

        sol = optimistix.minimise(
            cost_fn, self.minimiser, self.prior.z_b,
            adjoint=self.minimiser_adjoint,
        )
        return self.latent_map.decode(sol.value)

    def as_analysis_step(self):
        return _LatentThreeDVarAnalysisStep(self)
```

`_LatentThreeDVarAnalysisStep` is the same five-line adapter used by
the existing seven peers; it adds the canonical
`(forecast, obs, *, obs_op, obs_err_cov) → analysis` signature so
`pipekit_cycle.LatentDACycle` can drive it.

### 5.2  `LatentStrongFourDVar`

```python
class LatentStrongFourDVar(eqx.Module):
    """Latent strong-constraint 4DVar — minimise J_4D(z_0) over z_0."""

    latent_map: LatentMap
    forward: LatentForwardModel                       # M_z (or EncodedForwardModel)
    obs_op: ObservationOperator                       # x-space
    prior: LatentPrior
    obs_cov_op: lineax.AbstractLinearOperator
    n_steps: int = eqx.field(static=True, default=10)

    minimiser: optimistix.AbstractMinimiser = BFGS(...)
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()
    forward_adjoint: Any = None                       # if M_z uses diffrax

    def __call__(self, batch):
        ys, masks = batch.input, batch.mask
        lifted_H = LiftedObservationOperator(
            decoder=self.latent_map, inner=self.obs_op,
        )

        def rollout(z0):
            def step(z, _):
                z_new = self.forward.step(z, self.forward.dt)
                return z_new, z_new
            _, traj = jax.lax.scan(step, z0, None, length=self.n_steps)
            return jnp.concatenate([z0[None, :], traj], axis=0)

        def cost_fn(z0, _):
            zs = rollout(z0)
            obs_pred = jax.vmap(lifted_H)(zs)            # (K+1, ...)
            return (self.prior.cost(z0)
                    + obs_misfit_latent_seq(
                        obs_pred=obs_pred, y_seq=ys, mask_seq=masks,
                        R_op=self.obs_cov_op,
                    ))

        sol = optimistix.minimise(
            cost_fn, self.minimiser, self.prior.z_b,
            adjoint=self.minimiser_adjoint,
        )
        return self.latent_map.decode(sol.value)
```

### 5.3  `LatentHybridFourDVar`

```python
class LatentHybridFourDVar(eqx.Module):
    """Hybrid latent 4DVar — physics forecast in x, control in z."""

    latent_map: LatentMap
    forward: ForwardModel                            # M_x  (physics)
    obs_op: ObservationOperator
    prior: LatentPrior
    obs_cov_op: lineax.AbstractLinearOperator
    n_steps: int = eqx.field(static=True, default=10)
    minimiser: optimistix.AbstractMinimiser = BFGS(...)
    minimiser_adjoint: optimistix.AbstractAdjoint = ImplicitAdjoint()

    def __call__(self, batch):
        ys, masks = batch.input, batch.mask

        def rollout_x(x0):
            def step(x, _):
                x_new = self.forward.step(x, self.forward.dt)
                return x_new, x_new
            _, traj = jax.lax.scan(step, x0, None, length=self.n_steps)
            return jnp.concatenate([x0[None, :], traj], axis=0)

        def cost_fn(z0, _):
            x0 = self.latent_map.decode(z0)
            xs = rollout_x(x0)                       # x-space rollout
            obs_pred = jax.vmap(self.obs_op)(xs)
            # Decoder Jacobian appears only at x0 = ψ(z_0); the rest
            # of the chain is the x-space adjoint (see §18.4.3).
            return (self.prior.cost(z0)
                    + obs_misfit_latent_seq(
                        obs_pred=obs_pred, y_seq=ys, mask_seq=masks,
                        R_op=self.obs_cov_op,
                    ))

        sol = optimistix.minimise(
            cost_fn, self.minimiser, self.prior.z_b,
            adjoint=self.minimiser_adjoint,
        )
        return rollout_x(self.latent_map.decode(sol.value))[0]   # x_0^*
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

```python
def variational_cost_latent(
    z: Array, z_b: Array, B_z_op,
    obs_pred: Array, y: Array, mask: Array, R_op,
) -> Array:
    """Single-time latent variational cost.

        J(z) = 1/2 (z - z_b)^T B_z^{-1} (z - z_b)
             + 1/2 (y - obs_pred)^T R^{-1} (y - obs_pred)

    B_z^{-1} is applied via lineax.linear_solve(B_z_op, ·); R^{-1}
    likewise.  obs_pred is whatever the lifted operator produces
    (a y-space array).
    """
    ...

def obs_misfit_latent_seq(
    obs_pred: Array, y_seq: Array, mask_seq: Array, R_op,
) -> Array:
    """Time-summed, R-weighted, masked obs misfit.

        sum_k 1/2 (y_k - obs_pred_k)^T R^{-1} (y_k - obs_pred_k)

    Used by the 4DVar latent variants where the prior term is
    accumulated separately via `LatentPrior.cost`.
    """
    ...

def latent_incremental_cost(
    dz: Array, z_b: Array, B_z_op,
    innovation: Array, R_op, tangent_H: Array,
) -> Array:
    """Incremental form for Gauss-Newton outer + CG inner."""
    ...
```

Plus a test fixture that's exported from the top-level namespace
(see §9):

```python
def identity_latent_map(dim: int) -> LatentMap:
    """phi = psi = identity.  Reduces latent methods to the x-space baseline."""
    ...
```

---

## 7  Posterior — `LatentLaplaceCovariance`

The existing `LaplaceCovariance` lives in `vardax/_src/posterior/`. A
new sibling computes Laplace at the optimum in $\mathcal{Z}$ and
returns either the $z$-space covariance or its pushed-forward image
via $\psi$:

```python
class LatentLaplaceCovariance(eqx.Module):
    """Laplace approximation at z*.

    cov_z = (∇^2 J(z*))^{-1}.
    cov_x = ψ'(z*) · cov_z · ψ'(z*)^T  (when ``project=True``).
    """

    project: bool = False

    def __call__(self, model, z_star, batch):
        H_z = jax.hessian(lambda z: _cost(model, z, batch))(z_star)
        cov_z = jnp.linalg.pinv(H_z)
        if not self.project:
            return _LatentPosterior(mean_z=z_star, cov_z=cov_z, decoder=model.latent_map)
        psi_lin = jax.jacfwd(model.latent_map.decode)(z_star)
        cov_x = psi_lin @ cov_z @ psi_lin.T
        x_star = model.latent_map.decode(z_star)
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

```python
# Layer 2 — latent models
from vardax._src.models.latent import (
    LatentThreeDVar,
    LatentStrongFourDVar,
    LatentHybridFourDVar,
)

# Layer 1 — latent components
from vardax._src.latent import (
    LatentPrior,
    NeuralLatentForwardModel,
)

# Layer 1 — posterior
from vardax._src.posterior.latent import LatentLaplaceCovariance

# Layer 0 — cost primitives
from vardax._src.costs import (
    variational_cost_latent,
    obs_misfit_latent_seq,
    latent_incremental_cost,
)

# Layer 0 — test fixture (also handy as a baseline check in user code)
from vardax._src.latent import identity_latent_map
```

Re-exports from pipekit-cycle (for the user's convenience):

```python
from pipekit_cycle import (
    LatentMap, LatentForwardModel,
    LiftedObservationOperator, EncodedForwardModel,
    LatentDACycle, LatentDAState,
)
```

No existing exports are removed or renamed (per the v0.4 stability
contract). `BilinAEPrior*`, `MLPAEPrior*`, `ConvAEPrior*` gain the two
new properties (`latent_dim`, `state_signature`) but keep their current
signatures.

---

## 10  Worked example — Lorenz-96, latent 4DVar

```python
import jax
import vardax as vdx
import pipekit_cycle as pc
import lineax as lx
from gaussx.matern import MaternKernel

# 1.  Pretrained AE on L96 trajectories.
ae = vdx.BilinAEPrior1D(state_dim=40, latent_dim=8, ...)   # already satisfies LatentMap

# 2.  Either learn a latent M_z, or wrap the physics one.
M_z = pc.EncodedForwardModel(latent_map=ae, inner=l96_diffrax_model)

# 3.  Background error covariance in z (Matérn with short scale).
B_z = MaternKernel(nu=1.5, length_scale=0.5).to_lineax_op(dim=8)

prior = vdx.LatentPrior(latent_map=ae, z_b=z_climatology, B_z_op=B_z)

# 4.  Latent strong-4DVar model.
model = vdx.LatentStrongFourDVar(
    latent_map=ae,
    forward=M_z,
    obs_op=vdx.MaskedIdentity(),
    prior=prior,
    obs_cov_op=lx.IdentityLinearOperator(40) * sigma_obs**2,
    n_steps=20,
)

# 5.  Drive it through pipekit-cycle.
cycle = pc.LatentDACycle(
    forward_model=M_z,
    latent_map=ae,
    obs_op=vdx.MaskedIdentity(),
    analysis_step=model.as_analysis_step(),
    obs_source=satellite_iter,
    forecast_space="z", update_space="z", re_encode_every=10**9,
    n_steps=24,
)

state0 = pc.LatentDAState(t=0.0, cycle_count=0,
                          obs_err_cov=R, latent_state=ae.encode(x0))
analyses, _ = cycle(x0, state0)
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
