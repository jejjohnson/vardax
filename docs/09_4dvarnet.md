# 4DVarNet — Learned 4DVar

4DVarNet replaces two pieces of classical 4DVar with learned
counterparts: the Gaussian prior $\|x - x_b\|^2_{B^{-1}}$ becomes a
learned reconstruction $\|x - \varphi_\theta(x)\|^2$, and the inner
gradient-descent solver becomes a learned iteration with a ConvLSTM
gradient modulator. The result is trainable end-to-end on
(input, target) pairs.

The "Net" naming follows Fablet et al. (2021). vardax keeps it as the
name of the *method*, not the family — 4DVarNet is one of seven peer
analysis methods (Decision D14), distinguished by its learned components.
It is not the parent of classical 4DVar.

## Cost function

The variational cost is structurally familiar:

$$
J(x) = \alpha_\text{obs}\,\tfrac{1}{2} \|H(x) - y\|^2_{R^{-1}} +
       \alpha_\text{prior}\, \|x - \varphi_\theta(x)\|^2,
$$

with $\varphi_\theta$ a learned autoencoder. The Gaussian background
term $\|x - x_b\|^2_{B^{-1}}$ is **replaced** (not added). The learned
prior is expressive enough to encode the same information that $B$ +
$x_b$ would, plus structure that classical Gaussian priors can't
represent (multimodality, nonlinear manifolds, conditional
dependencies).

## The learned inner solver

Classical 4DVar minimises $J$ via gradient descent or a quasi-Newton
method. 4DVarNet replaces the inner step with a learned modulator:

$$
x_{k+1} = x_k - \Phi_\phi(\nabla_x J(x_k),\; h_k), \quad k = 0, \ldots, K-1
$$

where $\Phi_\phi$ is a small neural network (ConvLSTM, MLP, or
attention) with parameters $\phi$ and a recurrent hidden state $h_k$.
The network sees the current gradient and decides how to step — it can
learn momentum, adaptive step sizes, or problem-specific
preconditioning that hand-coded solvers don't capture.

After $K$ iterations the analysis is $x^* = x_K$. The whole pipeline —
forward, gradient, modulator, update — is differentiable end-to-end.

## Training

The training objective minimises reconstruction error against
ground-truth states:

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{(y, x_\text{true})}\, \|x^*(\theta, \phi; y) - x_\text{true}\|^2.
$$

The training gradient $\nabla_{\theta, \phi} \mathcal{L}$ flows through
the inner solver. This is where the adjoint choice (Decision D15)
matters.

### Adjoint choices for the inner solver

Vardax exposes `solver_adjoint: optimistix.AbstractAdjoint` as a
constructor slot. Three viable choices:

| Adjoint | Memory | Convergence requirement | Notes |
|---|---|---|---|
| `optimistix.RecursiveCheckpointAdjoint()` | $O(K)$ checkpoints | None | Standard backprop with recursive checkpointing |
| `vardax.adjoints.OneStepAdjoint()` | $O(1)$ | None | Bolte et al. 2023; only the last step differentiable |
| `optimistix.ImplicitAdjoint()` | $O(1)$ | Yes | IFT-based at the fixed point; exact at convergence |

**`RecursiveCheckpointAdjoint`** — the default. Standard
backpropagation through all $K$ unrolled steps, with recursive
checkpointing to bound memory. Works always, costs $O(K)$ checkpoints.

**`OneStepAdjoint`** — runs $K-1$ steps with `jax.lax.stop_gradient`
applied to the trajectory, then a single differentiable step. The
training gradient picks up only the last iteration's contribution.
Bolte et al. (2023) prove this is exact at convergence; in practice it
works well for converged or near-converged inner solvers. Memory cost
$O(1)$. The right choice for $K \ge 20$ when memory is constrained.

vardax ships `OneStepAdjoint` in `vardax._src.adjoints.one_step` with
the goal of upstreaming to `optimistix` once stable (Decision D6).

**`ImplicitAdjoint`** — uses the implicit function theorem at the
fixed point of the inner iteration:

$$
\frac{d x^*}{d \theta} = -\Big(I - \frac{\partial f}{\partial x}\Big)^{-1} \frac{\partial f}{\partial \theta},
$$

where $f$ is the inner-step map. Memory $O(1)$, gradient exact at
convergence. Requires the inner solver to actually converge; if $K$ is
too small, the IFT gradient is wrong. Use when the inner iteration is
known to be a contraction.

## Implementation in vardax

```python
import equinox as eqx
import optax
from vardax.models import FourDVarNet
from vardax import SolverConfig
from vardax.adjoints import OneStepAdjoint
from vardax.priors import ConvAEPrior
from vardax.obs_operators import MaskedIdentity
from vardax.grad_mod import ConvLSTMGradMod2D
from vardax.training import train_step

model = FourDVarNet(
    prior=ConvAEPrior(encoder=enc_2d, decoder=dec_2d),
    obs_op=MaskedIdentity(),
    grad_mod=ConvLSTMGradMod2D(hidden_dim=64),
    config=SolverConfig(n_steps=15, alpha=0.2, prior_weight=1.0),
    solver_adjoint=OneStepAdjoint(),                # O(1) memory training
)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def step(model, opt_state, batch):
    model, opt_state, loss = train_step(model, batch, optimizer, opt_state)
    return model, opt_state, loss

for epoch in range(100):
    for batch in dataloader:
        model, opt_state, loss = step(model, opt_state, batch)
```

After training, `model.as_analysis_step()` exposes the operational
interface for `pipekit_cycle.DACycle`.

## The prior family

`vardax.priors` ships:

| Prior | Form | When |
|---|---|---|
| `BilinAEPrior1D/2D/2DMultivar` | $\varphi(x) = \text{decode}(\text{ReLU}(Ax) \odot \tanh(Bx))$ | Standard 4DVarNet starting point |
| `ConvAEPrior1D` | Periodic 1D convolutional AE | Periodic 1D fields (Lorenz-96) |
| `MLPAEPrior1D` | Dense MLP AE | Low-dimensional states (Lorenz-63) |
| `IdentityPrior` | $\varphi(x) = x$ | Pure obs-driven baseline; recovers classical 4DVar with `IdentityGradMod` |
| `DynamicalPrior` | $\varphi(x) = M^n(x)$ via somax / plumax forward | Physics-informed 4DVarNet |

`DynamicalPrior` is interesting: it wraps any
`pipekit_cycle.ForwardModel` as the variational prior, so the regulariser
becomes "consistency with the physical dynamics" rather than
"reconstruction by an autoencoder". The gradient back through the
dynamics uses the `forward_adjoint` slot (chapter 3).

## The gradient modulator family

`vardax.grad_mod` ships:

| Modulator | Form |
|---|---|
| `ConvLSTMGradMod1D/2D` | ConvLSTM over spatial dims, recurrent over solver iterations |
| `MLPGradMod` | Dense MLP, dimension-agnostic via flatten |
| `AttentionGradMod` | Self-attention over spatial axis (planned) |
| `IdentityGradMod` | $\text{update} = -\alpha \cdot \text{grad}$ — classical fixed-step descent |

`FourDVarNet(IdentityPrior, IdentityGradMod, n_steps=large)` is
mathematically equivalent to gradient descent on the variational cost
— it's the classical 4DVar baseline, useful for sanity-checking
agreement with the linear-Gaussian baseline (Decision D14 invariant).

## Linear-Gaussian agreement

With `IdentityPrior` (so $\varphi(x) = x$ and the prior cost vanishes
to zero) and `IdentityGradMod(alpha=small)` and `n_steps=large`,
`FourDVarNet` performs many small gradient-descent steps on the
observation cost alone — and converges to the same answer as
`OptimalInterpolation` in the linear-Gaussian limit:

```python
def test_fourdvarnet_recovers_oi():
    oi = OptimalInterpolation(linear_H, x_b, B_op, R_op)
    net = FourDVarNet(
        prior=IdentityPrior(),
        obs_op=linear_H,
        grad_mod=IdentityGradMod(alpha=0.05),
        config=SolverConfig(n_steps=200, prior_weight=0.0),
    )
    batch = make_linear_gaussian_batch()
    assert jnp.allclose(oi(batch), net(batch), atol=1e-2)
```

This is the floor. A trained `FourDVarNet` with a non-trivial prior
and grad modulator typically does better in nonlinear regimes — but
the floor confirms the implementation is correct.

## When to use `FourDVarNet`

Use `FourDVarNet` when:

- You have a training set of (observation, ground-truth) pairs
- The classical $B$ is hard to specify or known to be inadequate
- The regime is data-rich and learning-friendly (smooth manifolds, no
  catastrophic distribution shift between train and test)

Don't use `FourDVarNet` when:

- You have no training pairs — go with classical methods (chapters
  4–8)
- The training distribution doesn't match deployment — `FourDVarNet`
  extrapolates poorly outside its training manifold
- You need defensible posterior covariance — the Laplace approximation
  around a learned-prior MAP can be miscalibrated; `IncrementalFourDVar`
  with a classical Matérn prior gives a more honest posterior

## Posterior

The Laplace approximation works the same way as for classical methods:

```python
from vardax.posterior import LaplaceCovariance

posterior = LaplaceCovariance()(model(batch), model.as_analysis_step(), batch)
```

The Hessian assembled at MAP uses the learned prior's curvature, which
may be poorly conditioned away from the training manifold. Validate
the posterior via the six-step cycle gates (chapter 14): does the
posterior diagonal agree with the spread of an ensemble of analyses
from perturbed inputs?

## A note on history

4DVarNet originated as a research line (Fablet et al. 2021, 2023) that
demonstrated end-to-end learning of variational DA on satellite SSH.
The original codebase (CIA-Oceanix/4dvarnet-starter) is PyTorch and
ocean-specific. Vardax brings the same algorithm into JAX, peers it
with the classical DA hierarchy, and connects it to the broader
ecosystem (somax / plumax for physics, gaussx for structured priors,
filterax for hybrid EnVar, pipekit-cycle for orchestration).

In the v0.4 design, 4DVarNet is **one method among seven**, not the
canonical case. It earns its place by being a useful learned variant
of strong-constraint 4DVar — but the classical methods are the
foundation, not legacy.

## See also

- Chapter 6 — `StrongFourDVar` (classical counterpart)
- Chapter 8 — `IncrementalFourDVar` (operational baseline)
- Chapter 12 — adjoint composition (the inner-solver adjoint choice)
- Chapter 14 — six-step cycle (validation methodology for learned methods)
- Design doc: [`design/decisions.md#d14`](design/decisions.md#d14-da-hierarchy-as-horizontal-peer-classes)

## References

- Fablet, R., Chapron, B., Drumetz, L., Mémin, E., Pannekoucke, O., &
  Rousseau, F. (2021). *Learning variational data assimilation models
  and solvers.* JAMES 13(10).
- Fablet, R., Febvre, Q., & Chapron, B. (2023). *Multimodal 4DVarNets
  for the reconstruction of sea surface dynamics from NADIR and
  wide-swath altimetry.* IEEE TGRS 61.
- Bolte, J., Pauwels, E., & Vaiter, S. (2023). *One-step differentiation
  of iterative algorithms.* NeurIPS.
