---
numbering:
  enumerator: "1.%s"
---

# Problem Setting

Data assimilation is the problem of estimating the state of a dynamical system given partial, noisy observations and prior knowledge. It sits at the intersection of Bayesian inference, optimisation and numerical analysis; the prior comes from a physical model (or a learned one), the likelihood comes from instruments, and the posterior is what we want.

## The Bayesian picture

Let $\vect{x} \in \mathbb{R}^{D_x}$ denote the state and $\vect{y} \in \mathbb{R}^{D_y}$ the observations. Bayes' theorem gives the posterior

$$p(\vect{x} \mid \vect{y}) \propto p(\vect{y} \mid \vect{x}) p(\vect{x})$$ (eq-bayes)

which has three ingredients:

- **Prior $p(\vect{x})$.** What we know about the state before seeing the data. Most commonly it is Gaussian, $\vect{x} \sim \mathcal{N}(\gvect{\mu}_x, \gmat{\Sigma}_x)$ ([](#eq-oi-prior)), with prior mean $\gvect{\mu}_x \in \mathbb{R}^{D_x}$ and prior covariance $\gmat{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}$. Data assimilation papers call them the background $\vect{x}_b$ and the background-error covariance $\mat{B}$. In `FourDVarNet` the prior is replaced by a learned autoencoder; in `AmortizedPosterior` it is absorbed into the trained head.
- **Likelihood $p(\vect{y} \mid \vect{x})$.** How observations relate to the state. Chapter 2 models them as $h(\vect{x})$ plus Gaussian noise with covariance $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$ ([](#eq-obs-model)), so $p(\vect{y} \mid \vect{x}) = \mathcal{N}(\vect{y}; h(\vect{x}), \gmat{\Sigma}_\varepsilon)$, where the observation operator $h : \mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ maps state space to observation space. Its linear form or Jacobian is $\mat{H} \in \mathbb{R}^{D_y \times D_x}$.
- **Posterior $p(\vect{x} \mid \vect{y})$.** What we infer about the state given the data. For a Gaussian prior and a Gaussian likelihood with linear $h$ it is Gaussian in closed form ([](#eq-oi-posterior), chapter 4); otherwise we approximate it.

The maximum a posteriori (MAP) estimator minimises the negative log-posterior,

$$\vect{x}^\star = \underset{\vect{x}}{\op{argmin}} \left\{-\log p(\vect{y} \mid \vect{x}) - \log p(\vect{x})\right\}$$ (eq-map)

with $\vect{x}^\star \in \mathbb{R}^{D_x}$. With the Gaussian prior and likelihood above the bracket is, up to a constant, the 3DVar cost $\mathcal{J}(\vect{x})$ of [](#eq-3dvar-cost), chapter 5. When the state evolves through dynamics $f$, we add a time index and a forward rollout, which gives 4DVar (chapters 6 to 8).

## When dynamics enter: the 4D problem

Consider a state $\vect{x}_0$ at the start of an assimilation window and observations $\vect{y}_t$ at times $t = 0, 1, \ldots, T$, so the window has $T$ steps and $T + 1$ times. A dynamical model $f : \mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ propagates $\vect{x}_0$ forward, and its $t$-fold composition $f_t$ gives

$$\vect{x}_t = f_t(\vect{x}_0)$$ (eq-rollout)

where $\vect{x}_t \in \mathbb{R}^{D_x}$ is the state at time $t$ and $\vect{y}_t \in \mathbb{R}^{D_y}$ the observations at time $t$.

The 4D variational cost adds an observation term for every time to the background term. In strong-constraint 4DVar (chapter 6) the dynamics are exact, the control variable is $\vect{x}_0$ alone and the cost is [](#eq-strong4dvar-cost). In weak-constraint 4DVar (chapter 7) each step carries a model error $\gvect{\eta}_t \in \mathbb{R}^{D_x}$ with covariance $\gmat{\Sigma}_{\eta,t} \in \mathbb{R}^{D_x \times D_x}$, the cost [](#eq-weak4dvar-cost) gains a model-error term, and the control becomes the augmented vector $(\vect{x}_0, \gvect{\eta}_1, \ldots, \gvect{\eta}_T)$.

In the linear-Gaussian limit (linear $h_t$, linear $f_t$, Gaussian $\gmat{\Sigma}_x$,
$\gmat{\Sigma}_{\varepsilon,t}$, $\gmat{\Sigma}_{\eta,t}$) the posterior is Gaussian in closed form. In all other
cases we minimise $\mathcal{J}$ iteratively. The cost itself is the same; the
solver is where the methods diverge.

## The seven analysis methods

vardax implements seven peer methods, each specialising the general
problem above:

| Method | Setting | Solver |
|---|---|---|
| **`OptimalInterpolation`** | Linear $h$, Gaussian $\gmat{\Sigma}_x$, $\gmat{\Sigma}_\varepsilon$; $T = 0$ | Closed-form |
| **`ThreeDVar`** | Nonlinear $h$; $T = 0$ | `optimistix` minimiser |
| **`StrongFourDVar`** | Multi-time; dynamics treated as exact | `optimistix` minimiser + `diffrax` adjoint |
| **`WeakFourDVar`** | Multi-time; model error active | Same, augmented control |
| **`IncrementalFourDVar`** | Operational fast path of `StrongFourDVar` | GN outer + CG inner |
| **`FourDVarNet`** | Learned prior + learned grad modulator | Learned inner iteration |
| **`AmortizedPosterior`** | Direct $q(\vect{x} \mid \vect{y}; \gvect{\nu})$ head | Single forward pass |

`ReducedBasisFourDVar` is `StrongFourDVar` with the control moved into a reduced basis, $\vect{x}_0 = \gvect{\mu}_x + \gmat{\Phi} \vect{z}$, where $\gmat{\Phi} \in \mathbb{R}^{D_x \times D_z}$ is the basis and $\vect{z} \in \mathbb{R}^{D_z}$ its coefficients (see [chapter 23](23_reduced_order.md)). It was called `Incremental4DVar` before 0.1.11; that name is deprecated.

They are siblings, not subclasses, and each one is appropriate to a different regime.

## The role of the forward model

The dynamics $f$ and the observation operator $h_t$ are the **physics**. vardax does not own them; `somax` (geophysics), `plumax` (atmospheric transport and methane) or any user code can supply them, as long as they satisfy `pipekit_cycle.ForwardModel` and `pipekit_cycle.ObservationOperator` respectively.

This means the same analysis class accepts any forward. A
`StrongFourDVar` doesn't know whether $f$ is a shallow-water solver
from `somax`, a Lagrangian particle simulator from `plumax`, or a
neural emulator trained from physics outputs. The interface is fixed;
the implementation is a slot.

This separation is what makes vardax composable with the broader
JAX ecosystem (`diffrax`, `optimistix`, `lineax`, `gaussx`,
`pipekit-cycle`) without bespoke per-method wiring.

## Computational ingredients

The four computations that recur in every DA method:

1. **Forward model rollout** [](#eq-rollout). ODE integration via `diffrax`; vardax composes it but does not implement it.
2. **Tangent-linear and adjoint** $\mat{F}_t$, $\mat{F}_t^\top$, the Jacobian of $f_t$ and its transpose. Computed via `diffrax.AbstractAdjoint` (recursive checkpointing, continuous backsolve or forward sensitivity; see chapter 12).
3. **Observation operator** $h_t(\vect{x}_t)$ and its tangent linear $\mat{H}_t$.
   See chapter 2 (`linearize()` contract via `lineax`).
4. **Inner minimisation** $\op{argmin}_{\vect{x}} \mathcal{J}(\vect{x})$. Via
   `optimistix.AbstractMinimiser` (Gauss-Newton, BFGS, NonlinearCG,
   fixed-point iteration). For incremental 4DVar, $\op{argmin}$ is done by
   CG on the linearised quadratic subproblem.

Each computation is owned by an upstream library. vardax composes them
into DA algorithms; it does not reimplement them.

## Posterior, not point estimate

A MAP $\vect{x}^\star$ is the start of the answer, not the end. Real DA produces a posterior (mean, covariance and provenance) that downstream consumers (population models, alert systems, decision pipelines) need.

vardax exposes a `Posterior(mean, cov, samples, provenance)` container
returned by every analysis. Approximations include the Laplace
(`LaplaceCovariance`) at MAP, the Gauss-Newton Hessian inversion
(`GaussNewtonHessian`, free for `IncrementalFourDVar`), the ensemble
covariance (`EnsembleCovariance` via `filterax`), or direct sampling
from amortized heads. See chapter 13.

For the closed-form case (`OptimalInterpolation`) the posterior is exact, with the covariance [](#eq-blue-cov) derived in chapter 4.

## Reading order

Chapters 1 to 3 establish the shared foundation: this chapter (the problem), chapter 2 (the observation model) and chapter 3 (the dynamical model and its adjoint).

Chapters 4 to 10 cover one analysis method each, in the order of the table above, from `OptimalInterpolation` in chapter 4 to `AmortizedPosterior` in chapter 10, so in increasing order of sophistication.

Chapters 11 to 14 cover cross-cutting concerns: observation operators (the averaging-kernel and multi-instrument family), adjoint composition via `optimistix` and `diffrax`, posterior covariance, and the six-step research-to-operations methodology.

Chapters 15 to 17 are end-to-end examples on Lorenz, SSH and methane.

The companion design docs in [`design/`](design/README.md) document the API
contracts and architectural decisions behind these methods.
