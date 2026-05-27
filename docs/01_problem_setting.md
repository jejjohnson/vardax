# Problem Setting

Data assimilation is the problem of estimating the state of a dynamical
system given partial, noisy observations and prior knowledge. It sits
at the intersection of Bayesian inference, optimisation, and numerical
analysis: the prior comes from a physical model (or a learned one), the
likelihood comes from instruments, and the posterior is what we want.

## The Bayesian picture

Let $x \in \mathbb{R}^n$ denote the state and $y \in \mathbb{R}^m$ the
observations. Bayes' theorem gives the posterior

$$
p(x \mid y) \propto p(y \mid x)\, p(x).
$$

Three ingredients:

- **Prior $p(x)$.** What we know about the state before seeing the
  data. Most commonly Gaussian: $x \sim \mathcal{N}(x_b, B)$, with
  background mean $x_b$ and background-error covariance $B$. In
  `FourDVarNet` the prior is replaced by a learned autoencoder; in
  `AmortizedPosterior` it is absorbed into the trained head.
- **Likelihood $p(y \mid x)$.** How observations relate to the state.
  We model $y = H(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0,
  R)$ giving $p(y \mid x) = \mathcal{N}(y; H(x), R)$. The observation
  operator $H$ is the projection from state space to observation space
  (see chapter 2).
- **Posterior $p(x \mid y)$.** What we infer about the state given the
  data. For Gaussian prior and Gaussian likelihood with linear $H$,
  this is Gaussian in closed form — that's chapter 4. Otherwise we
  approximate.

The maximum a posteriori (MAP) estimator minimises the negative
log-posterior:

$$
x^* = \underset{x}{\arg\min}\; \big\{-\log p(y \mid x) - \log p(x)\big\}
    = \underset{x}{\arg\min}\;
        \tfrac{1}{2} \|x - x_b\|^2_{B^{-1}} + \tfrac{1}{2} \|y - H(x)\|^2_{R^{-1}}.
$$

This is the 3DVar cost (chapter 5). When the state evolves through
dynamics $M_t$, we add a time index and a forward rollout — that's
4DVar (chapters 6–8).

## When dynamics enter — the 4D problem

Consider a state $x_0$ at the start of an assimilation window and
observations $y_t$ at times $t = 0, 1, \ldots, T$. A dynamical model
$M_t$ propagates $x_0$ forward:

$$
x_t = M_t(x_0).
$$

The 4D variational cost combines the background term, all observation
terms across time, and (optionally) a model-error term:

$$
J(x_0, \boldsymbol{\eta}) =
  \underbrace{\tfrac{1}{2} \|x_0 - x_b\|^2_{B^{-1}}}_{\text{background}} +
  \underbrace{\tfrac{1}{2} \sum_{t=0}^{T} \|y_t - H_t(M_t(x_0; \boldsymbol{\eta}))\|^2_{R_t^{-1}}}_{\text{observations}} +
  \underbrace{\tfrac{1}{2} \sum_{t=1}^{T} \|\eta_t\|^2_{Q_t^{-1}}}_{\text{model error (weak constraint only)}}.
$$

The model-error term is absent in strong-constraint 4DVar (chapter 6,
control variable $x_0$ alone) and present in weak-constraint 4DVar
(chapter 7, augmented control vector $(x_0, \eta_1, \ldots, \eta_T)$).

In the linear-Gaussian limit (linear $H_t$, linear $M_t$, Gaussian $B$,
$R_t$, $Q_t$) the posterior is Gaussian in closed form. In all other
cases we minimise $J$ iteratively. The cost itself is the same; the
solver is where the methods diverge.

## The seven analysis methods

vardax implements seven peer methods, each specialising the general
problem above:

| Method | Setting | Solver |
|---|---|---|
| **`OptimalInterpolation`** | Linear $H$, Gaussian $B$, $R$; $T = 0$ | Closed-form |
| **`ThreeDVar`** | Nonlinear $H$; $T = 0$ | `optimistix` minimiser |
| **`StrongFourDVar`** | Multi-time; dynamics treated as exact | `optimistix` minimiser + `diffrax` adjoint |
| **`WeakFourDVar`** | Multi-time; model error active | Same, augmented control |
| **`IncrementalFourDVar`** | Operational fast path of `StrongFourDVar` | GN outer + CG inner + CVT |
| **`FourDVarNet`** | Learned prior + learned grad modulator | Learned inner iteration |
| **`AmortizedPosterior`** | Direct $q_\phi(x \mid y)$ head | Single forward pass |

They are siblings, not subclasses. Each is appropriate to a different
regime. Chapters 4–10 give one method per chapter.

## The role of the forward model

The dynamics $M_t$ and the observation operator $H_t$ are the
**physics**. vardax does not own them — `somax` (geophysics), `plumax`
(atmospheric transport / methane), or any user code can supply them as
long as the operator satisfies `pipekit_cycle.ForwardModel` and
`pipekit_cycle.ObservationOperator` respectively.

This means the same analysis class accepts any forward. A
`StrongFourDVar` doesn't know whether $M_t$ is a shallow-water solver
from `somax`, a Lagrangian particle simulator from `plumax`, or a
neural emulator trained from physics outputs. The interface is fixed;
the implementation is a slot.

This separation is what makes vardax composable with the broader
JAX ecosystem (`diffrax`, `optimistix`, `lineax`, `gaussx`,
`pipekit-cycle`) without bespoke per-method wiring.

## Computational ingredients

The four computations that recur in every DA method:

1. **Forward model rollout** $x_t = M_t(x_0)$. ODE integration via
   `diffrax`; vardax composes but does not implement.
2. **Tangent-linear and adjoint** $M'_t$, $M'^\top_t$. Computed via
   `diffrax.AbstractAdjoint` (recursive checkpointing, continuous
   backsolve, or forward sensitivity — chapter 12).
3. **Observation operator** $H_t(x_t)$ and its tangent linear $H'_t$.
   See chapter 2 (`linearize()` contract via `lineax`).
4. **Inner minimisation** $\arg\min_x J(x)$. Via
   `optimistix.AbstractMinimiser` (Gauss-Newton, BFGS, NonlinearCG,
   fixed-point iteration). For incremental 4DVar, $\arg\min$ is done by
   CG on the linearised quadratic subproblem.

Each computation is owned by an upstream library. vardax composes them
into DA algorithms; it does not reimplement them.

## Posterior, not point estimate

A MAP $x^*$ is the start of the answer, not the end. Real DA produces a
posterior — mean, covariance, and provenance — that downstream
consumers (population models, alert systems, decision pipelines) need.

vardax exposes a `Posterior(mean, cov, samples, provenance)` container
returned by every analysis. Approximations include the Laplace
(`LaplaceCovariance`) at MAP, the Gauss-Newton Hessian inversion
(`GaussNewtonHessian`, free for `IncrementalFourDVar`), the ensemble
covariance (`EnsembleCovariance` via `filterax`), or direct sampling
from amortized heads. See chapter 13.

For the closed-form case (`OptimalInterpolation`), the posterior is
exact: $P^* = (B^{-1} + H^\top R^{-1} H)^{-1}$. Chapter 4 derives it.

## Reading order

Chapters 1–3 establish the shared foundation: this chapter (the
problem), chapter 2 (the observation model), chapter 3 (the dynamical
model and its adjoint).

Chapters 4–10 cover one analysis method each, in increasing order of
sophistication:

- Chapter 4 — **`OptimalInterpolation`** (closed form)
- Chapter 5 — **`ThreeDVar`** (nonlinear $H$)
- Chapter 6 — **`StrongFourDVar`** (add dynamics)
- Chapter 7 — **`WeakFourDVar`** (allow model error)
- Chapter 8 — **`IncrementalFourDVar`** (operational fast path)
- Chapter 9 — **`FourDVarNet`** (learned prior + solver)
- Chapter 10 — **`AmortizedPosterior`** (direct head)

Chapters 11–14 cover cross-cutting concerns: observation operators (the
averaging-kernel / multi-instrument family), posterior covariance,
adjoint composition via `optimistix` and `diffrax`, and the six-step
research-to-operations methodology.

Chapters 15–17 are end-to-end examples on Lorenz, SSH, and methane.

The companion design docs in [`design/`](design/README.md) document the API
contracts and architectural decisions behind these methods.
