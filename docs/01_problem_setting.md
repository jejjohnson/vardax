---
numbering:
  enumerator: "1.%s"
---

# Problem Setting

Data assimilation is the problem of estimating the state of a dynamical system given partial, noisy observations and prior knowledge. It sits at the intersection of Bayesian inference, optimisation and numerical analysis; the prior comes from a physical model (or a learned one), the likelihood comes from instruments, and the posterior is what we want.

## The Bayesian picture

Let $x \in \mathbb{R}^n$ denote the state and $y \in \mathbb{R}^m$ the observations. Bayes' theorem gives the posterior

$$p(x \mid y) \propto p(y \mid x) p(x)$$ (eq-bayes)

which has three ingredients:

- **Prior $p(x)$.** What we know about the state before seeing the data. Most commonly it is Gaussian, $x \sim \mathcal{N}(x_b, B)$ ([](#eq-oi-prior)), with background mean $x_b$ and background-error covariance $B$. In `FourDVarNet` the prior is replaced by a learned autoencoder; in `AmortizedPosterior` it is absorbed into the trained head.
- **Likelihood $p(y \mid x)$.** How observations relate to the state. Chapter 2 models them as $H(x)$ plus Gaussian noise with covariance $R$ ([](#eq-obs-model)), so $p(y \mid x) = \mathcal{N}(y; H(x), R)$, where the observation operator $H$ maps state space to observation space.
- **Posterior $p(x \mid y)$.** What we infer about the state given the data. For a Gaussian prior and a Gaussian likelihood with linear $H$ it is Gaussian in closed form ([](#eq-oi-posterior), chapter 4); otherwise we approximate it.

The maximum a posteriori (MAP) estimator minimises the negative log-posterior,

$$x^* = \underset{x}{\arg\min} \left\{-\log p(y \mid x) - \log p(x)\right\}$$ (eq-map)

and with the Gaussian prior and likelihood above the bracket is, up to a constant, the 3DVar cost [](#eq-3dvar-cost) of chapter 5. When the state evolves through dynamics $M_t$, we add a time index and a forward rollout, which gives 4DVar (chapters 6 to 8).

## When dynamics enter: the 4D problem

Consider a state $x_0$ at the start of an assimilation window and observations $y_t$ at times $t = 0, 1, \ldots, T$. A dynamical model $M_t$ propagates $x_0$ forward,

$$x_t = M_t(x_0)$$ (eq-rollout)

where $x_t$ is the state at time $t$.

The 4D variational cost adds an observation term for every time to the background term. In strong-constraint 4DVar (chapter 6) the dynamics are exact, the control variable is $x_0$ alone and the cost is [](#eq-strong4dvar-cost). In weak-constraint 4DVar (chapter 7) each step carries a model error $\eta_t$ with covariance $Q_t$, the cost [](#eq-weak4dvar-cost) gains a model-error term, and the control becomes the augmented vector $(x_0, \eta_1, \ldots, \eta_T)$.

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
| **`IncrementalFourDVar`** | Operational fast path of `StrongFourDVar` | GN outer + CG inner |
| **`FourDVarNet`** | Learned prior + learned grad modulator | Learned inner iteration |
| **`AmortizedPosterior`** | Direct $q_\phi(x \mid y)$ head | Single forward pass |

`ReducedBasisFourDVar` is `StrongFourDVar` with the control moved into a reduced basis, $x_0 = x_b + \Phi X$ (see [chapter 6](06_strong_4dvar.md#reduced-basis-4dvar)). It was called `Incremental4DVar` before 0.1.11; that name is deprecated.

They are siblings, not subclasses, and each one is appropriate to a different regime.

## The role of the forward model

The dynamics $M_t$ and the observation operator $H_t$ are the **physics**. vardax does not own them; `somax` (geophysics), `plumax` (atmospheric transport and methane) or any user code can supply them, as long as they satisfy `pipekit_cycle.ForwardModel` and `pipekit_cycle.ObservationOperator` respectively.

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

1. **Forward model rollout** [](#eq-rollout). ODE integration via `diffrax`; vardax composes it but does not implement it.
2. **Tangent-linear and adjoint** $M'_t$, $M'^\top_t$. Computed via `diffrax.AbstractAdjoint` (recursive checkpointing, continuous backsolve or forward sensitivity; see chapter 12).
3. **Observation operator** $H_t(x_t)$ and its tangent linear $H'_t$.
   See chapter 2 (`linearize()` contract via `lineax`).
4. **Inner minimisation** $\arg\min_x J(x)$. Via
   `optimistix.AbstractMinimiser` (Gauss-Newton, BFGS, NonlinearCG,
   fixed-point iteration). For incremental 4DVar, $\arg\min$ is done by
   CG on the linearised quadratic subproblem.

Each computation is owned by an upstream library. vardax composes them
into DA algorithms; it does not reimplement them.

## Posterior, not point estimate

A MAP $x^*$ is the start of the answer, not the end. Real DA produces a posterior (mean, covariance and provenance) that downstream consumers (population models, alert systems, decision pipelines) need.

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
