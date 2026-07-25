# Uncertainty Quantification

Consolidated from the `mfourdvar` uncertainty notes
(`content/uncertainty/`), which sketched the taxonomy below as an
outline; this chapter fills it in and maps each branch to the vardax
machinery that implements it.

Chapter [13](13_posterior_covariance.md) covers *how* to compute a
posterior covariance; this chapter is the map of *what is uncertain in
the first place* — where stochasticity enters a variational
assimilation system, and which lever addresses which source.

## Three kinds of uncertainty

- **Aleatoric** — irreducible randomness in the measurement process:
  sensor noise, representativeness error, sampling gaps. More data of
  the same kind does not remove it; it belongs in the likelihood
  ($R$).
- **Epistemic** — reducible ignorance about the system: imperfect
  model physics, unknown parameters, limited training data. More
  information (observations, better physics, more training) shrinks
  it; it belongs in priors and posteriors.
- **Intrinsic variability** — chaotic sensitivity of the dynamics
  itself. Even a perfect model with perfect parameters diverges from
  truth under infinitesimal initial-condition error; it bounds
  predictability (forecast horizon) rather than estimation quality.

The mfourdvar notes organise the *loci* of these uncertainties into
four groups — data, model, parameters, estimation — which map onto
vardax as follows.

## Uncertain data

The likelihood side. Observation error enters the cost through the
observation-error covariance $R$ (chapter
[2](02_observation_model.md)); real data additionally arrive with
gaps.

- **Error covariance** — every analysis method takes an `obs_cov_op`
  ($R$ as a `lineax` operator); per-instrument $R$ blocks compose
  through [`MultiInstrumentFusion`](11_observation_operators.md).
- **Gaps and NaNs** — real products encode missing data as `NaN`
  rather than a clean mask; the `nan_to_num` flag of
  [`obs_cost_1d` / `obs_cost_2d`](api/costs_priors.md) and the
  NaN-stripping analysis steps keep gradients finite (ported from
  mfourdvar's NaN-safe observation operator).
- **Augmentation and latent inputs** — the mfourdvar notes flag two
  learning-side treatments: perturbing inputs with samples of the
  observation noise during training, and letting an encoder treat the
  clean field as a latent variable behind noisy inputs. In vardax
  these live in the training loop (noise injection via
  `vardax.utils` noise helpers) and in the amortized encoder heads
  (chapter [10](10_amortized_inference.md)) respectively.

## Uncertain model

The dynamics side. A forward model is wrong in ways that range from
"ignore it" to "model it":

- **Deterministic (perfect-model)** — strong-constraint 4DVar
  (chapter [6](06_strong_4dvar.md)) treats $M_t$ as exact; all model
  error is silently folded into the initial-condition posterior.
- **Additive model error** — weak-constraint 4DVar (chapter
  [7](07_weak_4dvar.md)) promotes per-step error terms
  $\boldsymbol{\eta}_t$ to control variables with their own covariance
  $Q$; the dynamical-residual priors
  ([`DynIncrements`](19_physical_models.md)) are the functional
  version of the same idea — dynamics as a soft penalty rather than a
  hard constraint.
- **Fully Bayesian** — placing distributions over the model structure
  itself is out of vardax's scope; the pragmatic surrogate is
  ensembles over model variants, scored through the same analysis
  seams.

## Uncertain parameters

Between data and model sit the parameters $\theta$ — ODE forcings,
closure coefficients, neural-prior weights:

- **Deterministic point estimates** — fit $\theta$ by gradient
  descent through the solve
  ([`DynamicalPrior.loss(..., params=θ)`](19_physical_models.md) is
  differentiable in $\theta$).
- **Probabilistic** — the amortized family (chapter
  [10](10_amortized_inference.md)) returns distributions, not points:
  `ConditionalFlowHead` for flow-based posteriors, with the
  reverse-SDE `ScoreDiffusionHead` stubbed for multimodal cases.
- **Approximate Bayesian** — the mfourdvar outline lists dropout and
  deep ensembles. Neither needs library support beyond what exists:
  ensembles are `jax.vmap` over `PRNGKey`s at construction time, and
  their spread feeds the ensemble posterior adapter below.

## Uncertain estimation

The output side: attaching error bars to the analysis itself.

- **Curvature-based** — the posterior adapters of chapter
  [13](13_posterior_covariance.md): `LaplaceCovariance` (inverse
  Hessian at the mode), `GaussNewtonHessian` (drops second-order
  terms), both matrix-free via `lineax`.
- **Monte Carlo / ensemble** — `EnsembleCovariance` estimates the
  posterior spread from an ensemble of analyses (perturbed
  observations, perturbed parameters, or both).
- **Amortized** — the heads of chapter
  [10](10_amortized_inference.md) emit $q_\phi(x \mid y)$ directly.
- **Conformal prediction** — distribution-free calibrated intervals
  wrapped around any point estimator; listed in the mfourdvar outline
  as future work and deliberately left outside vardax (it is a
  post-processing wrapper, not an assimilation component).

## Trust, but verify

Every uncertainty estimate above is an approximation, and the library
treats "is it calibrated?" as a first-class question. The validation
gates of the [six-step cycle](14_six_step_cycle.md) are the
enforcement mechanism:

```python
from vardax import (
    simulation_based_calibration,   # SBC ranks must be uniform
    assert_posterior_agreement,     # two adapters must agree
    assert_adjoint_calibrated,      # cheap adjoint must track exact grad
)
```

Run simulation-based calibration before believing any posterior; if
the rank histogram is not uniform, the reported uncertainty is
mis-calibrated regardless of how principled its derivation looked.
