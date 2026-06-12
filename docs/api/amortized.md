# Amortized Inference

Where the variational methods on the [Models](models.md) page solve an
optimisation problem *per analysis*, an amortized posterior pays the cost
once at training time: a network is fit to map observations directly to an
approximate posterior, so inference is a single forward pass. See
[Amortized Inference](../10_amortized_inference.md) in the Mathematical
Reference for the underlying theory and the fidelity/speed trade-offs.

`AmortizedPosterior` composes two exchangeable parts: an *observation
encoder* that summarises (possibly masked, possibly irregular) observations
into a conditioning vector, and a *posterior head* that turns that vector
into a distribution over states. Heads span the fidelity ladder — point
estimates (`RegressionHead`), full densities via conditional normalizing
flows (`ConditionalFlowHead`), and score-based diffusion sampling
(`ScoreDiffusionHead`). Amortized posteriors should pass the same
[validation gates](utils.md) (simulation-based calibration, posterior
agreement) as their variational counterparts.

## Posterior and configuration

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AmortizedPosterior, AmortizedConfig]

## Observation encoders

`IdentityObsEncoder` passes observations straight through — appropriate
when they are already a fixed-size vector; `MLPObsEncoder` learns the
summary.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [IdentityObsEncoder, MLPObsEncoder]

## Posterior heads

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [RegressionHead, ConditionalFlowHead, ScoreDiffusionHead]
