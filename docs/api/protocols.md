# Protocols & Types

vardax components are coupled structurally: anything with the right
methods conforms, with no base class to inherit. Three of the Protocols —
`AnalysisStep`, `ForwardModel`, and `ObservationOperator` — are re-exported
from [pipekit-cycle](https://github.com/jejjohnson/pipekit) and define the
seam along which vardax plugs into assimilation
[cycles](cycle.md); the rest are vardax-specific and define the seams
*inside* a variational method (prior, cost, gradient modulator, posterior
adapter, minimiser). All are runtime-checkable, so `isinstance` checks work
at the boundaries.

## pipekit-cycle protocols

The orchestration contract. `ForwardModel` propagates state between cycle
times, `ObservationOperator` maps state to observation space (with
`linearize` for the tangent-linear), and `AnalysisStep` is what every
vardax model's `.as_analysis_step()` returns.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AnalysisStep, ForwardModel, ObservationOperator]

## vardax protocols

The internal seams: implement these to swap in custom
[priors](costs_priors.md), cost functions,
[gradient modulators](training.md),
[posterior adapters](posterior.md), or inner-loop minimisers without
touching the model classes.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Prior, CostFunction, GradModulator, PosteriorAdapter, Minimiser]

## Batch & state types

The typed carriers that flow through the solvers and training loops:
1D and 2D (single- and multi-variable) observation batches, and the
recurrent-state containers of the ConvLSTM gradient modulators.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Batch1D, Batch2D, Batch2DMultivar, LSTMState1D, LSTMState2D]
