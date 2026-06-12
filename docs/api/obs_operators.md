# Observation Operators

Observation operators map model state to observation space. All of them
satisfy the pipekit-cycle
[`ObservationOperator`](protocols.md) Protocol — `__call__(state)` produces
the predicted observation, and `linearize(state)` returns the tangent-linear
(Jacobian) operator that the incremental and posterior-covariance machinery
needs. Because conformance is structural, custom operators plug in by
matching that signature; nothing here needs to be subclassed. See
[Observation Operators](../11_observation_operators.md) in the Mathematical
Reference for the modelling background.

## Core operators

`LinearObs` wraps an explicit observation matrix; `MaskedIdentity` handles
the ubiquitous "observe a subset of grid points" case (satellite tracks,
sparse sensor networks); `AveragingKernel` implements the smoothing kernels
of retrieval products such as atmospheric-composition column averages.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LinearObs, MaskedIdentity, AveragingKernel]

## Multi-instrument fusion

Assimilating several instruments at once: each instrument is described by an
`InstrumentSpec`, registered in an `InstrumentRegistry`, and
`MultiInstrumentFusion` stacks the per-instrument operators into a single
observation operator over the concatenated observation vector.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [InstrumentSpec, InstrumentRegistry, MultiInstrumentFusion]
