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
      members: [LinearObs, MaskedIdentity, AveragingKernel, InterpObs, interp_obs_from_coords]

## Scattered point observations

Observations at arbitrary coordinates and times, carried by a
[`PointObsBatch`](protocols.md) with per-observation $\sigma_i$ and a
`valid` mask for padding. `GridPointObs` evaluates a gridded state or
trajectory at the points (linear / bilinear in space, at each
observation's `time_index`); `FieldPointObs` evaluates a callable field;
`point_obs_cost` is the masked, $\sigma$-weighted cost, exactly zero in
value and gradient on padding rows. The classical models dispatch on
the batch type.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GridPointObs, FieldPointObs, point_obs_cost]

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
