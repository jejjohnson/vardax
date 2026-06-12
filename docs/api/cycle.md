# Cycle Integration

Sequential assimilation — forecast, observe, analyse, repeat — is
orchestrated by [pipekit-cycle](https://github.com/jejjohnson/pipekit), not
reimplemented here. The two factories on this page are thin wrappers that
assemble a `pipekit_cycle.DACycle` (filtering) or
`pipekit_cycle.SmootherCycle` (fixed-lag smoothing) from vardax parts: any
[model](models.md)'s `.as_analysis_step()`, any
[observation operator](obs_operators.md), and a forward model satisfying
the [`ForwardModel`](protocols.md) Protocol.

Because the coupling is purely structural (runtime-checkable Protocols, no
inheritance), the same cycle accepts every vardax method interchangeably —
swapping [3DVar](../05_threedvar.md) for
[strong-constraint 4DVar](../06_strong_4dvar.md) inside a cycle is a
one-argument change. The full forecast/analysis loop is worked through in
[Six-Step Inference Cycle](../14_six_step_cycle.md). Both factories are also
accessible via the `vardax.cycle` submodule namespace.

## Factories

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [VarDACycle, VarSmootherCycle]
