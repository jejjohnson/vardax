# Models

The seven analysis methods are *peers* — sibling `equinox.Module` classes
with no inheritance between them, each owning exactly the assumptions its
math requires. Pick by problem structure: a single analysis time with a
linear observation operator wants [OI](https://jejjohnson.github.io/vardax/oi-blue/) or
[3DVar](https://jejjohnson.github.io/vardax/threedvar/); an assimilation window with perfect-model
dynamics wants [strong-constraint 4DVar](https://jejjohnson.github.io/vardax/strong-4dvar/); admitting
model error turns that into [weak-constraint 4DVar](https://jejjohnson.github.io/vardax/weak-4dvar/);
the operational linearise-and-iterate formulation with control-variable
transform is [incremental 4DVar](https://jejjohnson.github.io/vardax/incremental-4dvar/); and replacing
the inner-loop optimiser with a trained ConvLSTM gives
[4DVarNet](https://jejjohnson.github.io/vardax/dvarnet/).

Every model exposes `.as_analysis_step()`, returning a lightweight wrapper
that satisfies the pipekit-cycle [`AnalysisStep`](protocols.md) Protocol —
that is the seam through which all seven plug into
[`VarDACycle` / `VarSmootherCycle`](cycle.md) interchangeably.

## Classical methods

Closed-form and optimisation-based analyses. `OptimalInterpolation` is the
linear-Gaussian BLUE solution and refuses non-linear observation operators
at construction; `ThreeDVar` minimises the same cost iteratively and accepts
non-linear operators; the three 4DVar variants extend the cost over a time
window. `IncrementalConfig` collects the outer/inner-loop knobs of
`IncrementalFourDVar`.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [OptimalInterpolation, ThreeDVar, StrongFourDVar, WeakFourDVar, IncrementalFourDVar, IncrementalConfig, ReducedBasisFourDVar]

## Neural-field 4DVar

[`SpaceTimeFieldVar`](https://jejjohnson.github.io/vardax/neural-field-4dvar/) fits one continuous
field $u(t, s) = f_\theta(t, s)$ over the whole window to point
observations, with an optional PDE residual as a soft constraint and
weight decay in place of $B$; no rollout. It returns a
`SpaceTimeFieldResult`, queryable anywhere, and `ensemble(...)` gives a
deep-ensemble spread.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SpaceTimeFieldVar, SpaceTimeFieldResult]

## Learned solvers — 4DVarNet

End-to-end-trainable 4DVar: the variational cost is kept explicit, but the
inner-loop descent direction is produced by a ConvLSTM
[gradient modulator](training.md) instead of a hand-tuned optimiser. The 1D
variant operates on `Batch1D` (e.g. Lorenz-96 trajectories); the 2D variant
on `Batch2D` / `Batch2DMultivar` fields (e.g. SSH reconstruction). The
inner-loop iteration functions live on the
[Costs, Priors & Solvers](costs_priors.md) page; training utilities on
[Training & Adjoints](training.md).

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [FourDVarNet1D, FourDVarNet2D]
