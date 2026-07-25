# Costs, Priors & Solvers

The functional core under the [model classes](models.md): pure cost
functions that score a candidate state against observations and prior,
prior modules that supply the regularisation term, and the inner-loop
solver functions that drive the [4DVarNet](../09_4dvarnet.md) iteration.
The model classes are thin, stateful-looking wrappers over these pieces —
drop down to this layer when building custom methods or instrumenting the
optimisation.

## Cost functions

The variational cost $J(x) = J_\text{obs}(x) + J_\text{prior}(x)$ and its
gradient, with the observation and prior terms also available separately
(`decomposed_loss` returns them unsummed for logging). The `_1d` / `_2d`
suffixes match the [`Batch1D` / `Batch2D`](protocols.md) carriers. See
[3DVar](../05_threedvar.md) and
[strong-constraint 4DVar](../06_strong_4dvar.md) for the math each term
implements.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [variational_cost, variational_cost_grad, obs_cost_1d, obs_cost_2d, prior_cost, decomposed_loss, strong_variational_cost, background_cost]

## Priors

Implementations of the [`Prior`](protocols.md) Protocol. `IdentityPrior`
gives plain Tikhonov regularisation; `L63Prior` / `L96Prior` encode Lorenz
dynamics as a model-consistency penalty; the autoencoder priors (MLP,
convolutional, and bilinear variants in 1D, 2D, and 2D-multivariate) are
*learned* priors that penalise distance from a trained reconstruction
manifold.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [IdentityPrior, L63Prior, L96Prior, MLPAEPrior1D, ConvAEPrior1D, BilinAEPrior1D, BilinAEPrior2D, BilinAEPrior2DMultivar]

## Dynamical priors

ODE-based temporal priors ported from mfourdvar (Decision D18):
`DynIncrements` scores one-step increments, `DynTrajectory` scores the
full rollout from the initial state. Both satisfy the
[`TemporalPrior`](protocols.md) protocol; `bind(ts)` adapts either to
the static [`Prior`](protocols.md) seam, and `as_forward_model(dt)`
adapts the wrapped ODE to `pipekit_cycle.ForwardModel`.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [DynamicalPrior, DynIncrements, DynTrajectory]

## 4DVarNet inner-loop solvers

The unrolled (and fixed-point) inner loop of 4DVarNet, exposed as pure
functions over an explicit `SolverState`: initialise with
`init_solver_state_*`, advance one modulated-gradient step with
`solver_step_*` (or `fp_solver_step_1d` for the fixed-point formulation),
or run the whole loop with `solve_4dvarnet_*`. The `one_step_*` variants
pair with [`OneStepAdjoint`](training.md) for memory-frugal training.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SolverState1D, SolverState2D, init_solver_state_1d, init_solver_state_2d, solver_step_1d, solver_step_2d, fp_solver_step_1d, solve_4dvarnet_1d, solve_4dvarnet_2d, solve_4dvarnet_1d_fixedpoint, one_step_solve_4dvarnet_1d, one_step_solve_4dvarnet_2d]
