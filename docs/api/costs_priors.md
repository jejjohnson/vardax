# Costs, Priors & Solvers

The functional core under the [model classes](models.md): pure cost
functions that score a candidate state against observations and prior,
prior modules that supply the regularisation term, and the inner-loop
solver functions that drive the [4DVarNet](https://jejjohnson.github.io/vardax/dvarnet/) iteration.
The model classes are thin, stateful-looking wrappers over these pieces —
drop down to this layer when building custom methods or instrumenting the
optimisation.

## Cost functions

The variational cost $\mathcal{J}(\vect{x}) = \mathcal{J}_o(\vect{x}) + \mathcal{J}_p(\vect{x})$ and its
gradient, with the observation and prior terms also available separately
(`decomposed_loss` returns them unsummed for logging). The `_1d` / `_2d`
suffixes match the [`Batch1D` / `Batch2D`](protocols.md) carriers. See
[3DVar](https://jejjohnson.github.io/vardax/threedvar/) and
[strong-constraint 4DVar](https://jejjohnson.github.io/vardax/strong-4dvar/) for the math each term
implements.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [variational_cost, variational_cost_grad, obs_cost_1d, obs_cost_2d, prior_cost, decomposed_loss, strong_variational_cost, background_cost]

## Physics residual penalty

`ResidualPenalty` adds a PDE as a weak constraint on a continuous field
(a neural field, or any differentiable callable of one point): the
residual $r(\vect{c}; \gvect{\theta}) \in \mathbb{R}^{D_r}$ is evaluated by autodiff at the $N_c$ collocation points $\vect{c}_j$,
fixed or redrawn from a key on every call, and scored as
$\mathcal{J}_\text{res}(\gvect{\theta}) = \frac{\beta_\text{res}}{2 N_c} \sum_{j=1}^{N_c} \Vert r(\vect{c}_j; \gvect{\theta}) \Vert^2$.
Residuals are plain callables `(field, s) -> (R,)` built on
`geonnax.diffops`; vardax ships two references in `vardax.examples`,
2-D advection–diffusion and 2-D incompressible Navier–Stokes in
vorticity form (the field gives the streamfunction $\psi$, with
$u = \nabla^\perp\psi$ and $\omega = \Delta\psi$).

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ResidualPenalty]

::: vardax.examples
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [advection_diffusion_2d, navier_stokes_vorticity_2d]

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

## Reduced bases

Control-variable transforms: a `ReducedBasis` parameterises the analysis
increment in a lower-dimensional space, so the inner solve runs over basis
coefficients rather than grid points. The constructors build the common
families; `CompositeBasis` concatenates several.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LinearBasis, CompositeBasis, ModelErrorBasis, linear_basis, composite_basis, eof_basis, fourier_basis, rbf_basis, wavelet_basis]

## State parameterizations

Implementations of the [`StateParameterization`](protocols.md) protocol,
accepted as `parameterization=` by `ThreeDVar`, `StrongFourDVar`,
`WeakFourDVar` and `ReducedBasisFourDVar`: the control becomes any pytree
$\gvect{\theta}$, the state $g_\text{dec}(\gvect{\theta})$, and the penalty $\mathcal{R}(\gvect{\theta})$ replaces
the background term. `LinearParameterization` adapts any reduced basis
($g_\text{dec}(\vect{z}) = \gvect{\mu}_x + \gmat{\Phi} \vect{z}$, with $\gmat{\Phi} \in \mathbb{R}^{D_x \times D_z}$ and $\vect{z} \in \mathbb{R}^{D_z}$); `NeuralFieldParameterization` makes the state
a coordinate network on the grid, with weight decay as its penalty.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LinearParameterization, NeuralFieldParameterization]

## Minimisers

Implementations of the [`Minimiser`](protocols.md) protocol, accepted as
`minimiser=` by `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar` and
`ReducedBasisFourDVar`. `OptimistixMinimiser` wraps an optimistix solver
(the models' default, BFGS); `OptaxMinimiser` runs a first-order optax
optimiser with early stopping for controls too large for BFGS, and
reports the final gradient norm the implicit adjoint relies on. Both
return a `MinimiseResult` from `run`.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [OptaxMinimiser, OptimistixMinimiser, MinimiseResult]

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
