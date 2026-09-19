# Utilities & Diagnostics

Supporting cast: a reference dynamical system for experiments, the
statistical gates that decide whether an inference setup can be trusted,
and the plotting helpers used throughout the
[end-to-end examples](../15_lorenz_examples.md).

## Dynamical systems

Lorenz-63 and Lorenz-96 are the standard chaotic testbeds for
assimilation experiments; the `simulate_*` helpers generate trajectories
for the [Lorenz examples](../15_lorenz_examples.md) and the test suite,
and `time_patches` produces the consecutive time pairs consumed by the
one-step increment losses.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Lorenz63, Lorenz96, simulate_lorenz63, simulate_lorenz96, time_patches]

## Forward-model adapters

`SoftBoundedForward` wraps any `pipekit_cycle.ForwardModel` so that the
line search of a variational solver cannot drive an explicit ODE step
off the attractor into overflow: inside a box the adapter is the
identity, outside it saturates smoothly. Notebooks
[13](../notebooks/13_neural_closure_L96.py) and
[14](../notebooks/14_posterior_uncertainty_L63.py) use it around
Lorenz-96 and Lorenz-63 forwards.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SoftBoundedForward]

## Validation gates

The six-step methodology's go/no-go checks (Decision D12):
simulation-based calibration ranks the truth within posterior samples and
must be uniform; `assert_posterior_agreement` cross-checks two
[posterior adapters](posterior.md) against each other; and
`assert_adjoint_calibrated` verifies that a cheap
[adjoint](training.md) tracks the exact gradient before it is used for
training. Run these before believing any uncertainty estimate.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [simulation_based_calibration, assert_posterior_agreement, assert_adjoint_calibrated]

## Visualization

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [plot_l96_trajectories, plot_l96_grid, plot_reconstruction_comparison]
