# Training & Adjoints

Training a learned solver means differentiating *through* an inner
optimisation loop, and how you do that determines memory cost and gradient
quality. The adjoint strategies here make that choice explicit and
swappable — see [Adjoint Methods](../12_adjoint_methods.md) in the
Mathematical Reference for the trade-offs. Around them sit the loss
functions and train/eval steps for [4DVarNet](models.md) and the
[amortized posteriors](amortized.md), and the ConvLSTM gradient modulators
that 4DVarNet learns in place of a hand-tuned inner optimiser.

## Adjoint strategies

Implementations of `vardax.adjoints` (Decision D15): full
backpropagation with checkpointed memory (`RecursiveCheckpointAdjoint`),
truncated one-step gradients (`OneStepAdjoint`, pairing with the
`one_step_solve_*` [solver functions](costs_priors.md)), and
implicit differentiation at a fixed point (`ImplicitAdjoint`, pairing with
`solve_4dvarnet_1d_fixedpoint`). All three are also accessible via the
`vardax.adjoints` submodule namespace. Use
[`assert_adjoint_calibrated`](utils.md) to verify a cheap adjoint against
the exact one before trusting it.

`RecursiveCheckpointAdjoint` and `ImplicitAdjoint` are re-exported from
[optimistix](https://docs.kidger.site/optimistix/) for one-stop import;
see the optimistix documentation for their full signatures:

- `vardax.RecursiveCheckpointAdjoint` —
  [`optimistix.RecursiveCheckpointAdjoint`](https://docs.kidger.site/optimistix/api/adjoints/),
  exact reverse-mode backpropagation through the unrolled inner loop with
  binomial checkpointing (the default).
- `vardax.ImplicitAdjoint` —
  [`optimistix.ImplicitAdjoint`](https://docs.kidger.site/optimistix/api/adjoints/),
  implicit-function-theorem differentiation at a fixed point; pair with
  `solve_4dvarnet_1d_fixedpoint`.

`OneStepAdjoint` is vardax's own truncated adjoint (Bolte, Pauwels &
Vaiter, NeurIPS 2023):

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [OneStepAdjoint]

## Gradient modulators

The learned components of 4DVarNet: ConvLSTM cells that map the raw
variational-cost gradient to a descent update, satisfying the
[`GradModulator`](protocols.md) Protocol. Their recurrent state is carried
in the [`LSTMState1D` / `LSTMState2D`](protocols.md) containers.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ConvLSTMGradMod1D, ConvLSTMGradMod2D]

## Losses & steps

Outer-loop training: reconstruction-based losses for 4DVarNet, the
negative-log-likelihood loss for amortized posteriors, and the
optax-driven `train_step` / `amortized_train_step` / `eval_step` that
consume them.

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [reconstruction_loss, train_loss_fn, train_step, eval_step, amortized_nll_loss_fn, amortized_train_step]
