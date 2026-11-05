"""Training primitives: losses and single optimiser steps, with no outer loop.

``train_step`` and ``eval_step`` differentiate through the FourDVarNet inner
solver. The outer loop (epochs, logging, checkpointing) is left to
``pipekit-train``, with ``vardax.adapters.pipekit_train`` wrapping the loss.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import optax

from vardax._src.errors import MissingInputError

from ._types import Batch1D, Batch2D


def reconstruction_loss(
    pred: Float[Array, ...],
    target: Float[Array, ...],
) -> Float[Array, ""]:
    """Mean-squared reconstruction loss.

        ℒ_MSE(x̂, x) = mean((x̂ − x)²)

    Args:
        pred: Model prediction x̂.
        target: Ground-truth target x, broadcastable against ``pred``.

    Returns:
        Scalar mean-squared error ℒ_MSE.

    Examples:
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> prediction = jnp.array([1.0, 3.0])
        >>> float(vdx.reconstruction_loss(prediction, jnp.array([1.0, 1.0])))
        2.0

        A perfect prediction has zero loss, whatever the shape:

        >>> x = jnp.ones((2, 3, 4))
        >>> float(vdx.reconstruction_loss(x, x))
        0.0
    """
    # ℒ_MSE = mean((x̂ − x)²)
    return jnp.mean((pred - target) ** 2)


def train_loss_fn(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Compute the training loss for a single batch.

        ℒ_MSE = mean((x̂ − x)²),   x̂ = model(batch),   x = batch.target

    Algorithm:
        1. Run the model on the batch to obtain the prediction x̂.
        2. Compare it with the target x using
           [`reconstruction_loss`][vardax.reconstruction_loss].

    Args:
        model: Equinox module implementing ``__call__(batch) -> prediction``.
        batch: Training batch (must have a non-``None`` ``target``).

    Returns:
        Scalar reconstruction loss.

    Raises:
        MissingInputError: If ``batch.target`` is ``None``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import optax
        >>> import vardax as vdx
        >>> model = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=1,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> ones = jnp.ones((2, 3, 4))
        >>> batch = vdx.Batch1D(input=ones, mask=ones, target=jnp.zeros((2, 3, 4)))
        >>> vdx.train_loss_fn(model, batch).shape
        ()

        It differentiates with respect to the model, as ``train_step`` does:

        >>> import equinox as eqx
        >>> loss, grads = eqx.filter_value_and_grad(vdx.train_loss_fn)(model, batch)
        >>> bool(jnp.isfinite(loss))
        True

        A batch without a target is rejected:

        >>> unlabelled = vdx.Batch1D(input=ones, mask=ones)
        >>> vdx.train_loss_fn(model, unlabelled)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        MissingInputError: train_loss_fn requires batch.target to be set.
    """
    # x̂ = model(batch)
    prediction = model(batch)
    target = batch.target
    if target is None:
        raise MissingInputError("train_loss_fn requires batch.target to be set.")
    # ℒ_MSE = mean((x̂ − x)²)
    return reconstruction_loss(prediction, target)


@eqx.filter_jit
def train_step(
    model: Any,
    batch: Batch1D | Batch2D,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Any, optax.OptState, Float[Array, ""]]:
    """Perform a single training step (forward, backward and update).

        θ⁽ⁱ⁺¹⁾ = θ⁽ⁱ⁾ + Δθ⁽ⁱ⁾,   (Δθ⁽ⁱ⁾, s⁽ⁱ⁺¹⁾) = optimiser(∇_θ ℒ_MSE(θ⁽ⁱ⁾), s⁽ⁱ⁾)

    with s the optimiser state. Gradients flow through the FourDVarNet inner
    solver according to the model's ``solver_adjoint``. Compose this step into a
    notebook-level loop or ``pipekit_train.TrainingLoop`` rather than
    reimplementing it.

    Algorithm:
        1. Evaluate [`train_loss_fn`][vardax.train_loss_fn] and its gradient
           ∇_θ ℒ_MSE with respect to the model parameters θ.
        2. Turn the gradient into an update Δθ with the optax optimiser.
        3. Apply the update to the model.

    Args:
        model: Equinox module to optimise.
        batch: Training batch.
        optimizer: Optax gradient transformation (e.g. ``optax.adam(1e-3)``).
        opt_state: Current optimiser state.

    Returns:
        The updated model, the updated optimiser state and the scalar loss
        (evaluated before the update).

    Raises:
        MissingInputError: If ``batch.target`` is ``None``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> import optax
        >>> import vardax as vdx
        >>> model = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=1,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> ones = jnp.ones((2, 3, 4))
        >>> batch = vdx.Batch1D(input=ones, mask=ones, target=jnp.zeros((2, 3, 4)))
        >>> optimizer = optax.adam(1e-2)
        >>> opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        >>> new_model, opt_state, loss = vdx.train_step(
        ...     model, batch, optimizer, opt_state
        ... )
        >>> loss.shape
        ()

        The step returns the pre-update loss, so it equals ``eval_step`` on the
        old model, and the parameters have moved:

        >>> bool(jnp.allclose(loss, vdx.eval_step(model, batch)))
        True
        >>> bool(eqx.tree_equal(new_model, model))
        False
    """
    # ℒ_MSE(θ⁽ⁱ⁾) and ∇_θ ℒ_MSE(θ⁽ⁱ⁾)
    loss, grads = eqx.filter_value_and_grad(train_loss_fn)(model, batch)
    # Δθ⁽ⁱ⁾ from the optimiser
    updates, opt_state = optimizer.update(grads, opt_state, model)
    # θ⁽ⁱ⁺¹⁾ = θ⁽ⁱ⁾ + Δθ⁽ⁱ⁾
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Compute the evaluation loss for a single batch (no gradient).

        ℒ_MSE = mean((x̂ − x)²),   x̂ = model(batch),   x = batch.target

    Args:
        model: Equinox module implementing ``__call__(batch) -> prediction``.
        batch: Evaluation batch (must have a non-``None`` ``target``).

    Returns:
        Scalar reconstruction loss.

    Raises:
        MissingInputError: If ``batch.target`` is ``None``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import optax
        >>> import vardax as vdx
        >>> model = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=1,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> ones = jnp.ones((2, 3, 4))
        >>> batch = vdx.Batch1D(input=ones, mask=ones, target=jnp.zeros((2, 3, 4)))
        >>> vdx.eval_step(model, batch).shape
        ()

        It agrees with the training loss on the same batch:

        >>> bool(
        ...     jnp.allclose(
        ...         vdx.eval_step(model, batch), vdx.train_loss_fn(model, batch)
        ...     )
        ... )
        True

        A batch without a target is rejected:

        >>> vdx.eval_step(
        ...     model, vdx.Batch1D(input=ones, mask=ones)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        MissingInputError: eval_step requires batch.target to be set.
    """
    # x̂ = model(batch)
    prediction = model(batch)
    target = batch.target
    if target is None:
        raise MissingInputError("eval_step requires batch.target to be set.")
    # ℒ_MSE = mean((x̂ − x)²)
    return reconstruction_loss(prediction, target)


def amortized_nll_loss_fn(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Negative log-likelihood for amortized inference.

        ℒ_NLL(ν) = −mean(log q(x | y; ν))

    averaged over the simulated pairs (x, y) of the batch, for an
    ``AmortizedPosterior`` with any head that has a ``.log_prob``.

    Algorithm:
        1. Evaluate log q(x | y; ν) for each sample with
           ``model.log_prob(batch.target, batch)``.
        2. Return the negative batch mean.

    Args:
        model: ``AmortizedPosterior`` (any head with a ``.log_prob``
            method).
        batch: Training batch with ``target = x`` and ``input = y``.

    Returns:
        Scalar NLL averaged over the batch.

    Raises:
        MissingInputError: If ``batch.target`` is ``None``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import optax
        >>> import vardax as vdx
        >>> key = jax.random.PRNGKey(0)
        >>> posterior = vdx.AmortizedPosterior(
        ...     encoder=vdx.MLPObsEncoder(input_size=12, context_dim=6, key=key),
        ...     head=vdx.RegressionHead(context_dim=6, state_shape=(3, 4), key=key),
        ...     config=vdx.AmortizedConfig(),
        ... )
        >>> ones = jnp.ones((2, 3, 4))
        >>> batch = vdx.Batch1D(input=ones, mask=ones, target=jnp.zeros((2, 3, 4)))
        >>> vdx.amortized_nll_loss_fn(posterior, batch).shape
        ()

        It is the negative mean of the per-sample log-density:

        >>> expected = -jnp.mean(posterior.log_prob(batch.target, batch))
        >>> bool(jnp.allclose(vdx.amortized_nll_loss_fn(posterior, batch), expected))
        True

        The true state is required:

        >>> blind = vdx.Batch1D(input=ones, mask=ones)
        >>> vdx.amortized_nll_loss_fn(
        ...     posterior, blind
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        MissingInputError: amortized_nll_loss_fn requires batch.target ...
    """
    target = batch.target
    if target is None:
        raise MissingInputError(
            "amortized_nll_loss_fn requires batch.target (the true x) to be set."
        )
    # log q(x | y; ν) per sample
    log_prob = model.log_prob(target, batch)
    # ℒ_NLL = −mean(log q(x | y; ν))
    return -jnp.mean(log_prob)


@eqx.filter_jit
def amortized_train_step(
    model: Any,
    batch: Batch1D | Batch2D,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Any, optax.OptState, Float[Array, ""]]:
    """Single training step for ``AmortizedPosterior``.

        ν⁽ⁱ⁺¹⁾ = ν⁽ⁱ⁾ + Δν⁽ⁱ⁾,   (Δν⁽ⁱ⁾, s⁽ⁱ⁺¹⁾) = optimiser(∇_ν ℒ_NLL(ν⁽ⁱ⁾), s⁽ⁱ⁾)

    Same shape as [`train_step`][vardax.train_step] but uses
    [`amortized_nll_loss_fn`][vardax.amortized_nll_loss_fn] instead of the MSE
    reconstruction loss. Use this for simulation-based training of amortized
    variants; use ``train_step`` for ``FourDVarNet`` and classical models that
    reconstruct fields.

    Algorithm:
        1. Evaluate ℒ_NLL and its gradient ∇_ν ℒ_NLL with respect to the
           parameters ν.
        2. Turn the gradient into an update Δν with the optax optimiser.
        3. Apply the update to the model.

    Args:
        model: ``AmortizedPosterior`` to optimise.
        batch: Training batch with ``target = x`` and ``input = y``.
        optimizer: Optax gradient transformation.
        opt_state: Current optimiser state.

    Returns:
        The updated model, the updated optimiser state and the scalar NLL
        (evaluated before the update).

    Raises:
        MissingInputError: If ``batch.target`` is ``None``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import optax
        >>> import vardax as vdx
        >>> key = jax.random.PRNGKey(0)
        >>> posterior = vdx.AmortizedPosterior(
        ...     encoder=vdx.MLPObsEncoder(input_size=12, context_dim=6, key=key),
        ...     head=vdx.RegressionHead(context_dim=6, state_shape=(3, 4), key=key),
        ...     config=vdx.AmortizedConfig(),
        ... )
        >>> ones = jnp.ones((2, 3, 4))
        >>> batch = vdx.Batch1D(input=ones, mask=ones, target=jnp.zeros((2, 3, 4)))
        >>> import equinox as eqx
        >>> optimizer = optax.adam(1e-2)
        >>> opt_state = optimizer.init(eqx.filter(posterior, eqx.is_array))
        >>> new, opt_state, loss = vdx.amortized_train_step(
        ...     posterior, batch, optimizer, opt_state
        ... )
        >>> loss.shape
        ()

        A few steps reduce the loss on a fixed batch:

        >>> first = loss
        >>> for _ in range(5):
        ...     new, opt_state, loss = vdx.amortized_train_step(
        ...         new, batch, optimizer, opt_state
        ...     )
        >>> bool(loss < first)
        True
    """
    # ℒ_NLL(ν⁽ⁱ⁾) and ∇_ν ℒ_NLL(ν⁽ⁱ⁾)
    loss, grads = eqx.filter_value_and_grad(amortized_nll_loss_fn)(model, batch)
    # Δν⁽ⁱ⁾ from the optimiser
    updates, opt_state = optimizer.update(grads, opt_state, model)
    # ν⁽ⁱ⁺¹⁾ = ν⁽ⁱ⁾ + Δν⁽ⁱ⁾
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
