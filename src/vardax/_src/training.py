"""Training primitives for FourDVarNet.

Per Decision D5, vardax ships ``train_step`` and ``eval_step`` as
Layer 0 primitives — they encode the correctness-critical
differentiation through the FourDVarNet inner solver, which is the part
that's library-grade. The outer training loop (epoch iteration, logging,
checkpointing, distributed) is delegated to ``pipekit-train``
(``[train]`` extra) — see ``vardax.adapters.pipekit_train`` for the
``Loss`` adapter that wraps ``train_loss_fn`` into the
``pipekit_train.Loss`` protocol.

This module contains *only* the inner-step primitives. There is no
``fit()`` helper; users either compose ``train_step`` into a small
notebook-level loop or wire it through ``pipekit_train.TrainingLoop``.
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
    r"""Mean-squared reconstruction loss.

    $$
    \mathcal{L}(\hat x, x) = \frac{1}{n} \sum_i (\hat x_i - x_i)^2
    $$

    Args:
        pred: Model predictions $\hat x$.
        target: Ground-truth targets $x$, broadcastable against ``pred``.

    Returns:
        Scalar mean-squared error.

    Examples:
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> float(vdx.reconstruction_loss(jnp.array([1.0, 3.0]), jnp.array([1.0, 1.0])))
        2.0

        A perfect prediction has zero loss, whatever the shape:

        >>> x = jnp.ones((2, 3, 4))
        >>> float(vdx.reconstruction_loss(x, x))
        0.0
    """
    return jnp.mean((pred - target) ** 2)


def train_loss_fn(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    r"""Compute the training loss for a single batch.

    $$
    \mathcal{L} = \operatorname{mean}\big((\mathcal{M}(b) - x_\text{target})^2\big)
    $$

    Algorithm:
        1. Run the model on the batch to obtain the prediction.
        2. Compare it with ``batch.target`` using
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
    pred = model(batch)
    target = batch.target
    if target is None:
        raise MissingInputError("train_loss_fn requires batch.target to be set.")
    return reconstruction_loss(pred, target)


@eqx.filter_jit
def train_step(
    model: Any,
    batch: Batch1D | Batch2D,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Any, optax.OptState, Float[Array, ""]]:
    """Perform a single training step (forward + backward + update).

    This is the correctness-critical primitive: gradients flow through
    the FourDVarNet inner solver according to whichever differentiation
    strategy (the model's ``solver_adjoint``) the model is configured with.
    Users should compose this primitive into their training loop
    (notebook-level or ``pipekit_train.TrainingLoop``) rather than
    reimplementing it.

    Algorithm:
        1. Evaluate [`train_loss_fn`][vardax.train_loss_fn] and its gradient with
           respect to the model parameters.
        2. Turn the gradient into an update with the optax optimiser.
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
    loss, grads = eqx.filter_value_and_grad(train_loss_fn)(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Compute the evaluation loss for a single batch (no gradient).

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
    pred = model(batch)
    target = batch.target
    if target is None:
        raise MissingInputError("eval_step requires batch.target to be set.")
    return reconstruction_loss(pred, target)


def amortized_nll_loss_fn(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    r"""Negative log-likelihood for amortized inference (Epic 8).

    For ``AmortizedPosterior`` with flow / regression heads the maximum-
    likelihood objective on simulated pairs is

    $$
    \mathcal{L}_\text{MLE}(\phi) = -\mathbb{E}_{(x, y)} \log q_\phi(x \mid y).
    $$

    Algorithm:
        1. Evaluate $\log q_\phi(x \mid y)$ for each sample with
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
    log_prob = model.log_prob(target, batch)
    return -jnp.mean(log_prob)


@eqx.filter_jit
def amortized_train_step(
    model: Any,
    batch: Batch1D | Batch2D,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Any, optax.OptState, Float[Array, ""]]:
    """Single training step for ``AmortizedPosterior``.

    Same shape as [`train_step`][vardax.train_step] but uses
    [`amortized_nll_loss_fn`][vardax.amortized_nll_loss_fn] instead of the MSE
    reconstruction loss. Use this for simulation-based training of amortized
    variants; use ``train_step`` for ``FourDVarNet`` and classical models that
    reconstruct fields.

    Algorithm:
        1. Evaluate the NLL and its gradient with respect to the parameters.
        2. Turn the gradient into an update with the optax optimiser.
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
    loss, grads = eqx.filter_value_and_grad(amortized_nll_loss_fn)(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
