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

from ._types import Batch1D, Batch2D


def reconstruction_loss(
    pred: Float[Array, ...],
    target: Float[Array, ...],
) -> Float[Array, ""]:
    """Mean-squared reconstruction loss.

    Args:
        pred: Model predictions, arbitrary shape.
        target: Ground-truth targets, same shape as ``pred``.

    Returns:
        Scalar mean-squared error.
    """
    return jnp.mean((pred - target) ** 2)


def train_loss_fn(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Compute the training loss for a single batch.

    Args:
        model: Equinox module implementing ``__call__(batch) -> prediction``.
        batch: Training batch (must have a non-``None`` ``target``).

    Returns:
        Scalar reconstruction loss.
    """
    pred = model(batch)
    target = batch.target
    if target is None:
        raise ValueError("train_loss_fn requires batch.target to be set.")
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
    strategy (``"unrolled"`` / ``"one_step"`` / ``"implicit"``) the
    model is configured with. Users should compose this primitive into
    their training loop (notebook-level or ``pipekit_train.TrainingLoop``)
    rather than reimplementing it.

    Args:
        model: Equinox module to optimise.
        batch: Training batch.
        optimizer: Optax gradient transformation (e.g. ``optax.adam(1e-3)``).
        opt_state: Current optimiser state.

    Returns:
        Tuple of (updated model, updated optimiser state, scalar loss).
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
        model: Equinox module.
        batch: Evaluation batch (must have a non-``None`` ``target``).

    Returns:
        Scalar reconstruction loss.
    """
    pred = model(batch)
    target = batch.target
    if target is None:
        raise ValueError("eval_step requires batch.target to be set.")
    return reconstruction_loss(pred, target)


def amortized_nll_loss_fn(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Negative log-likelihood for amortized inference (Epic 8).

    For ``AmortizedPosterior`` with flow / regression heads the maximum-
    likelihood objective on simulated pairs is

    .. math::

        \\mathcal{L}_\\text{MLE}(\\phi) = -\\mathbb{E}_{(x, y)} \\log q_\\phi(x \\mid y).

    Args:
        model: ``AmortizedPosterior`` (any head with a ``.log_prob``
            method).
        batch: Training batch with ``target = x`` and ``input = y``.

    Returns:
        Scalar NLL averaged over the batch.
    """
    target = batch.target
    if target is None:
        raise ValueError(
            "amortized_nll_loss_fn requires batch.target (the true x) to be set."
        )
    log_p = model.log_prob(target, batch)
    return -jnp.mean(log_p)


@eqx.filter_jit
def amortized_train_step(
    model: Any,
    batch: Batch1D | Batch2D,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Any, optax.OptState, Float[Array, ""]]:
    """Single training step for ``AmortizedPosterior``.

    Same shape as ``train_step`` but uses ``amortized_nll_loss_fn``
    instead of the MSE reconstruction loss. Use this for simulation-
    based training of amortized variants; use ``train_step`` for
    ``FourDVarNet`` and classical models that reconstruct fields.
    """
    loss, grads = eqx.filter_value_and_grad(amortized_nll_loss_fn)(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
