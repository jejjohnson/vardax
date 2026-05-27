"""Training utilities for 4DVarNet.

Provides loss functions, per-step training / evaluation functions, and a
high-level ``fit`` loop compatible with Flax NNX and ``optax``.
"""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax.numpy as jnp
from jaxtyping import Array, Float
import optax

from ._types import Batch1D, Batch2D

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


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
        model: Flax NNX module.
        batch: Training batch.

    Returns:
        Scalar reconstruction loss.
    """
    pred = model(batch)
    return reconstruction_loss(pred, batch.target)


# ---------------------------------------------------------------------------
# Training / evaluation steps
# ---------------------------------------------------------------------------


def train_step(
    model: Any,
    optimizer: nnx.Optimizer,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Perform a single training step (forward + backward + update).

    Args:
        model: Flax NNX module.
        optimizer: NNX optimizer wrapping the model parameters.
        batch: Training batch.

    Returns:
        Scalar training loss.
    """
    loss, grads = nnx.value_and_grad(train_loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss


def eval_step(
    model: Any,
    batch: Batch1D | Batch2D,
) -> Float[Array, ""]:
    """Compute the evaluation loss for a single batch (no gradient).

    Args:
        model: Flax NNX module.
        batch: Evaluation batch.

    Returns:
        Scalar reconstruction loss.
    """
    pred = model(batch)
    return reconstruction_loss(pred, batch.target)


# ---------------------------------------------------------------------------
# High-level fit loop
# ---------------------------------------------------------------------------


def fit(
    model: Any,
    train_batches: list[Batch1D | Batch2D],
    *,
    lr: float = 1e-3,
    n_epochs: int = 10,
    val_batches: list[Batch1D | Batch2D] | None = None,
    verbose: bool = True,
) -> tuple[nnx.Optimizer, list[float], list[float]]:
    """Train a 4DVarNet model for multiple epochs.

    Iterates over epochs and batches, logging train (and optional validation)
    losses.  The ``model`` is updated in-place.

    Args:
        model: Flax NNX module (already initialised).
        train_batches: List of training batches.
        lr: Learning rate for the Adam optimiser.
        n_epochs: Number of training epochs.
        val_batches: Optional list of validation batches.
        verbose: Whether to print per-epoch losses.

    Returns:
        Tuple of (optimizer, train loss history, val loss history).
    """
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(n_epochs):
        epoch_train_losses = []
        for batch in train_batches:
            loss = train_step(model, optimizer, batch)
            epoch_train_losses.append(float(loss))

        mean_train = float(jnp.mean(jnp.array(epoch_train_losses)))
        train_losses.append(mean_train)

        mean_val = float("nan")
        if val_batches is not None:
            epoch_val_losses = [float(eval_step(model, b)) for b in val_batches]
            mean_val = float(jnp.mean(jnp.array(epoch_val_losses)))
        val_losses.append(mean_val)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs} — "
                f"train loss: {mean_train:.6f}"
                + (f"  val loss: {mean_val:.6f}" if val_batches else "")
            )

    return optimizer, train_losses, val_losses
