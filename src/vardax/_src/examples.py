"""Notebook-demo helpers (not part of the library API).

These helpers exist for the tutorial notebooks under ``notebooks/``.
Production code should compose ``train_step`` (Layer 0 primitive)
with ``pipekit-train`` (``[train]`` extra) or with any other training
framework — vardax does not own training loops.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optax

from ._types import Batch1D, Batch2D
from .training import eval_step, train_step


def fit_demo(
    model: Any,
    train_batches: list[Batch1D | Batch2D],
    *,
    lr: float = 1e-3,
    n_epochs: int = 10,
    val_batches: list[Batch1D | Batch2D] | None = None,
    verbose: bool = True,
) -> tuple[Any, list[float], list[float]]:
    """Minimal notebook-demo training loop.

    Composes ``optax.adam`` + ``vardax.train_step`` inline. Not a library
    API — exists for pedagogical use in ``notebooks/``. For production
    training, use ``pipekit-train.TrainingLoop`` via
    ``vardax.adapters.pipekit_train``.

    Args:
        model: Equinox module to train.
        train_batches: List of training batches.
        lr: Adam learning rate.
        n_epochs: Number of epochs.
        val_batches: Optional validation batches.
        verbose: Print per-epoch losses.

    Returns:
        Tuple of (trained model, per-epoch mean train losses, per-epoch
        mean val losses (NaN when ``val_batches`` is None)).
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(n_epochs):
        epoch_train_losses: list[float] = []
        for batch in train_batches:
            model, opt_state, loss = train_step(model, batch, optimizer, opt_state)
            epoch_train_losses.append(float(loss))
        mean_train = float(jnp.mean(jnp.array(epoch_train_losses)))
        train_history.append(mean_train)

        mean_val = float("nan")
        if val_batches is not None:
            epoch_val_losses = [float(eval_step(model, b)) for b in val_batches]
            mean_val = float(jnp.mean(jnp.array(epoch_val_losses)))
        val_history.append(mean_val)

        if verbose:
            print(
                f"epoch {epoch + 1}/{n_epochs} — "
                f"train loss: {mean_train:.6f}"
                + (f"  val loss: {mean_val:.6f}" if val_batches else "")
            )

    return model, train_history, val_history
