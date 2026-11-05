"""The documented ``vardax.adapters`` import path works."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import pytest

import vardax as vdx


def _batch(target):
    return vdx.Batch1D(
        input=jnp.zeros((1, 1, 2)), mask=jnp.ones((1, 1, 2)), target=target
    )


def test_recon_loss_is_the_mean_squared_error_to_the_target():
    pipekit_train = importlib.import_module("vardax.adapters.pipekit_train")
    prediction = jnp.array([[[1.0, 3.0]]])
    target = jnp.array([[[0.0, 1.0]]])

    loss, metrics = pipekit_train.VardaxReconLoss()(prediction, _batch(target))

    # mean((1, 2)²) = 5 / 2
    assert jnp.allclose(loss, 2.5)
    assert jnp.allclose(metrics["recon_mse"], 2.5)


def test_recon_loss_needs_a_target():
    pipekit_train = importlib.import_module("vardax.adapters.pipekit_train")
    with pytest.raises(vdx.MissingInputError, match="non-None `target`"):
        pipekit_train.VardaxReconLoss()(jnp.zeros((1, 1, 2)), _batch(None))
