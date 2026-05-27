"""Tests for vardax._src.training."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from vardax import FourDVarNet1D, reconstruction_loss
from vardax._src.training import eval_step, train_loss_fn, train_step


@pytest.fixture
def model_and_optimizer(rng, batch_1d):
    _, T, N = batch_1d.input.shape
    model = FourDVarNet1D(
        state_dim=N,
        n_time=T,
        latent_dim=8,
        hidden_dim=16,
        n_solver_steps=2,
        key=rng,
    )
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return model, optimizer, opt_state


class TestReconstructionLoss:
    def test_zero_for_identical(self):
        x = jnp.ones((2, 5, 16))
        loss = reconstruction_loss(x, x)
        assert float(loss) == pytest.approx(0.0)

    def test_positive(self):
        pred = jnp.ones((2, 5, 16))
        target = jnp.zeros((2, 5, 16))
        loss = reconstruction_loss(pred, target)
        assert float(loss) > 0.0

    def test_scalar(self):
        loss = reconstruction_loss(jnp.ones((2, 3)), jnp.zeros((2, 3)))
        assert loss.ndim == 0


class TestTrainLossFn:
    def test_returns_scalar(self, batch_1d, model_and_optimizer):
        model, _, _ = model_and_optimizer
        loss = train_loss_fn(model, batch_1d)
        assert loss.ndim == 0
        assert float(loss) >= 0.0


class TestTrainStep:
    def test_params_change(self, batch_1d, model_and_optimizer):
        model, optimizer, opt_state = model_and_optimizer
        params_before = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        new_model, _, _ = train_step(model, batch_1d, optimizer, opt_state)
        params_after = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
        changed = any(
            not jnp.allclose(a, b)
            for a, b in zip(params_before, params_after, strict=True)
        )
        assert changed

    def test_loss_is_finite(self, batch_1d, model_and_optimizer):
        model, optimizer, opt_state = model_and_optimizer
        _, _, loss = train_step(model, batch_1d, optimizer, opt_state)
        assert jnp.isfinite(loss)


class TestEvalStep:
    def test_returns_scalar(self, batch_1d, model_and_optimizer):
        model, _, _ = model_and_optimizer
        loss = eval_step(model, batch_1d)
        assert loss.ndim == 0
        assert jnp.isfinite(loss)
