"""Tests for vardax._src.training."""

from flax import nnx
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
        rngs=nnx.Rngs(rng),
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    return model, optimizer


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
    def test_returns_scalar(self, rng, batch_1d, model_and_optimizer):
        model, _ = model_and_optimizer
        loss = train_loss_fn(model, batch_1d)
        assert loss.ndim == 0
        assert float(loss) >= 0.0


class TestTrainStep:
    def test_params_change(self, rng, batch_1d, model_and_optimizer):
        model, optimizer = model_and_optimizer
        params_before = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
        train_step(model, optimizer, batch_1d)
        params_after = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
        changed = any(
            not jnp.allclose(a, b)
            for a, b in zip(params_before, params_after, strict=True)
        )
        assert changed

    def test_loss_is_finite(self, rng, batch_1d, model_and_optimizer):
        model, optimizer = model_and_optimizer
        loss = train_step(model, optimizer, batch_1d)
        assert jnp.isfinite(loss)


class TestEvalStep:
    def test_returns_scalar(self, batch_1d, model_and_optimizer):
        model, _ = model_and_optimizer
        loss = eval_step(model, batch_1d)
        assert loss.ndim == 0
        assert jnp.isfinite(loss)
