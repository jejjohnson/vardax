"""Tests for vardax._src.costs."""

import jax.numpy as jnp
import pytest

from vardax import obs_cost_1d, obs_cost_2d, prior_cost


class TestObsCost1D:
    def test_zero_when_perfect(self, batch_1d):
        # When state == obs everywhere the mask is 1, cost should be ~0
        cost = obs_cost_1d(batch_1d.input, batch_1d.input, batch_1d.mask)
        assert float(cost) == pytest.approx(0.0)

    def test_positive(self, batch_1d):
        cost = obs_cost_1d(batch_1d.target, batch_1d.input, batch_1d.mask)
        assert float(cost) >= 0.0

    def test_scalar_output(self, batch_1d):
        cost = obs_cost_1d(batch_1d.target, batch_1d.input, batch_1d.mask)
        assert cost.ndim == 0


class TestObsCost2D:
    def test_zero_when_perfect(self, batch_2d):
        cost = obs_cost_2d(batch_2d.input, batch_2d.input, batch_2d.mask)
        assert float(cost) == pytest.approx(0.0)

    def test_positive(self, batch_2d):
        cost = obs_cost_2d(batch_2d.target, batch_2d.input, batch_2d.mask)
        assert float(cost) >= 0.0

    def test_scalar_output(self, batch_2d):
        cost = obs_cost_2d(batch_2d.target, batch_2d.input, batch_2d.mask)
        assert cost.ndim == 0


class TestPriorCost:
    def test_zero_when_identical(self):
        x = jnp.ones((2, 5, 16))
        cost = prior_cost(x, x)
        assert float(cost) == pytest.approx(0.0)

    def test_positive(self):
        x = jnp.ones((2, 5, 16))
        x_recon = jnp.zeros((2, 5, 16))
        cost = prior_cost(x, x_recon)
        assert float(cost) > 0.0

    def test_scalar_output(self):
        x = jnp.ones((2, 5, 16))
        x_recon = jnp.zeros((2, 5, 16))
        cost = prior_cost(x, x_recon)
        assert cost.ndim == 0


class TestVariationalCost:
    def test_scalar_output(self, batch_1d):
        from vardax import variational_cost

        identity_fn = lambda x: x
        cost = variational_cost(batch_1d.target, batch_1d, identity_fn)
        assert cost.ndim == 0

    def test_non_negative(self, batch_1d):
        from vardax import variational_cost

        identity_fn = lambda x: x
        cost = variational_cost(batch_1d.target, batch_1d, identity_fn)
        assert float(cost) >= 0.0

    def test_differentiable(self, batch_1d):
        import jax

        from vardax import variational_cost

        identity_fn = lambda x: x
        grad = jax.grad(variational_cost)(batch_1d.target, batch_1d, identity_fn)
        assert grad.shape == batch_1d.target.shape


class TestVariationalCostGrad:
    def test_output_shape(self, batch_1d):
        from vardax import variational_cost_grad

        identity_fn = lambda x: x
        grad = variational_cost_grad(batch_1d.target, batch_1d, identity_fn)
        assert grad.shape == batch_1d.target.shape


class TestDecomposedLoss:
    def test_keys(self, batch_1d):
        from vardax import decomposed_loss

        identity_fn = lambda x: x
        result = decomposed_loss(batch_1d.target, batch_1d, identity_fn)
        assert set(result.keys()) == {"obs", "prior", "total"}

    def test_total_equals_sum(self, batch_1d):
        import pytest

        from vardax import decomposed_loss

        identity_fn = lambda x: x
        result = decomposed_loss(batch_1d.target, batch_1d, identity_fn)
        assert float(result["total"]) == pytest.approx(
            float(result["obs"]) + float(result["prior"])
        )
