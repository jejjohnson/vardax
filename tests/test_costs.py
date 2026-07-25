"""Tests for vardax._src.costs."""

import jax
import jax.numpy as jnp
import pytest

from vardax import (
    Batch1D,
    background_cost,
    obs_cost_1d,
    obs_cost_2d,
    prior_cost,
    strong_variational_cost,
)


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


class TestObsCostNaNSafe:
    """NaN-safe observation cost (issue #16, ported from mfourdvar)."""

    def _nan_obs_1d(self):
        state = jnp.ones((1, 1, 4))
        obs = jnp.array([[[0.0, jnp.nan, 0.0, jnp.nan]]])
        mask = jnp.array([[[1.0, 0.0, 1.0, 0.0]]])
        return state, obs, mask

    def test_nan_poisons_without_flag(self):
        state, obs, mask = self._nan_obs_1d()
        # 0 * NaN == NaN in JAX, so even masked-out NaNs poison the cost.
        assert bool(jnp.isnan(obs_cost_1d(state, obs, mask)))

    def test_nan_safe_value(self):
        state, obs, mask = self._nan_obs_1d()
        # Only the two masked-in points (obs 0) contribute: (1-0)^2 twice / 4.
        cost = obs_cost_1d(state, obs, mask, nan_to_num=True)
        assert float(cost) == pytest.approx(0.5)

    def test_nan_safe_gradient_finite(self):
        state, obs, mask = self._nan_obs_1d()
        grad = jax.grad(lambda s: obs_cost_1d(s, obs, mask, nan_to_num=True))(state)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_matches_plain_when_no_nans(self, batch_1d):
        plain = obs_cost_1d(batch_1d.target, batch_1d.input, batch_1d.mask)
        safe = obs_cost_1d(
            batch_1d.target, batch_1d.input, batch_1d.mask, nan_to_num=True
        )
        assert float(safe) == pytest.approx(float(plain))

    def test_2d_nan_safe(self):
        state = jnp.ones((1, 1, 2, 2))
        obs = jnp.array([[[[0.0, jnp.nan], [0.0, jnp.nan]]]])
        mask = jnp.array([[[[1.0, 0.0], [1.0, 0.0]]]])
        assert bool(jnp.isnan(obs_cost_2d(state, obs, mask)))
        cost = obs_cost_2d(state, obs, mask, nan_to_num=True)
        assert float(cost) == pytest.approx(0.5)


class TestBackgroundCost:
    def test_zero_when_identical(self):
        x0 = jnp.ones((4,))
        assert float(background_cost(x0, x0)) == pytest.approx(0.0)

    def test_value(self):
        x0 = jnp.ones((4,))
        xb = jnp.zeros((4,))
        assert float(background_cost(x0, xb)) == pytest.approx(1.0)


class TestStrongVariationalCost:
    def _setup(self):
        x0 = jnp.ones((1, 4))
        ts = jnp.linspace(0.0, 1.0, 3)
        batch = Batch1D(input=jnp.zeros((1, 3, 4)), mask=jnp.ones((1, 3, 4)))

        def forward_fn(x0_, ts_):
            # Constant-in-time rollout: (B, N) -> (B, T, N).
            return jnp.broadcast_to(x0_[:, None, :], (1, 3, 4))

        return x0, ts, batch, forward_fn

    def test_obs_only(self):
        x0, ts, batch, forward_fn = self._setup()
        # obs term: mean((1-0)^2) = 1; bg term 0 (xb defaults to x0).
        cost = strong_variational_cost(
            x0, ts, batch, forward_fn, alpha_obs=0.5, alpha_bg=0.5
        )
        assert float(cost) == pytest.approx(0.5)

    def test_background_term(self):
        x0, ts, batch, forward_fn = self._setup()
        # obs = 1, bg = mean((1-0)^2) = 1, both unit-weighted -> 2.
        cost = strong_variational_cost(
            x0,
            ts,
            batch,
            forward_fn,
            xb=jnp.zeros((1, 4)),
            alpha_obs=1.0,
            alpha_bg=1.0,
        )
        assert float(cost) == pytest.approx(2.0)

    def test_nan_safe(self):
        x0, ts, _, forward_fn = self._setup()
        obs = jnp.full((1, 3, 4), jnp.nan)
        batch = Batch1D(input=obs, mask=jnp.zeros((1, 3, 4)))
        # All masked out; with nan_to_num the fully-NaN obs must not poison.
        cost = strong_variational_cost(
            x0, ts, batch, forward_fn, alpha_obs=1.0, alpha_bg=0.0, nan_to_num=True
        )
        assert float(cost) == pytest.approx(0.0)

    def test_grad_wrt_x0(self):
        x0, ts, batch, forward_fn = self._setup()
        grad = jax.grad(lambda z: strong_variational_cost(z, ts, batch, forward_fn))(x0)
        assert grad.shape == x0.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
