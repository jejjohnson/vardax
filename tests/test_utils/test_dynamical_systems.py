"""Tests for vardax._src.utils.dynamical_systems."""

import jax
import jax.numpy as jnp
import pytest

from vardax._src.utils.dynamical_systems import (
    Lorenz63,
    Lorenz96,
    simulate_lorenz63,
    simulate_lorenz96,
)


class TestLorenz63Module:
    def test_output_shape(self):
        model = Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        y = jnp.array([1.0, 1.0, 1.0])
        dy = model(0.0, y, None)
        assert dy.shape == (3,)

    def test_fixed_point(self):
        """At the fixed point the vector field should be zero."""
        beta = 8.0 / 3.0
        rho = 28.0
        sigma = 10.0
        fp = jnp.array(
            [
                jnp.sqrt(beta * (rho - 1)),
                jnp.sqrt(beta * (rho - 1)),
                rho - 1.0,
            ]
        )
        model = Lorenz63(sigma=sigma, rho=rho, beta=beta)
        dy = model(0.0, fp, None)
        assert jnp.allclose(dy, jnp.zeros(3), atol=1e-5)


class TestSimulateLorenz63:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        n_steps = 200
        n_burn_in = 50
        time_coords, states = simulate_lorenz63(
            key, n_steps=n_steps, n_burn_in=n_burn_in
        )
        # After burn-in we have n_steps + 1 saved points
        assert states.shape == (n_steps + 1, 3)
        assert time_coords.shape == (n_steps + 1,)

    def test_time_starts_at_zero(self):
        key = jax.random.PRNGKey(1)
        time_coords, _ = simulate_lorenz63(key, n_steps=100, n_burn_in=50)
        assert float(time_coords[0]) == pytest.approx(0.0, abs=1e-6)

    def test_deterministic(self):
        key = jax.random.PRNGKey(42)
        _, states1 = simulate_lorenz63(key, n_steps=100, n_burn_in=50)
        _, states2 = simulate_lorenz63(key, n_steps=100, n_burn_in=50)
        assert jnp.allclose(states1, states2)

    def test_different_keys_give_different_results(self):
        k1 = jax.random.PRNGKey(0)
        k2 = jax.random.PRNGKey(1)
        _, s1 = simulate_lorenz63(k1, n_steps=100, n_burn_in=50)
        _, s2 = simulate_lorenz63(k2, n_steps=100, n_burn_in=50)
        assert not jnp.allclose(s1, s2)

    def test_bounded_attractor(self):
        """Classic L63 attractor stays bounded — no blow-up."""
        key = jax.random.PRNGKey(7)
        _, states = simulate_lorenz63(key, n_steps=2000, n_burn_in=500)
        assert jnp.all(jnp.abs(states) < 200)

    def test_explicit_x0(self):
        key = jax.random.PRNGKey(0)
        x0 = jnp.array([1.0, 0.0, 0.0])
        _time_coords, states = simulate_lorenz63(key, n_steps=100, n_burn_in=10, x0=x0)
        assert states.shape[1] == 3

    def test_burn_in_excluded(self):
        """Burn-in points must not appear in the returned trajectory.

        We verify this indirectly: running with ``n_burn_in=0`` and
        ``n_burn_in=200`` on the same key should give different final
        states, because the extra burn-in steps advance the attractor.
        """
        key = jax.random.PRNGKey(3)
        x0 = jax.random.normal(key, (3,)) * 0.01 + jnp.array([8.5, 8.5, 27.0])
        _, s_no_burn = simulate_lorenz63(key, n_steps=100, n_burn_in=0, x0=x0)
        _, s_burn = simulate_lorenz63(key, n_steps=100, n_burn_in=200, x0=x0)
        assert not jnp.allclose(s_no_burn[0], s_burn[0])


class TestLorenz96Module:
    def test_output_shape(self):
        model = Lorenz96(F=8.0)
        y = jnp.ones((40,))
        dy = model(0.0, y, None)
        assert dy.shape == (40,)

    def test_uniform_fixed_point(self):
        """Uniform state x_k = F is a fixed point of the L96 ODE."""
        F = 8.0
        N = 40
        model = Lorenz96(F=F)
        y = jnp.full((N,), F)
        dy = model(0.0, y, None)
        assert jnp.allclose(dy, jnp.zeros(N), atol=1e-5)


class TestSimulateLorenz96:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        N = 40
        n_steps = 100
        n_burn_in = 20
        time_coords, states = simulate_lorenz96(
            key, N=N, n_steps=n_steps, n_burn_in=n_burn_in
        )
        assert states.shape == (n_steps + 1, N)
        assert time_coords.shape == (n_steps + 1,)

    def test_time_starts_at_zero(self):
        key = jax.random.PRNGKey(1)
        time_coords, _ = simulate_lorenz96(key, n_steps=100, n_burn_in=20)
        assert float(time_coords[0]) == pytest.approx(0.0, abs=1e-6)

    def test_deterministic(self):
        key = jax.random.PRNGKey(42)
        _, states1 = simulate_lorenz96(key, n_steps=100, n_burn_in=20)
        _, states2 = simulate_lorenz96(key, n_steps=100, n_burn_in=20)
        assert jnp.allclose(states1, states2)

    def test_different_keys_give_different_results(self):
        k1 = jax.random.PRNGKey(0)
        k2 = jax.random.PRNGKey(1)
        _, s1 = simulate_lorenz96(k1, n_steps=100, n_burn_in=20)
        _, s2 = simulate_lorenz96(k2, n_steps=100, n_burn_in=20)
        assert not jnp.allclose(s1, s2)

    def test_bounded_attractor(self):
        """L96 attractor stays bounded — no blow-up."""
        key = jax.random.PRNGKey(7)
        _, states = simulate_lorenz96(key, n_steps=500, n_burn_in=200)
        assert jnp.all(jnp.abs(states) < 200)

    def test_custom_n_and_f(self):
        key = jax.random.PRNGKey(5)
        time_coords, states = simulate_lorenz96(
            key, N=20, F=10.0, n_steps=50, n_burn_in=10
        )
        assert states.shape == (51, 20)
        assert time_coords.shape == (51,)
