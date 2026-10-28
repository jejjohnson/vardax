"""Constructors keep the caller's precision instead of forcing float32."""

import einx
import jax
import jax.numpy as jnp

import vardax as vdx


def test_linear_basis_variance_follows_phi_dtype_in_float64():
    with jax.enable_x64(True):
        phi = jnp.eye(3, dtype=jnp.float64)
        basis = vdx.linear_basis(phi, variance=0.5)
        assert basis.variance.dtype == jnp.float64


def test_linear_basis_variance_follows_phi_dtype_in_float32():
    phi = jnp.eye(3, dtype=jnp.float32)
    basis = vdx.linear_basis(phi, variance=0.5)
    assert basis.variance.dtype == jnp.float32


def test_interp_obs_weights_are_float64_under_x64():
    with jax.enable_x64(True):
        y_axis = jnp.linspace(0.0, 1.0, 4)
        x_axis = jnp.linspace(0.0, 1.0, 5)
        obs_coords = jnp.array([[0.3, 0.7], [0.5, 0.1]])
        obs_op = vdx.interp_obs_from_coords((y_axis, x_axis), obs_coords)
        assert obs_op.weights.dtype == jnp.float64
        # bilinear weights must still sum to one at full precision
        weight_sums = einx.sum("o [k]", obs_op.weights)
        assert jnp.allclose(weight_sums, 1.0, rtol=1e-12)
