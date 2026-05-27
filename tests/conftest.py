"""Shared pytest fixtures for vardax tests."""

import jax
import jax.numpy as jnp
import pytest

from vardax import Batch1D, Batch2D, Batch2DMultivar


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def batch_1d():
    key = jax.random.PRNGKey(42)
    B, T, N = 2, 5, 16
    k1, k2 = jax.random.split(key)
    target = jax.random.normal(k1, (B, T, N))
    mask = (jax.random.uniform(k2, (B, T, N)) > 0.3).astype(jnp.float32)
    inp = target * mask
    return Batch1D(input=inp, mask=mask, target=target)


@pytest.fixture
def batch_2d():
    key = jax.random.PRNGKey(7)
    B, T, H, W = 2, 3, 8, 8
    k1, k2 = jax.random.split(key)
    target = jax.random.normal(k1, (B, T, H, W))
    mask = (jax.random.uniform(k2, (B, T, H, W)) > 0.3).astype(jnp.float32)
    inp = target * mask
    return Batch2D(input=inp, mask=mask, target=target)


@pytest.fixture
def batch_2d_multivar():
    key = jax.random.PRNGKey(99)
    B, T, C, H, W = 2, 3, 2, 8, 8
    k1, k2 = jax.random.split(key)
    target = jax.random.normal(k1, (B, T, C, H, W))
    mask = (jax.random.uniform(k2, (B, T, C, H, W)) > 0.3).astype(jnp.float32)
    inp = target * mask
    return Batch2DMultivar(input=inp, mask=mask, target=target)
