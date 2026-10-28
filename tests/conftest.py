"""Shared pytest fixtures for vardax tests."""

import sys

import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import install_import_hook
import pytest

# The hook only instruments modules imported after it is installed, so
# vardax must not be imported before this point.
assert "vardax" not in sys.modules, "vardax imported before the jaxtyping hook"
with install_import_hook("vardax", "beartype.beartype"):
    import vardax  # noqa: F401

from vardax import Batch1D, Batch2D, Batch2DMultivar


@pytest.fixture(autouse=True)
def _unchecked_types(request):
    """Switch the runtime type checks off for ``@pytest.mark.unchecked_types``.

    Validation tests pass deliberately malformed inputs to check vardax's
    own errors, which users without the import hook see; the type checker
    would otherwise reject those inputs first.
    """
    if request.node.get_closest_marker("unchecked_types") is None:
        yield
        return
    jaxtyping.config.update("jaxtyping_disable", True)
    try:
        yield
    finally:
        jaxtyping.config.update("jaxtyping_disable", False)


@pytest.fixture
def x64():
    """Run the test in float64, for closed-form oracles at ``rtol=1e-12``."""
    with jax.enable_x64(True):
        yield


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
