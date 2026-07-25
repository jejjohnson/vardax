"""Tests for `vardax.InterpObs` (issue #49).

Forward bilinear interpolation, mask kwarg, adjoint identity via the
`linearize().transpose()` path, JAX traceability, and protocol
conformance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vardax import (
    ObservationOperator,
    interp_obs_from_coords,
)


@pytest.fixture
def grid():
    y = np.linspace(0.0, 1.0, 11)
    x = np.linspace(0.0, 2.0, 21)
    return y, x


def test_constant_field_returns_constant(grid):
    """A constant field returns that constant at every obs point."""
    y, x = grid
    field = jnp.full((y.size, x.size), 3.7)
    pts = np.array([[0.1, 0.2], [0.55, 1.13], [0.95, 1.91]])
    op = interp_obs_from_coords((y, x), pts)
    out = op(field)
    assert out.shape == (3,)
    np.testing.assert_allclose(np.asarray(out), 3.7, atol=1e-5)


def test_weights_sum_to_one(grid):
    """Each row of the cached bilinear weights sums to 1."""
    y, x = grid
    pts = np.array([[0.1, 0.2], [0.55, 1.13], [0.95, 1.91], [0.0, 0.0]])
    op = interp_obs_from_coords((y, x), pts)
    row_sums = np.asarray(op.weights).sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_adjoint_identity_via_linearize_transpose(grid):
    """``⟨Hx, d⟩ ≈ ⟨x, H^T d⟩`` via the `linearize` → `lineax` path."""
    y, x = grid
    rng = np.random.default_rng(0)
    pts = rng.uniform([0.05, 0.05], [0.95, 1.95], size=(7, 2))
    op = interp_obs_from_coords((y, x), pts)

    field = jnp.asarray(rng.standard_normal((y.size, x.size)).astype(np.float32))
    d = jnp.asarray(rng.standard_normal(7).astype(np.float32))

    H = op.linearize(field)
    hx = H.mv(field)
    htd = H.transpose().mv(d)
    lhs = float(jnp.sum(hx * d))
    rhs = float(jnp.sum(field * htd))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-4)


def test_satisfies_observation_operator_protocol(grid):
    y, x = grid
    op = interp_obs_from_coords((y, x), np.array([[0.5, 1.0]]))
    assert isinstance(op, ObservationOperator)


def test_linearize_returns_lineax_op(grid):
    """`linearize` returns a `lineax.AbstractLinearOperator` so it
    plugs into vardax's GN Hessian and `LaplaceCovariance`."""
    import lineax as lx

    y, x = grid
    op = interp_obs_from_coords((y, x), np.array([[0.5, 1.0]]))
    L = op.linearize(jnp.zeros((y.size, x.size), jnp.float32))
    assert isinstance(L, lx.AbstractLinearOperator)


def test_linearize_preserves_flat_state_shape(grid):
    """When variational models carry flat ``(N,)`` state vectors, the
    operator structure must follow the input — not be forced through
    ``grid_shape`` — so it composes with flat prior covariances."""
    y, x = grid
    rng = np.random.default_rng(1)
    pts = rng.uniform([0.05, 0.05], [0.95, 1.95], size=(5, 2))
    op = interp_obs_from_coords((y, x), pts)

    flat = jnp.asarray(rng.standard_normal(y.size * x.size).astype(np.float32))
    L_flat = op.linearize(flat)
    in_struct = L_flat.in_structure()
    assert in_struct.shape == (y.size * x.size,)

    grid2d = flat.reshape(y.size, x.size)
    np.testing.assert_allclose(
        np.asarray(L_flat.mv(flat)), np.asarray(op(grid2d)), atol=1e-5
    )


def test_mask_kwarg_zeroes_masked_obs(grid):
    """Optional `mask` matches the `MaskedIdentity` style for
    `_accepts_mask` dispatch."""
    y, x = grid
    pts = np.array([[0.5, 1.0], [0.3, 0.5]])
    op = interp_obs_from_coords((y, x), pts)
    field = jnp.ones((y.size, x.size))
    mask = jnp.array([1.0, 0.0])
    out = op(field, mask=mask)
    np.testing.assert_allclose(np.asarray(out), [1.0, 0.0], atol=1e-5)


def test_jit_grad_traceable(grid):
    """Forward is `jit`-able and `grad`-able through the cost."""
    y, x = grid
    op = interp_obs_from_coords((y, x), np.array([[0.5, 1.0], [0.3, 0.5]]))

    def loss(f):
        return jnp.sum(op(f) ** 2)

    field = jnp.ones((y.size, x.size)) * 0.5
    g = jax.grad(loss)(field)
    # Two obs, 4-point stencils, possibly overlapping: between 4 and 8
    # nonzero entries depending on whether the stencils touch a shared
    # cell. The gradient itself must be finite + JAX-traced.
    assert jnp.all(jnp.isfinite(g))
    assert int(jnp.sum(g != 0)) >= 3


def test_rejects_higher_order():
    with pytest.raises(NotImplementedError, match="order=1"):
        interp_obs_from_coords(
            (np.arange(5.0), np.arange(5.0)), np.zeros((1, 2)), order=2
        )


def test_rejects_bad_obs_shape(grid):
    y, x = grid
    with pytest.raises(ValueError, match=r"\(n_obs, 2\)"):
        interp_obs_from_coords((y, x), np.zeros((3, 4)))


def test_rejects_bad_grid_arity():
    with pytest.raises(ValueError, match="grid_coords"):
        interp_obs_from_coords((np.arange(5.0),), np.zeros((1, 2)))


def test_rejects_descending_axis():
    """Descending axes (common for latitude grids) would silently produce
    wrong stencils; reject them at construction with a clear error."""
    y_desc = np.linspace(1.0, 0.0, 11)
    x_axis = np.linspace(0.0, 2.0, 21)
    with pytest.raises(ValueError, match="strictly increasing"):
        interp_obs_from_coords((y_desc, x_axis), np.array([[0.5, 1.0]]))


def test_rejects_non_monotonic_axis():
    y_axis = np.array([0.0, 0.5, 0.4, 1.0])
    x_axis = np.linspace(0.0, 2.0, 21)
    with pytest.raises(ValueError, match="strictly increasing"):
        interp_obs_from_coords((y_axis, x_axis), np.array([[0.5, 1.0]]))


def test_is_pytree_with_grid_shape_static(grid):
    """`InterpObs` is an `eqx.Module` pytree; `grid_shape` is static."""
    y, x = grid
    op = interp_obs_from_coords((y, x), np.array([[0.5, 1.0]]))
    leaves = jax.tree_util.tree_leaves(op)
    # `indices` and `weights` are JAX leaves; `grid_shape` is static.
    assert len(leaves) == 2
