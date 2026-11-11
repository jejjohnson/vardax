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
    `accepts_mask` dispatch."""
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
    # off the grid nodes, where a stencil has all four corners
    op = interp_obs_from_coords((y, x), np.array([[0.52, 1.03], [0.33, 0.47]]))

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


@pytest.mark.unchecked_types
def test_rejects_bad_obs_shape(grid):
    y, x = grid
    with pytest.raises(ValueError, match=r"\(n_obs, 2\)"):
        interp_obs_from_coords((y, x), np.zeros((3, 4)))


@pytest.mark.unchecked_types
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


def test_a_narrow_cell_on_a_wide_axis_keeps_its_offsets():
    """Scaling by the whole span would collapse this cell in float32."""
    y = np.array([0.0, 1e8, 1e8 + 1.0, 2e8])
    x = np.array([0.0, 1.0])
    op = interp_obs_from_coords(
        (y, x), np.array([[1e8 + 0.25, 0.0], [1e8 + 0.75, 0.0]])
    )
    # in the narrow cell, at offsets 0.25 and 0.75; x sits on its first node
    np.testing.assert_array_equal(np.asarray(op.indices[:, 0]), [2, 2])
    np.testing.assert_allclose(np.asarray(op.weights[:, 0]), [0.75, 0.25], atol=1e-6)
    np.testing.assert_allclose(np.asarray(op.weights[:, 2]), [0.25, 0.75], atol=1e-6)


def test_building_the_stencil_inside_jit():
    """The coordinates are concrete, so building under jit stays possible."""
    axis = np.arange(3.0)

    def predict(state):
        return interp_obs_from_coords((axis, axis), np.array([[0.5, 0.5]]))(state)

    assert jnp.allclose(jax.jit(predict)(jnp.ones((3, 3))), 1.0)


def test_projected_coordinates_keep_their_offset_in_a_cell():
    """Coordinates of order 1e6 are computed in float64, then cast."""
    y = 4.0e6 + 1000.0 * np.arange(5.0)
    x = 5.0e5 + 1000.0 * np.arange(5.0)
    op = interp_obs_from_coords((y, x), np.array([[4.0e6 + 1250.0, 5.0e5 + 2500.0]]))
    # in cell (1, 2) at offsets (0.25, 0.5)
    expected = [0.375, 0.375, 0.125, 0.125]
    np.testing.assert_allclose(np.asarray(op.weights[0]), expected, atol=1e-5)


def test_an_empty_batch_has_an_empty_stencil(grid):
    y, x = grid
    op = interp_obs_from_coords((y, x), np.zeros((0, 2)))
    assert op.indices.shape == (0, 4)
    assert op.weights.shape == (0, 4)
    assert op(jnp.ones((y.size, x.size))).shape == (0,)


@pytest.mark.parametrize("flat", [False, True], ids=["grid", "flat"])
def test_linearize_and_its_transpose_match_the_dense_jacobian(grid, flat):
    y, x = grid
    op = interp_obs_from_coords((y, x), np.array([[0.52, 1.03], [0.33, 0.47]]))
    state = jnp.ones(y.size * x.size if flat else (y.size, x.size))
    tangent = op.linearize(state)
    dense = jax.jacfwd(op)(state).reshape(2, -1)
    direction = jnp.arange(state.size, dtype=state.dtype).reshape(state.shape)
    np.testing.assert_allclose(
        tangent.mv(direction), dense @ jnp.ravel(direction), rtol=1e-5
    )
    cotangent = tangent.transpose().mv(jnp.array([1.0, -2.0]))
    assert cotangent.shape == state.shape
    np.testing.assert_allclose(
        jnp.ravel(cotangent), dense.T @ jnp.array([1.0, -2.0]), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("flat", [False, True], ids=["grid", "flat"])
def test_empty_linearize_and_adjoint_preserve_structure_and_jit(grid, flat):
    y, x = grid
    op = interp_obs_from_coords((y, x), np.zeros((0, 2)))
    state = jnp.ones(y.size * x.size if flat else (y.size, x.size), dtype=jnp.float32)
    tangent = op.linearize(state)
    empty = jnp.empty((0,), dtype=state.dtype)
    for forward, adjoint in (
        (tangent.mv(state), tangent.transpose().mv(empty)),
        (
            jax.jit(lambda z: op.linearize(z).mv(z))(state),
            jax.jit(lambda z: tangent.transpose().mv(z))(empty),
        ),
    ):
        assert forward.shape == (0,)
        assert forward.dtype == state.dtype
        assert adjoint.shape == state.shape
        assert adjoint.dtype == state.dtype
        np.testing.assert_array_equal(adjoint, jnp.zeros_like(state))
