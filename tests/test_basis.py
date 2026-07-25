"""Tests for the reduced-basis operators (issue #49, split from PR #50).

One protocol-conformance suite parametrized over every family
constructor, plus per-family checks (bump geometry, compact support,
spectral-density priors, orthonormality, explained variance).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vardax as vdx
from vardax import (
    CompositeBasis,
    LinearBasis,
    ReducedBasis,
    composite_basis,
    linear_basis,
)

GRID_Y = np.linspace(0.0, 1.0, 8)
GRID_X = np.linspace(0.0, 2.0, 16)
GRID_SHAPE = (GRID_Y.size, GRID_X.size)


def _make_rbf() -> LinearBasis:
    centers = np.array([[0.5, 1.0], [0.2, 0.4], [0.9, 1.7]])
    return vdx.rbf_basis((GRID_Y, GRID_X), centers, widths=0.1)


def _make_fourier() -> LinearBasis:
    return vdx.fourier_basis((GRID_Y, GRID_X), 3)


def _make_wavelet() -> LinearBasis:
    return vdx.wavelet_basis(GRID_SHAPE, wavelet="haar")


def _make_eof() -> LinearBasis:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(10, GRID_Y.size * GRID_X.size))
    return vdx.eof_basis(data, 4, output_shape=GRID_SHAPE)


def _make_linear() -> LinearBasis:
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(GRID_Y.size * GRID_X.size, 5))
    return linear_basis(phi, variance=2.0, output_shape=GRID_SHAPE)


def _make_composite() -> CompositeBasis:
    return composite_basis(_make_rbf(), _make_fourier())


FAMILIES = {
    "rbf": _make_rbf,
    "fourier": _make_fourier,
    "wavelet": _make_wavelet,
    "eof": _make_eof,
    "linear": _make_linear,
    "composite": _make_composite,
}


@pytest.fixture(params=sorted(FAMILIES), ids=sorted(FAMILIES))
def basis(request):
    return FAMILIES[request.param]()


class TestReducedBasisConformance:
    """Every family satisfies the same `ReducedBasis` contract."""

    def test_satisfies_protocol(self, basis):
        assert isinstance(basis, ReducedBasis)

    def test_operg_maps_coefficients_to_grid(self, basis):
        X = jnp.ones((basis.nbasis,))
        out = basis.operg(0.0, X)
        assert out.shape == GRID_SHAPE

    def test_operg_is_linear(self, basis):
        rng = np.random.default_rng(2)
        x1 = jnp.asarray(rng.normal(size=basis.nbasis), dtype=jnp.float32)
        x2 = jnp.asarray(rng.normal(size=basis.nbasis), dtype=jnp.float32)
        lhs = basis.operg(0.0, 2.0 * x1 + x2)
        rhs = 2.0 * basis.operg(0.0, x1) + basis.operg(0.0, x2)
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-4)

    def test_prior_inv_shape_and_positivity(self, basis):
        X = jnp.ones((basis.nbasis,))
        out = basis.prior_inv(X)
        assert out.shape == (basis.nbasis,)
        # Q is a positive diagonal, so X^T Q^{-1} X > 0 for X != 0.
        assert float(jnp.sum(X * out)) > 0.0

    def test_jit_traceable(self, basis):
        @jax.jit
        def call(X):
            return basis.operg(0.0, X)

        out = call(jnp.ones((basis.nbasis,)))
        assert out.shape == GRID_SHAPE


class TestRBF:
    def test_single_coefficient_makes_a_bump(self):
        centers = np.array([[0.5, 1.0], [0.2, 0.4]])
        basis = vdx.rbf_basis((GRID_Y, GRID_X), centers, widths=0.05)
        out = basis.operg(0.0, jnp.array([1.0, 0.0]))
        iy, ix = np.unravel_index(int(jnp.argmax(out)), GRID_SHAPE)
        assert abs(GRID_Y[iy] - 0.5) < 0.1
        assert abs(GRID_X[ix] - 1.0) < 0.1

    def test_wendland_has_compact_support(self):
        centers = np.array([[0.5, 1.0]])
        basis = vdx.rbf_basis(
            (GRID_Y, GRID_X), centers, widths=0.2, kernel="wendland_c2"
        )
        out = np.asarray(basis.operg(0.0, jnp.array([1.0])))
        gy, gx = np.meshgrid(GRID_Y, GRID_X, indexing="ij")
        dist = np.hypot(gy - 0.5, gx - 1.0)
        assert out[dist > 0.2].max() == 0.0
        assert out[dist < 0.1].min() > 0.0

    def test_per_atom_widths(self):
        centers = np.array([[0.5, 1.0], [0.5, 1.0]])
        basis = vdx.rbf_basis((GRID_Y, GRID_X), centers, widths=np.array([0.05, 0.5]))
        narrow = basis.operg(0.0, jnp.array([1.0, 0.0]))
        wide = basis.operg(0.0, jnp.array([0.0, 1.0]))
        # The wider atom carries more total mass over the same grid.
        assert float(jnp.sum(wide)) > float(jnp.sum(narrow))

    def test_prior_inv_divides_by_variance(self):
        centers = np.array([[0.5, 1.0], [0.2, 0.4]])
        basis = vdx.rbf_basis((GRID_Y, GRID_X), centers, widths=0.05, variance=4.0)
        out = basis.prior_inv(jnp.array([2.0, -3.0]))
        np.testing.assert_allclose(np.asarray(out), [0.5, -0.75])

    def test_works_in_one_dimension(self):
        basis = vdx.rbf_basis((GRID_Y,), np.array([[0.5]]), widths=0.1)
        assert basis.operg(0.0, jnp.array([1.0])).shape == (GRID_Y.size,)

    def test_rejects_mismatched_centers(self):
        with pytest.raises(ValueError, match=r"centers must be \(M, 2\)"):
            vdx.rbf_basis((GRID_Y, GRID_X), np.zeros((3, 4)), widths=0.1)

    def test_rejects_non_positive_widths(self):
        with pytest.raises(ValueError, match="widths"):
            vdx.rbf_basis((GRID_Y, GRID_X), np.array([[0.5, 1.0]]), widths=0.0)

    def test_rejects_unknown_kernel(self):
        with pytest.raises(ValueError, match="kernel"):
            vdx.rbf_basis(
                (GRID_Y, GRID_X), np.array([[0.5, 1.0]]), widths=0.1, kernel="cubic"
            )

    def test_rejects_non_positive_variance(self):
        with pytest.raises(ValueError, match="variance must be strictly positive"):
            vdx.rbf_basis(
                (GRID_Y, GRID_X), np.array([[0.5, 1.0]]), widths=0.1, variance=0.0
            )
        with pytest.raises(ValueError, match="variance must be strictly positive"):
            vdx.rbf_basis(
                (GRID_Y, GRID_X),
                np.array([[0.5, 1.0], [0.2, 0.4]]),
                widths=0.1,
                variance=np.array([1.0, -1.0]),
            )


class TestFourier:
    def test_nbasis_is_product_of_modes(self):
        basis = vdx.fourier_basis((GRID_Y, GRID_X), (3, 4))
        assert basis.nbasis == 12

    def test_spectral_density_prior(self):
        """A spectral-density callable yields decreasing variance with
        increasing eigenvalue (HSGP prior), scaled by ``variance``."""
        basis = vdx.fourier_basis(
            (GRID_Y, GRID_X),
            3,
            variance=2.0,
            spectral_density=lambda lam: 1.0 / (1.0 + lam),
        )
        var = np.asarray(basis.variance)
        assert var.shape == (9,)
        assert var.max() <= 2.0
        assert var.min() < var.max()

    def test_boundary_scale_keeps_edges_nonzero(self):
        """With ``boundary_scale > 1`` the increment need not vanish on
        the grid edge rows/columns."""
        basis = vdx.fourier_basis((GRID_Y, GRID_X), 3, boundary_scale=1.5)
        out = np.abs(np.asarray(basis.operg(0.0, jnp.ones(9))))
        assert out[0, :].max() > 1e-4
        assert out[:, -1].max() > 1e-4

    def test_rejects_boundary_scale_below_one(self):
        with pytest.raises(ValueError, match="boundary_scale"):
            vdx.fourier_basis((GRID_Y, GRID_X), 3, boundary_scale=0.5)

    def test_rejects_degenerate_axis(self):
        with pytest.raises(ValueError, match="non-degenerate"):
            vdx.fourier_basis((GRID_Y, np.array([1.0])), 3)


class TestWavelet:
    def test_columns_are_orthonormal(self):
        basis = vdx.wavelet_basis(16, wavelet="db2")
        gram = np.asarray(basis.phi.T @ basis.phi)
        np.testing.assert_allclose(gram, np.eye(16), atol=1e-5)

    def test_complete_basis_on_2d_grid(self):
        basis = vdx.wavelet_basis((8, 16))
        assert basis.nbasis == 8 * 16
        assert basis.output_shape == (8, 16)

    def test_accepts_int_shorthand(self):
        basis = vdx.wavelet_basis(8)
        assert basis.output_shape == (8,)

    def test_rejects_3d_grid(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            vdx.wavelet_basis((4, 4, 4))

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            vdx.wavelet_basis(12)


class TestEOF:
    def test_explained_variance_matches_singular_values(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(10, 32))
        basis = vdx.eof_basis(data, 3)
        centred = data - data.mean(axis=0, keepdims=True)
        svals = np.linalg.svd(centred, compute_uv=False)[:3]
        np.testing.assert_allclose(
            np.asarray(basis.variance), svals**2 / 9.0, rtol=1e-4
        )

    def test_columns_are_orthonormal(self):
        rng = np.random.default_rng(4)
        basis = vdx.eof_basis(rng.normal(size=(10, 32)), 4)
        gram = np.asarray(basis.phi.T @ basis.phi)
        np.testing.assert_allclose(gram, np.eye(4), atol=1e-5)

    def test_explicit_variance_overrides_explained(self):
        rng = np.random.default_rng(5)
        basis = vdx.eof_basis(rng.normal(size=(10, 32)), 2, variance=3.0)
        np.testing.assert_allclose(np.asarray(basis.variance), [3.0, 3.0])

    def test_rejects_explained_with_single_sample(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            vdx.eof_basis(np.ones((1, 32)), 1)

    def test_rejects_unknown_variance_string(self):
        with pytest.raises(ValueError, match="explained"):
            vdx.eof_basis(np.ones((4, 32)), 1, variance="climatology")


class TestLinearBasis:
    def test_rejects_non_2d_phi(self):
        with pytest.raises(ValueError, match="phi must be 2-D"):
            linear_basis(jnp.ones((4,)))

    def test_rejects_output_shape_mismatch(self):
        with pytest.raises(ValueError, match="output_shape"):
            linear_basis(jnp.ones((6, 2)), output_shape=(2, 2))

    def test_default_output_shape_is_flat(self):
        basis = linear_basis(jnp.ones((6, 2)))
        assert basis.operg(0.0, jnp.ones(2)).shape == (6,)

    def test_is_pytree_with_two_leaves(self):
        basis = _make_rbf()
        # `phi` + `variance` are JAX leaves; `output_shape` is static.
        assert len(jax.tree_util.tree_leaves(basis)) == 2


class TestComposite:
    def test_nbasis_sums(self):
        comp = composite_basis(_make_rbf(), _make_fourier())
        assert comp.nbasis == 3 + 9

    def test_operg_sums_components(self):
        b1, b2 = _make_rbf(), _make_fourier()
        comp = composite_basis(b1, b2)
        X = jnp.arange(comp.nbasis, dtype=jnp.float32)
        expected = b1.operg(0.0, X[:3]) + b2.operg(0.0, X[3:])
        np.testing.assert_allclose(
            np.asarray(comp.operg(0.0, X)), np.asarray(expected), atol=1e-5
        )

    def test_prior_inv_concatenates_blocks(self):
        b1, b2 = _make_rbf(), _make_eof()
        comp = composite_basis(b1, b2)
        X = jnp.arange(1.0, comp.nbasis + 1.0)
        out = comp.prior_inv(X)
        np.testing.assert_allclose(np.asarray(out[:3]), np.asarray(b1.prior_inv(X[:3])))
        np.testing.assert_allclose(np.asarray(out[3:]), np.asarray(b2.prior_inv(X[3:])))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            composite_basis()
