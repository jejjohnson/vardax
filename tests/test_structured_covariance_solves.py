"""The weighted norms solve structured covariances by structure (#192).

Each model evaluates ½ rᵀ C⁻¹ r through ``gaussx.solve``, so a diagonal, Kronecker
or block-diagonal C skips the iterative ``cov_solver`` and gives the same analysis
as its dense form.
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest

import vardax as vdx
from vardax._src.utils.covariance import as_covariance, half_weighted_norm

N = 6
PSD = lx.positive_semidefinite_tag


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


class _Decay(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return 0.9 * state + 0.05 * jnp.roll(state, 1)


def _factors():
    return (
        jnp.array([[2.0, 0.5], [0.5, 1.0]]),
        jnp.array([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 2.0]]),
    )


def _structured_covs():
    left, right = _factors()
    variances = jnp.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.25])
    return {
        "diagonal": lx.DiagonalLinearOperator(variances),
        "kronecker": gaussx.Kronecker(
            lx.MatrixLinearOperator(left, PSD), lx.MatrixLinearOperator(right, PSD)
        ),
        "block_diag": gaussx.BlockDiag(
            lx.MatrixLinearOperator(jnp.kron(left, left), PSD),
            lx.MatrixLinearOperator(left, PSD),
        ),
    }


def _analysis(model):
    return np.asarray(model[0] if isinstance(model, tuple) else model)


def _make(name, prior, noise):
    common = dict(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.asarray(np.random.default_rng(1).normal(size=N)),
        prior_cov_op=prior,
        obs_cov_op=noise,
    )
    tight = dict(
        minimiser=optx.BFGS(rtol=1e-11, atol=1e-11),
        cov_solver=lx.CG(atol=1e-13, rtol=1e-13),
        max_steps=2000,
    )
    if name == "3dvar":
        return vdx.ThreeDVar(**common, **tight)
    if name == "strong":
        return vdx.StrongFourDVar(forward=_Decay(), **common, **tight)
    if name == "weak":
        identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float64))
        return vdx.WeakFourDVar(
            forward=_Decay(), **common, model_err_cov_op=identity, **tight
        )
    if name == "reduced-basis":
        return vdx.ReducedBasisFourDVar(forward=_Decay(), **common, **tight)
    return vdx.IncrementalFourDVar(
        forward=_Decay(),
        **common,
        config=vdx.IncrementalConfig(
            n_outer=2, n_inner=100, cg_atol=1e-12, cg_rtol=1e-12, covariance_tol=1e-12
        ),
    )


def _batch(name):
    rng = np.random.default_rng(2)
    n_times = 1 if name == "3dvar" else 3
    obs = jnp.asarray(rng.normal(size=(n_times, N)))
    if name == "3dvar":
        return vdx.Batch1D(input=obs, mask=jnp.ones_like(obs))
    return vdx.Batch1D(input=obs[None], mask=jnp.ones_like(obs)[None])


@pytest.mark.slow
@pytest.mark.parametrize("structure", ["diagonal", "kronecker", "block_diag"])
@pytest.mark.parametrize(
    "name", ["3dvar", "strong", "weak", "reduced-basis", "incremental"]
)
def test_a_structured_covariance_matches_its_dense_form(x64, name, structure):
    cov = _structured_covs()[structure]
    dense = lx.MatrixLinearOperator(cov.as_matrix(), PSD)
    batch = _batch(name)
    actual = _analysis(_make(name, cov, cov)(batch))
    expected = _analysis(_make(name, dense, dense)(batch))
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("structure", ["diagonal", "kronecker", "block_diag"])
def test_the_weighted_norm_of_each_structure(x64, structure):
    cov = _structured_covs()[structure]
    vector = jnp.asarray(np.random.default_rng(3).normal(size=N))
    expected = 0.5 * vector @ jnp.linalg.solve(cov.as_matrix(), vector)
    solver = lx.CG(atol=1e-13, rtol=1e-13)
    actual = half_weighted_norm(as_covariance(cov), vector, solver)
    assert jnp.allclose(actual, expected, rtol=1e-9)


def test_a_diagonal_never_reaches_the_solver(x64):
    """A CG that would fail at once, were it ever called."""
    cov = _structured_covs()["diagonal"]
    vector = jnp.ones(N)
    never = lx.CG(atol=1e-13, rtol=1e-13, max_steps=0)
    actual = half_weighted_norm(as_covariance(cov), vector, never)
    assert jnp.allclose(actual, 0.5 * jnp.sum(1.0 / lx.diagonal(cov)), rtol=1e-12)


def test_a_two_dimensional_diagonal_keeps_its_shape(x64):
    variances = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vector = jnp.ones((2, 3))
    actual = half_weighted_norm(
        lx.DiagonalLinearOperator(variances), vector, lx.CG(atol=1e-6, rtol=1e-6)
    )
    assert jnp.allclose(actual, 0.5 * jnp.sum(1.0 / variances), rtol=1e-10)


def _untagged_composite(name):
    left, right = _factors()
    kronecker = gaussx.Kronecker(
        lx.MatrixLinearOperator(left), lx.MatrixLinearOperator(right)
    )
    blocks = gaussx.BlockDiag(
        lx.MatrixLinearOperator(jnp.kron(left, left)),
        lx.MatrixLinearOperator(left),
    )
    return {
        "kronecker": kronecker,
        "block_diag": blocks,
        "tagged_kronecker": lx.TaggedLinearOperator(kronecker, PSD),
        "tagged_block_diag": lx.TaggedLinearOperator(blocks, PSD),
        "scaled_kronecker": 2.0 * kronecker,
    }[name]


@pytest.mark.parametrize(
    "name",
    [
        "kronecker",
        "block_diag",
        "tagged_kronecker",
        "tagged_block_diag",
        "scaled_kronecker",
    ],
)
def test_untagged_parts_work_under_the_default_cg(x64, name):
    """lx.CG refuses an untagged part, so as_covariance tags each one."""
    untagged = _untagged_composite(name)
    vector = jnp.asarray(np.random.default_rng(4).normal(size=N))
    covariance = as_covariance(untagged)
    assert lx.is_positive_semidefinite(covariance)
    actual = half_weighted_norm(covariance, vector, lx.CG(atol=1e-13, rtol=1e-13))
    expected = 0.5 * vector @ jnp.linalg.solve(untagged.as_matrix(), vector)
    assert jnp.allclose(actual, expected, rtol=1e-9)
    model = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N),
        prior_cov_op=untagged,
        obs_cov_op=lx.DiagonalLinearOperator(jnp.ones(N)),
    )
    batch = vdx.Batch1D(input=jnp.ones((1, N)), mask=jnp.ones((1, N)))
    assert bool(jnp.all(jnp.isfinite(model(batch))))
