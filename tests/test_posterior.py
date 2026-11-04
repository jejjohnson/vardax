"""Tests for the posterior-adapter family (Epic 5)."""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from vardax import (
    Batch1D,
    EnsembleCovariance,
    GaussianMarkLikelihood,
    GaussNewtonHessian,
    LaplaceCovariance,
    MaskedIdentity,
    OptimalInterpolation,
    Posterior,
    ThreeDVar,
)


@pytest.fixture
def linear_gaussian_oi():
    """Standard OI setup for testing posteriors."""
    T, N, B = 1, 4, 2
    prior_mean = jnp.zeros((T, N))
    B_op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((T, N), jnp.float32))
    R_op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((T, N), jnp.float32))
    oi = OptimalInterpolation(
        obs_op=MaskedIdentity(),
        prior_mean=prior_mean,
        prior_cov_op=B_op,
        obs_cov_op=R_op,
    )
    batch = Batch1D(
        input=jax.random.normal(jax.random.PRNGKey(0), (B, T, N)),
        mask=jnp.ones((B, T, N)),
        target=None,
    )
    return {"oi": oi, "batch": batch, "B_op": B_op, "R_op": R_op}


@pytest.mark.parametrize(
    "make_model",
    [
        pytest.param(OptimalInterpolation, id="oi"),
        pytest.param(ThreeDVar, id="3dvar"),
    ],
)
def test_multi_time_diagonal_r(make_model):
    """A (T, N) diagonal R: P* = (B⁻¹ + R⁻¹)⁻¹ = (1 + 2)⁻¹ per cell (#149)."""
    shape = (2, 3)
    model = make_model(
        obs_op=MaskedIdentity(),
        prior_mean=jnp.zeros(shape),
        prior_cov_op=lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(shape, jnp.float32)
        ),
        obs_cov_op=lx.DiagonalLinearOperator(jnp.full(shape, 0.5)),
    )
    batch = Batch1D(input=jnp.ones((1, *shape)), mask=jnp.ones((1, *shape)))
    spread = model.posterior(batch).cov.mv(jnp.ones(shape))
    assert jnp.allclose(spread, jnp.full(shape, 1.0 / 3.0), rtol=1e-3)


class TestPosteriorContainer:
    def test_basic_construction(self):
        post = Posterior(mean=jnp.zeros(4))
        assert post.mean.shape == (4,)
        assert post.cov is None
        assert post.samples is None
        assert post.provenance == {}

    def test_with_provenance(self):
        post = Posterior(
            mean=jnp.zeros(4),
            provenance={"forward_model_id": "test", "n_iter": 5},
        )
        assert post.provenance["forward_model_id"] == "test"


class TestLaplaceCovariance:
    def test_returns_posterior(self, linear_gaussian_oi):
        s = linear_gaussian_oi
        analysis = s["oi"](s["batch"])
        laplace = LaplaceCovariance(prior_cov_op=s["B_op"], obs_cov_op=s["R_op"])
        post = laplace(analysis[0], s["oi"].as_analysis_step(), s["batch"])
        assert isinstance(post, Posterior)
        assert post.cov is not None
        assert post.provenance["adapter"] == "LaplaceCovariance"

    def test_cov_matvec(self, linear_gaussian_oi):
        """Covariance op should be mat-vec-able (without materialising)."""
        s = linear_gaussian_oi
        analysis = s["oi"](s["batch"])
        laplace = LaplaceCovariance(prior_cov_op=s["B_op"], obs_cov_op=s["R_op"])
        post = laplace(analysis[0], s["oi"].as_analysis_step(), s["batch"])
        v = jnp.ones_like(post.mean)
        # For B=R=I and fully-observed H=I: P* = (I + I)^{-1} = I/2
        cov_v = post.cov.mv(v)
        assert jnp.allclose(cov_v, v / 2, atol=1e-2)

    def test_sample_batch_of_unbatched_states(self):
        """``(S, N)`` batches of ``(N,)`` states, as ``__call__`` accepts (#142)."""
        eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float32))
        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=jnp.zeros(4),
            prior_cov_op=eye,
            obs_cov_op=eye,
        )
        mask = jnp.array([[1.0, 1.0, 1.0, 0.0]])
        post = oi.posterior(Batch1D(input=jnp.ones((1, 4)), mask=mask))
        assert post.mean.shape == (4,)
        # P* = (I + M)⁻¹: 1/2 where observed, 1 at the missing cell
        expected = jnp.array([0.5, 0.5, 0.5, 1.0])
        assert jnp.allclose(post.cov.mv(jnp.ones(4)), expected, atol=1e-4)


class TestGaussNewtonHessian:
    def test_returns_posterior(self, linear_gaussian_oi):
        s = linear_gaussian_oi
        analysis = s["oi"](s["batch"])
        gn = GaussNewtonHessian(prior_cov_op=s["B_op"], obs_cov_op=s["R_op"])
        post = gn(analysis[0], s["oi"].as_analysis_step(), s["batch"])
        assert isinstance(post, Posterior)
        assert post.provenance["adapter"] == "GaussNewtonHessian"


class TestEnsembleCovariance:
    def test_sample_cov(self, linear_gaussian_oi):
        s = linear_gaussian_oi
        # Build an ensemble of 32 samples around the analysis mean
        analysis = s["oi"](s["batch"])
        rng = jax.random.PRNGKey(42)
        perturbations = jax.random.normal(rng, (32, *analysis.shape[1:]))
        ensemble = analysis[0] + perturbations  # (32, T, N)

        ec = EnsembleCovariance(n_members=32)
        post = ec(ensemble, s["oi"].as_analysis_step(), s["batch"])
        assert isinstance(post, Posterior)
        assert post.samples is not None
        assert post.samples.shape == ensemble.shape
        assert post.provenance["n_members"] == 32


class TestGaussianMarkLikelihood:
    def test_to_dict_keys(self, linear_gaussian_oi):
        s = linear_gaussian_oi
        analysis = s["oi"](s["batch"])
        laplace = LaplaceCovariance(prior_cov_op=s["B_op"], obs_cov_op=s["R_op"])
        post = laplace(analysis[0], s["oi"].as_analysis_step(), s["batch"])
        mark = GaussianMarkLikelihood(
            posterior=post,
            event_metadata={"event_id": "test", "time": "2026-05-27"},
        )
        d = mark.to_dict()
        assert set(d.keys()) == {
            "mean",
            "cov_diag",
            "samples",
            "provenance",
            "event_metadata",
        }
        assert d["event_metadata"]["event_id"] == "test"

    def test_serialises_to_json_compatible_types(self, linear_gaussian_oi):
        """Result should be json.dumps-able."""
        import json

        s = linear_gaussian_oi
        analysis = s["oi"](s["batch"])
        laplace = LaplaceCovariance(prior_cov_op=s["B_op"], obs_cov_op=s["R_op"])
        post = laplace(analysis[0], s["oi"].as_analysis_step(), s["batch"])
        # Cov_diag may be None for lazy operators — strip it for the JSON test
        mark = GaussianMarkLikelihood(posterior=Posterior(mean=post.mean))
        d = mark.to_dict()
        # Should serialise without error
        json.dumps(d)


def _precisions():
    psd = lx.positive_semidefinite_tag
    dense = jnp.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 2.0]])
    return [
        pytest.param(lx.MatrixLinearOperator(dense, psd), id="dense"),
        pytest.param(
            lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 4.0])), id="diagonal"
        ),
        pytest.param(
            gaussx.Kronecker(
                lx.MatrixLinearOperator(jnp.array([[2.0, 0.5], [0.5, 1.0]]), psd),
                lx.MatrixLinearOperator(dense, psd),
            ),
            id="kronecker",
        ),
    ]


@pytest.mark.parametrize("precision", _precisions())
def test_marginal_variances_from_the_stored_precision(precision):
    """diag(P) = diag(A⁻¹), and to_dict's keys do not depend on the precision (#195)."""
    size = precision.in_size()
    posterior = Posterior(
        mean=jnp.zeros(size), cov=gaussx.inv(precision), precision=precision
    )
    mark = GaussianMarkLikelihood(posterior=posterior)
    expected = jnp.diag(jnp.linalg.inv(precision.as_matrix()))
    assert jnp.allclose(mark._cov_diag(), expected, rtol=1e-5)
    without = GaussianMarkLikelihood(
        posterior=Posterior(mean=jnp.zeros(size), cov=gaussx.inv(precision))
    )
    assert mark.to_dict().keys() == without.to_dict().keys()
    assert jnp.allclose(
        jnp.asarray(mark.to_dict()["cov_diag"]),
        jnp.asarray(without.to_dict()["cov_diag"]),
        rtol=1e-5,
    )


def test_a_large_unstructured_precision_is_never_estimated(monkeypatch):
    """Above the dense limit, the result comes from cov and not from a random probe."""
    size = 2049
    precision = lx.FunctionLinearOperator(
        lambda v: 2.0 * v,
        jax.ShapeDtypeStruct((size,), jnp.float32),
        lx.positive_semidefinite_tag,
    )
    calls = []
    monkeypatch.setattr(gaussx, "diag_inv", lambda *a, **k: calls.append(1))
    cov = lx.DiagonalLinearOperator(jnp.full(size, 0.5))
    mark = GaussianMarkLikelihood(
        posterior=Posterior(mean=jnp.zeros(size), cov=cov, precision=precision)
    )
    assert jnp.allclose(mark._cov_diag(), jnp.full(size, 0.5))
    assert not calls


@pytest.mark.parametrize(
    "make_model",
    [
        pytest.param(OptimalInterpolation, id="oi-laplace"),
        pytest.param(ThreeDVar, id="3dvar"),
    ],
)
def test_adapters_store_the_precision(make_model):
    B = lx.MatrixLinearOperator(
        jnp.array([[2.0, 0.5], [0.5, 1.0]]), lx.positive_semidefinite_tag
    )
    model = make_model(
        obs_op=MaskedIdentity(),
        prior_mean=jnp.zeros(2),
        prior_cov_op=B,
        obs_cov_op=lx.DiagonalLinearOperator(jnp.ones(2)),
    )
    batch = Batch1D(input=jnp.ones((1, 2)), mask=jnp.ones((1, 2)))
    posterior = model.posterior(batch, LaplaceCovariance())
    assert posterior.precision is not None
    assert jnp.allclose(
        posterior.precision.as_matrix(),
        jnp.linalg.inv(posterior.cov.as_matrix()),
        rtol=1e-3,
    )
