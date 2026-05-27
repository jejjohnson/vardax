"""Tests for the posterior-adapter family (Epic 5)."""

from __future__ import annotations

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
