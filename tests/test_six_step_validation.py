"""Tests for the six-step cycle validation gates (Decision D12 / Epic 8).

Exercises ``assert_posterior_agreement``, ``assert_adjoint_calibrated``,
and ``simulation_based_calibration`` on tractable problems so the
behaviour of each gate is pinned down. Real-world usage of these gates
lives in the validation harness of an amortized training pipeline; these
tests verify the gate logic itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from vardax import (
    Posterior,
    assert_adjoint_calibrated,
    assert_posterior_agreement,
    simulation_based_calibration,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


class TestAssertPosteriorAgreement:
    def test_close_means_pass(self):
        N = 4
        cov = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
        mean = jnp.array([0.1, -0.2, 0.3, 0.4])
        p_oracle = Posterior(mean=mean, cov=cov)
        p_fast = Posterior(mean=mean + 0.1, cov=cov)
        # |0.1| / std=1 well within tolerance 1.0
        assert_posterior_agreement(p_fast, p_oracle, tolerance_sigma=1.0)

    def test_far_means_fail(self):
        N = 4
        cov = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
        mean = jnp.zeros(N)
        p_oracle = Posterior(mean=mean, cov=cov)
        p_fast = Posterior(mean=mean + 2.5, cov=cov)
        with pytest.raises(AssertionError, match="posterior agreement"):
            assert_posterior_agreement(p_fast, p_oracle, tolerance_sigma=1.0)

    def test_requires_oracle_cov(self):
        mean = jnp.zeros(3)
        p_oracle = Posterior(mean=mean, cov=None)
        p_fast = Posterior(mean=mean, cov=None)
        with pytest.raises(ValueError, match=r"p_oracle\.cov"):
            assert_posterior_agreement(p_fast, p_oracle)

    def test_shape_mismatch(self):
        cov = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
        p_oracle = Posterior(mean=jnp.zeros(3), cov=cov)
        p_fast = Posterior(mean=jnp.zeros(4), cov=None)
        with pytest.raises(ValueError, match="same shape"):
            assert_posterior_agreement(p_fast, p_oracle)


class TestAssertAdjointCalibrated:
    def test_identical_pass(self, rng):
        def fn(y):
            return 2.0 * y + 1.0

        assert_adjoint_calibrated(
            fn, fn, jnp.zeros(4), key=rng, threshold=1e-6, n_probes=5
        )

    def test_close_pass(self, rng):
        def fn(y):
            return 2.0 * y

        def fn_close(y):
            return 2.001 * y

        assert_adjoint_calibrated(
            fn, fn_close, jnp.zeros(4), key=rng, threshold=0.01, n_probes=5
        )

    def test_far_fail(self, rng):
        def fn(y):
            return 2.0 * y

        def fn_far(y):
            return 4.0 * y

        with pytest.raises(AssertionError, match="adjoint calibration"):
            assert_adjoint_calibrated(
                fn, fn_far, jnp.zeros(4), key=rng, threshold=0.05, n_probes=3
            )


class TestSimulationBasedCalibration:
    def test_returns_ranks_in_range(self, rng):
        def sample_prior(k):
            return jax.random.normal(k, (3,))

        def simulate(x, k):
            return x + 0.1 * jax.random.normal(k, x.shape)

        n_samples = 50

        def sample_posterior(y, k, n):
            return y[None] + jax.random.normal(k, (n, *y.shape))

        ranks = simulation_based_calibration(
            sample_posterior=sample_posterior,
            sample_prior=sample_prior,
            simulate_obs=simulate,
            key=rng,
            n_runs=30,
            n_samples=n_samples,
        )
        assert ranks.shape == (30,)
        assert jnp.all(ranks >= 0)
        assert jnp.all(ranks <= n_samples)

    def test_biased_posterior_piles_ranks_low(self, rng):
        """A posterior whose samples are biased *higher* than the true
        scalar reduction produces ranks concentrated near 0.

        Concretely: samples = y + drift with drift large and positive,
        so ‖samples‖ ≫ ‖x_true‖ and the count of samples below
        ``‖x_true‖`` is 0 almost surely.
        """

        def sample_prior(k):
            return 0.1 * jax.random.normal(k, (3,))

        def simulate(x, k):
            return x  # noise-free

        def sample_posterior(y, k, n):
            return y[None] + 5.0 + 0.01 * jax.random.normal(k, (n, *y.shape))

        n_samples = 50
        ranks = simulation_based_calibration(
            sample_posterior=sample_posterior,
            sample_prior=sample_prior,
            simulate_obs=simulate,
            key=rng,
            n_runs=30,
            n_samples=n_samples,
        )
        # ‖x_true‖ ≈ 0.17 ≪ ‖biased samples‖ ≈ 8.7, so rank ≈ 0.
        assert int(jnp.median(ranks)) < 5
