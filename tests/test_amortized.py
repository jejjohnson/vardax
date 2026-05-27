"""Tests for ``AmortizedPosterior`` (Epic 8)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from vardax import (
    AmortizedConfig,
    AmortizedPosterior,
    AnalysisStep,
    Batch1D,
    ConditionalFlowHead,
    IdentityObsEncoder,
    MLPObsEncoder,
    RegressionHead,
    ScoreDiffusionHead,
    amortized_train_step,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def shapes():
    return {"B": 3, "T": 2, "N": 4}


@pytest.fixture
def regression_model(rng, shapes):
    T, N = shapes["T"], shapes["N"]
    head = RegressionHead(
        context_dim=2 * T * N,
        state_shape=(T, N),
        hidden_dim=8,
        depth=2,
        key=rng,
    )
    return AmortizedPosterior(
        encoder=IdentityObsEncoder(),
        head=head,
        config=AmortizedConfig(head_type="regression", n_samples=8),
    )


@pytest.fixture
def batch(rng, shapes):
    B, T, N = shapes["B"], shapes["T"], shapes["N"]
    k1, k2 = jax.random.split(rng)
    return Batch1D(
        input=jax.random.normal(k1, (B, T, N)),
        mask=jnp.ones((B, T, N)),
        target=jax.random.normal(k2, (B, T, N)),
    )


class TestRegressionHead:
    def test_map_shape(self, regression_model, batch, shapes):
        out = regression_model(batch)
        assert out.shape == (shapes["B"], shapes["T"], shapes["N"])

    def test_sample_shape(self, regression_model, batch, rng, shapes):
        samples = regression_model.sample(batch, rng, n=5)
        assert samples.shape == (shapes["B"], 5, shapes["T"], shapes["N"])

    def test_log_prob_shape(self, regression_model, batch, shapes):
        log_p = regression_model.log_prob(batch.target, batch)
        assert log_p.shape == (shapes["B"],)

    def test_log_prob_finite(self, regression_model, batch):
        log_p = regression_model.log_prob(batch.target, batch)
        assert jnp.all(jnp.isfinite(log_p))


class TestMLPObsEncoder:
    def test_runs(self, rng, shapes):
        T, N = shapes["T"], shapes["N"]
        ctx_dim = 6
        enc = MLPObsEncoder(input_size=T * N, context_dim=ctx_dim, key=rng)
        ctx = enc(jnp.zeros((T, N)), jnp.ones((T, N)))
        assert ctx.shape == (ctx_dim,)


class TestAnalysisStep:
    def test_satisfies_protocol(self, regression_model):
        step = regression_model.as_analysis_step()
        assert isinstance(step, AnalysisStep)

    def test_handles_nan_obs(self, regression_model, shapes):
        B, T, N = shapes["B"], shapes["T"], shapes["N"]
        step = regression_model.as_analysis_step()
        forecast = jnp.zeros((B, T, N))
        obs = forecast.at[:, 0, 0].set(jnp.nan)
        out = step(forecast, obs, obs_op=None, obs_err_cov=None)
        assert out.shape == forecast.shape
        assert jnp.all(jnp.isfinite(out))


class TestAmortizedTraining:
    def test_loss_decreases(self, regression_model, batch):
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(eqx.filter(regression_model, eqx.is_array))
        model, opt_state, loss0 = amortized_train_step(
            regression_model, batch, optimizer, opt_state
        )
        for _ in range(40):
            model, opt_state, loss = amortized_train_step(
                model, batch, optimizer, opt_state
            )
        assert float(loss) < float(loss0)

    def test_requires_target(self, regression_model, shapes):
        B, T, N = shapes["B"], shapes["T"], shapes["N"]
        batch = Batch1D(
            input=jnp.zeros((B, T, N)),
            mask=jnp.ones((B, T, N)),
            target=None,
        )
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(regression_model, eqx.is_array))
        with pytest.raises(ValueError, match=r"batch\.target"):
            amortized_train_step(regression_model, batch, optimizer, opt_state)


class TestStubHeads:
    def test_flow_head_raises(self):
        with pytest.raises(NotImplementedError, match="gauss_flows"):
            ConditionalFlowHead()

    def test_score_head_raises(self):
        with pytest.raises(NotImplementedError, match="reverse-SDE"):
            ScoreDiffusionHead()
