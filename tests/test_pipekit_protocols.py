"""Conformance suite for pipekit-cycle and vardax-specific protocols.

Per Decision D8, vardax classes satisfy ``pipekit_cycle.{ForwardModel,
ObservationOperator, AnalysisStep}`` directly (no parallel ``Abstract*``
hierarchy). This module verifies that conformance by:

1. Importing the runtime-checkable protocols.
2. Building concrete instances of every Layer 1/2 class.
3. Asserting ``isinstance(instance, Protocol)`` succeeds.

When new Layer 2 analysis methods are added (OptimalInterpolation,
ThreeDVar, StrongFourDVar, WeakFourDVar, IncrementalFourDVar,
AmortizedPosterior in Phases 2 and 3), each one must also be added to
``test_model_yields_analysis_step``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from vardax import (
    AnalysisStep,
    BilinAEPrior1D,
    ConvLSTMGradMod1D,
    CostFunction,
    FourDVarNet1D,
    GradModulator,
    Prior,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


class TestPriorProtocol:
    def test_bilin_ae_prior_satisfies_prior(self, rng):
        prior = BilinAEPrior1D(state_dim=4, latent_dim=2, n_time=3, key=rng)
        assert isinstance(prior, Prior)

    def test_callable_returns_array(self, rng):
        prior = BilinAEPrior1D(state_dim=4, latent_dim=2, n_time=3, key=rng)
        x = jnp.zeros((2, 3, 4))
        out = prior(x)
        assert out.shape == x.shape


class TestGradModulatorProtocol:
    def test_conv_lstm_satisfies_grad_modulator(self, rng):
        grad_mod = ConvLSTMGradMod1D(state_channels=3, hidden_dim=4, key=rng)
        assert isinstance(grad_mod, GradModulator)


class TestAnalysisStepProtocol:
    """Every Layer 2 model must expose ``.as_analysis_step()``."""

    def test_fourdvarnet1d_yields_analysis_step(self, rng):
        model = FourDVarNet1D(
            state_dim=4,
            n_time=3,
            latent_dim=2,
            hidden_dim=4,
            n_solver_steps=2,
            key=rng,
        )
        step = model.as_analysis_step()
        assert isinstance(step, AnalysisStep)

    def test_analysis_step_callable(self, rng):
        model = FourDVarNet1D(
            state_dim=4,
            n_time=3,
            latent_dim=2,
            hidden_dim=4,
            n_solver_steps=2,
            key=rng,
        )
        step = model.as_analysis_step()
        # Forecast + obs with NaN-masked entries
        forecast = jnp.zeros((2, 3, 4))
        obs = forecast.at[:, 0, 0].set(jnp.nan)  # one missing pixel
        analysis = step(forecast, obs, obs_op=None, obs_err_cov=None)
        assert analysis.shape == forecast.shape


class TestCostFunctionProtocol:
    """``CostFunction`` is satisfied by any callable with the right signature."""

    def test_lambda_satisfies(self):
        def cost(x, batch):
            return jnp.sum(x**2)

        assert isinstance(cost, CostFunction)
