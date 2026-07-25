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
    AmortizedConfig,
    AmortizedPosterior,
    AnalysisStep,
    BilinAEPrior1D,
    ConvLSTMGradMod1D,
    CostFunction,
    FourDVarNet1D,
    GradModulator,
    IdentityObsEncoder,
    Prior,
    RegressionHead,
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


class TestAmortizedPosteriorProtocol:
    """`AmortizedPosterior` is the seventh AnalysisStep (Epic 8)."""

    def _model(self, rng):
        head = RegressionHead(
            context_dim=2 * 3 * 4,
            state_shape=(3, 4),
            hidden_dim=4,
            depth=2,
            key=rng,
        )
        return AmortizedPosterior(
            encoder=IdentityObsEncoder(),
            head=head,
            config=AmortizedConfig(head_type="regression"),
        )

    def test_yields_analysis_step(self, rng):
        model = self._model(rng)
        step = model.as_analysis_step()
        assert isinstance(step, AnalysisStep)

    def test_analysis_step_callable(self, rng):
        model = self._model(rng)
        step = model.as_analysis_step()
        forecast = jnp.zeros((2, 3, 4))
        obs = forecast.at[:, 0, 0].set(jnp.nan)
        analysis = step(forecast, obs, obs_op=None, obs_err_cov=None)
        assert analysis.shape == forecast.shape


class TestCostFunctionProtocol:
    """``CostFunction`` is satisfied by any callable with the right signature."""

    def test_lambda_satisfies(self):
        def cost(x, batch):
            return jnp.sum(x**2)

        assert isinstance(cost, CostFunction)


class TestObservationOperatorConformance:
    """Obs operators satisfy ``pipekit_cycle.ObservationOperator`` (D8)."""

    def test_linear_obs(self):
        import lineax as lx

        from vardax import LinearObs, ObservationOperator

        op = LinearObs(H_mat=lx.MatrixLinearOperator(jnp.eye(3)))
        assert isinstance(op, ObservationOperator)
        assert isinstance(op.linearize(jnp.zeros(3)), lx.AbstractLinearOperator)

    def test_masked_identity(self):
        from vardax import MaskedIdentity, ObservationOperator

        op = MaskedIdentity()
        assert isinstance(op, ObservationOperator)

    def test_averaging_kernel(self):
        import lineax as lx

        from vardax import AveragingKernel, ObservationOperator

        op = AveragingKernel(
            A=lx.MatrixLinearOperator(0.5 * jnp.eye(3)),
            x_a=jnp.zeros(3),
            h=jnp.ones(3),
        )
        assert isinstance(op, ObservationOperator)

    def test_multi_instrument_flattened_wrapper(self):
        import lineax as lx

        from vardax import (
            InstrumentRegistry,
            InstrumentSpec,
            LinearObs,
            MultiInstrumentFusion,
            ObservationOperator,
        )

        spec = InstrumentSpec(
            obs_op=LinearObs(H_mat=lx.MatrixLinearOperator(jnp.eye(2))),
            mask=jnp.ones(2),
            R_op=lx.DiagonalLinearOperator(0.1 * jnp.ones(2)),
            instrument_id="a",
        )
        fusion = MultiInstrumentFusion(registry=InstrumentRegistry(entries={"a": spec}))
        assert isinstance(fusion.to_observation_operator(), ObservationOperator)


class TestTemporalPriorProtocol:
    """Decision D18: temporal prior seam + adapters."""

    def _decay(self):
        def decay(t, y, args):
            return -y

        return decay

    def test_dyn_increments_satisfies_temporal_prior(self):
        from vardax import DynIncrements, TemporalPrior

        assert isinstance(DynIncrements(model=self._decay()), TemporalPrior)

    def test_dyn_trajectory_satisfies_temporal_prior(self):
        from vardax import DynTrajectory, TemporalPrior

        assert isinstance(DynTrajectory(model=self._decay()), TemporalPrior)

    def test_static_prior_does_not_satisfy_temporal_prior(self, rng):
        from vardax import TemporalPrior

        prior = BilinAEPrior1D(state_dim=4, latent_dim=2, n_time=3, key=rng)
        # Static priors have no ``loss`` member — the seams stay distinct.
        assert not isinstance(prior, TemporalPrior)

    def test_bound_prior_satisfies_prior(self):
        from vardax import DynTrajectory

        ts = jnp.linspace(0.0, 0.5, 6)
        bound = DynTrajectory(model=self._decay()).bind(ts)
        assert isinstance(bound, Prior)

    def test_as_forward_model_satisfies_forward_model(self):
        from pipekit_cycle import ForwardModel

        from vardax import DynTrajectory

        fwd = DynTrajectory(model=self._decay()).as_forward_model(dt=0.1)
        assert isinstance(fwd, ForwardModel)
