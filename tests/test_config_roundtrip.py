"""Hyperparameter-only modules round-trip through get_config / from_config."""

import json

import equinox as eqx
import jax.numpy as jnp
import pytest

import vardax as vdx

INSTANCES = [
    vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0),
    vdx.Lorenz96(F=8.0),
    vdx.EnsembleCovariance(n_members=8, inflation=1.1),
    vdx.KStepAdjoint(k=3),
    vdx.OneStepAdjoint(k=2),
    vdx.AmortizedConfig(head_type="regression", n_samples=16),
    vdx.IncrementalConfig(n_outer=2, n_inner=10, cvt=False),
]


@pytest.mark.parametrize("module", INSTANCES, ids=lambda m: type(m).__name__)
def test_round_trip_rebuilds_an_equal_module(module):
    rebuilt = type(module).from_config(module.get_config())
    assert type(rebuilt) is type(module)
    assert eqx.tree_equal(rebuilt, module)


@pytest.mark.parametrize("module", INSTANCES, ids=lambda m: type(m).__name__)
def test_config_is_plain_json(module):
    # hydra-zen and experiment logs need configs without arrays
    config = module.get_config()
    assert json.loads(json.dumps(config)) == config


def test_array_parameters_come_back_as_floats():
    # parameter estimation swaps arrays into the leaves with tree_at
    lorenz = eqx.tree_at(
        lambda model: (model.sigma, model.rho, model.beta),
        vdx.Lorenz63(sigma=10.0, rho=28.0, beta=2.5),
        (jnp.asarray(10.0), jnp.asarray(28.0), jnp.asarray(2.5)),
    )
    config = lorenz.get_config()
    assert all(isinstance(value, float) for value in config.values())


def test_from_config_reruns_validation():
    # from_config goes through __init__, so a bad config fails like a bad call
    with pytest.raises(vdx.ConfigurationError, match="k >= 1"):
        vdx.KStepAdjoint.from_config({"k": 0})
    with pytest.raises(vdx.ConfigurationError, match="head_type"):
        vdx.AmortizedConfig.from_config({"head_type": "gan", "n_samples": 4})
