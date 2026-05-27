"""Smoke tests for ``vardax.cycle`` factories (Epic 7)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pipekit_cycle as pc
import pytest

from vardax import (
    LinearObs,
    MaskedIdentity,
    OptimalInterpolation,
    VarDACycle,
    VarSmootherCycle,
)


class IdentityForward(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    @property
    def state_signature(self):
        return None

    def step(self, state, dt):
        return state


@pytest.fixture
def oi_model():
    N = 4
    H = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, N), jnp.float32))
    cov = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, N), jnp.float32))
    return OptimalInterpolation(
        obs_op=LinearObs(H_mat=H),
        prior_mean=jnp.zeros((1, N)),
        prior_cov_op=cov,
        obs_cov_op=cov,
    )


class TestVarDACycle:
    def test_returns_dacycle(self, oi_model):
        cycle = VarDACycle(
            forward=IdentityForward(),
            obs_op=MaskedIdentity(),
            model=oi_model,
            n_steps=1,
        )
        assert isinstance(cycle, pc.DACycle)

    def test_carries_analysis_step(self, oi_model):
        cycle = VarDACycle(
            forward=IdentityForward(), obs_op=MaskedIdentity(), model=oi_model
        )
        assert isinstance(cycle.analysis_step, pc.AnalysisStep)

    def test_rejects_model_without_as_analysis_step(self):
        with pytest.raises(AttributeError, match="as_analysis_step"):
            VarDACycle(
                forward=IdentityForward(),
                obs_op=MaskedIdentity(),
                model=object(),
            )


class TestVarSmootherCycle:
    def test_returns_smoothercycle(self, oi_model):
        cycle = VarSmootherCycle(
            forward=IdentityForward(),
            obs_op=MaskedIdentity(),
            model=oi_model,
            window=3,
        )
        assert isinstance(cycle, pc.SmootherCycle)
