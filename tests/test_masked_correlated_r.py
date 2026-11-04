"""Masked observations under a correlated R are marginalised out (#189).

B = I, H = I, R = [[1, ρ], [ρ, 1]] with ρ = 0.9 and only entry 0 observed
(y₀ = 1). The likelihood of the observed entry alone has variance R₀₀ = 1,
so the exact analysis is x = [1 / (1 + 1), 0] = [½, 0]. Solving the full R
against the zeroed residual instead weights y₀ by (R⁻¹)₀₀ = 1 / (1 − ρ²)
and returns 0.840.
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

RHO = 0.9
EXACT = np.array([0.5, 0.0])


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


class _Stay(eqx.Module):
    """x_{t+1} = x_t."""

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return state


def _correlated_r():
    return lx.MatrixLinearOperator(
        jnp.array([[1.0, RHO], [RHO, 1.0]]), lx.positive_semidefinite_tag
    )


def _common():
    return dict(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(2),
        prior_cov_op=lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float64)),
        obs_cov_op=_correlated_r(),
    )


def _single_time_batch():
    return vdx.Batch1D(input=jnp.array([[1.0, jnp.nan]]), mask=jnp.array([[1.0, 0.0]]))


def _window_batch():
    """One-time window, (1, T + 1 = 1, M = 2)."""
    return vdx.Batch1D(
        input=jnp.array([[[1.0, jnp.nan]]]), mask=jnp.array([[[1.0, 0.0]]])
    )


def _bfgs():
    return dict(
        minimiser=optx.BFGS(rtol=1e-12, atol=1e-12),
        cov_solver=lx.Cholesky(),
        max_steps=1000,
    )


SINGLE_TIME = [
    pytest.param(
        lambda: vdx.ThreeDVar(**_common(), cov_solver=lx.Cholesky()), id="3dvar"
    ),
    pytest.param(
        lambda: vdx.OptimalInterpolation(**_common(), solver=lx.Cholesky()),
        id="oi-observation",
    ),
    pytest.param(
        lambda: vdx.OptimalInterpolation(
            **_common(), solver=lx.Cholesky(), form="state"
        ),
        id="oi-state",
    ),
]

WINDOW = [
    pytest.param(
        lambda: vdx.StrongFourDVar(forward=_Stay(), **_common(), **_bfgs()),
        id="strong",
    ),
    pytest.param(
        lambda: vdx.WeakFourDVar(
            forward=_Stay(),
            **_common(),
            model_err_cov_op=lx.IdentityLinearOperator(
                jax.ShapeDtypeStruct((2,), jnp.float64)
            ),
            **_bfgs(),
        ),
        id="weak",
    ),
    pytest.param(
        lambda: vdx.IncrementalFourDVar(
            forward=_Stay(),
            **_common(),
            config=vdx.IncrementalConfig(
                n_outer=1,
                n_inner=50,
                cg_atol=1e-13,
                cg_rtol=1e-13,
                covariance_tol=1e-13,
            ),
        ),
        id="incremental",
    ),
    pytest.param(
        lambda: vdx.ReducedBasisFourDVar(forward=_Stay(), **_common(), **_bfgs()),
        id="reduced-basis",
    ),
]


@pytest.mark.parametrize("make_model", SINGLE_TIME)
def test_single_time_models_use_the_observed_likelihood(x64, make_model):
    analysis = make_model()(_single_time_batch())[0]
    np.testing.assert_allclose(analysis, EXACT, atol=1e-8)


@pytest.mark.parametrize("make_model", WINDOW)
def test_window_models_use_the_observed_likelihood(x64, make_model):
    analysis = make_model()(_window_batch())
    # WeakFourDVar returns (x₀, η)
    initial_state = analysis[0] if isinstance(analysis, tuple) else analysis
    np.testing.assert_allclose(initial_state[0], EXACT, atol=1e-8)


def test_threedvar_posterior_variances(x64):
    """P = (I + Hᵀ R₀₀⁻¹ H)⁻¹ on the observed entry: variances [½, 1]."""
    model = vdx.ThreeDVar(**_common(), cov_solver=lx.Cholesky())
    posterior = model.posterior(_single_time_batch())
    np.testing.assert_allclose(
        jnp.diag(posterior.cov.as_matrix()), [0.5, 1.0], atol=1e-8
    )


def test_fusion_cost_uses_the_observed_likelihood(x64):
    """J(0) = ½ (y₀ − 0)² / R₀₀ = ½, not ½ / (1 − ρ²)."""
    spec = vdx.InstrumentSpec(
        vdx.MaskedIdentity(), jnp.array([1.0, 0.0]), _correlated_r(), "a"
    )
    fusion = vdx.MultiInstrumentFusion(vdx.InstrumentRegistry(entries={"a": spec}))
    cost = fusion.cost(jnp.zeros(2), {"a": jnp.array([1.0, jnp.nan])})
    np.testing.assert_allclose(cost, 0.5, rtol=1e-12)


def test_diagonal_r_stays_diagonal(x64):
    """A diagonal R gives a diagonal R̃: m ⊙ diag(R) + (1 − m)."""
    R = lx.DiagonalLinearOperator(jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
    mask = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    masked = gaussx.masked_obs_covariance(R, mask)
    assert lx.is_diagonal(masked)
    np.testing.assert_allclose(
        lx.diagonal(masked), [2.0, 1.0, 4.0, 1.0, 6.0, 7.0], rtol=0.0
    )


def test_fully_observed_r_is_unchanged(x64):
    """With m = 1 everywhere, R̃ = R."""
    masked = gaussx.masked_obs_covariance(_correlated_r(), jnp.ones(2))
    np.testing.assert_allclose(
        masked.as_matrix(), _correlated_r().as_matrix(), rtol=0.0
    )
