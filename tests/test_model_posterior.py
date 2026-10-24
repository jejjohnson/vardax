"""``model.posterior(batch)`` and adapters without explicit B, R (issue #99).

The docs called ``model.posterior(batch)`` and ``LaplaceCovariance()``,
neither of which existed. ``posterior`` now runs the analysis and the
default adapter on one window; an adapter given no covariances reads the
model's own.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx

N_STATE = 3


class Decay:
    dt = 1.0

    def step(self, x, dt):
        return 0.8 * x


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _diagonal(value, shape, dtype=jnp.float64):
    diagonal = lx.DiagonalLinearOperator(jnp.full(shape, value, dtype=dtype))
    return lx.TaggedLinearOperator(diagonal, lx.positive_semidefinite_tag)


def _ops(dtype=jnp.float64):
    return _diagonal(2.0, N_STATE, dtype), _diagonal(0.5, N_STATE, dtype)


def _window(n_windows=1, n_times=3):
    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.normal(size=(n_windows, n_times, N_STATE)))
    return vdx.Batch1D(input=data, mask=jnp.ones_like(data))


def _four_dvar_models():
    B_op, R_op = _ops()
    common = dict(
        forward=Decay(),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
    )
    return {
        "strong": vdx.StrongFourDVar(**common),
        "incremental": vdx.IncrementalFourDVar(**common),
        "reduced_basis": vdx.ReducedBasisFourDVar(**common),
        "weak": vdx.WeakFourDVar(**common, model_err_cov_op=B_op),
    }


@pytest.mark.parametrize("name", ["strong", "incremental", "reduced_basis", "weak"])
def test_posterior_equals_explicit_adapter(x64, name):
    model = _four_dvar_models()[name]
    batch = _window()
    posterior = model.posterior(batch)

    B_op, R_op = _ops()
    analysis = model(batch)
    if name == "weak":
        window_analysis = (analysis[0][0], analysis[1][0])
        mean = analysis[0][0]
    else:
        window_analysis = mean = analysis[0]
    explicit = vdx.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)(
        window_analysis, model, batch
    )
    probe = jnp.arange(1.0, N_STATE + 1)
    assert isinstance(posterior, vdx.Posterior)
    np.testing.assert_allclose(posterior.mean, mean)
    np.testing.assert_allclose(posterior.cov.mv(probe), explicit.cov.mv(probe), rtol=1e-6)


def test_posterior_with_gauss_newton_adapter(x64):
    model = _four_dvar_models()["incremental"]
    posterior = model.posterior(_window(), adapter=vdx.GaussNewtonHessian(n_krylov=100))
    assert posterior.provenance["adapter"] == "GaussNewtonHessian"
    assert posterior.provenance["structure"] == "strong_4dvar"


def test_single_time_models(x64):
    B_op = _diagonal(2.0, (1, N_STATE))
    R_op = _diagonal(0.5, (1, N_STATE))
    common = dict(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros((1, N_STATE)),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
    )
    batch = _window(n_times=1)
    for model in (vdx.OptimalInterpolation(**common), vdx.ThreeDVar(**common)):
        posterior = model.posterior(batch)
        # P* = (B⁻¹ + R⁻¹)⁻¹ = (1/2 + 2)⁻¹ = 0.4
        np.testing.assert_allclose(
            posterior.cov.mv(jnp.ones((1, N_STATE))), 0.4, rtol=1e-5
        )


def test_posterior_rejects_several_windows(x64):
    model = _four_dvar_models()["strong"]
    with pytest.raises(ValueError, match="takes one window"):
        model.posterior(_window(n_windows=2))


def test_adapter_reads_model_covariances(x64):
    model = _four_dvar_models()["strong"]
    batch = _window()
    analysis = model(batch)[0]
    B_op, R_op = _ops()
    implicit = vdx.LaplaceCovariance()(analysis, model, batch)
    explicit = vdx.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)(
        analysis, model, batch
    )
    probe = jnp.ones(N_STATE)
    np.testing.assert_allclose(implicit.cov.mv(probe), explicit.cov.mv(probe))


def test_adapter_without_covariances_anywhere_raises():
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N_STATE,), jnp.float32))
    view = SimpleNamespace(obs_op=vdx.LinearObs(H_mat=eye))
    with pytest.raises(ValueError, match="pass prior_cov_op="):
        vdx.LaplaceCovariance()(jnp.zeros(N_STATE), view, batch=None)
