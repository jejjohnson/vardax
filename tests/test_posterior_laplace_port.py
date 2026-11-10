"""The posterior adapters build their covariance on ``argminax.curvature.Laplace``.

Port check for vardax#278: on dense linear-Gaussian problems the covariance equals
the closed form (Hᵀ R⁻¹ H + B⁻¹)⁻¹ to 1e-10 in x64, as it did before the port (the
pre-port vardax at bfb00ed matched it to 3e-16 on these problems).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import vardax as vdx

# float64 for the 1e-10 closed forms, scoped to this module (conftest.py)
pytestmark = pytest.mark.usefixtures("x64")


def _spd(key, n):
    factor = jax.random.normal(key, (n, n))
    return factor @ factor.T / n + 0.5 * jnp.eye(n)


def _psd(matrix):
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


CG = lx.CG(rtol=1e-14, atol=1e-14, max_steps=2000)
ADAPTERS = [
    lambda b, r: vdx.LaplaceCovariance(prior_cov_op=b, obs_cov_op=r, solver=CG),
    lambda b, r: vdx.GaussNewtonHessian(
        prior_cov_op=b, obs_cov_op=r, n_krylov=2000, tol=1e-14
    ),
]


@pytest.mark.parametrize("make_adapter", ADAPTERS, ids=["laplace", "gauss_newton"])
def test_three_dvar_covariance_is_the_closed_form(make_adapter):
    keys = jax.random.split(jax.random.key(20261109), 5)
    n_state, n_obs = 4, 5
    observation = jax.random.normal(keys[0], (n_obs, n_state))
    prior_cov = _spd(keys[1], n_state)
    noise_cov = _spd(keys[2], n_obs)
    model = vdx.ThreeDVar(
        obs_op=vdx.LinearObs(lx.MatrixLinearOperator(observation)),
        prior_mean=jax.random.normal(keys[3], (n_state,)),
        prior_cov_op=_psd(prior_cov),
        obs_cov_op=_psd(noise_cov),
    )
    batch = vdx.Batch1D(
        input=jax.random.normal(keys[4], (1, n_obs)), mask=jnp.ones((1, n_obs))
    )
    adapter = make_adapter(_psd(prior_cov), _psd(noise_cov))
    posterior = adapter(model(batch)[0], model.as_analysis_step(), batch)
    dense = jax.vmap(posterior.cov.mv)(jnp.eye(n_state))
    # Σ = (Hᵀ R⁻¹ H + B⁻¹)⁻¹
    expected = jnp.linalg.inv(
        observation.T @ jnp.linalg.solve(noise_cov, observation)
        + jnp.linalg.inv(prior_cov)
    )
    assert jnp.allclose(dense, expected, atol=1e-10)
    assert lx.is_positive_semidefinite(posterior.cov)


class _Linear(eqx.Module):
    matrix: jax.Array
    dt: float = 1.0

    def step(self, state, dt):
        return self.matrix @ state


@pytest.mark.parametrize("make_adapter", ADAPTERS, ids=["laplace", "gauss_newton"])
def test_strong_four_dvar_covariance_is_the_closed_form(make_adapter):
    keys = jax.random.split(jax.random.key(7), 5)
    n_state, n_times = 3, 3
    dynamics = 0.8 * jnp.eye(n_state) + 0.2 * jax.random.normal(keys[0], (3, 3))
    prior_cov = _spd(keys[1], n_state)
    noise_cov = _spd(keys[2], n_state)
    model = vdx.StrongFourDVar(
        forward=_Linear(dynamics),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jax.random.normal(keys[3], (n_state,)),
        prior_cov_op=_psd(prior_cov),
        obs_cov_op=_psd(noise_cov),
    )
    window = jax.random.normal(keys[4], (1, n_times, n_state))
    batch = vdx.Batch1D(input=window, mask=jnp.ones((1, n_times, n_state)))
    adapter = make_adapter(_psd(prior_cov), _psd(noise_cov))
    posterior = adapter(model(batch)[0], model, batch)
    dense = jax.vmap(posterior.cov.mv)(jnp.eye(n_state))
    # Σ = (∑ₜ (Aᵗ)ᵀ R⁻¹ Aᵗ + B⁻¹)⁻¹, observed at t = 0, 1, 2
    powers = [jnp.linalg.matrix_power(dynamics, t) for t in range(n_times)]
    precision = jnp.linalg.inv(prior_cov) + sum(
        power.T @ jnp.linalg.solve(noise_cov, power) for power in powers
    )
    assert jnp.allclose(dense, jnp.linalg.inv(precision), atol=1e-10)


def test_a_gridded_state_keeps_its_shape():
    """A (T, N) single-time state, the reason argminax#36 lets Laplace hold one."""
    shape = (2, 3)
    model = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(shape),
        prior_cov_op=lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(shape, jnp.float64)
        ),
        obs_cov_op=lx.DiagonalLinearOperator(jnp.full(shape, 0.5)),
    )
    batch = vdx.Batch1D(input=jnp.ones((1, *shape)), mask=jnp.ones((1, *shape)))
    spread = model.posterior(batch).cov.mv(jnp.ones(shape))
    # (B⁻¹ + R⁻¹)⁻¹ = (1 + 2)⁻¹ per cell
    assert spread.shape == shape
    assert jnp.allclose(spread, 1.0 / 3.0, atol=1e-8)
