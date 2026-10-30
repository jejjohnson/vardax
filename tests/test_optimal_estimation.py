"""Rodgers optimal estimation, ``vdx.optimal_estimation`` (#73)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx

N_STATE, N_OBS = 3, 6


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _linear_problem():
    rng = np.random.default_rng(0)
    K = jnp.asarray(rng.normal(size=(N_OBS, N_STATE)))
    root = rng.normal(size=(N_STATE, N_STATE))
    S_a = jnp.asarray(root @ root.T + np.eye(N_STATE))
    S_e = jnp.asarray(np.diag(rng.uniform(0.1, 0.5, size=N_OBS)))
    x_a = jnp.asarray(rng.normal(size=N_STATE))
    y = jnp.asarray(rng.normal(size=N_OBS))
    return K, S_a, S_e, x_a, y


def _psd(matrix):
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


@pytest.mark.parametrize("dense_noise", [False, True], ids=["diagonal", "dense"])
def test_linear_gaussian_is_the_closed_form(x64, dense_noise):
    K, S_a, S_e, x_a, y = _linear_problem()
    noise = _psd(S_e) if dense_noise else lx.DiagonalLinearOperator(jnp.diag(S_e))
    result = vdx.optimal_estimation(
        lambda x: K @ x, y, x_a, _psd(S_a), noise, rtol=1e-14, atol=1e-14
    )
    # Ŝ = (Kᵀ S_ε⁻¹ K + S_a⁻¹)⁻¹, x̂ = x_a + Ŝ Kᵀ S_ε⁻¹ (y − K x_a)
    S_e_inv = jnp.linalg.inv(S_e)
    S_hat = jnp.linalg.inv(K.T @ S_e_inv @ K + jnp.linalg.inv(S_a))
    x_hat = x_a + S_hat @ K.T @ S_e_inv @ (y - K @ x_a)
    A = S_hat @ K.T @ S_e_inv @ K
    np.testing.assert_allclose(result.x_hat, x_hat, rtol=1e-12)
    np.testing.assert_allclose(result.S_hat.as_matrix(), S_hat, rtol=1e-12)
    np.testing.assert_allclose(result.averaging_kernel, A, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.dofs, jnp.trace(A), rtol=1e-12)
    information = 0.5 * jnp.linalg.slogdet(S_a @ jnp.linalg.inv(S_hat))[1]
    np.testing.assert_allclose(result.information, information, rtol=1e-12)
    assert bool(result.converged)
    assert 0.0 <= float(result.dofs) <= N_STATE


@pytest.mark.slow
def test_coverage_of_one_sigma(x64):
    """Over 200 retrievals the truth lies within 1σ at a rate near 68.3 %."""
    n_trials = 200
    K, S_a, S_e, x_a, _ = _linear_problem()

    def forward(x):
        # mildly nonlinear: K x + 0.05 (K x)²
        linear = K @ x
        return linear + 0.05 * linear**2

    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    prior_root = jnp.linalg.cholesky(S_a)
    truths = x_a + jax.random.normal(keys[0], (n_trials, N_STATE)) @ prior_root.T
    noise = jax.random.normal(keys[1], (n_trials, N_OBS)) * jnp.sqrt(jnp.diag(S_e))
    observations = jax.vmap(forward)(truths) + noise

    def retrieve(y):
        return vdx.optimal_estimation(
            forward, y, x_a, _psd(S_a), lx.DiagonalLinearOperator(jnp.diag(S_e))
        )

    results = eqx.filter_vmap(retrieve)(observations)
    sigma = jnp.sqrt(jax.vmap(lambda r: jnp.diag(r.S_hat.as_matrix()))(results))
    inside = jnp.abs(results.x_hat - truths) < sigma
    rate = float(jnp.mean(inside[:, 0]))
    standard_error = np.sqrt(0.683 * 0.317 / n_trials)
    assert abs(rate - 0.683) < 5 * standard_error
    assert bool(jnp.all(results.converged))
    assert bool(jnp.all((results.dofs >= 0.0) & (results.dofs <= N_STATE)))


@pytest.mark.slow
def test_gradient_in_y_matches_central_differences(x64):
    K, S_a, S_e, x_a, y = _linear_problem()

    def forward(x):
        return K @ x + 0.1 * jnp.sin(K @ x)

    def loss(measurements):
        result = vdx.optimal_estimation(
            forward,
            measurements,
            x_a,
            _psd(S_a),
            lx.DiagonalLinearOperator(jnp.diag(S_e)),
            rtol=1e-13,
            atol=1e-13,
            max_steps=100,
        )
        return jnp.sum(result.x_hat**2)

    grad = jax.grad(loss)(y)
    direction = jnp.asarray(np.random.default_rng(1).normal(size=N_OBS))
    # a wide step: the error of a narrow one is the solver's stopping
    # tolerance divided by the step (1.7e-6 at 1e-5, 4e-9 at 1e-3)
    step = 1e-3
    central = (loss(y + step * direction) - loss(y - step * direction)) / (2 * step)
    np.testing.assert_allclose(jnp.dot(grad, direction), central, rtol=1e-6)


def test_as_observation_operator_is_the_affine_retrieval(x64):
    K, S_a, S_e, x_a, y = _linear_problem()
    result = vdx.optimal_estimation(
        lambda x: K @ x, y, x_a, _psd(S_a), lx.DiagonalLinearOperator(jnp.diag(S_e))
    )
    obs_op = result.as_observation_operator()
    state = jnp.ones(N_STATE)
    expected = x_a + result.averaging_kernel @ (state - x_a)
    np.testing.assert_allclose(obs_op(state), expected, rtol=1e-12)
