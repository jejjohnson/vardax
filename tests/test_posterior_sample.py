"""``Posterior.sample`` (#165).

x⁽ᵉ⁾ = x* + P^{1/2} ξ⁽ᵉ⁾, ξ⁽ᵉ⁾ ~ 𝒩(0, I), or x* + Φ P_θ^{1/2} ξ⁽ᵉ⁾ for a
linear control. The sample mean and covariance of N_E draws converge to
x* and P at rate N_E^{-1/2}, so the checks allow five Monte Carlo standard
errors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx

N_DRAWS = 10_000


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _problem(n=4, m=6, r=0.25):
    rng = np.random.default_rng(0)
    F = jnp.asarray(rng.normal(size=(m, n)))
    B = jnp.asarray(np.diag([1.0, 2.0, 0.5, 1.5]) + 0.2)
    y = jnp.asarray(rng.normal(size=m))
    common = dict(
        obs_op=vdx.CallableObs(fn=lambda q: q @ F.T),
        prior_mean=jnp.zeros((1, n)),
        prior_cov_op=lx.FunctionLinearOperator(
            lambda z: z @ B,
            jax.ShapeDtypeStruct((1, n), jnp.float64),
            lx.positive_semidefinite_tag,
        ),
        obs_cov_op=lx.DiagonalLinearOperator(jnp.full((1, m), r)),
    )
    batch = vdx.Batch1D(input=y[None, None], mask=jnp.ones((1, 1, m)))
    return F, B, r, common, batch


def _assert_moments(draws, mean, cov):
    """Sample mean and covariance within 5 standard errors of the truth."""
    flat = draws.reshape(draws.shape[0], -1)
    n_draws = flat.shape[0]
    variance = jnp.diag(cov)
    mean_error = jnp.abs(flat.mean(0) - mean.reshape(-1))
    assert jnp.all(mean_error < 5 * jnp.sqrt(variance / n_draws))
    # Var(Ĉᵢⱼ) ≈ (Cᵢᵢ Cⱼⱼ + Cᵢⱼ²) / N_E for Gaussian draws
    cov_error = jnp.abs(jnp.cov(flat.T) - cov)
    cov_se = jnp.sqrt((jnp.outer(variance, variance) + cov**2) / n_draws)
    assert jnp.all(cov_error < 5 * cov_se)


def test_threedvar_draws_match_the_closed_form(x64):
    F, B, r, common, batch = _problem()
    model = vdx.ThreeDVar(**common, cov_solver=lx.Cholesky())
    posterior = model.posterior(batch)
    draws = posterior.sample(jax.random.PRNGKey(0), N_DRAWS)
    assert draws.shape == (N_DRAWS, *posterior.mean.shape)
    # P* = (B⁻¹ + Fᵀ F / r)⁻¹
    expected = jnp.linalg.inv(jnp.linalg.inv(B) + F.T @ F / r)
    _assert_moments(draws, posterior.mean, expected)


def test_linear_control_draws_in_coefficient_space(x64):
    F, _, r, common, batch = _problem()
    n = F.shape[1]
    phi = jnp.asarray(np.random.default_rng(3).normal(size=(n, 2)))
    variance = jnp.asarray([0.5, 2.0])
    param = vdx.LinearParameterization(
        vdx.linear_basis(phi, variance=variance), background=common["prior_mean"]
    )
    model = vdx.ThreeDVar(**common, parameterization=param)
    posterior = model.posterior(batch)
    assert posterior.basis is not None
    assert posterior.coefficient_cov is not None
    draws = posterior.sample(jax.random.PRNGKey(1), N_DRAWS)
    # P_x = Φ (Q⁻¹ + Φᵀ Fᵀ F Φ / r)⁻¹ Φᵀ, rank 2
    p_theta = jnp.linalg.inv(jnp.diag(1.0 / variance) + phi.T @ F.T @ F @ phi / r)
    expected = phi @ p_theta @ phi.T
    _assert_moments(draws, posterior.mean, expected)
    # every draw stays in x* + span(Φ)
    offsets = (draws - posterior.mean).reshape(N_DRAWS, n)
    projector = phi @ jnp.linalg.pinv(phi)
    assert jnp.allclose(offsets @ projector.T, offsets, atol=1e-8)
    # the state-space draw (no basis) has the same distribution
    state_space = vdx.Posterior(mean=posterior.mean, cov=posterior.cov)
    _assert_moments(
        state_space.sample(jax.random.PRNGKey(2), N_DRAWS), posterior.mean, expected
    )


def test_draws_are_reproducible_and_jit():
    cov = lx.MatrixLinearOperator(
        jnp.array([[2.0, 0.5], [0.5, 1.0]]), lx.positive_semidefinite_tag
    )
    posterior = vdx.Posterior(mean=jnp.ones(2), cov=cov)
    key = jax.random.PRNGKey(0)
    eager = posterior.sample(key, 8)
    jitted = jax.jit(lambda p, k: p.sample(k, 8))(posterior, key)
    assert jnp.array_equal(eager, posterior.sample(key, 8))
    assert jnp.allclose(eager, jitted, atol=1e-6)


def test_stored_samples_return_a_subset():
    samples = jnp.arange(10.0)[:, None] * jnp.ones((10, 3))
    posterior = vdx.Posterior(mean=jnp.zeros(3), samples=samples)
    draws = posterior.sample(jax.random.PRNGKey(0), 4)
    assert draws.shape == (4, 3)
    rows = draws[:, 0]
    assert len(set(rows.tolist())) == 4
    assert all(row in set(samples[:, 0].tolist()) for row in rows.tolist())
    with pytest.raises(vdx.ShapeError):
        posterior.sample(jax.random.PRNGKey(0), 11)


def test_without_cov_or_samples_raises():
    with pytest.raises(vdx.UnsupportedInputError):
        vdx.Posterior(mean=jnp.zeros(2)).sample(jax.random.PRNGKey(0), 3)
