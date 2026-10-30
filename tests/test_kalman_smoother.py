"""``KalmanSmoother`` against the dense Gaussian conditional (#164).

An Ornstein–Uhlenbeck prior on irregular times t₁ … t_T has covariance
σ² exp(−|tᵢ − tⱼ| / ℓ), and on a gap Δt it discretises to
F = exp(−Δt / ℓ), Σ_η = σ² (1 − exp(−2 Δt / ℓ)). The smoothed marginals
must equal the conditional of the stacked x₁:T given y₁:T.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx

# numpy, so that the x64 tests see them in float64
TIMES = np.array([0.0, 0.3, 1.1, 1.4, 2.9, 3.0, 4.2, 5.5])
OBS = np.array([0.4, 0.1, -0.6, -0.2, 0.9, 1.2, 0.3, -0.5])
OBS_VAR = 0.05


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _ou_smoother(sigma, length, times=TIMES):
    gaps = jnp.diff(jnp.asarray(times))
    decay = jnp.exp(-gaps / length)
    return vdx.KalmanSmoother(
        # F = e^{−Δt/ℓ}, Σ_η = σ² (1 − e^{−2Δt/ℓ})
        transition=decay[:, None, None],
        transition_cov=(sigma**2 * (1 - decay**2))[:, None, None],
        obs_op=jnp.ones((1, 1)),
        obs_cov=OBS_VAR * jnp.ones((1, 1)),
        prior_mean=jnp.zeros(1),
        prior_cov=sigma**2 * jnp.ones((1, 1)),
    )


def _dense_conditional(sigma, length, observed, times=TIMES):
    """Mean and marginal variance of x₁:T given the observed entries."""
    times = jnp.asarray(times)
    prior = sigma**2 * jnp.exp(-jnp.abs(times[:, None] - times[None, :]) / length)
    H = jnp.eye(times.shape[0])[observed]
    innovation = H @ prior @ H.T + OBS_VAR * jnp.eye(int(observed.sum()))
    gain = prior @ H.T @ jnp.linalg.inv(innovation)
    mean = gain @ jnp.asarray(OBS)[observed]
    cov = prior - gain @ H @ prior
    return mean, jnp.diag(cov)


def _batch(mask=None):
    mask = jnp.ones(OBS.shape) if mask is None else mask
    obs = jnp.asarray(OBS)
    return vdx.Batch1D(input=obs[None, :, None], mask=mask[None, :, None])


def test_matches_the_dense_conditional(x64):
    smoother = _ou_smoother(0.8, 1.5)
    posterior = smoother.posterior(_batch())
    mean, variance = _dense_conditional(0.8, 1.5, np.ones(8, bool))
    np.testing.assert_allclose(posterior.mean[:, 0], mean, rtol=1e-10)
    marginals = jax.vmap(posterior.cov.mv)(jnp.eye(8)).diagonal()
    np.testing.assert_allclose(marginals, variance, rtol=1e-10)
    np.testing.assert_allclose(smoother(_batch())[0, :, 0], mean, rtol=1e-10)


def test_from_sde_matches_the_hand_discretisation(x64):
    import gaussx as gx

    sde = gx.MaternSDE(
        variance=jnp.asarray(0.64), lengthscale=jnp.asarray(1.5), order=0
    )
    smoother = vdx.KalmanSmoother.from_sde(
        sde,
        jnp.asarray(TIMES),
        obs_op=jnp.ones((1, 1)),
        obs_cov=OBS_VAR * jnp.ones((1, 1)),
    )
    np.testing.assert_allclose(
        smoother(_batch()), _ou_smoother(0.8, 1.5)(_batch()), rtol=1e-10
    )


def test_one_time_is_optimal_interpolation(x64):
    B = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    R = jnp.diag(jnp.array([0.1, 0.2, 0.3]))
    y = jnp.array([0.5, -0.2, 0.8])
    smoother = vdx.KalmanSmoother(
        transition=jnp.zeros((0, 2, 2)),
        transition_cov=jnp.zeros((0, 2, 2)),
        obs_op=H,
        obs_cov=R,
        prior_mean=jnp.zeros(2),
        prior_cov=B,
    )
    oi = vdx.OptimalInterpolation(
        obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(H)),
        prior_mean=jnp.zeros(2),
        prior_cov_op=lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
        obs_cov_op=lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag),
        solver=lx.Cholesky(),
    )
    smoothed = smoother(vdx.Batch1D(input=y[None, None], mask=jnp.ones((1, 1, 3))))
    analysis = oi(vdx.Batch1D(input=y[None], mask=jnp.ones((1, 3))))
    np.testing.assert_allclose(smoothed[0, 0], analysis[0], rtol=1e-10)


def test_a_fully_masked_time_is_skipped_and_filled(x64):
    smoother = _ou_smoother(0.8, 1.5)
    observed = np.ones(8, bool)
    observed[3] = False
    # NaN at the missing time, which the filter never reads
    gappy = jnp.where(jnp.asarray(observed), jnp.asarray(OBS), jnp.nan)
    batch = vdx.Batch1D(
        input=gappy[None, :, None], mask=jnp.asarray(observed, float)[None, :, None]
    )
    filtered_means, filtered_covs = smoother.filter(batch)
    # the filter only predicts at the gap: x₄|₃ = F₄ x₃|₃
    decay = jnp.exp(-(TIMES[3] - TIMES[2]) / 1.5)
    np.testing.assert_allclose(filtered_means[3], decay * filtered_means[2], rtol=1e-10)
    assert filtered_covs.shape == (8, 1, 1)
    mean, _ = _dense_conditional(0.8, 1.5, observed)
    smoothed = smoother(batch)[0, :, 0]
    assert jnp.all(jnp.isfinite(smoothed))
    np.testing.assert_allclose(smoothed, mean, rtol=1e-10)


def test_jit_and_vmap_match_a_loop(x64):
    smoother = _ou_smoother(0.8, 1.5)
    obs = jnp.asarray(OBS)
    rows = jnp.stack([obs, -obs, 2.0 * obs])[:, :, None]
    batch = vdx.Batch1D(input=rows, mask=jnp.ones_like(rows))
    batched = jax.jit(lambda s, b: s(b))(smoother, batch)
    for i in range(3):
        one = vdx.Batch1D(input=rows[i : i + 1], mask=jnp.ones((1, 8, 1)))
        np.testing.assert_allclose(batched[i], smoother(one)[0], rtol=1e-10)


def test_gradients_match_central_differences(x64):
    def loss(params):
        sigma, length = params
        return jnp.sum(_ou_smoother(sigma, length)(_batch()) ** 2)

    params = jnp.array([0.8, 1.5])
    grad = jax.grad(loss)(params)
    step = 1e-6
    for i in range(2):
        shift = jnp.zeros(2).at[i].set(step)
        central = (loss(params + shift) - loss(params - shift)) / (2 * step)
        np.testing.assert_allclose(grad[i], central, rtol=1e-6)


def test_posterior_takes_one_sequence():
    smoother = _ou_smoother(0.8, 1.5)
    two = vdx.Batch1D(input=jnp.zeros((2, 8, 1)), mask=jnp.ones((2, 8, 1)))
    with pytest.raises(vdx.ShapeError):
        smoother.posterior(two)
