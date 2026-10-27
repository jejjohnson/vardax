"""4DVar posterior adapters use the tangent-linear model (issue #94).

Before #94, ``LaplaceCovariance`` and ``GaussNewtonHessian`` built
B⁻¹ + H'ᵀ R⁻¹ H' from the observation operator alone, the 3DVar Hessian,
even for a 4DVar analysis. These tests pin the window's Gauss–Newton
precision B⁻¹ + ∑ₜ (H'ₜ M'ₜ)ᵀ R⁻¹ (H'ₜ M'ₜ):

- linear-Gaussian oracle: the adapters' covariance of x₀ equals the
  covariance of a Kalman filter + Rauch–Tung–Striebel smoother at t = 0
  (float64, 1e-8), for strong (Q = 0) and weak (Q > 0) constraints;
- partial observation: never-observed sites coupled to observed ones by
  the dynamics have posterior variance below their prior variance;
- single-time analyses (T = 0, OI / 3DVar) are unchanged;
- misuse raises instead of returning a wrong covariance.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx

N_STATE, N_STEPS = 4, 3


class LinearForward:
    """xₜ₊₁ = A xₜ."""

    dt = 1.0

    def __init__(self, transition):
        self.transition = transition

    def step(self, x, dt):
        return self.transition @ x


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _problem(observed_sites=(0, 1)):
    rng = np.random.default_rng(0)
    # a rotation-dominated transition couples every site to every other
    skew = rng.normal(size=(N_STATE, N_STATE))
    transition = np.asarray(jax.scipy.linalg.expm(0.4 * (skew - skew.T))) * 0.95
    root = rng.normal(size=(N_STATE, N_STATE))
    background_cov = root @ root.T / N_STATE + 0.5 * np.eye(N_STATE)
    obs_variances = np.array([0.3, 0.5, 0.4, 0.6])
    mask = np.zeros((N_STEPS + 1, N_STATE))
    mask[:, list(observed_sites)] = 1.0
    return {
        "A": jnp.asarray(transition),
        "B": jnp.asarray(background_cov),
        "r": jnp.asarray(obs_variances),
        "mask": jnp.asarray(mask),
    }


def _ops(problem):
    B_op = lx.MatrixLinearOperator(problem["B"], lx.positive_semidefinite_tag)
    R_op = lx.DiagonalLinearOperator(problem["r"])
    return B_op, R_op


def _strong(problem, **extra):
    B_op, R_op = _ops(problem)
    return vdx.StrongFourDVar(
        forward=LinearForward(problem["A"]),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
        **extra,
    )


def _batch(problem):
    observations = jnp.ones((1, N_STEPS + 1, N_STATE)) * problem["mask"]
    return vdx.Batch1D(input=observations, mask=problem["mask"][None])


def _materialise(operator):
    identity = jnp.eye(N_STATE)
    return jax.vmap(operator.mv)(identity).T


def _smoother_covariance_x0(problem, model_error_cov):
    """Kalman filter + RTS smoother, return Cov(x₀ | y₀ … y_T)."""
    A, B, r, mask = (np.asarray(problem[k]) for k in ("A", "B", "r", "mask"))
    Q = np.asarray(model_error_cov)
    filtered_covs, predicted_covs = [], []
    predicted = B
    for t in range(N_STEPS + 1):
        observed = np.flatnonzero(mask[t])
        # observe only the masked sites: H = rows of I
        H = np.eye(N_STATE)[observed]
        R = np.diag(r[observed])
        innovation_cov = H @ predicted @ H.T + R
        gain = predicted @ H.T @ np.linalg.inv(innovation_cov)
        filtered = (np.eye(N_STATE) - gain @ H) @ predicted
        filtered_covs.append(filtered)
        predicted = A @ filtered @ A.T + Q
        predicted_covs.append(predicted)
    smoothed = filtered_covs[-1]
    for t in range(N_STEPS - 1, -1, -1):
        # Gₜ = Pₜ|ₜ Aᵀ Pₜ₊₁|ₜ⁻¹ ; Pₜ|T = Pₜ|ₜ + Gₜ (Pₜ₊₁|T - Pₜ₊₁|ₜ) Gₜᵀ
        smoother_gain = filtered_covs[t] @ A.T @ np.linalg.inv(predicted_covs[t])
        correction = smoothed - predicted_covs[t]
        smoothed = filtered_covs[t] + smoother_gain @ correction @ smoother_gain.T
    return smoothed


@pytest.mark.parametrize("adapter_cls", [vdx.LaplaceCovariance, vdx.GaussNewtonHessian])
@pytest.mark.parametrize("model_kind", ["strong", "incremental", "reduced_basis"])
def test_strong_matches_kalman_smoother(x64, adapter_cls, model_kind):
    problem = _problem()
    B_op, R_op = _ops(problem)
    common = dict(
        forward=LinearForward(problem["A"]),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
    )
    model = {
        "strong": vdx.StrongFourDVar,
        "incremental": vdx.IncrementalFourDVar,
        "reduced_basis": vdx.ReducedBasisFourDVar,
    }[model_kind](**common)
    if adapter_cls is vdx.LaplaceCovariance:
        adapter = adapter_cls(prior_cov_op=B_op, obs_cov_op=R_op, solver=lx.Cholesky())
    else:
        adapter = adapter_cls(
            prior_cov_op=B_op, obs_cov_op=R_op, n_krylov=200, tol=1e-13
        )
    # linear model: the Gauss-Newton covariance does not depend on x₀*
    analysis = jnp.full(N_STATE, 0.3)
    post = adapter(analysis, model.as_analysis_step(), _batch(problem))
    expected = _smoother_covariance_x0(problem, np.zeros((N_STATE, N_STATE)))
    np.testing.assert_allclose(_materialise(post.cov), expected, atol=1e-8)
    assert post.provenance["structure"] == "strong_4dvar"


def test_weak_matches_kalman_smoother(x64):
    problem = _problem()
    B_op, R_op = _ops(problem)
    model_error_cov = 0.2 * jnp.eye(N_STATE)
    model = vdx.WeakFourDVar(
        forward=LinearForward(problem["A"]),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
        model_err_cov_op=lx.MatrixLinearOperator(
            model_error_cov, lx.positive_semidefinite_tag
        ),
    )
    adapter = vdx.LaplaceCovariance(
        prior_cov_op=B_op, obs_cov_op=R_op, solver=lx.Cholesky()
    )
    analysis = (jnp.zeros(N_STATE), jnp.zeros((N_STEPS, N_STATE)))
    post = adapter(analysis, model, _batch(problem))
    expected = _smoother_covariance_x0(problem, model_error_cov)
    np.testing.assert_allclose(_materialise(post.cov), expected, atol=1e-8)
    # model error loosens the window's constraint on x₀
    strong = _smoother_covariance_x0(problem, np.zeros((N_STATE, N_STATE)))
    assert np.all(np.diag(expected) > np.diag(strong))


def test_weak_single_state_window(x64):
    problem = _problem()
    B_op, R_op = _ops(problem)
    model = vdx.WeakFourDVar(
        forward=LinearForward(problem["A"]),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
        model_err_cov_op=lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct((N_STATE,), jnp.float64)
        ),
    )
    batch = vdx.Batch1D(input=jnp.ones((1, 1, N_STATE)), mask=jnp.ones((1, 1, N_STATE)))
    adapter = vdx.LaplaceCovariance(
        prior_cov_op=B_op, obs_cov_op=R_op, solver=lx.Cholesky()
    )
    post = adapter((jnp.zeros(N_STATE), jnp.zeros((0, N_STATE))), model, batch)
    # P* = (B⁻¹ + R⁻¹)⁻¹
    expected = np.linalg.inv(np.linalg.inv(problem["B"]) + np.diag(1 / problem["r"]))
    np.testing.assert_allclose(_materialise(post.cov), expected, atol=1e-8)


def test_unobserved_sites_learn_through_dynamics(x64):
    problem = _problem(observed_sites=(0, 1))
    B_op, R_op = _ops(problem)
    post = vdx.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)(
        jnp.zeros(N_STATE), _strong(problem), _batch(problem)
    )
    posterior_variance = jnp.diag(_materialise(post.cov))
    prior_variance = jnp.diag(problem["B"])
    # sites 2, 3 are never observed; the rotation carries them into 0, 1
    assert jnp.all(posterior_variance[2:] < 0.9 * prior_variance[2:])


def test_window_beats_single_time(x64):
    """More observed times than the 3DVar Hessian sees ⇒ smaller variance."""
    problem = _problem(observed_sites=(0, 1, 2, 3))
    B_op, R_op = _ops(problem)
    adapter = vdx.LaplaceCovariance(
        prior_cov_op=B_op, obs_cov_op=R_op, solver=lx.Cholesky()
    )
    window = _materialise(
        adapter(jnp.zeros(N_STATE), _strong(problem), _batch(problem)).cov
    )
    single_time = np.linalg.inv(np.linalg.inv(problem["B"]) + np.diag(1 / problem["r"]))
    assert np.all(np.diag(window) < np.diag(single_time) - 1e-3)


class TestSingleTimeUnchanged:
    def test_strong_with_one_state_equals_3dvar(self, x64):
        problem = _problem(observed_sites=(0, 1, 2, 3))
        B_op, R_op = _ops(problem)
        batch = vdx.Batch1D(
            input=jnp.ones((1, 1, N_STATE)), mask=jnp.ones((1, 1, N_STATE))
        )
        adapter = vdx.LaplaceCovariance(
            prior_cov_op=B_op, obs_cov_op=R_op, solver=lx.Cholesky()
        )
        post = adapter(jnp.zeros(N_STATE), _strong(problem), batch)
        expected = np.linalg.inv(
            np.linalg.inv(problem["B"]) + np.diag(1 / problem["r"])
        )
        np.testing.assert_allclose(_materialise(post.cov), expected, atol=1e-8)

    def test_threedvar_uses_obs_operator(self, x64):
        problem = _problem()
        B_op, R_op = _ops(problem)
        model = vdx.ThreeDVar(
            obs_op=vdx.LinearObs(
                H_mat=lx.IdentityLinearOperator(
                    jax.ShapeDtypeStruct((N_STATE,), jnp.float64)
                )
            ),
            prior_mean=jnp.zeros(N_STATE),
            prior_cov_op=B_op,
            obs_cov_op=R_op,
        )
        adapter = vdx.LaplaceCovariance(
            prior_cov_op=B_op, obs_cov_op=R_op, solver=lx.Cholesky()
        )
        post = adapter(jnp.zeros(N_STATE), model, batch=None)
        expected = np.linalg.inv(
            np.linalg.inv(problem["B"]) + np.diag(1 / problem["r"])
        )
        np.testing.assert_allclose(_materialise(post.cov), expected, atol=1e-8)
        assert post.provenance["structure"] == "single_time"


class TestMisuseRaises:
    def test_multi_sample_batch(self):
        problem = _problem()
        B_op, R_op = _ops(problem)
        batch = vdx.Batch1D(
            input=jnp.ones((2, N_STEPS + 1, N_STATE)),
            mask=jnp.ones((2, N_STEPS + 1, N_STATE)),
        )
        adapter = vdx.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)
        with pytest.raises(ValueError, match="holds 2 windows"):
            adapter(jnp.zeros(N_STATE), _strong(problem), batch)

    def test_missing_batch(self):
        problem = _problem()
        B_op, R_op = _ops(problem)
        adapter = vdx.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)
        with pytest.raises(ValueError, match="needs the window's Batch1D"):
            adapter(jnp.zeros(N_STATE), _strong(problem), None)

    def test_nonlinear_parameterization(self):
        # a linear basis= / LinearParameterization control is supported
        # (tests/test_linear_control_posterior.py); a non-affine one raises
        problem = _problem()
        B_op, R_op = _ops(problem)
        field = eqx.nn.MLP(1, 1, 4, 1, key=jax.random.key(0))
        param = vdx.NeuralFieldParameterization(
            field, jnp.linspace(0.0, 1.0, N_STATE)[:, None], 1e-3
        )
        model = _strong(problem, parameterization=param)
        adapter = vdx.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)
        with pytest.raises(ValueError, match="decode is not affine"):
            adapter(jnp.zeros(N_STATE), model, _batch(problem))

    def test_weak_needs_the_pair(self):
        problem = _problem()
        B_op, R_op = _ops(problem)
        model = vdx.WeakFourDVar(
            forward=LinearForward(problem["A"]),
            obs_op=vdx.MaskedIdentity(),
            prior_mean=jnp.zeros(N_STATE),
            prior_cov_op=B_op,
            obs_cov_op=R_op,
            model_err_cov_op=B_op,
        )
        adapter = vdx.LaplaceCovariance(prior_cov_op=B_op, obs_cov_op=R_op)
        with pytest.raises(ValueError, match=r"analysis=\(x0, eta\)"):
            adapter(jnp.zeros(N_STATE), model, _batch(problem))
