"""Posterior of a linear control x = x* + Φθ (#118).

With ``parameterization=LinearParameterization(basis, x_b)`` (or
``ReducedBasisFourDVar(basis=...)``) the analysis minimises
``½ θᵀ Q⁻¹ θ + ½ ‖y − G(x_b + Φθ)‖²_{R⁻¹}``, so the Gauss–Newton posterior is

    P_θ = (Q⁻¹ + Φᵀ G'ᵀ R⁻¹ G' Φ)⁻¹,   P_x = Φ P_θ Φᵀ.

The whitened control Φ = L, Q = I with B = L Lᵀ must reproduce the
state-space posterior exactly. Any other parameterization must raise rather
than return a covariance built from B.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


ADAPTERS = [
    pytest.param(lambda R: vdx.LaplaceCovariance(obs_cov_op=R), id="laplace"),
    pytest.param(
        lambda R: vdx.GaussNewtonHessian(obs_cov_op=R, tol=1e-12, n_krylov=200),
        id="gauss_newton",
    ),
]


def _dense(operator, n, shape=None):
    shape = (n,) if shape is None else shape
    columns = jax.vmap(lambda e: operator.mv(e.reshape(shape)).reshape(-1))
    return columns(jnp.eye(n)).T


# ──────────────────────────────────────────────────────────────────────
# Single time: ThreeDVar
# ──────────────────────────────────────────────────────────────────────


def _single_time(n: int = 6, m: int = 15, r: float = 1e-2):
    rng = np.random.default_rng(0)
    F = jnp.asarray(rng.uniform(size=(m, n)))
    x = jnp.linspace(0.0, 1.0, n)
    s = jnp.sqrt(3.0) * jnp.abs(x[:, None] - x[None, :]) / 0.4
    B = 0.5 * (1.0 + s) * jnp.exp(-s) + 1e-3 * jnp.eye(n)
    y = F @ jnp.linspace(0.5, 1.5, n) + jnp.sqrt(r) * rng.normal(size=m)
    R = lx.DiagonalLinearOperator(jnp.full((m,), r))
    common = dict(
        obs_op=vdx.CallableObs(fn=lambda q: q @ F.T),
        prior_mean=jnp.ones((1, n)),
        prior_cov_op=lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
        obs_cov_op=lx.FunctionLinearOperator(
            lambda z: r * z,
            jax.ShapeDtypeStruct((1, m), jnp.float64),
            lx.positive_semidefinite_tag,
        ),
    )
    batch = vdx.Batch1D(input=jnp.asarray(y)[None, None], mask=jnp.ones((1, 1, m)))
    return F, B, r, R, common, batch


@pytest.mark.slow
@pytest.mark.parametrize("make_adapter", ADAPTERS)
def test_whitened_control_matches_state_space(x64, make_adapter):
    F, B, r, _, common, batch = _single_time()
    n = B.shape[0]
    L = jnp.linalg.cholesky(B)
    whitened = vdx.LinearParameterization(
        vdx.linear_basis(L, variance=1.0), x_b=common["prior_mean"]
    )
    model = vdx.ThreeDVar(**common, parameterization=whitened)
    analysis = model(batch)[0]
    R_op = common["obs_cov_op"]
    posterior = make_adapter(R_op)(analysis, model, batch)
    cov = _dense(posterior.cov, n, (1, n))
    expected = jnp.linalg.inv(jnp.linalg.inv(B) + F.T @ F / r)
    assert jnp.max(jnp.abs(cov - expected)) / jnp.max(jnp.abs(expected)) < 1e-8
    assert jnp.array_equal(posterior.mean, analysis)


@pytest.mark.slow
def test_reduced_control_is_phi_p_theta_phi_t(x64):
    """Φ not square and Q ≠ I: the covariance lives in span(Φ)."""
    F, _, r, _, common, batch = _single_time()
    n = F.shape[1]
    phi = jnp.asarray(np.random.default_rng(3).normal(size=(n, 3)))
    variance = jnp.asarray([0.5, 2.0, 4.0])
    param = vdx.LinearParameterization(
        vdx.linear_basis(phi, variance=variance), x_b=common["prior_mean"]
    )
    model = vdx.ThreeDVar(**common, parameterization=param)
    analysis = model(batch)[0]
    posterior = vdx.LaplaceCovariance()(analysis, model, batch)
    cov = _dense(posterior.cov, n, (1, n))
    p_theta = jnp.linalg.inv(jnp.diag(1.0 / variance) + phi.T @ F.T @ F @ phi / r)
    expected = phi @ p_theta @ phi.T
    assert jnp.max(jnp.abs(cov - expected)) / jnp.max(jnp.abs(expected)) < 1e-8
    assert int(jnp.linalg.matrix_rank(cov, rtol=1e-8)) == 3


def test_neural_field_parameterization_raises(x64):
    """It used to return the B-based covariance silently."""
    _, B, _, _, common, batch = _single_time()
    n = B.shape[0]
    field = eqx.nn.MLP(1, 1, 8, 2, key=jax.random.key(0))
    param = vdx.NeuralFieldParameterization(
        field, jnp.linspace(0.0, 1.0, n)[:, None], 1e-3
    )
    model = vdx.ThreeDVar(**common, parameterization=param)
    with pytest.raises(ValueError, match="decode is not affine"):
        vdx.LaplaceCovariance()(jnp.zeros((1, n)), model, batch)


# ──────────────────────────────────────────────────────────────────────
# Rollout: strong 4DVar and ReducedBasisFourDVar
# ──────────────────────────────────────────────────────────────────────

N_STATE, N_STEPS = 4, 3


class _LinearForward:
    dt = 1.0

    def __init__(self, transition):
        self.transition = transition

    def step(self, x, dt):
        return self.transition @ x


def _rollout_problem():
    rng = np.random.default_rng(0)
    skew = rng.normal(size=(N_STATE, N_STATE))
    A = jnp.asarray(jax.scipy.linalg.expm(0.4 * (skew - skew.T))) * 0.95
    root = rng.normal(size=(N_STATE, N_STATE))
    B = jnp.asarray(root @ root.T / N_STATE + 0.5 * np.eye(N_STATE))
    r = jnp.asarray([0.3, 0.5, 0.4, 0.6])
    mask = jnp.zeros((N_STEPS + 1, N_STATE)).at[:, :2].set(1.0)
    batch = vdx.Batch1D(input=mask[None], mask=mask[None])
    common = dict(
        forward=_LinearForward(A),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
        obs_cov_op=lx.DiagonalLinearOperator(r),
    )
    # G = [mₜ ⊙ Aᵗ]ₜ, stacked: ((T + 1) N, N)
    G = jnp.concatenate(
        [mask[t][:, None] * jnp.linalg.matrix_power(A, t) for t in range(N_STEPS + 1)]
    )
    R_inv = jnp.diag(jnp.tile(1.0 / r, N_STEPS + 1))
    return B, G, R_inv, common, batch


@pytest.mark.slow
@pytest.mark.parametrize("make_adapter", ADAPTERS)
def test_strong_whitened_control_matches_state_space(x64, make_adapter):
    B, _, _, common, batch = _rollout_problem()
    whitened = vdx.LinearParameterization(
        vdx.linear_basis(jnp.linalg.cholesky(B), variance=1.0),
        x_b=common["prior_mean"],
    )
    state_model = vdx.StrongFourDVar(**common)
    param_model = vdx.StrongFourDVar(**common, parameterization=whitened)
    R_op = common["obs_cov_op"]
    x0 = jnp.zeros(N_STATE)
    expected = _dense(make_adapter(R_op)(x0, state_model, batch).cov, N_STATE)
    cov = _dense(make_adapter(R_op)(x0, param_model, batch).cov, N_STATE)
    assert jnp.max(jnp.abs(cov - expected)) / jnp.max(jnp.abs(expected)) < 1e-8


def test_reduced_basis_fourdvar_basis(x64):
    """``basis=`` is the same linear control; it used to raise."""
    _, G, R_inv, common, batch = _rollout_problem()
    phi = jnp.eye(N_STATE)[:, :2] + 0.1
    variance = jnp.asarray([0.5, 2.0])
    model = vdx.ReducedBasisFourDVar(
        **common, basis=vdx.linear_basis(phi, variance=variance)
    )
    posterior = vdx.LaplaceCovariance()(jnp.zeros(N_STATE), model, batch)
    cov = _dense(posterior.cov, N_STATE)
    reduced = G @ phi
    p_theta = jnp.linalg.inv(jnp.diag(1.0 / variance) + reduced.T @ R_inv @ reduced)
    expected = phi @ p_theta @ phi.T
    assert jnp.max(jnp.abs(cov - expected)) / jnp.max(jnp.abs(expected)) < 1e-8
    assert posterior.provenance["structure"] == "strong_4dvar"


def test_weak_fourdvar_parameterization_still_raises(x64):
    B, _, _, common, batch = _rollout_problem()
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N_STATE,), jnp.float64))
    param = vdx.LinearParameterization(
        vdx.linear_basis(jnp.linalg.cholesky(B)), x_b=common["prior_mean"]
    )
    model = vdx.WeakFourDVar(**common, model_err_cov_op=eye, parameterization=param)
    analysis = (jnp.zeros(N_STATE), jnp.zeros((N_STEPS, N_STATE)))
    with pytest.raises(ValueError, match="basis= or parameterization="):
        vdx.LaplaceCovariance()(analysis, model, batch)


# ──────────────────────────────────────────────────────────────────────
# Correlated R with a missing observation: the posterior uses the same
# per-time precision as the cost
# ──────────────────────────────────────────────────────────────────────


class _Stay:
    dt = 1.0

    def step(self, x, dt):
        return x


def _two_site(cls, **extra):
    """One time, identity H, R = [[1, ½], [½, 1]], site 1 unobserved."""
    R = jnp.asarray([[1.0, 0.5], [0.5, 1.0]])
    mask = jnp.asarray([[1.0, 0.0]])
    model = cls(
        forward=_Stay(),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(2),
        prior_cov_op=lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag),
        obs_cov_op=lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag),
        **extra,
    )
    batch = vdx.Batch1D(input=mask[None], mask=mask[None])
    return model, batch, R


def test_reduced_basis_marginalises_missing_observations(x64):
    """Φ = [1, 0]ᵀ, Q = 1: the cost sees only R_oo = 1, so P_θ = ½ (not 3/7)."""
    basis = vdx.linear_basis(jnp.asarray([[1.0], [0.0]]), variance=1.0)
    model, batch, _ = _two_site(vdx.ReducedBasisFourDVar, basis=basis)
    cov = _dense(vdx.LaplaceCovariance()(jnp.zeros(2), model, batch).cov, 2)
    assert jnp.allclose(cov, jnp.asarray([[0.5, 0.0], [0.0, 0.0]]), atol=1e-12)


def test_reduced_basis_state_control_marginalises_too(x64):
    """B = I: P = (I + Hᵀ R_oo⁻¹ H)⁻¹ = diag(½, 1)."""
    model, batch, _ = _two_site(vdx.ReducedBasisFourDVar)
    cov = _dense(vdx.LaplaceCovariance()(jnp.zeros(2), model, batch).cov, 2)
    assert jnp.allclose(cov, jnp.diag(jnp.asarray([0.5, 1.0])), atol=1e-12)


def test_strong_fourdvar_keeps_full_r_inverse(x64):
    """StrongFourDVar's cost applies R⁻¹ to the zeroed residual; unchanged."""
    model, batch, R = _two_site(vdx.StrongFourDVar)
    cov = _dense(vdx.LaplaceCovariance()(jnp.zeros(2), model, batch).cov, 2)
    H = jnp.diag(jnp.asarray([1.0, 0.0]))
    expected = jnp.linalg.inv(jnp.eye(2) + H.T @ jnp.linalg.inv(R) @ H)
    assert jnp.allclose(cov, expected, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Single time with a missing observation: the posterior uses M H'
# ──────────────────────────────────────────────────────────────────────


def _single_time_masked(cls, **extra):
    """Identity H on two sites, B = R = I, site 1 unobserved."""
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 2), jnp.float64))
    model = cls(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros((1, 2)),
        prior_cov_op=eye,
        obs_cov_op=eye,
        **extra,
    )
    mask = jnp.asarray([[[1.0, 0.0]]])
    batch = vdx.Batch1D(input=mask, mask=mask)
    return model, batch


EXPECTED_MASKED = jnp.diag(jnp.asarray([0.5, 1.0]))


@pytest.mark.parametrize("make_adapter", ADAPTERS)
def test_single_time_linear_control_masks_missing_obs(x64, make_adapter):
    """Φ = I, Q = R = I, mask [1, 0]: diag(½, 1), not diag(½, ½)."""
    param = vdx.LinearParameterization(
        vdx.linear_basis(jnp.eye(2), variance=1.0), x_b=jnp.zeros((1, 2))
    )
    model, batch = _single_time_masked(vdx.ThreeDVar, parameterization=param)
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 2), jnp.float64))
    posterior = make_adapter(eye)(jnp.zeros((1, 2)), model, batch)
    assert jnp.allclose(_dense(posterior.cov, 2, (1, 2)), EXPECTED_MASKED, atol=1e-10)


@pytest.mark.parametrize("cls", [vdx.ThreeDVar, vdx.OptimalInterpolation])
def test_single_time_state_control_masks_missing_obs(x64, cls):
    model, batch = _single_time_masked(cls)
    posterior = vdx.LaplaceCovariance()(jnp.zeros((1, 2)), model, batch)
    assert jnp.allclose(_dense(posterior.cov, 2, (1, 2)), EXPECTED_MASKED, atol=1e-10)


def test_multi_window_batch_with_missing_obs_raises(x64):
    model, batch = _single_time_masked(vdx.ThreeDVar)
    two = jax.tree.map(lambda a: jnp.concatenate([a, a]), batch)
    with pytest.raises(ValueError, match="windows with missing observations"):
        vdx.LaplaceCovariance()(jnp.zeros((1, 2)), model, two)
