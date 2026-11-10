"""``ThreeDVar`` on whitened residuals and the dual ``DualThreeDVar`` (vardax#279).

A least-squares ``minimiser`` (``argminax.solve.GaussNewtonCG``) solves 3D-Var as
Gauss–Newton on the whitened residuals; ``DualThreeDVar`` solves the same increment in
observation space on ``argminax.solve.dual_solve``. For a linear h both are the BLUE,
x⋆ = μ_x + Σ_x Gᵀ (G Σ_x Gᵀ + Σ_ε,oo)⁻¹ (y_o − G μ_x) over the observed rows o.
"""

from __future__ import annotations

from argminax.solve import GaussNewtonCG
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest

import vardax as vdx

N, M = 7, 4
PSD = lx.positive_semidefinite_tag


def _smooth(n, length, jitter):
    distance = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return np.exp(-0.5 * (distance / length) ** 2) + jitter * np.eye(n)


def _linear_problem():
    """Fewer observations than states, a correlated Σ_x and Σ_ε, one masked row."""
    rng = np.random.default_rng(3)
    H = rng.normal(size=(M, N))
    B = _smooth(N, 2.0, 1e-2)
    R = 0.3 * _smooth(M, 1.0, 0.2)
    x_b = rng.normal(size=N)
    y = rng.normal(size=(2, M))
    mask = np.ones((2, M))
    mask[1, 2] = 0.0
    return H, B, R, x_b, y, mask


def _blue(H, B, R, x_b, y, mask):
    observed = mask > 0
    G = H[observed]
    gain = B @ G.T @ np.linalg.inv(G @ B @ G.T + R[np.ix_(observed, observed)])
    return x_b + gain @ (y[observed] - G @ x_b)


def _common(H, B, R, x_b):
    return dict(
        obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(jnp.asarray(H))),
        prior_mean=jnp.asarray(x_b),
        prior_cov_op=lx.MatrixLinearOperator(jnp.asarray(B), PSD),
        obs_cov_op=lx.MatrixLinearOperator(jnp.asarray(R), PSD),
    )


def _batch(y, mask):
    return vdx.Batch1D(input=jnp.asarray(y * mask), mask=jnp.asarray(mask))


def _expected(problem):
    H, B, R, x_b, y, mask = problem
    return np.stack([_blue(H, B, R, x_b, y[k], mask[k]) for k in range(len(y))])


def test_gauss_newton_threedvar_is_the_blue(x64):
    problem = _linear_problem()
    H, B, R, x_b, y, mask = problem
    model = vdx.ThreeDVar(
        **_common(H, B, R, x_b),
        minimiser=GaussNewtonCG(rtol=1e-12, atol=1e-12, max_cg_steps=100),
    )
    np.testing.assert_allclose(
        model(_batch(y, mask)), _expected(problem), rtol=1e-10, atol=1e-12
    )


def test_dual_threedvar_is_the_blue(x64):
    problem = _linear_problem()
    H, B, R, x_b, y, mask = problem
    model = vdx.DualThreeDVar(**_common(H, B, R, x_b))
    np.testing.assert_allclose(
        model(_batch(y, mask)), _expected(problem), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("tol", [1e-6, 1e-10])
def test_dual_equals_the_primal_step_to_the_cg_tolerance(x64, tol):
    """Linear h: one dual solve is the primal Gauss–Newton step."""
    H, B, R, x_b, y, mask = _linear_problem()
    batch = _batch(y, mask)
    primal = vdx.ThreeDVar(
        **_common(H, B, R, x_b),
        minimiser=GaussNewtonCG(rtol=tol, atol=tol, max_cg_steps=100),
    )(batch)
    dual = vdx.DualThreeDVar(
        **_common(H, B, R, x_b), solver=lx.CG(rtol=tol, atol=tol, max_steps=100)
    )(batch)
    scale = np.max(np.abs(np.asarray(primal)))
    # the condition number of the dual system turns its residual tolerance into the
    # error on the analysis; 100 bounds it for this problem
    np.testing.assert_allclose(dual, primal, atol=100 * tol * scale)


def test_a_masked_observation_has_no_multiplier(x64):
    H, B, R, x_b, y, mask = _linear_problem()
    _, solution = vdx.DualThreeDVar(**_common(H, B, R, x_b)).analyse(_batch(y, mask))
    assert float(solution.multiplier[1, 2]) == 0.0
    assert bool(jnp.all(solution.result == lx.RESULTS.successful))


def _nonlinear(n_outer=None, minimiser=None):
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float64))
    common = dict(
        obs_op=vdx.CallableObs(fn=lambda x: x + 0.1 * x**3),
        prior_mean=jnp.array([0.5, -0.2, 1.0]),
        prior_cov_op=eye,
        obs_cov_op=0.5 * eye,
    )
    if n_outer is not None:
        return vdx.DualThreeDVar(**common, n_outer=n_outer)
    return vdx.ThreeDVar(**common, minimiser=minimiser, max_steps=500)


def _nonlinear_batch():
    y = jnp.array([[1.5, -1.0, 2.0], [0.2, 0.4, -0.3]])
    mask = jnp.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    return vdx.Batch1D(input=y * mask, mask=mask)


def test_nonlinear_h_reaches_the_primal_minimum(x64):
    batch = _nonlinear_batch()
    bfgs = _nonlinear(minimiser=optx.BFGS(rtol=1e-12, atol=1e-12))(batch)
    gauss_newton = _nonlinear(minimiser=GaussNewtonCG(rtol=1e-12, atol=1e-12))(batch)
    dual = _nonlinear(n_outer=12)(batch)
    np.testing.assert_allclose(gauss_newton, bfgs, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(dual, bfgs, rtol=1e-7, atol=1e-9)
    # one dual outer iteration is one Gauss–Newton step from μ_x, short of the minimum
    assert not np.allclose(_nonlinear(n_outer=1)(batch), bfgs, atol=1e-3)


def test_gauss_newton_analyse_reports_in_the_whitened_control(x64):
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float64))
    model = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros((1, 3)),
        prior_cov_op=4.0 * eye,
        obs_cov_op=eye,
        minimiser=GaussNewtonCG(rtol=1e-10, atol=1e-10),
    )
    y = jnp.stack([jnp.ones((1, 3)), 2.0 * jnp.ones((1, 3))])
    batch = vdx.Batch1D(input=y, mask=jnp.ones_like(y))
    analysis, result = model.analyse(batch)
    np.testing.assert_array_equal(analysis, model(batch))
    # Σ_x = 4 I, Σ_ε = I: x⋆ = 4 y / 5, and x = 2 z, so z⋆ = 2 y / 5
    np.testing.assert_allclose(analysis, 0.8 * y, rtol=1e-10)
    np.testing.assert_allclose(result.value, 0.4 * y, rtol=1e-10)
    # 𝒥(x⋆) = ½ ‖x⋆‖² / 4 + ½ ‖y − x⋆‖² = y² / 10 per entry
    np.testing.assert_allclose(result.cost, jnp.sum(y**2, axis=(1, 2)) / 10, rtol=1e-10)
    assert bool(jnp.all(result.converged))
    assert bool(jnp.all(result.grad_norm < 1e-8))


def test_gauss_newton_threedvar_on_a_point_batch(x64):
    axis = jnp.arange(float(N))
    _, B, _, x_b, _, _ = _linear_problem()
    points = vdx.PointObsBatch(
        coords=jnp.array([[0.5], [2.0], [4.25], [6.0]]),
        values=jnp.array([1.0, -0.5, 0.3, 2.0]),
        sigma=jnp.array([0.5, 1.0, 0.8, 0.4]),
        valid=jnp.array([True, True, False, True]),
    )
    # Σ_x on the (1, N) state a GridPointObs reads; its dense root acts on (N,)
    prior_cov = lx.TaggedLinearOperator(
        lx.FunctionLinearOperator(
            lambda x: (jnp.asarray(B) @ x[0])[None],
            jax.ShapeDtypeStruct((1, N), jnp.float64),
        ),
        PSD,
    )
    common = dict(
        obs_op=vdx.GridPointObs((axis,)),
        prior_mean=jnp.asarray(x_b)[None],
        prior_cov_op=prior_cov,
        obs_cov_op=None,
    )
    bfgs = vdx.ThreeDVar(
        **common,
        minimiser=optx.BFGS(rtol=1e-12, atol=1e-12),
        cov_solver=lx.CG(rtol=1e-13, atol=1e-13),
        max_steps=1000,
    )
    gauss_newton = vdx.ThreeDVar(
        **common, minimiser=GaussNewtonCG(rtol=1e-12, atol=1e-12, max_cg_steps=100)
    )
    analysis = gauss_newton(points)
    assert analysis.shape == (1, N)
    np.testing.assert_allclose(analysis, bfgs(points), atol=1e-8)


@pytest.mark.parametrize("make_model", ["threedvar", "dual"])
def test_the_analysis_step_centres_on_the_forecast(x64, make_model):
    problem = _linear_problem()
    H, B, R, _, y, _ = problem
    forecast = np.linspace(-1.0, 1.0, N)
    common = _common(H, B, R, np.zeros(N))
    if make_model == "dual":
        model = vdx.DualThreeDVar(**common)
    else:
        model = vdx.ThreeDVar(**common, minimiser=GaussNewtonCG(rtol=1e-12, atol=1e-12))
    obs = jnp.asarray(y[1]).at[2].set(jnp.nan)
    analysis = model.as_analysis_step()(
        jnp.asarray(forecast), obs, obs_op=None, obs_err_cov=None
    )
    expected = _blue(H, B, R, forecast, y[1], np.array([1.0, 1.0, 0.0, 1.0]))
    np.testing.assert_allclose(analysis, expected, rtol=1e-8, atol=1e-10)


def test_the_dual_posterior_is_the_blue_posterior(x64):
    problem = _linear_problem()
    H, B, R, x_b, y, mask = problem
    one = _batch(y[:1], mask[:1])
    posterior = vdx.DualThreeDVar(**_common(H, B, R, x_b)).posterior(one)
    np.testing.assert_allclose(posterior.mean, _expected(problem)[0], atol=1e-10)
    # Σ_x|y = Σ_x − Σ_x Hᵀ (H Σ_x Hᵀ + Σ_ε)⁻¹ H Σ_x, everything observed
    gain = B @ H.T @ np.linalg.inv(H @ B @ H.T + R)
    np.testing.assert_allclose(
        posterior.cov.as_matrix(), B - gain @ H @ B, rtol=1e-5, atol=1e-8
    )


def test_dual_runs_under_filter_jit(x64):
    H, B, R, x_b, y, mask = _linear_problem()
    model = vdx.DualThreeDVar(**_common(H, B, R, x_b))
    batch = _batch(y, mask)
    np.testing.assert_allclose(eqx.filter_jit(model)(batch), model(batch))


def test_dual_refuses_a_point_batch():
    model = vdx.DualThreeDVar(
        obs_op=vdx.GridPointObs((jnp.arange(3.0),)),
        prior_mean=jnp.zeros(3),
        prior_cov_op=lx.DiagonalLinearOperator(jnp.ones(3)),
        obs_cov_op=None,
    )
    points = vdx.PointObsBatch(
        coords=jnp.zeros((1, 1)),
        values=jnp.ones(1),
        sigma=jnp.ones(1),
        valid=jnp.ones(1, bool),
    )
    with pytest.raises(vdx.UnsupportedInputError, match="Batch1D"):
        model.analyse(points)


@pytest.mark.unchecked_types
@pytest.mark.parametrize("n_outer", [0, 2.0, False])
def test_dual_n_outer_must_be_a_positive_int(n_outer):
    with pytest.raises(vdx.ConfigurationError, match="n_outer"):
        vdx.DualThreeDVar(
            obs_op=vdx.MaskedIdentity(),
            prior_mean=jnp.zeros(3),
            prior_cov_op=lx.DiagonalLinearOperator(jnp.ones(3)),
            obs_cov_op=lx.DiagonalLinearOperator(jnp.ones(3)),
            n_outer=n_outer,
        )


def test_a_least_squares_minimiser_refuses_a_parameterization():
    basis = jnp.eye(3)[:, :2]
    with pytest.raises(vdx.ConfigurationError, match="whitened residuals"):
        vdx.ThreeDVar(
            obs_op=vdx.MaskedIdentity(),
            prior_mean=jnp.zeros(3),
            prior_cov_op=lx.DiagonalLinearOperator(jnp.ones(3)),
            obs_cov_op=lx.DiagonalLinearOperator(jnp.ones(3)),
            minimiser=GaussNewtonCG(rtol=1e-6, atol=1e-6),
            parameterization=vdx.LinearParameterization(
                background=jnp.zeros(3), basis=basis
            ),
        )


@pytest.mark.parametrize("make_model", ["threedvar", "dual"])
def test_a_scaled_identity_on_a_field_stays_diagonal(x64, make_model):
    """σ² I over a (2, 2, 3) observation field, masked: Σ̃_ε stays diagonal.

    Composed lazily, gaussx's dense fallback solve of Σ̃_ε fails on a non-flat
    observation shape; the closed form is x⋆ = ∑ᵢ mᵢ yᵢ / (σ² + ∑ᵢ mᵢ) per state.
    """
    structure = jax.ShapeDtypeStruct((2, 2, 3), jnp.float64)
    common = dict(
        obs_op=vdx.CallableObs(
            fn=lambda x: jnp.broadcast_to(x[:, None, None], (2, 2, 3))
        ),
        prior_mean=jnp.zeros(2),
        prior_cov_op=lx.DiagonalLinearOperator(jnp.ones(2)),
        obs_cov_op=0.5 * lx.IdentityLinearOperator(structure),
    )
    if make_model == "dual":
        model = vdx.DualThreeDVar(**common)
    else:
        model = vdx.ThreeDVar(**common, minimiser=GaussNewtonCG(1e-12, 1e-12))
    y = jnp.arange(12.0).reshape(1, 2, 2, 3)
    mask = jnp.ones_like(y).at[0, 0, 1, 2].set(0.0)
    analysis = model(vdx.Batch1D(input=y * mask, mask=mask))[0]
    observed = jnp.sum(mask * y, axis=(2, 3))[0]
    counts = jnp.sum(mask, axis=(2, 3))[0]
    np.testing.assert_allclose(analysis, observed / (0.5 + counts), rtol=1e-8)
