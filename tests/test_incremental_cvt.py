"""Control-variable transform for ``IncrementalFourDVar`` (issue #97).

With ``IncrementalConfig(cvt=True)`` the inner Gauss–Newton system is
solved over χ, x₀ = x_b + B^{1/2} χ:

    (I + Sᵀ G'ᵀ R⁻¹ G' S) δχ = −∇_χ J,   S Sᵀ = B,

instead of (B⁻¹ + G'ᵀ R⁻¹ G') δx = −∇J in state space with B⁻¹ applied by
a nested CG. These tests pin that both give the analysis, that the
transform needs no more inner CG iterations under a smooth B, and that the
transform never solves against B.
"""

from __future__ import annotations

import pathlib
import re

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx

N_STATE, N_STEPS = 20, 3


class Shift:
    """xₜ₊₁ = 0.9 · roll(xₜ, 1): linear advection with decay."""

    dt = 1.0

    def step(self, x, dt):
        return 0.9 * jnp.roll(x, 1)


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _smooth_background_cov(length_scale=3.0, jitter=1e-3):
    # a Gaussian kernel of the line distance is PSD (of the periodic
    # distance it is not); the jitter sets λ_min(B) ≈ jitter
    sites = np.arange(N_STATE)
    distance = np.abs(sites[:, None] - sites[None, :])
    cov = np.exp(-0.5 * (distance / length_scale) ** 2) + jitter * np.eye(N_STATE)
    return jnp.asarray(cov)


def _problem(prior_cov):
    rng = np.random.default_rng(1)
    mask = np.zeros((N_STEPS + 1, N_STATE))
    mask[:, ::3] = 1.0
    observations = rng.normal(size=(N_STEPS + 1, N_STATE)) * mask
    batch = vdx.Batch1D(
        input=jnp.asarray(observations)[None], mask=jnp.asarray(mask)[None]
    )
    B_op = lx.MatrixLinearOperator(prior_cov, lx.positive_semidefinite_tag)
    R_op = lx.DiagonalLinearOperator(jnp.full(N_STATE, 0.25))
    return batch, B_op, R_op


def _exact_analysis(prior_cov, batch):
    """Linear model and H: x₀* = (B⁻¹ + GᵀR⁻¹G)⁻¹ GᵀR⁻¹ y (x_b = 0)."""
    transition = 0.9 * np.roll(np.eye(N_STATE), 1, axis=0)
    mask = np.asarray(batch.mask[0])
    rows = [
        np.diag(mask[t]) @ np.linalg.matrix_power(transition, t)
        for t in range(N_STEPS + 1)
    ]
    G = np.concatenate(rows)
    y = np.asarray(batch.input[0]).reshape(-1)
    precision = np.linalg.inv(np.asarray(prior_cov)) + G.T @ G / 0.25
    return np.linalg.solve(precision, G.T @ y / 0.25)


def _model(B_op, R_op, **config):
    return vdx.IncrementalFourDVar(
        forward=Shift(),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
        config=vdx.IncrementalConfig(**config),
    )


def _relative_error(estimate, exact):
    return float(np.linalg.norm(np.asarray(estimate) - exact) / np.linalg.norm(exact))


@pytest.mark.slow
@pytest.mark.parametrize("cvt", [True, False])
def test_both_paths_reach_the_exact_analysis(x64, cvt):
    prior_cov = _smooth_background_cov()
    batch, B_op, R_op = _problem(prior_cov)
    model = _model(
        B_op,
        R_op,
        n_outer=1,
        n_inner=400,
        cvt=cvt,
        cg_atol=1e-11,
        cg_rtol=1e-11,
        covariance_tol=1e-12,
    )
    analysis = model(batch)[0]
    exact = _exact_analysis(prior_cov, batch)
    assert _relative_error(analysis, exact) < 1e-6


@pytest.mark.slow
def test_cvt_and_state_space_agree(x64):
    prior_cov = _smooth_background_cov()
    batch, B_op, R_op = _problem(prior_cov)
    tight = dict(
        n_outer=2, n_inner=400, cg_atol=1e-11, cg_rtol=1e-11, covariance_tol=1e-12
    )
    with_cvt = _model(B_op, R_op, cvt=True, **tight)(batch)
    without_cvt = _model(B_op, R_op, cvt=False, **tight)(batch)
    np.testing.assert_allclose(with_cvt, without_cvt, rtol=1e-6, atol=1e-8)


def _iterations_to_converge(cvt, prior_cov, batch, B_op, R_op, budgets):
    exact = _exact_analysis(prior_cov, batch)
    for budget in budgets:
        model = _model(
            B_op,
            R_op,
            n_outer=1,
            n_inner=budget,
            cvt=cvt,
            cg_atol=0.0,
            cg_rtol=0.0,
            covariance_tol=1e-12,
        )
        if _relative_error(model(batch)[0], exact) < 1e-4:
            return budget
    return np.inf


@pytest.mark.slow
def test_cvt_needs_no_more_inner_iterations(x64):
    """Smooth B: B⁻¹'s huge eigenvalues slow state-space CG; χ-space does not."""
    prior_cov = _smooth_background_cov(length_scale=3.0, jitter=1e-3)
    batch, B_op, R_op = _problem(prior_cov)
    budgets = [2, 4, 8, 12, 16, 20, 30, 40, 60, 80, 120]
    with_cvt = _iterations_to_converge(True, prior_cov, batch, B_op, R_op, budgets)
    without_cvt = _iterations_to_converge(False, prior_cov, batch, B_op, R_op, budgets)
    assert with_cvt <= without_cvt
    assert with_cvt < np.inf


@pytest.mark.slow
def test_cvt_never_solves_against_the_prior(x64, monkeypatch):
    """Tag B and record every operator handed to lineax.linear_solve."""
    marker = object()
    # well conditioned, so the state-space comparison run is cheap
    prior_cov = _smooth_background_cov(jitter=0.1)
    batch, _, R_op = _problem(prior_cov)
    B_op = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(prior_cov),
        frozenset({lx.positive_semidefinite_tag, marker}),
    )
    solved_against_prior = []
    original_solve = lx.linear_solve

    def recording_solve(operator, vector, *args, **kwargs):
        tags = getattr(operator, "tags", frozenset())
        solved_against_prior.append(marker in tags)
        return original_solve(operator, vector, *args, **kwargs)

    monkeypatch.setattr(lx, "linear_solve", recording_solve)

    _model(B_op, R_op, n_outer=1, n_inner=10, cvt=True)(batch)
    assert solved_against_prior and not any(solved_against_prior)

    # the recorder does see B on the state-space path
    solved_against_prior.clear()
    _model(B_op, R_op, n_outer=1, n_inner=10, cvt=False)(batch)
    assert any(solved_against_prior)


def test_user_supplied_square_root(x64):
    """A Cholesky factor L (L Lᵀ = B) is as good as the symmetric root."""
    prior_cov = _smooth_background_cov()
    batch, B_op, R_op = _problem(prior_cov)
    cholesky = lx.MatrixLinearOperator(jnp.linalg.cholesky(prior_cov))
    tight = vdx.IncrementalConfig(
        n_outer=1, n_inner=400, cg_atol=1e-11, cg_rtol=1e-11, covariance_tol=1e-12
    )
    model = vdx.IncrementalFourDVar(
        forward=Shift(),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N_STATE),
        prior_cov_op=B_op,
        obs_cov_op=R_op,
        config=tight,
        prior_sqrt_op=cholesky,
    )
    exact = _exact_analysis(prior_cov, batch)
    assert _relative_error(model(batch)[0], exact) < 1e-6


def test_default_config_uses_cvt():
    assert vdx.IncrementalConfig().cvt is True


@pytest.mark.slow
def test_docs_08_linear_gaussian_example_runs():
    """The docs/08 blocks (once a TypeError: no cvt field) run and agree with OI."""
    page = pathlib.Path(__file__).parents[1] / "docs" / "08_incremental_4dvar.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.S)
    # the page's blocks build on each other, so they run in one namespace
    namespace: dict = {}
    exec("\n".join(blocks), namespace)
    closed_form = namespace["closed_form"]
    gauss_newton = namespace["gauss_newton"]
    assert jnp.allclose(closed_form[:, 0], gauss_newton, atol=1e-3)
