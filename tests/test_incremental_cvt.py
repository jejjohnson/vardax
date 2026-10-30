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

import equinox as eqx
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
    # ```python or a MyST ```{code-block} python with option lines first
    fence = r"```(?:python|\{code-block\} python)\n(?::[\w-]+:[^\n]*\n)*(.*?)```"
    blocks = re.findall(fence, page.read_text(), flags=re.S)
    # the page's blocks build on each other, so they run in one namespace
    namespace: dict = {}
    exec("\n".join(blocks), namespace)
    closed_form = namespace["closed_form"]
    gauss_newton = namespace["gauss_newton"]
    assert jnp.allclose(closed_form[:, 0], gauss_newton, atol=1e-3)


class _Static:
    """xₜ₊₁ = xₜ: a one-time window makes the outer loop Gauss–Newton 3DVar."""

    dt = 1.0

    def step(self, x, dt):
        return x


def _matern_window(cvt, length_scale=10.0):
    """Matérn-3/2 B on 60 cells, every third cell observed.

    The default length scale of 10 cells gives cond(B) ≈ 4.8e4.
    """
    n_cells, n_obs = 60, 20
    cells = jnp.arange(n_cells, dtype=float)
    gap = jnp.abs(cells[:, None] - cells[None, :]) / length_scale
    root_three_gap = jnp.sqrt(3.0) * gap
    matern = (1 + root_three_gap) * jnp.exp(-root_three_gap) + 1e-8 * jnp.eye(n_cells)
    observed = jnp.eye(n_cells)[::3]
    model = vdx.IncrementalFourDVar(
        forward=_Static(),
        obs_op=vdx.CallableObs(fn=lambda x: observed @ x),
        prior_mean=jnp.zeros(n_cells),
        prior_cov_op=lx.MatrixLinearOperator(matern, lx.positive_semidefinite_tag),
        obs_cov_op=lx.MatrixLinearOperator(
            0.01 * jnp.eye(n_obs), lx.positive_semidefinite_tag
        ),
        config=vdx.IncrementalConfig(
            n_outer=1, n_inner=200, cg_atol=1e-12, cg_rtol=1e-12, cvt=cvt
        ),
    )
    obs = jnp.sin(cells[::3] / 8.0)
    batch = vdx.Batch1D(input=obs[None, None], mask=jnp.ones((1, 1, n_obs)))
    # Σ_x|y = Σ_x − G H Σ_x, with G the Kalman gain
    innovation = observed @ matern @ observed.T + 0.01 * jnp.eye(n_obs)
    gain = matern @ observed.T @ jnp.linalg.inv(innovation)
    return model, batch, matern - gain @ observed @ matern


def test_cvt_posterior_on_a_smooth_prior(x64):
    """The posterior is built over the whitened control, never B⁻¹ (#161)."""
    model, batch, exact = _matern_window(cvt=True)
    posterior = model.posterior(batch)
    dense = jax.vmap(posterior.cov.mv)(jnp.eye(exact.shape[0]))
    assert _relative_error(dense, exact) < 1e-8


def test_state_space_posterior_is_unchanged(x64):
    """``cvt=False`` keeps the default Laplace adapter in state space."""
    # a rough B, which state-space CG can invert
    model, batch, _ = _matern_window(cvt=False, length_scale=0.5)
    analysis = model(batch)[0]
    default = model.posterior(batch).cov
    explicit = vdx.LaplaceCovariance()(analysis, model, batch).cov
    probe = jnp.eye(analysis.shape[0])[:3]
    assert jnp.array_equal(jax.vmap(default.mv)(probe), jax.vmap(explicit.mv)(probe))


# ──────────────────────────────────────────────────────────────────────
# jac="bwd": a reverse-mode-only Gauss–Newton (#162)
# ──────────────────────────────────────────────────────────────────────

N_REVERSE_STATE, N_REVERSE_OBS = 3, 5
FOOTPRINT = np.random.default_rng(0).normal(size=(N_REVERSE_OBS, N_REVERSE_STATE))


@jax.custom_vjp
def _reverse_only(x):
    """h(x) = F exp(x), with only a reverse-mode rule."""
    return jnp.asarray(FOOTPRINT) @ jnp.exp(x)


_reverse_only.defvjp(
    lambda x: (_reverse_only(x), x),
    lambda x, cotangent: ((jnp.asarray(FOOTPRINT).T @ cotangent) * jnp.exp(x),),
)


def _both_modes(x):
    return jnp.asarray(FOOTPRINT) @ jnp.exp(x)


def _static_model(obs_fn, mode, **obs_kwargs):
    def eye(n):
        return lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), jnp.float64))

    return vdx.IncrementalFourDVar(
        forward=_Static(),
        obs_op=vdx.CallableObs(fn=obs_fn, **obs_kwargs),
        prior_mean=jnp.zeros(N_REVERSE_STATE),
        prior_cov_op=eye(N_REVERSE_STATE),
        obs_cov_op=eye(N_REVERSE_OBS),
        config=vdx.IncrementalConfig(
            n_outer=20,
            n_inner=50,
            cg_atol=1e-14,
            cg_rtol=1e-14,
            covariance_tol=1e-14,
            jac=mode,
        ),
    )


def _reverse_batch(scale=1.0):
    obs = scale * (_both_modes(jnp.full(N_REVERSE_STATE, 0.2)) + 0.1)
    return vdx.Batch1D(input=obs[None, None], mask=jnp.ones((1, 1, N_REVERSE_OBS)))


@pytest.mark.slow
def test_bwd_runs_a_custom_vjp_map_to_the_minimum(x64):
    import optimistix as optx

    model = _static_model(_reverse_only, "bwd", jac="bwd", materialise=True)
    batch = _reverse_batch()
    analysis = model(batch)[0]
    obs = batch.input[0, 0]

    def cost(state, _args):
        # ½ ‖x‖² + ½ ‖y − F exp(x)‖²
        return 0.5 * state @ state + 0.5 * jnp.sum((obs - _both_modes(state)) ** 2)

    solver = optx.BFGS(rtol=1e-13, atol=1e-13)
    minimum = optx.minimise(cost, solver, jnp.zeros(N_REVERSE_STATE), max_steps=2000)
    assert _relative_error(analysis, minimum.value) < 1e-8


@pytest.mark.slow
def test_bwd_equals_fwd_on_a_map_with_both_modes(x64):
    batch = _reverse_batch()
    forward_mode = _static_model(_both_modes, "fwd")
    reverse_mode = _static_model(_both_modes, "bwd")
    np.testing.assert_allclose(
        reverse_mode(batch), forward_mode(batch), rtol=1e-10, atol=1e-12
    )
    eye = jnp.eye(N_REVERSE_STATE)
    cov_fwd = jax.vmap(forward_mode.posterior(batch).cov.mv)(eye)
    cov_bwd = jax.vmap(reverse_mode.posterior(batch).cov.mv)(eye)
    np.testing.assert_allclose(cov_bwd, cov_fwd, rtol=1e-10, atol=1e-12)


@pytest.mark.slow
def test_bwd_runs_a_recursive_checkpoint_diffrax_forward(x64):
    import diffrax as dfx

    def decay(t, y, args):
        return -y

    forward = vdx.DynTrajectory(model=decay).as_forward_model(
        dt=0.1, adjoint=dfx.RecursiveCheckpointAdjoint()
    )

    def model(jac):
        eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float64))
        return vdx.IncrementalFourDVar(
            forward=forward,
            obs_op=vdx.MaskedIdentity(),
            prior_mean=jnp.zeros(2),
            prior_cov_op=eye,
            obs_cov_op=eye,
            config=vdx.IncrementalConfig(n_outer=2, n_inner=10, jac=jac),
        )

    batch = vdx.Batch1D(input=jnp.ones((1, 3, 2)), mask=jnp.ones((1, 3, 2)))
    with pytest.raises(TypeError, match="custom_vjp"):
        model("fwd")(batch)
    assert jnp.all(jnp.isfinite(model("bwd")(batch)))


@pytest.mark.slow
def test_bwd_jit_and_vmap_match_a_loop(x64):
    model = _static_model(_reverse_only, "bwd", jac="bwd", materialise=True)
    windows = [_reverse_batch(scale) for scale in (1.0, 0.5, 2.0)]
    stacked = jax.tree.map(lambda *leaves: jnp.concatenate(leaves), *windows)
    batched = eqx.filter_jit(model)(stacked)
    for i, window in enumerate(windows):
        np.testing.assert_allclose(batched[i], model(window)[0], rtol=1e-10)


def test_jac_is_validated_and_serialised():
    config = vdx.IncrementalConfig(jac="bwd")
    assert vdx.IncrementalConfig.from_config(config.get_config()).jac == "bwd"
    with pytest.raises(vdx.ConfigurationError, match="jac"):
        vdx.IncrementalConfig(jac="central")
