"""Tests for vardax.SpaceTimeFieldVar (neural-field 4D-Var, issue #87)."""

from __future__ import annotations

import equinox as eqx
import geonnax
from geonnax import diffops
import jax
import jax.numpy as jnp
import lineax as lx
import optax
import pytest

import vardax
from vardax import (
    AnalysisStep,
    FieldPointObs,
    OptaxMinimiser,
    OptimalInterpolation,
    OptimistixMinimiser,
    PointObsBatch,
    ResidualPenalty,
    SpaceTimeFieldResult,
    SpaceTimeFieldVar,
    point_obs_cost,
)

KEY = jax.random.PRNGKey(0)


class FeatureField(eqx.Module):
    """$f_w(t, x) = \\phi(t, x)^\\top w$ with fixed Gaussian RBF features."""

    w: jax.Array
    centres: jax.Array  # frozen: excluded from the control by filter_spec
    width: float = eqx.field(static=True, default=0.5)

    def features(self, s):
        # φ_m(s) = exp(-‖s - c_m‖² / (2 ℓ²)): (M,)
        return jnp.exp(-jnp.sum((s - self.centres) ** 2, -1) / (2 * self.width**2))

    def __call__(self, s):
        return jnp.dot(self.features(s), self.w)[None]


def _feature_field(n_centres: int = 6) -> FeatureField:
    t = jnp.linspace(0.0, 1.0, n_centres)
    x = jnp.linspace(-1.0, 1.0, n_centres)
    return FeatureField(w=jnp.zeros(n_centres), centres=jnp.stack([t, x], -1))


def _only_w(field: FeatureField):
    return FeatureField(w=True, centres=False, width=field.width)  # ty:ignore[invalid-argument-type]


def _obs(n: int = 20, n_pad: int = 0, key=KEY, sigma: float = 0.1) -> PointObsBatch:
    k1, k2, k3 = jax.random.split(key, 3)
    coords = jnp.stack(
        [jax.random.uniform(k1, (n,)), jax.random.uniform(k2, (n,), minval=-1)], -1
    )
    values = jnp.sin(2 * coords[:, 1]) + sigma * jax.random.normal(k3, (n,))
    valid = jnp.ones(n, bool)
    sig = jnp.full(n, sigma)
    if n_pad:
        # Padding rows hold NaN everywhere: they must change nothing.
        nan = jnp.full(n_pad, jnp.nan)
        coords = jnp.concatenate([coords, jnp.stack([nan, nan], -1)])
        values = jnp.concatenate([values, nan])
        sig = jnp.concatenate([sig, nan])
        valid = jnp.concatenate([valid, jnp.zeros(n_pad, bool)])
    return PointObsBatch(coords=coords, values=values, sigma=sig, valid=valid)


def _linear_model(alpha: float = 1.0, residual=None) -> SpaceTimeFieldVar:
    field = _feature_field()
    return SpaceTimeFieldVar(
        field=field,
        residual=residual,
        weight_decay=alpha,
        minimiser=OptimistixMinimiser(),
        filter_spec=_only_w(field),
    )


def _siren(key, hidden: int = 16, depth: int = 2):
    return geonnax.SIREN.init(2, hidden, 1, depth=depth, key=key, first_omega=5.0)


# ---------------------------------------------------------------------------
# Linear-Gaussian sanity: λ = 0 and a linear-in-features field is OI
# ---------------------------------------------------------------------------


class _FeatureObs(eqx.Module):
    """$H w = \\Phi w$, $\\Phi_{km} = \\phi_m(s_k)$, on a ``(1, M)`` state."""

    Phi: jax.Array

    def __call__(self, x):
        return (self.Phi @ x[0])[None]


def test_linear_gaussian_matches_oi():
    # With λ = 0 the cost is ½‖y - Φw‖²_{R⁻¹} + alpha/2 ‖w‖²: the 3D-Var cost
    # with B = alpha⁻¹ I, x_b = 0 and H = Φ, whose minimiser is the OI / BLUE
    # analysis. Both are solved to ~1e-6 relative (BFGS rtol, CG tol) in
    # float32 on a problem with condition number < 1e3, hence atol 1e-3.
    alpha, sigma = 2.0, 0.1
    obs = _obs(sigma=sigma)
    model = _linear_model(alpha)
    result = model(obs)

    field = model.field
    Phi = jax.vmap(field.features)(obs.coords)  # (N_obs, M)
    m = Phi.shape[1]
    oi = OptimalInterpolation(
        obs_op=_FeatureObs(Phi),
        prior_mean=jnp.zeros((1, m)),
        prior_cov_op=lx.DiagonalLinearOperator(jnp.full((1, m), 1.0 / alpha)),
        obs_cov_op=lx.DiagonalLinearOperator(
            jnp.full((1, obs.values.shape[0]), sigma**2)
        ),
    )
    w_oi = oi(vardax.Batch1D(input=obs.values[None, None], mask=jnp.ones((1, 1, 20))))
    assert jnp.allclose(result.field.w, w_oi[0, 0], atol=1e-3)
    # The frozen centres are untouched.
    assert jnp.array_equal(result.field.centres, field.centres)


def test_padding_rows_change_nothing():
    # NaN padding must not reach the cost, the fit or the gradient.
    plain = _linear_model()(_obs())
    padded = _linear_model()(_obs(n_pad=5))
    assert jnp.allclose(padded.field.w, plain.field.w, atol=1e-6)
    assert jnp.allclose(padded.cost, plain.cost, rtol=1e-6)
    assert bool(jnp.isfinite(padded.grad_norm))


def test_cost_is_differentiable_in_the_observations():
    # Envelope theorem: dJ*/dy_k = ∂J/∂y_k at θ* = m_k (y_k - f*(s_k)) / sigma_k²,
    # zero on padding rows. BFGS (rtol = atol = 1e-6) in float32 leaves
    # f*(s_k) accurate to ~1e-5; the 1/σ² = 100 factor makes that ~1e-3 on
    # gradients of size ~40, so atol 1e-2 (0.03 % of the largest entry).
    obs = _obs(n_pad=3)
    model = _linear_model()

    def cost(values):
        return model(eqx.tree_at(lambda o: o.values, obs, values)).cost

    g = jax.grad(cost)(jnp.nan_to_num(obs.values))
    fitted = model(obs)
    pred = fitted.evaluate(jnp.nan_to_num(obs.coords))
    expected = jnp.where(obs.valid, (obs.values - pred) / obs.sigma**2, 0.0)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert jnp.allclose(g, expected, atol=1e-2)


# ---------------------------------------------------------------------------
# Cost assembly, residual, keys, jit
# ---------------------------------------------------------------------------


def _sampler(key):
    k1, k2 = jax.random.split(key)
    return jnp.stack(
        [jax.random.uniform(k1, (32,)), jax.random.uniform(k2, (32,), minval=-1)], -1
    )


def _advection_1d(c: float = 0.5):
    # N[u] = ∂ₜu + c ∂ₓu at s = (t, x)
    def residual(field, s):
        du = diffops.grad(lambda z: field(z)[0])(s)
        return (du[0] + c * du[1])[None]

    return residual


def test_result_cost_is_the_full_objective_with_drawn_collocation():
    # result.cost must equal J(θ*) assembled by hand from the merged pieces,
    # with the collocation points the sampler draws from the key.
    obs = _obs(n_pad=2)
    pen = ResidualPenalty(_advection_1d(), None, 0.5, sampler=_sampler)
    model = _linear_model(alpha=0.3, residual=pen)
    key = jax.random.PRNGKey(3)
    result = model(obs, key=key)

    fixed = ResidualPenalty(_advection_1d(), _sampler(key), 0.5)
    w = result.field.w
    expected = (
        point_obs_cost(FieldPointObs()(result.field, obs), obs)
        + fixed(result.field)
        + 0.5 * 0.3 * jnp.sum(w**2)
    )
    assert jnp.allclose(result.cost, expected, rtol=1e-6)
    # The residual changes the analysis (the physics term is not ignored).
    assert not jnp.allclose(w, _linear_model(alpha=0.3)(obs).field.w, atol=1e-3)
    # Same key, same collocation, same fit; another key, another fit.
    again = model(obs, key=key)
    other = model(obs, key=jax.random.PRNGKey(4))
    assert jnp.array_equal(again.field.w, w)
    assert not jnp.allclose(other.field.w, w, atol=1e-4)


def test_fixed_collocation_needs_no_key():
    pen = ResidualPenalty(_advection_1d(), _sampler(KEY), 0.5)
    result = _linear_model(residual=pen)(_obs())
    assert result.cost.shape == ()


def test_jit_does_not_retrace_on_new_observations():
    traces = []
    model = _linear_model()

    @eqx.filter_jit
    def fit(o):
        traces.append(None)
        return model(o)

    a = fit(_obs(key=jax.random.PRNGKey(1)))
    b = fit(_obs(key=jax.random.PRNGKey(2)))
    assert len(traces) == 1
    assert jnp.allclose(a.field.w, model(_obs(key=jax.random.PRNGKey(1))).field.w)
    assert not jnp.allclose(a.field.w, b.field.w)


def test_optax_smoke_with_siren():
    # One fast end-to-end fit of a neural field with the default kind of
    # minimiser: shapes, finite diagnostics and a decreasing cost.
    obs = _obs()
    pen = ResidualPenalty(_advection_1d(), None, 1.0, sampler=_sampler)
    model = SpaceTimeFieldVar(
        field=_siren(KEY),
        residual=pen,
        weight_decay=1e-6,
        minimiser=OptaxMinimiser(optax.adam(1e-2), max_steps=50),
    )
    result = eqx.filter_jit(lambda o, k: model(o, key=k))(obs, KEY)
    assert isinstance(result, SpaceTimeFieldResult)
    assert result.initial_condition(jnp.zeros((7, 1))).shape == (7,)
    assert result.evaluate(obs.coords).shape == (20,)
    assert int(result.num_steps) <= 50
    assert bool(jnp.isfinite(result.grad_norm))
    start = SpaceTimeFieldVar(
        field=_siren(KEY),
        residual=pen,
        weight_decay=1e-6,
        minimiser=OptaxMinimiser(optax.adam(1e-2), max_steps=1),
    )(obs, key=KEY)
    assert float(result.cost) < float(start.cost)


def test_initial_condition_uses_t0():
    model = SpaceTimeFieldVar(
        field=_feature_field(),
        minimiser=OptimistixMinimiser(),
        filter_spec=_only_w(_feature_field()),
        t0=0.25,
    )
    result = model(_obs())
    grid = jnp.linspace(-1, 1, 5)[:, None]
    direct = result.evaluate(jnp.concatenate([jnp.full((5, 1), 0.25), grid], -1))
    assert jnp.allclose(result.initial_condition(grid), direct)
    # An integer grid must not truncate t0 = 0.25 to 0.
    ints = jnp.arange(-1, 2)[:, None]
    direct = result.evaluate(jnp.stack([jnp.full(3, 0.25), jnp.arange(-1.0, 2.0)], -1))
    assert jnp.allclose(result.initial_condition(ints), direct)


# ---------------------------------------------------------------------------
# Ensemble and AnalysisStep
# ---------------------------------------------------------------------------


def test_ensemble_smoke():
    model = SpaceTimeFieldVar(
        field=_siren(KEY),
        minimiser=OptaxMinimiser(optax.adam(1e-2), max_steps=5),
    )
    ens = model.ensemble(_obs(), _siren, 3, key=KEY)
    assert ens.cost.shape == (3,)
    grid = jnp.zeros((4, 1))
    u0 = eqx.filter_vmap(lambda r: r.initial_condition(grid))(ens)
    assert u0.shape == (3, 4)
    # Different initialisations give different members.
    assert float(jnp.std(u0[:, 0])) > 0


def test_analysis_step():
    model = _linear_model()
    grid = jnp.linspace(-1, 1, 6)[:, None]
    step = model.as_analysis_step(grid)
    assert isinstance(step, AnalysisStep)
    obs = _obs()
    xa = step(jnp.zeros((2, 3)), obs, obs_op=None, obs_err_cov=None)
    expected = model(obs).initial_condition(grid)
    assert xa.shape == (2, 3)
    assert jnp.allclose(xa.reshape(-1), expected)
    assert step(None, obs, obs_op=FieldPointObs(), obs_err_cov=None).shape == (6,)
    with pytest.raises(ValueError, match="obs_err_cov"):
        step(None, obs, obs_op=None, obs_err_cov=jnp.ones(20))
    with pytest.raises(TypeError, match="FieldPointObs"):
        step(None, obs, obs_op=vardax.MaskedIdentity(), obs_err_cov=None)
    with pytest.raises(ValueError, match="analysis on the grid"):
        step(jnp.zeros(5), obs, obs_op=None, obs_err_cov=None)
    # Without a grid (as VarDACycle builds it) the step returns the field.
    bare = model.as_analysis_step()
    assert isinstance(bare, AnalysisStep)
    out = bare(None, obs, obs_op=None, obs_err_cov=None)
    assert isinstance(out, SpaceTimeFieldResult)
    assert jnp.allclose(out.initial_condition(grid), expected)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation():
    field = _feature_field()
    with pytest.raises(ValueError, match="weight_decay"):
        SpaceTimeFieldVar(field, weight_decay=-1.0)
    with pytest.raises(TypeError, match="ResidualPenalty"):
        SpaceTimeFieldVar(field, residual=lambda f: 0.0)  # ty:ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="minimiser"):
        SpaceTimeFieldVar(field, minimiser=optax.adam(1e-3))  # ty:ignore[invalid-argument-type]
    model = _linear_model()
    with pytest.raises(TypeError, match="PointObsBatch"):
        model(jnp.zeros(3))  # ty:ignore[invalid-argument-type]
    batched = jax.tree.map(lambda a: jnp.stack([a, a]), _obs())
    with pytest.raises(ValueError, match="one sample"):
        model(batched)
    pen = ResidualPenalty(_advection_1d(), None, 1.0, sampler=_sampler)
    with pytest.raises(ValueError, match="key"):
        _linear_model(residual=pen)(_obs())
    with pytest.raises(ValueError, match="n_members"):
        model.ensemble(_obs(), lambda k: field, 0, key=KEY)
    with pytest.raises(ValueError, match="grid"):
        model.as_analysis_step(jnp.zeros(3))
    with pytest.raises(ValueError, match="grid"):
        model(_obs()).initial_condition(jnp.zeros(3))


def test_default_minimiser_is_optax():
    assert isinstance(SpaceTimeFieldVar(_feature_field()).minimiser, OptaxMinimiser)


# ---------------------------------------------------------------------------
# Slow: recovery and ensemble spread
# ---------------------------------------------------------------------------

C, X0, WIDTH, SIGMA = 0.5, -0.4, 0.15, 0.05


def _bump(t, x):
    # The true solution of ∂ₜu + c ∂ₓu = 0: a Gaussian bump moving at speed c.
    return jnp.exp(-((x - C * t - X0) ** 2) / (2 * WIDTH**2))


def _box_sampler(key):
    k1, k2 = jax.random.split(key)
    return jnp.stack(
        [jax.random.uniform(k1, (512,)), jax.random.uniform(k2, (512,), minval=-1)], -1
    )


def _advection_fit(seed: int, weight: float) -> jax.Array:
    """Relative L2 error of u(0, ·) recovered from observations at t >= 0.3."""
    k_obs, k_x, k_noise, k_init, k_col = jax.random.split(jax.random.PRNGKey(seed), 5)
    n, steps = 100, 4000
    t = jax.random.uniform(k_obs, (n,), minval=0.3, maxval=1.0)
    x = jax.random.uniform(k_x, (n,), minval=-1.0, maxval=1.0)
    obs = PointObsBatch(
        coords=jnp.stack([t, x], -1),
        values=_bump(t, x) + SIGMA * jax.random.normal(k_noise, (n,)),
        sigma=jnp.full(n, SIGMA),
        valid=jnp.ones(n, bool),
    )
    residual = (
        ResidualPenalty(_advection_1d(C), None, weight, sampler=_box_sampler)
        if weight > 0
        else None
    )
    model = SpaceTimeFieldVar(
        field=geonnax.SIREN.init(2, 32, 1, depth=3, key=k_init, first_omega=5.0),
        residual=residual,
        weight_decay=1e-6,
        minimiser=OptaxMinimiser(
            optax.adam(optax.cosine_decay_schedule(3e-3, steps, 1e-3)),
            max_steps=steps,
            patience=None,
        ),
    )
    result = eqx.filter_jit(lambda o, k: model(o, key=k))(obs, k_col)
    # Characteristics through the observed region reach x <= 0.85 at t = 0;
    # beyond that u(0, ·) is constrained by the residual alone.
    grid = jnp.linspace(-1.0, 0.8, 181)[:, None]
    u0 = result.initial_condition(grid)
    truth = _bump(0.0, grid[:, 0])
    return jnp.linalg.norm(u0 - truth) / jnp.linalg.norm(truth)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_advection_recovers_initial_bump(seed):
    # No observation is earlier than t = 0.3, so u(0, ·) comes from the
    # physics carrying the later observations back along characteristics.
    # Calibration (seeds 0-5, 100 obs, sigma = 0.05, λ = 1e3): relative L2
    # errors 0.076, 0.068, 0.105, 0.101, 0.074, 0.158 (max 0.158); the bound
    # 0.3 leaves ~1.9x headroom over the worst of six seeds. Without the
    # residual (λ = 0) the errors are 2.25, 1.74, 2.72, 1.70, 1.04, 1.88:
    # nothing tells the field how to extrapolate back in time.
    err = _advection_fit(seed, weight=1e3)
    assert float(err) < 0.3
    assert float(err) < 0.5 * float(_advection_fit(seed, weight=0.0))


def _ensemble_spread(seed: int) -> tuple[float, float]:
    """Mean member std of u(0, ·) near (|x| <= 0.3) and far (|x| >= 0.8) from data."""
    k_t, k_x, k_noise, k_ens = jax.random.split(jax.random.PRNGKey(seed), 4)
    n = 30
    t = jax.random.uniform(k_t, (n,))
    x = jax.random.uniform(k_x, (n,), minval=-0.4, maxval=0.4)
    obs = PointObsBatch(
        coords=jnp.stack([t, x], -1),
        values=jnp.sin(3 * x) + SIGMA * jax.random.normal(k_noise, (n,)),
        sigma=jnp.full(n, SIGMA),
        valid=jnp.ones(n, bool),
    )
    model = SpaceTimeFieldVar(
        field=_siren(KEY),
        weight_decay=1e-4,
        minimiser=OptaxMinimiser(optax.adam(3e-3), max_steps=1000),
    )
    ens = eqx.filter_jit(lambda o, k: model.ensemble(o, _siren, 8, key=k))(obs, k_ens)
    near = jnp.linspace(-0.3, 0.3, 31)[:, None]
    far = jnp.concatenate([jnp.linspace(-1, -0.8, 11), jnp.linspace(0.8, 1, 11)])
    u_near = eqx.filter_vmap(lambda r: r.initial_condition(near))(ens)
    u_far = eqx.filter_vmap(lambda r: r.initial_condition(far[:, None]))(ens)
    s_near = jnp.mean(jnp.std(u_near, axis=0))
    s_far = jnp.mean(jnp.std(u_far, axis=0))
    return float(s_near), float(s_far)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ensemble_spread_grows_away_from_observations(seed):
    # Observations only in |x| < 0.4; members agree there (spread within
    # the noise level) and diverge where the field is unconstrained.
    # Calibration (seeds 0-5, 8 members): near spread 0.018-0.039, far
    # spread 0.16-0.47, far/near ratio 8.6-15.8. Bounds: near < 2 sigma = 0.1
    # (2.6x the largest of six seeds), ratio > 4 (2.2x below the smallest).
    s_near, s_far = _ensemble_spread(seed)
    assert s_near < 2 * SIGMA
    assert s_far > 4 * s_near
