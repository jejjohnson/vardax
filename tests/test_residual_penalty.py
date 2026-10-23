"""Tests for vardax.ResidualPenalty."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from vardax import ResidualPenalty
from vardax.examples import advection_diffusion_2d, navier_stokes_vorticity_2d

KEY = jax.random.PRNGKey(0)


def _mlp(key=KEY):
    # A small neural field (t, x, y) -> (1,)
    return eqx.nn.MLP(3, 1, 16, 2, activation=jnp.tanh, key=key)


def _uniform_sampler(n: int):
    return lambda key: jax.random.uniform(key, (n, 3))


# ---------------------------------------------------------------------------
# Closed form and shapes
# ---------------------------------------------------------------------------


def test_closed_form_constant_field():
    # N[f] = f, f ≡ (1, 2): J = λ/(2N) · N · 5 = 5λ/2
    pen = ResidualPenalty(lambda f, s: f(s), jnp.zeros((7, 3)), weight=0.3)
    out = pen(lambda s: jnp.array([1.0, 2.0]))
    assert out.shape == ()
    assert jnp.allclose(out, 0.75)


def test_matches_explicit_sum():
    pts = jax.random.uniform(KEY, (5, 3))
    res = advection_diffusion_2d((1.0, 0.5), 0.1)
    field = _mlp()
    pen = ResidualPenalty(res, pts, weight=2.0)
    point_residual = jax.jit(lambda p: res(field, p))
    r = jnp.stack([point_residual(p) for p in pts])  # (N_c, 1), one point at a time
    assert jnp.allclose(pen(field), 2.0 / (2 * 5) * jnp.sum(r**2), rtol=1e-5)


def test_scalar_residual_is_accepted():
    pen = ResidualPenalty(lambda f, s: f(s)[0], jnp.ones((4, 3)), weight=1.0)
    assert jnp.allclose(pen(lambda s: jnp.array([2.0])), 2.0)


def test_zero_weight_and_key_ignored_without_sampler():
    pen = ResidualPenalty(lambda f, s: f(s), jnp.ones((4, 3)), weight=0.0)
    assert float(pen(lambda s: s, key=KEY)) == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_needs_points_or_sampler():
    with pytest.raises(ValueError, match="collocation points or a sampler"):
        ResidualPenalty(lambda f, s: f(s), None, weight=1.0)


def test_collocation_must_be_2d():
    with pytest.raises(ValueError, match="shape"):
        ResidualPenalty(lambda f, s: f(s), jnp.ones(3), weight=1.0)


def test_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        ResidualPenalty(lambda f, s: f(s), jnp.ones((2, 3)), weight=-1.0)


def test_sampler_needs_key():
    pen = ResidualPenalty(lambda f, s: f(s), None, 1.0, sampler=_uniform_sampler(4))
    with pytest.raises(ValueError, match="key"):
        pen(lambda s: s)


def test_sampler_output_must_be_2d():
    pen = ResidualPenalty(lambda f, s: f(s), None, 1.0, sampler=lambda k: jnp.ones(3))
    with pytest.raises(ValueError, match="shape"):
        pen(lambda s: s, key=KEY)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def test_resamples_per_key():
    res = advection_diffusion_2d((1.0, 0.5), 0.1)
    pen = ResidualPenalty(res, None, 1.0, sampler=_uniform_sampler(32))
    field = _mlp()
    k1, k2 = jax.random.split(KEY)
    assert jnp.allclose(pen(field, key=k1), pen(field, key=k1))
    assert not jnp.allclose(pen(field, key=k1), pen(field, key=k2))


def test_resampled_estimate_is_unbiased():
    """E[J_resampled] = J_dense when points are drawn uniformly from the set.

    Each call draws N_c indices i.i.d. uniform over the M dense points, so
    J_s = λ/(2 N_c) Σⱼ q_{iⱼ} with qᵢ = ‖𝒩[f](cᵢ)‖² has mean λ/2 · mean(q) =
    J_dense and variance (λ/2)² Var(q) / N_c, with Var(q) the population
    variance over the dense set (known exactly). The mean of K independent
    calls then has standard deviation (λ/2) √(Var(q) / (N_c K)); the test
    allows 4 of them (two-sided false-failure rate ≈ 6e-5 by the CLT).
    """
    lam, n_c, n_draws = 0.7, 16, 4000
    dense = jax.random.uniform(KEY, (512, 3))
    res = advection_diffusion_2d((1.0, 0.5), 0.1)
    field = _mlp(jax.random.PRNGKey(1))

    def sampler(key):
        return dense[jax.random.randint(key, (n_c,), 0, dense.shape[0])]

    j_dense = eqx.filter_jit(ResidualPenalty(res, dense, lam))(field)
    pen = ResidualPenalty(res, None, lam, sampler=sampler)
    keys = jax.random.split(jax.random.PRNGKey(2), n_draws)
    draws = jax.jit(jax.vmap(lambda k: pen(field, key=k)))(keys)

    q = jnp.sum(jax.jit(jax.vmap(lambda c: res(field, c)))(dense) ** 2, axis=-1)
    sd = lam / 2 * jnp.sqrt(jnp.var(q) / (n_c * n_draws))
    assert jnp.allclose(j_dense, lam / 2 * jnp.mean(q), rtol=1e-5)
    assert jnp.abs(jnp.mean(draws) - j_dense) < 4 * sd


# ---------------------------------------------------------------------------
# jit and gradients
# ---------------------------------------------------------------------------


def test_jit_fixed_collocation():
    pen = ResidualPenalty(
        advection_diffusion_2d((1.0, 0.5), 0.1), jax.random.uniform(KEY, (16, 3)), 1.0
    )
    field = _mlp()
    eager = pen(field)
    assert jnp.allclose(jax.jit(lambda: pen(field))(), eager, rtol=1e-5)
    # The penalty itself through filter_jit (callables stay static).
    assert jnp.allclose(eqx.filter_jit(lambda p, f: p(f))(pen, field), eager, rtol=1e-5)


def test_jit_resampled_collocation():
    pen = ResidualPenalty(
        advection_diffusion_2d((1.0, 0.5), 0.1), None, 1.0, sampler=_uniform_sampler(16)
    )
    field = _mlp()
    jitted = jax.jit(lambda key: pen(field, key=key))
    k1, k2 = jax.random.split(KEY)
    assert jnp.allclose(jitted(k1), pen(field, key=k1), rtol=1e-5)
    # A new key reuses the compiled function and draws new points.
    assert not jnp.allclose(jitted(k2), jitted(k1))


@pytest.mark.parametrize(
    "residual",
    [
        pytest.param(advection_diffusion_2d((1.0, 0.5), 0.1), id="advection-diffusion"),
        # Fourth derivatives of the field, then a gradient: slow to compile.
        pytest.param(
            navier_stokes_vorticity_2d(0.01), id="navier-stokes", marks=pytest.mark.slow
        ),
    ],
)
@pytest.mark.parametrize("resample", [False, True], ids=["fixed", "resampled"])
def test_gradients_finite(residual, resample):
    sampler = _uniform_sampler(8) if resample else None
    pts = None if resample else jax.random.uniform(KEY, (8, 3))
    pen = ResidualPenalty(residual, pts, 1.0, sampler=sampler)
    loss = eqx.filter_jit(eqx.filter_value_and_grad(lambda f: pen(f, key=KEY)))
    value, grads = loss(_mlp())
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert sum(float(jnp.sum(g**2)) for g in leaves) > 0.0
