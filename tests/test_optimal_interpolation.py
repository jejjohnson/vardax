"""``OptimalInterpolation(form="state")``: the information form (#163).

    observation: x* = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ d     (D_y × D_y solve)
    state:       x* = x_b + (B⁻¹ + Hᵀ R⁻¹ H)⁻¹ Hᵀ R⁻¹ d   (D_x × D_x solve)

The two are equal by Sherman–Morrison–Woodbury. The state form is the one
to solve when D_y ≫ D_x, e.g. a few emission and bias parameters against
thousands of pixels.
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


def _case(n_state, n_obs, dtype=jnp.float64, obs_var=None, seed=0):
    rng = np.random.default_rng(seed)
    H = jnp.asarray(rng.normal(size=(n_obs, n_state)), dtype)
    root = rng.normal(size=(n_state, n_state))
    B = jnp.asarray(root @ root.T + n_state * np.eye(n_state), dtype)
    variances = (
        rng.uniform(0.5, 2.0, size=n_obs)
        if obs_var is None
        else np.full(n_obs, obs_var)
    )
    R = jnp.asarray(variances, dtype)
    y = jnp.asarray(rng.normal(size=n_obs), dtype)

    def model(form, solver=None):
        return vdx.OptimalInterpolation(
            obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(H)),
            prior_mean=jnp.zeros(n_state, dtype),
            prior_cov_op=lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
            obs_cov_op=lx.DiagonalLinearOperator(R),
            form=form,
            solver=solver,
        )

    return H, B, R, y, model


def _relative(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


@pytest.mark.parametrize(("n_state", "n_obs"), [(5, 12), (12, 5)])
def test_both_forms_agree(x64, n_state, n_obs):
    H, B, R, y, model = _case(n_state, n_obs)
    batch = vdx.Batch1D(input=y[None], mask=jnp.ones((1, n_obs)))
    observation = model("observation", lx.Cholesky())(batch)[0]
    state = model("state", lx.Cholesky())(batch)[0]
    # x* = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹ Hᵀ R⁻¹ y, with x_b = 0
    dense = jnp.linalg.solve(jnp.linalg.inv(B) + H.T @ (H / R[:, None]), H.T @ (y / R))
    assert _relative(state, dense) < 1e-10
    assert _relative(observation, state) < 1e-10


def test_fusion_shaped_case(x64):
    """D_x = 3 against 10⁴ pixels with R = 10⁻⁴ I, under the default CG."""
    H, B, R, y, model = _case(3, 10_000, obs_var=1e-4)
    batch = vdx.Batch1D(input=y[None], mask=jnp.ones((1, 10_000)))
    state = model("state")(batch)[0]
    dense = jnp.linalg.solve(jnp.linalg.inv(B) + H.T @ (H / R[:, None]), H.T @ (y / R))
    assert jnp.all(jnp.isfinite(state))
    assert _relative(state, dense) < 1e-8


def test_fusion_shaped_case_float32():
    _, _, _, y, model = _case(3, 10_000, dtype=jnp.float32, obs_var=1e-4)
    batch = vdx.Batch1D(input=y[None], mask=jnp.ones((1, 10_000), jnp.float32))
    assert jnp.all(jnp.isfinite(model("state")(batch)))


def test_masked_observations_agree(x64):
    _, _, _, y, model = _case(4, 9)
    mask = jnp.asarray([1, 0, 1, 1, 0, 1, 1, 1, 0], jnp.float64)[None]
    # NaN in the gaps, which fill_gaps ignores
    gappy = jnp.where(mask != 0, y[None], jnp.nan)
    batch = vdx.Batch1D(input=gappy, mask=mask)
    observation = model("observation", lx.Cholesky())(batch)
    state = model("state", lx.Cholesky())(batch)
    assert _relative(observation, state) < 1e-10


def test_jit_and_vmap_match_a_loop(x64):
    _, _, _, y, model = _case(4, 9)
    oi = model("state")
    rows = jnp.stack([y, 2.0 * y, -y])
    batch = vdx.Batch1D(input=rows, mask=jnp.ones_like(rows))
    batched = eqx.filter_jit(oi)(batch)
    for i in range(3):
        one = vdx.Batch1D(input=rows[i : i + 1], mask=jnp.ones((1, 9)))
        assert jnp.allclose(batched[i], oi(one)[0], rtol=1e-10)


def test_state_form_posterior_is_the_information_inverse(x64):
    H, B, R, y, model = _case(4, 9)
    batch = vdx.Batch1D(input=y[None], mask=jnp.ones((1, 9)))
    posterior = model("state").posterior(batch)
    dense = jax.vmap(posterior.cov.mv)(jnp.eye(4))
    # P* = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹
    expected = jnp.linalg.inv(jnp.linalg.inv(B) + H.T @ (H / R[:, None]))
    assert _relative(dense, expected) < 1e-5


def test_unknown_form_raises():
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
    with pytest.raises(vdx.ConfigurationError, match="form"):
        vdx.OptimalInterpolation(
            obs_op=vdx.MaskedIdentity(),
            prior_mean=jnp.zeros(2),
            prior_cov_op=eye,
            obs_cov_op=eye,
            form="dual",
        )
