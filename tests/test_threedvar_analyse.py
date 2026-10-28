"""``ThreeDVar.analyse``: the analysis plus the minimiser's diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import pytest

import vardax as vdx


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _model(**extra):
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float64))
    return vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros((1, 3)),
        prior_cov_op=eye,
        obs_cov_op=eye,
        **extra,
    )


def _batch(n=2):
    y = jnp.stack([jnp.ones((1, 3)), 2.0 * jnp.ones((1, 3))])[:n]
    return vdx.Batch1D(input=y, mask=jnp.ones_like(y))


@pytest.mark.parametrize(
    "minimiser",
    [
        pytest.param(None, id="optx-default"),
        pytest.param(vdx.OptimistixMinimiser(optx.LBFGS(1e-8, 1e-8)), id="wrapped"),
    ],
)
def test_analysis_matches_call_and_reports(x64, minimiser):
    model = _model(minimiser=minimiser) if minimiser else _model()
    batch = _batch()
    analysis, result = model.analyse(batch)
    assert jnp.array_equal(analysis, model(batch))
    # B = R = I: x* = y / 2 and J(x*) = ¼ ‖y‖²
    assert jnp.allclose(analysis, batch.input / 2, atol=1e-6)
    expected = 0.25 * jnp.sum(batch.input**2, axis=(1, 2))
    assert jnp.allclose(result.cost, expected, rtol=1e-6)
    assert result.num_steps.shape == (2,)
    assert bool(jnp.all(result.num_steps > 0))
    assert bool(jnp.all(result.grad_norm < 1e-5))


def test_step_cap_is_visible(x64):
    minimiser = vdx.OptimistixMinimiser(optx.LBFGS(1e-12, 1e-12), max_steps=1)
    _, result = _model(minimiser=minimiser).analyse(_batch(1))
    assert int(result.num_steps[0]) == 1


def test_parameterized_value_is_the_control(x64):
    basis = vdx.linear_basis(jnp.eye(3)[:, :2], variance=1.0)
    param = vdx.LinearParameterization(basis, background=jnp.zeros((1, 3)))
    analysis, result = _model(parameterization=param).analyse(_batch(1))
    assert result.value.shape == (1, 2)
    assert jnp.allclose(analysis[0, 0, :2], result.value[0], atol=1e-6)
    assert jnp.allclose(analysis[0, 0, 2], 0.0)


def test_point_batch_raises(x64):
    batch = vdx.PointObsBatch(
        coords=jnp.zeros((2, 1)),
        values=jnp.ones(2),
        sigma=jnp.ones(2),
        valid=jnp.ones(2, bool),
    )
    with pytest.raises(TypeError, match="takes a Batch1D"):
        _model().analyse(batch)
