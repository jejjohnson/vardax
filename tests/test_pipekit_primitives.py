"""pipekit-cycle's observation and forward primitives work in vardax (#217).

vardax's ``LinearObs`` and ``CallableObs`` are subclasses of pipekit-cycle's, and
the plain primitives run through every model and posterior, with the same answer
as vardax's own operators.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pipekit_cycle as pc
import pytest

import vardax as vdx

N = 4
PSD = lx.positive_semidefinite_tag


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _decay(state, dt):
    return 0.9 * state + 0.05 * jnp.roll(state, 1)


class _Decay(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return _decay(state, dt)


def _primitive(name):
    """A pipekit-cycle operator with its vardax equivalent, both H = I."""
    identity = jnp.eye(N)
    return {
        "identity": pc.IdentityObs(),
        "linear": pc.LinearObs(identity),
        "callable": pc.CallableObs(lambda state: state),
        "composite": pc.CompositeObs((pc.LinearObs(identity), pc.IdentityObs())),
    }[name]


def _common(obs_op):
    return dict(
        obs_op=obs_op,
        prior_mean=jnp.zeros(N),
        prior_cov_op=lx.MatrixLinearOperator(jnp.eye(N) * 2.0, PSD),
        obs_cov_op=lx.MatrixLinearOperator(jnp.eye(N) * 0.5, PSD),
    )


def _tight():
    return dict(
        minimiser=optx.BFGS(rtol=1e-10, atol=1e-10),
        cov_solver=lx.CG(atol=1e-12, rtol=1e-12),
        max_steps=2000,
    )


def _single_time():
    y = jnp.asarray(np.random.default_rng(0).normal(size=(1, N)))
    return vdx.Batch1D(input=y, mask=jnp.ones_like(y))


def _window():
    y = jnp.asarray(np.random.default_rng(1).normal(size=(3, N)))
    mask = jnp.asarray(np.random.default_rng(2).uniform(size=(3, N)) > 0.3, y.dtype)
    return vdx.Batch1D(input=y[None], mask=mask[None])


def _build(model_name, obs_op, forward):
    if model_name == "3dvar":
        return vdx.ThreeDVar(**_common(obs_op), **_tight()), _single_time()
    if model_name == "oi":
        return vdx.OptimalInterpolation(**_common(obs_op)), _single_time()
    if model_name == "strong":
        return vdx.StrongFourDVar(
            forward=forward, **_common(obs_op), **_tight()
        ), _window()
    if model_name == "reduced-basis":
        return (
            vdx.ReducedBasisFourDVar(forward=forward, **_common(obs_op), **_tight()),
            _window(),
        )
    return (
        vdx.IncrementalFourDVar(
            forward=forward,
            **_common(obs_op),
            config=vdx.IncrementalConfig(
                n_outer=2,
                n_inner=50,
                cg_atol=1e-10,
                cg_rtol=1e-10,
                covariance_tol=1e-10,
            ),
        ),
        _window(),
    )


@pytest.mark.slow
@pytest.mark.parametrize("primitive", ["identity", "linear", "callable", "composite"])
@pytest.mark.parametrize(
    "model_name", ["3dvar", "oi", "strong", "reduced-basis", "incremental"]
)
def test_primitives_match_vardax_operators(x64, model_name, primitive):
    forward = pc.CallableForward(_decay, dt=1.0)
    model, batch = _build(model_name, _primitive(primitive), forward)
    reference, _ = _build(model_name, vdx.MaskedIdentity(), _Decay())
    actual = np.asarray(model(batch))
    np.testing.assert_allclose(actual, np.asarray(reference(batch)), atol=1e-6)
    np.testing.assert_allclose(
        model.posterior(batch).cov.as_matrix(),
        reference.posterior(batch).cov.as_matrix(),
        atol=1e-6,
    )


def test_a_primitive_posterior_runs(x64):
    """pipekit's LinearObs linearises to itself, which has no ``mv``."""
    model = vdx.ThreeDVar(**_common(pc.LinearObs(jnp.eye(N))))
    covariance = model.posterior(_single_time()).cov.as_matrix()
    # P = (B⁻¹ + R⁻¹)⁻¹ = (½ + 2)⁻¹ I
    np.testing.assert_allclose(covariance, 0.4 * np.eye(N), atol=1e-5)


def test_vardax_operators_are_pipekit_operators():
    operator = lx.MatrixLinearOperator(jnp.eye(N))
    linear = vdx.LinearObs(operator=operator)
    assert isinstance(linear, pc.LinearObs)
    assert linear.H is operator
    assert isinstance(linear, pc.ObservationOperator)
    assert "H_shape" in linear.get_config()
    callable_obs = vdx.CallableObs(fn=jnp.sin)
    assert isinstance(callable_obs, pc.CallableObs)
    assert callable_obs.name == "sin"
    assert callable_obs.get_config() == {"name": "sin"}


def test_vardax_operators_stay_pytrees_under_every_transform():
    operator = lx.MatrixLinearOperator(2.0 * jnp.eye(N))
    linear = vdx.LinearObs(operator=operator)
    assert len(jax.tree.leaves(linear)) == 1
    state = jnp.ones(N)
    assert jnp.allclose(eqx.filter_jit(linear)(state), 2.0 * state)
    assert jnp.allclose(jax.vmap(linear)(jnp.ones((3, N))), 2.0)
    gradient = jax.grad(lambda x: jnp.sum(linear(x)))(state)
    assert jnp.allclose(gradient, 2.0)
    # the operator is a leaf that can be differentiated
    by_operator = eqx.filter_grad(lambda ob: jnp.sum(ob(state)))(linear)
    assert jnp.allclose(by_operator.operator.matrix, 1.0)


def test_vardax_operators_compose_in_a_pipekit_composite():
    linear = vdx.LinearObs(operator=lx.MatrixLinearOperator(2.0 * jnp.eye(N)))
    composite = pc.CompositeObs((linear, pc.IdentityObs()))
    assert jnp.allclose(composite(jnp.ones(N)), 2.0)
