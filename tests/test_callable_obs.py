"""Tests for ``CallableObs``: any JAX function as an observation operator.

The reference problem is parameter estimation through a linear map
``G(q) = F q`` with more observations than parameters (``M != N``), where
every analysis has the closed-form BLUE as its answer.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from vardax import (
    Batch1D,
    CallableObs,
    LaplaceCovariance,
    ObservationOperator,
    OptimalInterpolation,
    ThreeDVar,
)


@pytest.fixture
def x64():
    """Run a test in float64 and restore the previous setting afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _psd(fn, shape, dtype):
    return lx.FunctionLinearOperator(
        fn, jax.ShapeDtypeStruct(shape, dtype), lx.positive_semidefinite_tag
    )


def _problem(n: int = 6, m: int = 15, r: float = 1e-4, b: float = 0.5):
    """Linear-Gaussian problem with its closed-form posterior."""
    footprint = jax.random.uniform(jax.random.key(0), (m, n))
    q_true = jnp.linspace(0.5, 1.5, n)
    noise = jax.random.normal(jax.random.key(1), (m,))
    y = footprint @ q_true + jnp.sqrt(r) * noise
    q_b = jnp.ones(n)
    prior = b * jnp.eye(n)
    precision = jnp.linalg.inv(prior) + footprint.T @ footprint / r
    cov = jnp.linalg.inv(precision)
    mean = q_b + cov @ footprint.T @ (y - footprint @ q_b) / r
    dtype = footprint.dtype
    return {
        "obs_op": CallableObs(fn=lambda q: q @ footprint.T),
        "prior_mean": q_b[None],
        "prior_cov_op": _psd(lambda x: x @ prior, (1, n), dtype),
        "obs_cov_op": _psd(lambda x: r * x, (1, m), dtype),
        "batch": Batch1D(input=y[None, None], mask=jnp.ones((1, 1, m), dtype)),
        "mean": mean,
        "cov": cov,
    }


class TestContract:
    def test_satisfies_protocol(self):
        assert isinstance(CallableObs(fn=jnp.sin), ObservationOperator)

    def test_call_shape_differs_from_state(self):
        op = CallableObs(fn=lambda x: jnp.concatenate([x, x**2], axis=-1))
        assert op(jnp.ones((1, 3))).shape == (1, 6)

    @pytest.mark.parametrize("fn", [jnp.exp, lambda x: jnp.tanh(x) * x[::-1]])
    def test_linearize_is_jvp_and_vjp(self, x64, fn):
        x = jnp.linspace(-1.0, 1.0, 5)
        v = jnp.linspace(0.3, 1.1, 5)
        lin = CallableObs(fn=fn).linearize(x)
        _, jvp = jax.jvp(fn, (x,), (v,))
        _, vjp_fn = jax.vjp(fn, x)
        assert jnp.allclose(lin.mv(v), jvp, rtol=1e-12)
        assert jnp.allclose(lin.transpose().mv(v), vjp_fn(v)[0], rtol=1e-12)


class TestAgainstClosedForm:
    def test_optimal_interpolation_mean(self, x64):
        p = _problem()
        oi = OptimalInterpolation(
            obs_op=p["obs_op"],
            prior_mean=p["prior_mean"],
            prior_cov_op=p["prior_cov_op"],
            obs_cov_op=p["obs_cov_op"],
        )
        assert jnp.allclose(oi(p["batch"])[0, 0], p["mean"], rtol=1e-8)

    def test_threedvar_mean(self, x64):
        p = _problem()
        three = ThreeDVar(
            obs_op=p["obs_op"],
            prior_mean=p["prior_mean"],
            prior_cov_op=p["prior_cov_op"],
            obs_cov_op=p["obs_cov_op"],
        )
        assert jnp.allclose(three(p["batch"])[0, 0], p["mean"], rtol=1e-8)

    def test_laplace_covariance(self, x64):
        p = _problem()
        three = ThreeDVar(
            obs_op=p["obs_op"],
            prior_mean=p["prior_mean"],
            prior_cov_op=p["prior_cov_op"],
            obs_cov_op=p["obs_cov_op"],
        )
        analysis = three(p["batch"])[0]
        posterior = LaplaceCovariance(
            prior_cov_op=p["prior_cov_op"], obs_cov_op=p["obs_cov_op"]
        )(analysis, three, p["batch"])
        n = p["mean"].shape[0]
        cov = jax.vmap(lambda e: posterior.cov.mv(e[None])[0])(jnp.eye(n))
        scale = jnp.max(jnp.abs(p["cov"]))
        assert jnp.max(jnp.abs(cov - p["cov"])) / scale < 1e-8


class TestTransforms:
    def test_filter_jit(self):
        op = CallableObs(fn=jnp.sin)
        x = jnp.linspace(0.0, 1.0, 4)
        assert jnp.allclose(eqx.filter_jit(lambda o, z: o(z))(op, x), jnp.sin(x))

    def test_vmap_over_states(self):
        op = CallableObs(fn=lambda x: 2.0 * x)
        xs = jnp.ones((3, 4))
        assert jax.vmap(op)(xs).shape == (3, 4)

    def test_filter_grad_reaches_module_parameters(self):
        emulator = eqx.nn.Linear(3, 2, key=jax.random.key(0))
        op = CallableObs(fn=emulator)
        x = jnp.ones(3)

        def loss(o):
            return jnp.sum(o(x) ** 2)

        grads = eqx.filter_grad(loss)(op)
        assert grads.fn.weight.shape == (2, 3)
        assert jnp.any(grads.fn.weight != 0.0)


class TestReverseModeOnly:
    """``jac="bwd"`` linearises a model that defines only a VJP (#115)."""

    @staticmethod
    def _vjp_only(matrix):
        @jax.custom_vjp
        def f(x):
            return jnp.tanh(x @ matrix.T)

        def fwd(x):
            y = f(x)
            return y, (x, y)

        def bwd(res, g):
            _, y = res
            return (((1.0 - y**2) * g) @ matrix,)

        f.defvjp(fwd, bwd)
        return f

    def test_forward_mode_fails_and_bwd_matches_jacrev(self, x64):
        matrix = jax.random.normal(jax.random.key(0), (4, 3))
        f = self._vjp_only(matrix)
        x = jnp.linspace(-0.5, 0.5, 3)
        v = jnp.asarray([0.3, -1.0, 0.7])
        with pytest.raises(TypeError, match="custom_vjp"):
            CallableObs(fn=f).linearize(x).mv(v)
        lin = CallableObs(fn=f, jac="bwd").linearize(x)
        assert jnp.allclose(lin.mv(v), jax.jacrev(f)(x) @ v, rtol=1e-12)

    def test_posterior_with_vjp_only_model(self, x64):
        p = _problem()
        footprint = p["obs_op"].fn(jnp.eye(6)).T  # recover F (M, N)

        @jax.custom_vjp
        def g(q):
            return q @ footprint.T

        g.defvjp(lambda q: (g(q), None), lambda _, ct: (ct @ footprint,))
        three = ThreeDVar(
            obs_op=CallableObs(fn=g, jac="bwd"),
            prior_mean=p["prior_mean"],
            prior_cov_op=p["prior_cov_op"],
            obs_cov_op=p["obs_cov_op"],
        )
        analysis = three(p["batch"])[0]
        posterior = LaplaceCovariance(
            prior_cov_op=p["prior_cov_op"], obs_cov_op=p["obs_cov_op"]
        )(analysis, three, p["batch"])
        cov = jax.vmap(lambda e: posterior.cov.mv(e[None])[0])(jnp.eye(6))
        assert jnp.max(jnp.abs(cov - p["cov"])) / jnp.max(jnp.abs(p["cov"])) < 1e-8
        assert jnp.allclose(analysis[0], p["mean"], rtol=1e-8)
