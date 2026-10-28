"""``cov_solver=``: the covariance solves of the variational models (#117).

The cost applies ``B⁻¹``, ``R⁻¹`` (and ``Q⁻¹`` in weak 4D-Var) through a
``lineax`` solve. The default CG at ``atol = rtol = 1e-6`` returns NaN on a
diagonal ``R`` of 1e-14, the scale of a concentration-noise variance in
kg² m⁻⁶; a direct solver recovers the closed-form BLUE.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import vardax
from vardax import Batch1D, CallableObs, ThreeDVar

AUTO = lx.AutoLinearSolver(well_posed=True)


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _psd(fn, shape):
    return lx.FunctionLinearOperator(
        fn, jax.ShapeDtypeStruct(shape, jnp.float64), lx.positive_semidefinite_tag
    )


def _tiny_noise(r: float, n: int = 6, m: int = 15):
    """``y = F q + ε`` with ``F ~ 1e-6`` and ``ε ~ 𝒩(0, r I)``, plus its BLUE."""
    F = 1e-6 * jax.random.uniform(jax.random.key(0), (m, n))
    noise = jax.random.normal(jax.random.key(1), (m,))
    y = F @ jnp.linspace(0.5, 1.5, n) + jnp.sqrt(r) * noise
    q_b, P = jnp.ones(n), 0.5 * jnp.eye(n)
    cov = jnp.linalg.inv(jnp.linalg.inv(P) + F.T @ F / r)
    mean = q_b + cov @ F.T @ (y - F @ q_b) / r
    kwargs = dict(
        obs_op=CallableObs(fn=lambda q: q @ F.T),
        prior_mean=q_b[None],
        prior_cov_op=_psd(lambda x: x @ P, (1, n)),
        obs_cov_op=_psd(lambda x: r * x, (1, m)),
    )
    batch = Batch1D(input=y[None, None], mask=jnp.ones((1, 1, m)))
    return kwargs, batch, mean, q_b


def test_default_cg_fails_on_tiny_noise(x64):
    kwargs, batch, _, _ = _tiny_noise(1e-14)
    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        ThreeDVar(**kwargs)(batch)


@pytest.mark.slow
@pytest.mark.parametrize("r", [1e-12, 1e-14])
def test_direct_cov_solver_matches_blue(x64, r):
    kwargs, batch, mean, q_b = _tiny_noise(r)
    analysis = ThreeDVar(**kwargs, cov_solver=AUTO)(batch)[0, 0]
    rel = jnp.max(jnp.abs(analysis - mean)) / jnp.max(jnp.abs(mean - q_b))
    assert rel < 1e-7


@pytest.mark.slow
def test_default_is_unchanged(x64):
    kwargs, batch, _, _ = _tiny_noise(1e-12)
    default = ThreeDVar(**kwargs)(batch)
    explicit = ThreeDVar(**kwargs, cov_solver=lx.CG(atol=1e-6, rtol=1e-6))(batch)
    assert jnp.array_equal(default, explicit)


class _Identity:
    dt = 1.0

    def step(self, x, dt):
        return x


@pytest.mark.parametrize(
    "cls", [vardax.StrongFourDVar, vardax.WeakFourDVar, vardax.ReducedBasisFourDVar]
)
def test_fourdvar_models_accept_cov_solver(x64, cls):
    """Single-time identity dynamics reduce each to 3D-Var: x₀* = y / 2."""
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float64))
    extra = {"model_err_cov_op": eye} if cls is vardax.WeakFourDVar else {}
    model = cls(
        forward=_Identity(),
        obs_op=vardax.MaskedIdentity(),
        prior_mean=jnp.zeros(3),
        prior_cov_op=eye,
        obs_cov_op=eye,
        cov_solver=AUTO,
        **extra,
    )
    assert model.cov_solver is AUTO
    batch = Batch1D(input=jnp.ones((1, 1, 3)), mask=jnp.ones((1, 1, 3)))
    out = model(batch)
    x0 = out[0] if isinstance(out, tuple) else out
    assert jnp.allclose(x0, 0.5, atol=1e-5)
