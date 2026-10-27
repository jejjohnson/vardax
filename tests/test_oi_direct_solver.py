"""``OptimalInterpolation(solver=...)``: an exact direct solve (#116).

The default CG solve runs at a fixed tolerance with a step cap, so it is
inexact and stalls on an ill-conditioned innovation system. These cases use
a Matérn-3/2 prior, a sparse random ``H`` and diagonal ``R``, where the
second case makes the default CG hit its step cap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from vardax import Batch1D, CallableObs, OptimalInterpolation


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _case(n_grid: int, n_obs: int, r: float):
    x = jnp.linspace(0.0, 1.0, n_grid)
    s = jnp.sqrt(3.0) * jnp.abs(x[:, None] - x[None, :]) / 0.1
    B = (1.0 + s) * jnp.exp(-s)
    keep = jax.random.uniform(jax.random.key(1), (n_obs, n_grid)) < 0.05
    H = jax.random.uniform(jax.random.key(0), (n_obs, n_grid)) * keep
    noise = jax.random.normal(jax.random.key(2), (n_obs,))
    y = H @ (jnp.sin(6.0 * x) + 1.0) + jnp.sqrt(r) * noise
    x_b = jnp.ones(n_grid)
    S = H @ B @ H.T + r * jnp.eye(n_obs)
    gain = B @ H.T @ jnp.linalg.inv(S)
    mean = x_b + gain @ (y - H @ x_b)
    cov = B - gain @ H @ B
    dtype = x.dtype
    model = dict(
        obs_op=CallableObs(fn=lambda z: z @ H.T),
        prior_mean=x_b[None],
        prior_cov_op=lx.FunctionLinearOperator(
            lambda z: z @ B,
            jax.ShapeDtypeStruct((1, n_grid), dtype),
            lx.positive_semidefinite_tag,
        ),
        obs_cov_op=lx.FunctionLinearOperator(
            lambda z: r * z,
            jax.ShapeDtypeStruct((1, n_obs), dtype),
            lx.positive_semidefinite_tag,
        ),
    )
    batch = Batch1D(input=y[None, None], mask=jnp.ones((1, 1, n_obs), dtype))
    return model, batch, mean, cov


@pytest.mark.parametrize(
    ("n_grid", "n_obs", "r"),
    [(100, 40, 1e-2), (400, 150, 1e-3)],
    ids=["small", "stiff"],
)
def test_direct_solver_matches_closed_form(x64, n_grid, n_obs, r):
    model, batch, mean, _ = _case(n_grid, n_obs, r)
    oi = OptimalInterpolation(**model, solver=lx.Cholesky())
    analysis = oi(batch)[0, 0]
    scale = jnp.max(jnp.abs(mean - model["prior_mean"][0]))
    assert jnp.max(jnp.abs(analysis - mean)) / scale < 1e-10


def test_default_cg_stalls_on_the_stiff_case(x64):
    """Pins the motivating failure, so the default's limits stay documented."""
    model, batch, _, _ = _case(400, 150, 1e-3)
    with pytest.raises(Exception, match="maximum number of solver steps"):
        OptimalInterpolation(**model)(batch)


def test_default_solver_is_unchanged(x64):
    model, batch, _, _ = _case(100, 40, 1e-2)
    default = OptimalInterpolation(**model)(batch)[0, 0]
    explicit_cg = OptimalInterpolation(
        **model, solver=lx.CG(atol=1e-6, rtol=1e-6, max_steps=200)
    )(batch)[0, 0]
    assert jnp.array_equal(default, explicit_cg)


def test_posterior_uses_the_model_solver(x64):
    model, batch, _, cov = _case(100, 40, 1e-2)
    oi = OptimalInterpolation(**model, solver=lx.Cholesky())
    posterior = oi.posterior(batch)
    dense = jax.vmap(lambda e: posterior.cov.mv(e[None])[0])(jnp.eye(100))
    assert jnp.max(jnp.abs(dense - cov)) / jnp.max(jnp.abs(cov)) < 1e-8
