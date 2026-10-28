"""``MinimiseResult.converged``: the minimiser's own termination status (#127).

Inferring it from ``num_steps < max_steps`` is wrong both ways: a failed
solve that stops early looks converged, and one that meets its tolerance on
the last allowed step looks unconverged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import optax
import optimistix as optx
import pytest

import vardax as vdx
from vardax._src.minimisers import minimise_with_result


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


M = jnp.asarray([1.0, -2.0, 0.5])


def _quadratic(x, m):
    return 0.5 * jnp.sum((x - m) ** 2)


def _rosenbrock(x, _):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


# ── optimistix ──────────────────────────────────────────────────────────


def test_optimistix_converged(x64):
    result = vdx.OptimistixMinimiser().run(_quadratic, jnp.zeros(3), M)
    assert bool(result.converged)


def test_optimistix_step_cap_is_not_converged(x64):
    minimiser = vdx.OptimistixMinimiser(optx.BFGS(1e-12, 1e-12), max_steps=2)
    result = minimiser.run(_rosenbrock, jnp.full(4, -1.2), None)
    assert int(result.num_steps) == 2
    assert not bool(result.converged)


def test_optimistix_success_on_the_last_allowed_step(x64):
    """The count-based rule would call this unconverged."""
    free = vdx.OptimistixMinimiser(optx.BFGS(1e-8, 1e-8)).run(
        _quadratic, jnp.zeros(3), M
    )
    steps = int(free.num_steps)
    capped = vdx.OptimistixMinimiser(optx.BFGS(1e-8, 1e-8), max_steps=steps).run(
        _quadratic, jnp.zeros(3), M
    )
    assert int(capped.num_steps) == steps
    assert bool(capped.converged)


def test_raw_optimistix_solver_through_minimise_with_result(x64):
    result = minimise_with_result(
        optx.BFGS(1e-8, 1e-8),
        lambda x, _: _quadratic(x, M),
        jnp.zeros(3),
        adjoint=optx.ImplicitAdjoint(),
        max_steps=200,
    )
    assert bool(result.converged)


# ── optax ───────────────────────────────────────────────────────────────


def test_optax_gradient_stop_is_converged(x64):
    minimiser = vdx.OptaxMinimiser(optax.sgd(0.5), max_steps=500, gtol=1e-8)
    result = minimiser.run(_quadratic, jnp.zeros(3), M)
    assert bool(result.converged)


def test_optax_step_cap_is_not_converged(x64):
    minimiser = vdx.OptaxMinimiser(optax.sgd(1e-3), max_steps=3, gtol=1e-12)
    result = minimiser.run(_quadratic, jnp.zeros(3), M)
    assert int(result.num_steps) == 3
    assert not bool(result.converged)


def test_optax_patience_stop_is_not_converged(x64):
    # a learning rate this large makes every step worse than the start
    minimiser = vdx.OptaxMinimiser(
        optax.sgd(3.0), max_steps=500, gtol=1e-12, patience=5
    )
    result = minimiser.run(_quadratic, jnp.zeros(3), M)
    assert int(result.num_steps) < 500
    assert not bool(result.converged)


def test_optax_polish_decides(x64):
    minimiser = vdx.OptaxMinimiser(
        optax.sgd(1e-3), max_steps=2, polish=optax.sgd(0.5), polish_gtol=1e-8
    )
    result = minimiser.run(_quadratic, jnp.zeros(3), M)
    assert bool(result.converged)


# ── models and custom minimisers ────────────────────────────────────────


def test_threedvar_analyse_carries_the_flag(x64):
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float64))
    three = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros((1, 3)),
        prior_cov_op=eye,
        obs_cov_op=eye,
    )
    y = jnp.ones((2, 1, 3))
    _, result = three.analyse(vdx.Batch1D(input=y, mask=jnp.ones_like(y)))
    assert result.converged.shape == (2,)
    assert bool(jnp.all(result.converged))


def test_custom_minimiser_reports_none(x64):
    def gradient_descent(cost_fn, x0, batch):
        grad = jax.grad(cost_fn)
        return jax.lax.fori_loop(0, 200, lambda _, x: x - 0.5 * grad(x, batch), x0)

    result = minimise_with_result(
        gradient_descent,
        lambda x, _: _quadratic(x, M),
        jnp.zeros(3),
        adjoint=optx.ImplicitAdjoint(),
        max_steps=200,
    )
    assert result.converged is None
    assert int(result.num_steps) == -1
