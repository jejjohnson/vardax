"""``IncrementalFourDVar`` on ``argminax.solve.GaussNewtonCG`` (vardax#279).

The outer loop moved from a hand-written Gauss–Newton loop to
``optimistix.least_squares`` on the whitened residuals. ``_reference_analysis`` is
that loop as it was before the port (gaussx's Gauss–Newton operator, a truncated CG
of at most n_inner steps, n_outer fixed outer iterations), and the port must
reproduce it to 1e-10 relative wherever the old loop applied Σ_ε⁻¹ and Σ_x⁻¹ exactly
(identity and diagonal covariances).
"""

from __future__ import annotations

from argminax.solve import GaussNewtonCG
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest

import vardax as vdx

N, T = 6, 3
PORT_RTOL = 1e-10


class _Lorenz96(eqx.Module):
    """Lorenz-96 (F = 8) stepped with RK4, so the window is nonlinear."""

    @property
    def dt(self) -> float:
        return 0.05

    def step(self, x, dt):
        def tendency(state):
            return (
                (jnp.roll(state, -1) - jnp.roll(state, 2)) * jnp.roll(state, 1)
                - state
                + 8.0
            )

        k1 = tendency(x)
        k2 = tendency(x + 0.5 * dt * k1)
        k3 = tendency(x + 0.5 * dt * k2)
        k4 = tendency(x + dt * k3)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class _Decay(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    def step(self, x, dt):
        return 0.9 * x + 0.05 * jnp.roll(x, 1)


def _reference_analysis(
    forward, prior_mean, prior_cov, obs_cov, batch, *, n_outer, n_inner, cvt, jac
):
    """The pre-port loop of IncrementalFourDVar, for h = I and an exact Σ⁻¹."""
    tol = 1e-6

    def one_window(window_obs, window_mask):
        def apply_obs_precision(residuals):
            def solve_step(step_residual, step_mask):
                masked = gaussx.masked_obs_covariance(obs_cov, step_mask)
                return gaussx.solve(masked, step_residual, solver=lx.CG(tol, tol))

            return jax.vmap(solve_step)(residuals, window_mask)

        def residual(initial_state):
            states = [initial_state]
            for _ in range(T):
                states.append(forward.step(states[-1], forward.dt))
            return window_mask * window_obs - window_mask * jnp.stack(states)

        if cvt:
            prior_sqrt = gaussx.sqrt(prior_cov)

            def to_state(control):
                return prior_mean + prior_sqrt.mv(control)

            def background_cost(control):
                return 0.5 * jnp.sum(control**2)

            def apply_prior_precision(direction):
                return direction

            control = jnp.zeros_like(prior_mean)
        else:

            def to_state(control):
                return control

            def apply_prior_precision(direction):
                return gaussx.solve(prior_cov, direction, solver=lx.CG(tol, tol))

            def background_cost(control):
                departure = control - prior_mean
                return 0.5 * jnp.sum(departure * apply_prior_precision(departure))

            control = prior_mean

        def control_residual(control):
            return residual(to_state(control))

        def cost(control):
            misfit = control_residual(control)
            return background_cost(control) + 0.5 * jnp.sum(
                misfit * apply_obs_precision(misfit)
            )

        inner = gaussx.CGSolver(rtol=tol, atol=tol, max_steps=n_inner, throw=False)
        for _ in range(n_outer):
            hessian = gaussx.gauss_newton_operator(
                control_residual,
                control,
                noise_precision=apply_obs_precision,
                prior_precision=apply_prior_precision,
                jac=jac,
            )
            control = control + inner.solve(hessian, -jax.grad(cost)(control))
        return to_state(control)

    return jax.vmap(one_window)(batch.input, batch.mask)


def _problem(forward_name):
    rng = np.random.default_rng(0)
    offset, scale = (3.0, 0.5) if forward_name == "lorenz96" else (0.0, 1.0)
    obs = offset + scale * rng.normal(size=(2, T + 1, N))
    mask = (rng.uniform(size=(2, T + 1, N)) < 0.7).astype(float)
    batch = vdx.Batch1D(input=jnp.asarray(obs * mask), mask=jnp.asarray(mask))
    forward = _Lorenz96() if forward_name == "lorenz96" else _Decay()
    prior_mean = jnp.asarray(offset + 0.3 * rng.normal(size=N))
    prior_cov = lx.DiagonalLinearOperator(jnp.asarray(rng.uniform(0.5, 2.0, N)))
    obs_cov = lx.DiagonalLinearOperator(jnp.asarray(rng.uniform(0.2, 0.6, N)))
    return forward, prior_mean, prior_cov, obs_cov, batch


def _model(forward, prior_mean, prior_cov, obs_cov, **kwargs):
    return vdx.IncrementalFourDVar(
        forward=forward,
        obs_op=vdx.MaskedIdentity(),
        prior_mean=prior_mean,
        prior_cov_op=prior_cov,
        obs_cov_op=obs_cov,
        **kwargs,
    )


def _relative(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


@pytest.mark.slow
@pytest.mark.parametrize("forward_name", ["decay", "lorenz96"])
@pytest.mark.parametrize(
    ("n_outer", "n_inner", "cvt", "jac"),
    [
        pytest.param(3, 30, True, "fwd", id="default"),
        pytest.param(3, 3, True, "fwd", id="truncated-cg"),
        pytest.param(2, 30, False, "fwd", id="state-space"),
        pytest.param(3, 30, True, "bwd", id="bwd"),
    ],
)
def test_the_port_reproduces_the_old_loop(
    x64, forward_name, n_outer, n_inner, cvt, jac
):
    forward, prior_mean, prior_cov, obs_cov, batch = _problem(forward_name)
    expected = _reference_analysis(
        forward,
        prior_mean,
        prior_cov,
        obs_cov,
        batch,
        n_outer=n_outer,
        n_inner=n_inner,
        cvt=cvt,
        jac=jac,
    )
    solver = GaussNewtonCG(
        rtol=0.0, atol=0.0, max_cg_steps=n_inner, cg_rtol=1e-6, cg_atol=1e-6
    )
    model = _model(
        forward,
        prior_mean,
        prior_cov,
        obs_cov,
        config=vdx.IncrementalConfig(cvt=cvt, jac=jac),
        solver=solver,
        n_outer=n_outer,
    )
    assert _relative(model(batch), expected) < PORT_RTOL


@pytest.mark.slow
def test_the_default_is_the_old_default_budget(x64):
    """No solver and no budget: three outer, thirty inner, CG tolerances 1e-6."""
    forward, prior_mean, prior_cov, obs_cov, batch = _problem("lorenz96")
    expected = _reference_analysis(
        forward,
        prior_mean,
        prior_cov,
        obs_cov,
        batch,
        n_outer=3,
        n_inner=30,
        cvt=True,
        jac="fwd",
    )
    model = _model(forward, prior_mean, prior_cov, obs_cov)
    assert _relative(model(batch), expected) < PORT_RTOL


def test_the_deprecated_budget_builds_the_same_solver(x64):
    forward, prior_mean, prior_cov, obs_cov, batch = _problem("decay")
    with pytest.warns(DeprecationWarning, match="GaussNewtonCG") as caught:
        deprecated = _model(
            forward,
            prior_mean,
            prior_cov,
            obs_cov,
            config=vdx.IncrementalConfig(n_outer=2, n_inner=5, cg_rtol=1e-9),
        )
    # attributed to this line, not to vardax or equinox
    assert caught[0].filename == __file__
    explicit = _model(
        forward,
        prior_mean,
        prior_cov,
        obs_cov,
        solver=GaussNewtonCG(
            rtol=0.0, atol=0.0, max_cg_steps=5, cg_rtol=1e-9, cg_atol=1e-6
        ),
        n_outer=2,
    )
    assert deprecated.n_outer == 2
    assert eqx.tree_equal(deprecated.solver, explicit.solver)
    np.testing.assert_array_equal(deprecated(batch), explicit(batch))


def test_the_new_route_does_not_warn(recwarn):
    forward, prior_mean, prior_cov, obs_cov, _ = _problem("decay")
    _model(forward, prior_mean, prior_cov, obs_cov, n_outer=2)
    _model(forward, prior_mean, prior_cov, obs_cov, config=vdx.IncrementalConfig())
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_a_converging_solver_reaches_the_closed_form(x64):
    """Linear window: GaussNewtonCG with outer tolerances stops at the BLUE."""
    forward, prior_mean, prior_cov, obs_cov, batch = _problem("decay")
    transition = 0.9 * np.eye(N) + 0.05 * np.roll(np.eye(N), 1, axis=0)
    mask = np.asarray(batch.mask[0])
    rows = [
        np.diag(mask[t]) @ np.linalg.matrix_power(transition, t) for t in range(T + 1)
    ]
    G = np.concatenate(rows)
    R = np.tile(np.asarray(lx.diagonal(obs_cov)), T + 1)
    B = np.diag(np.asarray(lx.diagonal(prior_cov)))
    y = np.asarray(batch.input[0]).reshape(-1)
    x_b = np.asarray(prior_mean)
    gain = B @ G.T @ np.linalg.inv(G @ B @ G.T + np.diag(R))
    expected = x_b + gain @ (y - G @ x_b)
    model = _model(
        forward,
        prior_mean,
        prior_cov,
        obs_cov,
        solver=GaussNewtonCG(rtol=1e-10, atol=1e-10, max_cg_steps=100),
        n_outer=20,
    )
    np.testing.assert_allclose(model(batch)[0], expected, rtol=1e-8, atol=1e-10)


def test_it_runs_under_filter_jit(x64):
    forward, prior_mean, prior_cov, obs_cov, batch = _problem("decay")
    model = _model(forward, prior_mean, prior_cov, obs_cov)
    np.testing.assert_allclose(eqx.filter_jit(model)(batch), model(batch), rtol=1e-12)


def test_the_budget_is_passed_once():
    forward, prior_mean, prior_cov, obs_cov, _ = _problem("decay")
    with pytest.raises(vdx.ConfigurationError, match="budget once"):
        _model(
            forward,
            prior_mean,
            prior_cov,
            obs_cov,
            config=vdx.IncrementalConfig(n_inner=10),
            solver=GaussNewtonCG(rtol=1e-6, atol=1e-6),
        )


@pytest.mark.unchecked_types
def test_a_minimiser_is_not_a_least_squares_solver():
    forward, prior_mean, prior_cov, obs_cov, _ = _problem("decay")
    with pytest.raises(vdx.InputTypeError, match="AbstractLeastSquaresSolver"):
        _model(
            forward,
            prior_mean,
            prior_cov,
            obs_cov,
            solver=optx.BFGS(rtol=1e-6, atol=1e-6),
        )


@pytest.mark.unchecked_types
@pytest.mark.parametrize("n_outer", [0, 1.5, True])
def test_n_outer_must_be_a_positive_int(n_outer):
    forward, prior_mean, prior_cov, obs_cov, _ = _problem("decay")
    with pytest.raises(vdx.ConfigurationError, match="n_outer"):
        _model(forward, prior_mean, prior_cov, obs_cov, n_outer=n_outer)
