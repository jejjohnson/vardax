"""OptaxMinimiser, OptimistixMinimiser and ``minimiser=`` on the models (#85)."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
import optimistix as optx
import pytest

import vardax as vdx
from vardax import (
    Batch1D,
    Incremental4DVar,
    MaskedIdentity,
    Minimiser,
    OptaxMinimiser,
    OptimistixMinimiser,
    StrongFourDVar,
    ThreeDVar,
    WeakFourDVar,
)

# A well-conditioned SPD quadratic J(x) = ½ (x - m)ᵀ A (x - m), minimum m.
_KEY = jax.random.PRNGKey(0)
_Q = jnp.linalg.qr(jax.random.normal(_KEY, (5, 5)))[0]
_A = _Q @ jnp.diag(jnp.array([1.0, 1.5, 2.0, 3.0, 4.0])) @ _Q.T
_M = jnp.array([1.0, -2.0, 0.5, 3.0, -1.0])
_LAMBDA_MIN = 1.0


def _quadratic(x, m):
    d = x - m
    return 0.5 * d @ (_A @ d)


def _quiet(**kwargs):
    """An OptaxMinimiser that does not warn (for tests that check values)."""
    return OptaxMinimiser(warn_grad_norm=None, **kwargs)


class TestOptaxMinimiser:
    def test_satisfies_protocol(self):
        assert isinstance(OptaxMinimiser(), Minimiser)
        assert isinstance(OptimistixMinimiser(), Minimiser)

    def test_quadratic_converges_to_closed_form(self):
        minimiser = _quiet(
            optimizer=optax.adam(5e-2), max_steps=4000, tol=0.0, patience=None
        )
        result = minimiser.run(_quadratic, jnp.zeros(5), _M)
        # For a quadratic with Hessian A ⪰ λ_min I the error is bounded
        # by the gradient: ‖x - m‖ ≤ ‖A (x - m)‖ / λ_min = ‖∇J(x)‖ / λ_min.
        # The 1e-5 is float32 round-off in ‖∇J‖ (|m| ~ 3, eps ~ 1.2e-7,
        # times the condition number 4 and a few ops).
        err = jnp.linalg.norm(result.value - _M)
        assert err <= result.grad_norm / _LAMBDA_MIN + 1e-5
        assert result.grad_norm < 1e-3
        assert jnp.allclose(result.cost, _quadratic(result.value, _M))

    def test_lbfgs_polish_reaches_round_off(self):
        minimiser = OptaxMinimiser(
            optimizer=optax.adam(1e-1), max_steps=200, polish=optax.lbfgs()
        )
        result = minimiser.run(_quadratic, jnp.zeros(5), _M)
        # L-BFGS on a 5-D quadratic converges to float32 round-off.
        assert jnp.allclose(result.value, _M, atol=1e-4)
        assert result.grad_norm < 1e-3

    def test_pytree_control(self):
        def cost(x, m):
            return 0.5 * jnp.sum((x["a"] - m) ** 2) + 0.5 * jnp.sum(x["b"] ** 2)

        x0 = {"a": jnp.zeros(3), "b": jnp.ones((2, 2))}
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
        x = minimiser(cost, x0, _M[:3])
        assert jax.tree.structure(x) == jax.tree.structure(x0)
        assert jnp.allclose(x["a"], _M[:3], atol=1e-4)
        assert jnp.allclose(x["b"], 0.0, atol=1e-4)

    def test_stops_on_max_steps(self):
        result = _quiet(max_steps=7, tol=0.0, patience=None).run(
            _quadratic, jnp.zeros(5), _M
        )
        assert int(result.num_steps) == 7

    def test_stops_on_patience(self):
        # A constant cost never improves after the first step.
        result = _quiet(max_steps=1000, tol=0.0, patience=5).run(
            lambda x, _: jnp.sum(0.0 * x) + 1.0, jnp.zeros(3), None
        )
        assert int(result.num_steps) == 6

    def test_stops_on_relative_cost_change(self):
        result = _quiet(
            optimizer=optax.sgd(0.5), max_steps=1000, tol=1e-3, patience=None
        ).run(lambda x, _: 0.5 * jnp.sum((x - 1.0) ** 2) + 1.0, jnp.zeros(3), None)
        assert 1 < int(result.num_steps) < 1000

    def test_returns_lowest_cost_iterate(self):
        # SGD with too large a step diverges; the best iterate is x0.
        x0 = jnp.array([0.1, -0.1])
        result = _quiet(
            optimizer=optax.sgd(3.0), max_steps=10, tol=0.0, patience=None
        ).run(lambda x, _: 0.5 * jnp.sum(x**2), x0, None)
        assert jnp.array_equal(result.value, x0)

    def test_warns_when_not_stationary(self):
        minimiser = OptaxMinimiser(max_steps=1, warn_grad_norm=1e-3)
        with pytest.warns(UserWarning, match="gradient norm"):
            result = minimiser.run(_quadratic, jnp.zeros(5), _M)
            jax.effects_barrier()
        assert result.grad_norm > 1e-3

    def test_no_warning_when_stationary(self):
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            minimiser.run(_quadratic, jnp.zeros(5), _M)
            jax.effects_barrier()

    def test_jit_full_minimisation(self):
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())

        @jax.jit
        def solve(m):
            return minimiser(_quadratic, jnp.zeros(5), m)

        x_jit = solve(_M)
        x_eager = minimiser(_quadratic, jnp.zeros(5), _M)
        assert jnp.allclose(x_jit, x_eager, atol=1e-5)
        assert jnp.allclose(solve(2.0 * _M), 2.0 * _M, atol=1e-4)

    def test_implicit_adjoint_gradient(self):
        # x*(m) = m, so dx*/dm = I through the implicit function theorem.
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
        jac = jax.jacobian(lambda m: minimiser(_quadratic, jnp.zeros(5), m))(_M)
        assert jnp.allclose(jac, jnp.eye(5), atol=1e-4)

    def test_recursive_checkpoint_adjoint_gradient(self):
        # No L-BFGS polish: its zoom line search is an inner while_loop,
        # which reverse-mode cannot differentiate.
        minimiser = _quiet(optimizer=optax.adam(1e-1), max_steps=1000)

        def solve(m):
            return minimiser.run(
                _quadratic,
                jnp.zeros(5),
                m,
                adjoint=optx.RecursiveCheckpointAdjoint(),
            ).value

        jac = jax.jacrev(solve)(_M)
        # Unrolled differentiation of a converged loop: I up to the
        # residual contraction of the last iterates (looser than IFT).
        assert jnp.allclose(jac, jnp.eye(5), atol=1e-2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_steps": 0},
            {"tol": -1.0},
            {"patience": 0},
            {"polish_steps": 0},
            {"warn_grad_norm": -1.0},
        ],
    )
    def test_validation(self, kwargs):
        with pytest.raises(ValueError):
            OptaxMinimiser(**kwargs)

    def test_rejects_non_scalar_cost(self):
        with pytest.raises(ValueError, match="scalar"):
            OptaxMinimiser().run(lambda x, _: x, jnp.zeros(3), None)


class TestOptimistixMinimiser:
    def test_quadratic(self):
        result = OptimistixMinimiser().run(_quadratic, jnp.zeros(5), _M)
        # BFGS(rtol=atol=1e-6) on a 5-D quadratic: converged to round-off.
        assert jnp.allclose(result.value, _M, atol=1e-4)
        assert result.grad_norm < 1e-3

    def test_validation(self):
        with pytest.raises(ValueError):
            OptimistixMinimiser(max_steps=0)


# --- the models -----------------------------------------------------------


class IdentityForward(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return state


class Lorenz63RK4(eqx.Module):
    """Lorenz-63 stepped with classical RK4 (a ``ForwardModel``)."""

    dt: float = 0.01

    def step(self, x, dt):
        f = vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        k1 = f(0.0, x, None)
        k2 = f(0.0, x + 0.5 * dt * k1, None)
        k3 = f(0.0, x + 0.5 * dt * k2, None)
        k4 = f(0.0, x + dt * k3, None)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _eye(shape):
    return lx.IdentityLinearOperator(jax.ShapeDtypeStruct(shape, jnp.float32))


@pytest.fixture
def blue_problem():
    """B = R = I, H = I, everything observed: x* = (x_b + y) / 2."""
    key = jax.random.PRNGKey(3)
    y = jax.random.normal(key, (2, 1, 4))
    batch = Batch1D(input=y, mask=jnp.ones_like(y))
    return batch, y[:, 0] / 2.0


def _strong(minimiser=None, **kw):
    return StrongFourDVar(
        forward=IdentityForward(),
        obs_op=MaskedIdentity(),
        prior_mean=jnp.zeros(4),
        prior_cov_op=_eye((4,)),
        obs_cov_op=_eye((4,)),
        minimiser=minimiser,
        **kw,
    )


class TestModels:
    def test_default_is_bit_for_bit_optimistix_minimiser(self, blue_problem):
        batch, _ = blue_problem
        default = _strong()(batch)
        explicit = _strong(OptimistixMinimiser())(batch)
        assert jnp.array_equal(default, explicit)

    @pytest.mark.parametrize("which", ["three", "strong", "weak", "incremental"])
    def test_models_accept_optax_minimiser(self, blue_problem, which):
        batch, expected = blue_problem
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
        common = {
            "obs_op": MaskedIdentity(),
            "obs_cov_op": _eye((4,)),
            "minimiser": minimiser,
        }
        if which == "three":
            model = ThreeDVar(
                prior_mean=jnp.zeros((1, 4)),
                prior_cov_op=_eye((1, 4)),
                obs_cov_op=_eye((1, 4)),
                obs_op=MaskedIdentity(),
                minimiser=minimiser,
            )
            xa = model(batch)[:, 0]
        elif which == "strong":
            xa = _strong(minimiser)(batch)
        elif which == "weak":
            xa, _ = WeakFourDVar(
                forward=IdentityForward(),
                prior_mean=jnp.zeros(4),
                prior_cov_op=_eye((4,)),
                model_err_cov_op=_eye((4,)),
                **common,
            )(batch)
        else:
            xa = Incremental4DVar(
                forward=IdentityForward(),
                prior_mean=jnp.zeros(4),
                prior_cov_op=_eye((4,)),
                **common,
            )(batch)
        # BLUE with B = R = I: x* = y / 2; L-BFGS polish reaches round-off.
        assert jnp.allclose(xa, expected, atol=1e-4)

    def test_gradient_through_model_with_optax(self, blue_problem):
        batch, _ = blue_problem
        model = _strong(OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs()))

        def loss(y):
            return jnp.sum(model(Batch1D(input=y, mask=batch.mask)))

        # x* = y / 2 elementwise, so d(sum x*)/dy = 1/2 everywhere.
        g = jax.grad(loss)(batch.input)
        assert jnp.allclose(g, 0.5, atol=1e-4)

    @pytest.mark.slow
    def test_strong_4dvar_lorenz63_matches_bfgs(self):
        key_truth, key_b, key_o = jax.random.split(jax.random.PRNGKey(1), 3)
        forward = Lorenz63RK4()
        _, states = vdx.simulate_lorenz63(key_truth, n_steps=1, n_burn_in=500)
        x_true = states[0]
        T = 20  # 0.2 time units: weakly nonlinear window
        traj = [x_true]
        for _ in range(T):
            traj.append(forward.step(traj[-1], forward.dt))
        truth = jnp.stack(traj)
        y = truth + jax.random.normal(key_o, truth.shape)
        x_b = x_true + jax.random.normal(key_b, (3,))
        batch = Batch1D(input=y[None], mask=jnp.ones_like(y)[None])

        def model(minimiser=None):
            return StrongFourDVar(
                forward=forward,
                obs_op=MaskedIdentity(),
                prior_mean=x_b,
                prior_cov_op=_eye((3,)),
                obs_cov_op=_eye((3,)),
                minimiser=minimiser,
            )

        x_bfgs = model()(batch)[0]
        x_optax = model(
            # tol=0: stop on patience, not on a float32-noise cost change.
            OptaxMinimiser(
                optax.adam(5e-2),
                max_steps=3000,
                tol=0.0,
                patience=20,
                polish=optax.lbfgs(),
                polish_steps=100,
            )
        )(batch)[0]

        # The cost the model minimises, with B = R = I.
        def cost(x0):
            xs = [x0]
            for _ in range(T):
                xs.append(forward.step(xs[-1], forward.dt))
            return 0.5 * jnp.sum((x0 - x_b) ** 2) + 0.5 * jnp.sum(
                (y - jnp.stack(xs)) ** 2
            )

        # Strong convexity bound near the minimum: if ∇²J ⪰ λ I on the
        # segment between the two analyses, each lies within ‖∇J‖ / λ of
        # x*, so ‖x_optax - x_bfgs‖ ≤ (‖∇J(x_optax)‖ + ‖∇J(x_bfgs)‖) / λ.
        # λ is the smallest Hessian eigenvalue at x_bfgs halved, to allow
        # for curvature change over a segment this short; 1e-4 covers
        # float32 round-off in the 21-step rollout.
        lam = jnp.linalg.eigvalsh(jax.hessian(cost)(x_bfgs))[0] / 2.0
        assert lam > 0
        g_bfgs = jnp.linalg.norm(jax.grad(cost)(x_bfgs))
        g_optax = jnp.linalg.norm(jax.grad(cost)(x_optax))
        bound = (g_bfgs + g_optax) / lam + 1e-4
        assert jnp.linalg.norm(x_optax - x_bfgs) <= bound
        # And the analysis improves on the background.
        assert jnp.linalg.norm(x_optax - x_true) < jnp.linalg.norm(x_b - x_true)
