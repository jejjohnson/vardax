"""OptaxMinimiser, OptimistixMinimiser and ``minimiser=`` on the models (#85)."""

from __future__ import annotations

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
    ReducedBasisFourDVar,
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


EPS = float(jnp.finfo(jnp.float32).eps)
_COND = 4.0  # condition number of _A
# Float32 resolution of the minimiser: ∇J = A(x - m) is computed with an
# error of a few eps·‖A‖‖x‖, so x is only determined to ~eps·cond(A)·‖x‖;
# 8 covers the op count of the matvec and the norm.
_ROUND = 8 * EPS * _COND * float(jnp.linalg.norm(_M))


def _grad_bound(result):
    """‖x - m‖ ≤ ‖∇J(x)‖ / λ_min (exact for this quadratic) + round-off."""
    return float(result.grad_norm) / _LAMBDA_MIN + _ROUND


class TestOptaxMinimiser:
    def test_satisfies_protocol(self):
        assert isinstance(OptaxMinimiser(), Minimiser)
        assert isinstance(OptimistixMinimiser(), Minimiser)

    def test_quadratic_converges_to_closed_form(self):
        minimiser = OptaxMinimiser(
            optimizer=optax.adam(5e-2), max_steps=4000, gtol=1e-4, patience=None
        )
        result = minimiser.run(_quadratic, jnp.zeros(5), _M)
        assert jnp.linalg.norm(result.value - _M) <= _grad_bound(result)
        assert result.grad_norm < 1e-3
        assert jnp.allclose(result.cost, _quadratic(result.value, _M))

    def test_large_offset_does_not_stop_early(self):
        # Regression: a relative cost-change stop fired after 2 steps here,
        # because J - J_prev rounds to 0 next to a 1e7 offset (error 4.8).
        m = jnp.full(10, 5.0)
        result = OptaxMinimiser(optax.adam(1e-1)).run(
            lambda x, _: 0.5 * jnp.sum((x - m) ** 2) + 1e7, jnp.zeros(10), None
        )
        assert int(result.num_steps) > 50
        # Hessian I: ‖x - m‖ ≤ ‖∇J‖ + eps·‖m‖ round-off.
        bound = float(result.grad_norm) + 8 * EPS * float(jnp.linalg.norm(m))
        assert jnp.linalg.norm(result.value - m) <= bound
        assert bound < 1e-4

    def test_lbfgs_polish_reaches_round_off(self):
        minimiser = OptaxMinimiser(
            optimizer=optax.adam(1e-1), max_steps=200, polish=optax.lbfgs()
        )
        result = minimiser.run(_quadratic, jnp.zeros(5), _M)
        assert jnp.linalg.norm(result.value - _M) <= _grad_bound(result)
        assert result.grad_norm < 1e-5

    def test_polish_has_its_own_tight_tolerances(self):
        # A loose main loop (gtol = 0.1) must not loosen the polish.
        minimiser = OptaxMinimiser(
            optimizer=optax.adam(1e-1), gtol=1e-1, polish=optax.lbfgs()
        )
        result = minimiser.run(_quadratic, jnp.zeros(5), _M)
        assert result.grad_norm < 1e-5
        assert jnp.linalg.norm(result.value - _M) <= _ROUND + 1e-5

    def test_pytree_control(self):
        def cost(x, m):
            return 0.5 * jnp.sum((x["a"] - m) ** 2) + 0.5 * jnp.sum(x["b"] ** 2)

        x0 = {"a": jnp.zeros(3), "b": jnp.ones((2, 2))}
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
        result = minimiser.run(cost, x0, _M[:3])
        x = result.value
        assert jax.tree.structure(x) == jax.tree.structure(x0)
        # Hessian I: the error is bounded by the gradient norm.
        bound = float(result.grad_norm) + 8 * EPS * 3.0
        assert jnp.max(jnp.abs(x["a"] - _M[:3])) <= bound
        assert jnp.max(jnp.abs(x["b"])) <= bound

    def test_stops_on_max_steps(self):
        result = OptaxMinimiser(max_steps=7, gtol=0.0, patience=None).run(
            _quadratic, jnp.zeros(5), _M
        )
        assert int(result.num_steps) == 7

    def test_stops_on_gradient_norm(self):
        # SGD(0.5) on ½‖x - 1‖² halves the gradient exactly each step:
        # ‖g_k‖ = √3 / 2^k ≤ 1e-3 first at k = 11, so 12 evaluations.
        result = OptaxMinimiser(optax.sgd(0.5), gtol=1e-3, patience=None).run(
            lambda x, _: 0.5 * jnp.sum((x - 1.0) ** 2), jnp.zeros(3), None
        )
        assert int(result.num_steps) == 12

    def test_stops_on_step_size(self):
        # Same iteration, step ‖u_k‖ = ‖g_k‖ / 2 = √3 / 2^(k+1) ≤ 1e-3 first
        # at k = 10; step_patience = 3 needs k = 10, 11, 12, so 13 evaluations.
        result = OptaxMinimiser(optax.sgd(0.5), gtol=0.0, atol=1e-3, patience=None).run(
            lambda x, _: 0.5 * jnp.sum((x - 1.0) ** 2), jnp.zeros(3), None
        )
        assert int(result.num_steps) == 13

    def test_momentum_does_not_leave_a_stationary_point(self):
        # Codex P2: with J = x²/2 + 1e7 (costs tie in float32) heavy-ball SGD
        # reaches x = 0 in one step; the stored momentum must not then carry
        # it to -0.9. The stationary iterate wins the tie on gradient norm.
        result = OptaxMinimiser(optax.sgd(1.0, momentum=0.9)).run(
            lambda x, _: 0.5 * x**2 + 1e7, jnp.asarray(1.0), None
        )
        assert result.value == 0.0
        assert result.grad_norm == 0.0

    def test_zero_start_schedule_does_not_stop_at_once(self):
        # Codex P2: a schedule starting at 0 makes the first update exactly
        # zero; one small step must not stop the loop.
        schedule = optax.linear_schedule(0.0, 0.1, 100)
        result = OptaxMinimiser(optax.adam(schedule)).run(
            lambda x, _: 0.5 * jnp.sum((x - 3.0) ** 2), jnp.zeros(2), None
        )
        assert int(result.num_steps) > 100
        # Hessian I: ‖x - 3‖ ≤ ‖∇J‖ + round-off.
        bound = float(result.grad_norm) + 8 * EPS * 3.0 * 2
        assert jnp.linalg.norm(result.value - 3.0) <= bound

    def test_patience_and_lowest_cost_iterate(self):
        # SGD(3) on ½‖x‖² multiplies x by -2 each step, so the cost grows
        # 4x per step: x0 is the best, and patience 5 stops after 1 + 5.
        x0 = jnp.array([0.1, -0.1])
        result = OptaxMinimiser(optax.sgd(3.0), gtol=0.0, patience=5).run(
            lambda x, _: 0.5 * jnp.sum(x**2), x0, None
        )
        assert int(result.num_steps) == 6
        assert jnp.array_equal(result.value, x0)

    def test_flat_cost_does_not_exhaust_patience(self):
        # A cost flat to float32 round-off (offset 1e7) counts as improving:
        # patience=3 must not stop Adam while x is still moving.
        result = OptaxMinimiser(optax.adam(1e-2), gtol=0.0, patience=3).run(
            lambda x, _: 1e-3 * jnp.sum((x - 1.0) ** 2) + 1e7, jnp.zeros(3), None
        )
        # The cost is exactly 1e7 throughout (ulp(1e7) = 1); Adam walks x
        # to 1 regardless, ~100 steps per unit at lr 1e-2.
        assert int(result.num_steps) > 100
        assert jnp.allclose(result.value, 1.0, atol=2e-2)

    def test_not_stationary_is_reported_under_jit_and_vmap(self):
        # No in-graph warning: jit + vmap under warnings-as-errors works and
        # the gradient norm is returned instead.
        minimiser = OptaxMinimiser(max_steps=1)

        @jax.jit
        @jax.vmap
        def solve(m):
            return minimiser.run(_quadratic, jnp.zeros(5), m).grad_norm

        g = solve(jnp.stack([_M, 2 * _M]))
        assert g.shape == (2,)
        assert jnp.all(g > 1.0)

    def test_jit_full_minimisation(self):
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())

        @jax.jit
        def solve(m):
            return minimiser.run(_quadratic, jnp.zeros(5), m)

        r_jit = solve(_M)
        r_eager = minimiser.run(_quadratic, jnp.zeros(5), _M)
        # Both are within the gradient bound of m, hence of each other.
        tol = _grad_bound(r_jit) + _grad_bound(r_eager)
        assert jnp.linalg.norm(r_jit.value - r_eager.value) <= tol
        r2 = solve(2.0 * _M)
        assert jnp.linalg.norm(r2.value - 2.0 * _M) <= _grad_bound(r2) + _ROUND

    def test_implicit_adjoint_gradient(self):
        # x*(m) = m, so dx*/dm = I; the IFT tangent is A^{-1} A = I up to
        # the float32 solve error, ~ eps·cond(A) times a small op count.
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
        jac = jax.jacobian(lambda m: minimiser(_quadratic, jnp.zeros(5), m))(_M)
        assert jnp.allclose(jac, jnp.eye(5), atol=16 * EPS * _COND)

    def test_implicit_adjoint_matrix_derivative(self):
        # J = ½ xᵀAx - bᵀx, x* = A⁻¹b, dx*/dA[δA] = -A⁻¹ δA A⁻¹ b. Unlike
        # dx*/dm above, this depends on x* itself, so an inexact x* shows.
        Q = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (4, 4)))[0]
        A = Q @ jnp.diag(jnp.array([1.0, 2.0, 5.0, 10.0])) @ Q.T
        b = jnp.array([1.0, -1.0, 2.0, 0.5])
        dA = jax.random.normal(jax.random.PRNGKey(2), (4, 4))
        dA = dA + dA.T
        minimiser = OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())

        def solve(Am):
            return minimiser(lambda x, _: 0.5 * x @ Am @ x - b @ x, jnp.zeros(4), None)

        x, t = jax.jvp(solve, (A,), (dA,))
        A_inv = jnp.linalg.inv(A)
        x_true = A_inv @ b
        t_true = -A_inv @ dA @ x_true
        cond = 10.0
        # x* is resolved to float32: ~eps·cond(A)·‖x‖ (8 for the op count).
        x_floor = 8 * EPS * cond * float(jnp.linalg.norm(x_true))
        assert jnp.linalg.norm(x - x_true) <= x_floor
        # The IFT tangent is -A⁻¹ δA x̂, so its error is ‖A⁻¹‖‖δA‖‖x̂ - x‖
        # plus the float32 tangent solve, ~eps·cond(A)·‖t‖.
        bound = float(jnp.linalg.norm(dA, 2)) * x_floor + 8 * EPS * cond * float(
            jnp.linalg.norm(t_true)
        )
        assert jnp.linalg.norm(t - t_true) <= bound

    @pytest.mark.parametrize("which", ["optax", "optimistix"])
    def test_cost_gradient_is_envelope(self, which):
        # J*(b) = min_x ½ xᵀAx - bᵀx has dJ*/db = -x* = -A⁻¹b.
        b = jnp.array([1.0, -1.0, 2.0, 0.5, 0.0])
        minimiser = (
            OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
            if which == "optax"
            else OptimistixMinimiser()
        )

        def run(b):
            return minimiser.run(
                lambda x, _: 0.5 * x @ _A @ x - b @ x, jnp.zeros(5), None
            )

        g = jax.grad(lambda b: run(b).cost)(b)
        expected = -jnp.linalg.solve(_A, b)
        # dJ*/db = -x̂ exactly, so the error is ‖x̂ - x*‖ ≤ ‖∇J(x̂)‖ / λ_min
        # (exact for a quadratic) plus round-off.
        bound = float(run(b).grad_norm) / _LAMBDA_MIN + _ROUND
        assert jnp.linalg.norm(g - expected) <= bound

    @pytest.mark.parametrize("which", ["optax", "optimistix"])
    def test_cost_second_derivative(self, which):
        # Codex P2: J*(b) = min_x x²/2 - bx = -b²/2, so dJ*/db = -b and
        # d²J*/db² = -1; detaching x* gave 0. The first derivative is
        # -x̂ (round-off of x̂ = b); the second is the IFT tangent dx*/db = 1,
        # exact up to float32.
        minimiser = (
            OptaxMinimiser(optax.adam(1e-1), polish=optax.lbfgs())
            if which == "optax"
            else OptimistixMinimiser()
        )

        def j_star(b):
            return minimiser.run(
                lambda x, _: 0.5 * x**2 - b * x, jnp.asarray(0.0), None
            ).cost

        assert jnp.allclose(jax.grad(j_star)(2.0), -2.0, atol=1e-5)
        assert jnp.allclose(jax.grad(jax.grad(j_star))(2.0), -1.0, atol=8 * EPS)

    def test_recursive_checkpoint_adjoint_gradient(self):
        # 50 fixed SGD(0.1) steps on J = ½(x - m)ᵀA(x - m): x_50 = m - (I -
        # 0.1A)^50 (m - x0), so dx_50/dm = I - (I - 0.1A)^50 exactly. The
        # cost decreases monotonically (0.1·λ_max < 2), so the last iterate
        # is returned. Tolerance: float32 round-off of a 50-term product.
        minimiser = OptaxMinimiser(
            optax.sgd(0.1), max_steps=50, gtol=0.0, patience=None
        )

        def solve(m):
            return minimiser.run(
                _quadratic,
                jnp.zeros(5),
                m,
                adjoint=optx.RecursiveCheckpointAdjoint(),
            ).value

        jac = jax.jacrev(solve)(_M)
        expected = jnp.eye(5) - jnp.linalg.matrix_power(jnp.eye(5) - 0.1 * _A, 50)
        assert jnp.allclose(jac, expected, atol=50 * 8 * EPS)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_steps": 0},
            {"gtol": -1.0},
            {"rtol": -1.0},
            {"atol": -1.0},
            {"patience": 0},
            {"polish_steps": 0},
            {"polish_gtol": -1.0},
            {"polish_patience": 0},
            {"step_patience": 0},
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
        assert jnp.linalg.norm(result.value - _M) <= _grad_bound(result)
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


def _l63_case(seed):
    """Strong 4D-Var on Lorenz-63: BFGS and OptaxMinimiser analyses.

    Returns ``(‖x_optax - x_bfgs‖, bound, ‖x_optax - x_true‖, ‖x_b - x_true‖)``.
    """
    key_truth, key_b, key_o = jax.random.split(jax.random.PRNGKey(seed), 3)
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
        OptaxMinimiser(
            optax.adam(5e-2),
            max_steps=3000,
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
        return 0.5 * jnp.sum((x0 - x_b) ** 2) + 0.5 * jnp.sum((y - jnp.stack(xs)) ** 2)

    # Heuristic bound (not a proof): near a strict minimum each analysis
    # lies within ‖∇J‖ / λ of x*, with λ the smallest Hessian eigenvalue on
    # the segment. We use half the eigenvalue at x_bfgs for λ, plus 1e-4
    # for float32 round-off in the 21-step rollout. Calibrated on seeds
    # 1-6: observed distance / bound was at most 0.15, so the margin is
    # about 7x.
    lam = jnp.linalg.eigvalsh(jax.hessian(cost)(x_bfgs))[0] / 2.0
    assert lam > 0
    g_bfgs = jnp.linalg.norm(jax.grad(cost)(x_bfgs))
    g_optax = jnp.linalg.norm(jax.grad(cost)(x_optax))
    bound = (g_bfgs + g_optax) / lam + 1e-4
    return (
        float(jnp.linalg.norm(x_optax - x_bfgs)),
        float(bound),
        float(jnp.linalg.norm(x_optax - x_true)),
        float(jnp.linalg.norm(x_b - x_true)),
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
            xa = ReducedBasisFourDVar(
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
        dist, bound, err_a, err_b = _l63_case(1)
        assert dist <= bound
        # And the analysis improves on the background.
        assert err_a < err_b
