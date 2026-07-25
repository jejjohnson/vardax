"""Tests for vardax._src.priors_dynamical (issue #14, ported from mfourdvar)."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest

from vardax import DynamicalPrior, DynIncrements, DynTrajectory


def decay(t, y, args):
    """Linear decay ODE ``dy/dt = -a y`` (``a = 1`` when ``args is None``)."""
    a = 1.0 if args is None else args
    return -a * y


class TestDynamicalPriorBase:
    def test_defaults(self):
        prior = DynamicalPrior(model=decay)
        assert isinstance(prior.solver, dfx.Tsit5)
        assert isinstance(prior.stepsize, dfx.PIDController)
        assert isinstance(prior.adjoint, dfx.RecursiveCheckpointAdjoint)

    def test_base_call_not_implemented(self):
        prior = DynamicalPrior(model=decay)
        with pytest.raises(NotImplementedError):
            prior(jnp.ones(3), jnp.array([0.0, 0.1]))

    def test_custom_solver(self):
        prior = DynIncrements(model=decay, solver=dfx.Euler())
        assert isinstance(prior.solver, dfx.Euler)

    def test_fixed_step_solver_gets_constant_controller(self):
        # Euler provides no error estimate; the default controller must not
        # be adaptive, and integration must actually work.
        prior = DynTrajectory(model=decay, solver=dfx.Euler())
        assert isinstance(prior.stepsize, dfx.ConstantStepSize)
        ts = jnp.linspace(0.0, 0.5, 6)
        traj = prior(jnp.array([1.0, 2.0, -1.0]), ts)
        assert traj.shape == (6, 3)
        assert bool(jnp.all(jnp.isfinite(traj)))

    def test_adaptive_solver_keeps_pid_controller(self):
        prior = DynTrajectory(model=decay, solver=dfx.Tsit5())
        assert isinstance(prior.stepsize, dfx.PIDController)

    def test_explicit_stepsize_respected(self):
        controller = dfx.ConstantStepSize()
        prior = DynTrajectory(model=decay, stepsize=controller)
        assert prior.stepsize is controller


class TestDynTrajectory:
    def test_call_shape(self):
        prior = DynTrajectory(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        traj = prior(jnp.array([1.0, 2.0, -1.0]), ts)
        assert traj.shape == (6, 3)

    def test_first_state_is_initial_condition(self):
        prior = DynTrajectory(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        x0 = jnp.array([1.0, 2.0, -1.0])
        traj = prior(x0, ts)
        assert jnp.allclose(traj[0], x0)

    def test_matches_analytic_decay(self):
        # dy/dt = -y  =>  y(t) = y0 exp(-t)
        prior = DynTrajectory(model=decay)
        ts = jnp.linspace(0.0, 1.0, 5)
        x0 = jnp.array([1.0, 2.0])
        traj = prior(x0, ts)
        expected = x0[None, :] * jnp.exp(-ts)[:, None]
        assert jnp.allclose(traj, expected, atol=1e-4)

    def test_loss_zero_on_consistent(self):
        prior = DynTrajectory(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        traj = prior(jnp.array([1.0, 2.0, -1.0]), ts)
        assert float(prior.loss(traj, ts)) < 1e-4


class TestDynIncrements:
    def test_call_shape(self):
        prior = DynIncrements(model=decay)
        out = prior(jnp.ones(3), jnp.array([0.0, 0.1]))
        assert out.shape == (3,)

    def test_loss_zero_on_consistent(self):
        # A trajectory produced by the same model has ~zero increment loss.
        ts = jnp.linspace(0.0, 0.5, 6)
        traj = DynTrajectory(model=decay)(jnp.array([1.0, 2.0, -1.0]), ts)
        loss = DynIncrements(model=decay).loss(traj, ts)
        assert float(loss) < 1e-3

    def test_loss_positive_on_inconsistent(self):
        prior = DynIncrements(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        x = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
        assert float(prior.loss(x, ts)) > 0.0

    def test_loss_size_mismatch_raises(self):
        prior = DynIncrements(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        with pytest.raises(ValueError, match="Size mismatch"):
            prior.loss(jnp.ones((4, 3)), ts)

    def test_grad_wrt_state(self):
        prior = DynIncrements(model=decay)
        ts = jnp.linspace(0.0, 0.3, 4)
        x = jnp.ones((4, 3))
        grad = jax.grad(lambda z: prior.loss(z, ts))(x)
        assert grad.shape == x.shape
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_params_override_grad(self):
        # Gradient w.r.t. a scalar ODE parameter flows through the solve.
        ts = jnp.linspace(0.0, 0.3, 4)
        traj = DynTrajectory(model=decay, params=1.0)(jnp.ones(3), ts)
        prior = DynIncrements(model=decay)

        def loss_of_a(a):
            return prior.loss(traj, ts, params=a)

        grad = jax.grad(loss_of_a)(2.0)
        assert jnp.isfinite(grad)

    def test_jit(self):
        import equinox as eqx

        prior = DynIncrements(model=decay)
        ts = jnp.linspace(0.0, 0.3, 4)
        x = jnp.ones((4, 3))
        loss = eqx.filter_jit(prior.loss)(x, ts)
        assert jnp.isfinite(loss)


class TestReconstructAndBind:
    """Decision D18: reconstruction semantics and the Prior bridge."""

    def test_trajectory_reconstruct_matches_rollout(self):
        prior = DynTrajectory(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        x = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
        rec = prior.reconstruct(x, ts)
        assert rec.shape == x.shape
        assert jnp.allclose(rec, prior(x[0], ts))

    def test_increments_reconstruct_shape_and_anchor(self):
        prior = DynIncrements(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        x = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
        rec = prior.reconstruct(x, ts)
        assert rec.shape == x.shape
        # First element anchors to the input; residual there is zero.
        assert jnp.allclose(rec[0], x[0])

    def test_increments_residual_matches_loss(self):
        prior = DynIncrements(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        x = jax.random.normal(jax.random.PRNGKey(2), (6, 3))
        residual = jnp.sum((x - prior.reconstruct(x, ts)) ** 2)
        assert jnp.allclose(residual, prior.loss(x, ts), atol=1e-5)

    def test_bound_prior_in_variational_cost(self):
        # A bound temporal prior drops into the weak-constraint cost as-is.
        from vardax import Batch1D, variational_cost

        prior = DynTrajectory(model=decay)
        ts = jnp.linspace(0.0, 0.5, 6)
        traj = prior(jnp.ones(3), ts)
        x = traj[None, :, :]  # (B=1, T, N)
        batch = Batch1D(input=x, mask=jnp.ones_like(x))

        bound = prior.bind(ts)

        def batched_bound(z):
            return jax.vmap(bound)(z)

        # State equals both the observations and its own rollout -> ~0 cost.
        cost = variational_cost(x, batch, batched_bound)
        assert float(cost) < 1e-6

    def test_bind_closes_over_params(self):
        prior = DynTrajectory(model=decay, params=1.0)
        ts = jnp.linspace(0.0, 0.3, 4)
        x = jnp.ones((4, 3))
        fast = prior.bind(ts, params=5.0)(x)
        slow = prior.bind(ts)(x)
        assert not jnp.allclose(fast, slow)


class TestAsForwardModel:
    """Decision D18: pipekit_cycle.ForwardModel adapter."""

    def test_step_matches_one_step_integration(self):
        prior = DynTrajectory(model=decay)
        fwd = prior.as_forward_model(dt=0.1)
        state = jnp.array([1.0, 2.0, -1.0])
        stepped = fwd.step(state, 0.1)
        # dy/dt = -y  =>  y(dt) = y0 exp(-dt)
        assert jnp.allclose(stepped, state * jnp.exp(-0.1), atol=1e-4)

    def test_dt_and_signature(self):
        fwd = DynIncrements(model=decay).as_forward_model(dt=0.05)
        assert fwd.dt == 0.05
        assert fwd.state_signature is None

    def test_drives_rollout(self):
        # Repeated stepping reproduces the trajectory rollout.
        prior = DynTrajectory(model=decay)
        fwd = prior.as_forward_model(dt=0.1)
        ts = jnp.linspace(0.0, 0.3, 4)
        traj = prior(jnp.ones(3), ts)
        x = jnp.ones(3)
        for t in range(1, 4):
            x = fwd.step(x, 0.1)
            assert jnp.allclose(x, traj[t], atol=1e-4)
