"""Tests for vardax._src.solver."""

from flax import nnx
import jax
import jax.numpy as jnp

from vardax import (
    LSTMState1D,
    SolverState1D,
    fp_solver_step_1d,
    init_solver_state_1d,
    init_solver_state_2d,
    solve_4dvarnet_1d_fixedpoint,
)
from vardax._src.solver import solver_step_1d


class TestInitSolverState1D:
    def test_shape(self, batch_1d):
        state = init_solver_state_1d(batch_1d, hidden_dim=16)
        B, T, N = batch_1d.input.shape
        assert state.x.shape == (B, T, N)
        assert state.step == 0

    def test_initial_state_is_masked_input(self, batch_1d):
        state = init_solver_state_1d(batch_1d, hidden_dim=16)
        expected = batch_1d.input * batch_1d.mask
        assert jnp.allclose(state.x, expected)


class TestInitSolverState2D:
    def test_shape(self, batch_2d):
        state = init_solver_state_2d(batch_2d, hidden_dim=8)
        B, T, H, W = batch_2d.input.shape
        assert state.x.shape == (B, T, H, W)
        assert state.step == 0


class TestSolverStep1D:
    def test_step_increments(self, rng, batch_1d):
        from vardax import BilinAEPrior1D, ConvLSTMGradMod1D

        B, T, N = batch_1d.input.shape
        hidden_dim = 16
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(state_dim=N, latent_dim=8, n_time=T, rngs=nnx.Rngs(k1))
        grad_mod = ConvLSTMGradMod1D(
            state_channels=T, hidden_dim=hidden_dim, rngs=nnx.Rngs(k2)
        )

        x0 = batch_1d.input * batch_1d.mask
        lstm = LSTMState1D.zeros(B, hidden_dim, N)

        state = SolverState1D(x=x0, lstm=lstm, step=0)
        new_state = solver_step_1d(state, batch_1d, prior, grad_mod)
        assert new_state.step == 1

    def test_state_changes_after_step(self, rng, batch_1d):
        from vardax import BilinAEPrior1D, ConvLSTMGradMod1D

        B, T, N = batch_1d.input.shape
        hidden_dim = 16
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(state_dim=N, latent_dim=8, n_time=T, rngs=nnx.Rngs(k1))
        grad_mod = ConvLSTMGradMod1D(
            state_channels=T, hidden_dim=hidden_dim, rngs=nnx.Rngs(k2)
        )

        x0 = batch_1d.input * batch_1d.mask
        lstm = LSTMState1D.zeros(B, hidden_dim, N)

        state = SolverState1D(x=x0, lstm=lstm, step=0)
        new_state = solver_step_1d(state, batch_1d, prior, grad_mod)
        # State should differ after the step
        assert not jnp.allclose(state.x, new_state.x)


class TestFpSolverStep1D:
    def test_output_shape(self, batch_1d):
        x = batch_1d.input * batch_1d.mask
        identity_fn = lambda x_: x_
        x_new = fp_solver_step_1d(x, batch_1d, identity_fn)
        assert x_new.shape == x.shape

    def test_observed_locations_equal_input(self, batch_1d):
        """At observed locations (mask==1), result should equal the input."""
        x = jnp.zeros_like(batch_1d.input)
        identity_fn = lambda x_: x_
        x_new = fp_solver_step_1d(x, batch_1d, identity_fn)
        mask = batch_1d.mask.astype(bool)
        assert jnp.allclose(x_new[mask], batch_1d.input[mask])


class TestSolve4dvarnet1dFixedpoint:
    def test_output_shape(self, rng, batch_1d):
        from vardax import BilinAEPrior1D

        B, T, N = batch_1d.input.shape
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(rng))

        result = solve_4dvarnet_1d_fixedpoint(batch_1d, prior, n_fp_steps=5)
        assert result.shape == (B, T, N)

    def test_zero_steps_returns_masked_input(self, rng, batch_1d):
        from vardax import BilinAEPrior1D

        B, T, N = batch_1d.input.shape
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(rng))
        x0 = batch_1d.input * batch_1d.mask

        result = solve_4dvarnet_1d_fixedpoint(batch_1d, prior, n_fp_steps=0)
        assert result.shape == (B, T, N)
        assert jnp.allclose(result, x0)


class TestOneStepSolve4dvarnet1D:
    def test_output_shape(self, rng, batch_1d):
        from vardax import BilinAEPrior1D, ConvLSTMGradMod1D
        from vardax._src.solver import one_step_solve_4dvarnet_1d

        B, T, N = batch_1d.input.shape
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(k1))
        grad_mod = ConvLSTMGradMod1D(state_channels=T, hidden_dim=16, rngs=nnx.Rngs(k2))

        result = one_step_solve_4dvarnet_1d(
            batch_1d, prior, grad_mod, n_steps=5, hidden_dim=16
        )
        assert result.shape == (B, T, N)

    def test_output_differs_from_initial(self, rng, batch_1d):
        from vardax import BilinAEPrior1D, ConvLSTMGradMod1D
        from vardax._src.solver import one_step_solve_4dvarnet_1d

        _B, T, N = batch_1d.input.shape
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(k1))
        grad_mod = ConvLSTMGradMod1D(state_channels=T, hidden_dim=16, rngs=nnx.Rngs(k2))

        x0 = batch_1d.input * batch_1d.mask
        result = one_step_solve_4dvarnet_1d(
            batch_1d, prior, grad_mod, n_steps=5, hidden_dim=16
        )
        assert not jnp.allclose(result, x0)

    def test_gradients_flow(self, rng, batch_1d):
        """Verify that gradients w.r.t. model parameters are non-zero."""
        from vardax import BilinAEPrior1D, ConvLSTMGradMod1D
        from vardax._src.solver import one_step_solve_4dvarnet_1d

        _B, T, N = batch_1d.input.shape
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(k1))
        grad_mod = ConvLSTMGradMod1D(state_channels=T, hidden_dim=16, rngs=nnx.Rngs(k2))

        def loss_fn(prior, grad_mod):
            x_hat = one_step_solve_4dvarnet_1d(
                batch_1d, prior, grad_mod, n_steps=3, hidden_dim=16
            )
            return jnp.mean((x_hat - batch_1d.target) ** 2)

        grads = nnx.grad(loss_fn, argnums=(0, 1))(prior, grad_mod)
        prior_grads, grad_mod_grads = grads
        # at least some parameter gradients should be non-zero
        prior_leaves = jax.tree_util.tree_leaves(nnx.state(prior_grads))
        assert any(jnp.any(g != 0) for g in prior_leaves)
        grad_mod_leaves = jax.tree_util.tree_leaves(nnx.state(grad_mod_grads))
        assert any(jnp.any(g != 0) for g in grad_mod_leaves)

    def test_single_step_equals_solver_step(self, rng, batch_1d):
        """With n_steps=1, one-step solve should equal a single solver_step."""
        from vardax import (
            BilinAEPrior1D,
            ConvLSTMGradMod1D,
            LSTMState1D,
            SolverState1D,
        )
        from vardax._src.solver import one_step_solve_4dvarnet_1d, solver_step_1d

        B, T, N = batch_1d.input.shape
        k1, k2 = jax.random.split(rng)
        prior = BilinAEPrior1D(state_dim=N, latent_dim=4, n_time=T, rngs=nnx.Rngs(k1))
        grad_mod = ConvLSTMGradMod1D(state_channels=T, hidden_dim=16, rngs=nnx.Rngs(k2))

        result_one_step = one_step_solve_4dvarnet_1d(
            batch_1d, prior, grad_mod, n_steps=1, hidden_dim=16
        )

        x0 = batch_1d.input * batch_1d.mask
        lstm = LSTMState1D.zeros(B, 16, N)
        state = SolverState1D(x=x0, lstm=lstm, step=0)
        state = solver_step_1d(state, batch_1d, prior, grad_mod)

        assert jnp.allclose(result_one_step, state.x, atol=1e-5)
