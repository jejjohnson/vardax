"""Tests for vardax._src.model."""

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from vardax import FourDVarNet1D, FourDVarNet2D
from vardax.adjoints import OneStepAdjoint


class TestFourDVarNet1D:
    def test_output_shape(self, rng, batch_1d):
        B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=2,
            key=rng,
        )
        out = model(batch_1d)
        assert out.shape == (B, T, N)


class TestFourDVarNet2D:
    def test_output_shape(self, rng, batch_2d):
        B, T, H, W = batch_2d.input.shape
        model = FourDVarNet2D(
            n_time=T,
            height=H,
            width=W,
            latent_dim=8,
            hidden_dim=8,
            n_solver_steps=2,
            key=rng,
        )
        out = model(batch_2d)
        assert out.shape == (B, T, H, W)

    def test_implicit_adjoint_raises(self, rng, batch_2d):
        _B, T, H, W = batch_2d.input.shape
        model = FourDVarNet2D(
            n_time=T,
            height=H,
            width=W,
            latent_dim=8,
            hidden_dim=8,
            n_solver_steps=2,
            solver_adjoint=optx.ImplicitAdjoint(),
            key=rng,
        )
        with pytest.raises(NotImplementedError):
            model(batch_2d)


class TestFourDVarNet1DAdjointDispatch:
    """Test that ``solver_adjoint`` selects the correct inner path."""

    @pytest.mark.parametrize(
        "adjoint",
        [
            optx.RecursiveCheckpointAdjoint(),
            OneStepAdjoint(),
            optx.ImplicitAdjoint(),
        ],
        ids=["recursive_checkpoint", "one_step", "implicit"],
    )
    def test_output_shape(self, rng, batch_1d, adjoint):
        B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=3,
            solver_adjoint=adjoint,
            key=rng,
        )
        out = model(batch_1d)
        assert out.shape == (B, T, N)

    def test_one_step_gradients_are_finite(self, rng, batch_1d):
        import equinox as eqx

        _B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=3,
            solver_adjoint=OneStepAdjoint(),
            key=rng,
        )

        def loss_fn(model):
            x_hat = model(batch_1d)
            return jnp.mean((x_hat - batch_1d.target) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_default_is_recursive_checkpoint(self, rng, batch_1d):
        """No ``solver_adjoint`` arg → default to ``RecursiveCheckpointAdjoint``."""
        _B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=3,
            key=rng,
        )
        assert isinstance(model.solver_adjoint, optx.RecursiveCheckpointAdjoint)
