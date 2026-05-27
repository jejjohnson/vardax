"""Tests for vardax._src.model."""

import jax
import jax.numpy as jnp
import pytest

from vardax import FourDVarNet1D, FourDVarNet2D


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

    @pytest.mark.slow
    def test_output_changes_with_different_masks(self, rng, batch_1d):
        from vardax import Batch1D

        _, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=2,
            key=rng,
        )
        out1 = model(batch_1d)
        # All-zero mask
        batch_no_obs = Batch1D(
            input=batch_1d.input,
            mask=jnp.zeros_like(batch_1d.mask),
            target=batch_1d.target,
        )
        out2 = model(batch_no_obs)
        assert not jnp.allclose(out1, out2)


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

    def test_implicit_grad_mode_raises(self, rng, batch_2d):
        _B, T, H, W = batch_2d.input.shape
        model = FourDVarNet2D(
            n_time=T,
            height=H,
            width=W,
            latent_dim=8,
            hidden_dim=8,
            n_solver_steps=2,
            grad_mode="implicit",
            key=rng,
        )
        with pytest.raises(NotImplementedError):
            model(batch_2d)


class TestFourDVarNet1DGradMode:
    @pytest.mark.parametrize("grad_mode", ["unrolled", "one_step", "implicit"])
    def test_output_shape(self, rng, batch_1d, grad_mode):
        B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=3,
            grad_mode=grad_mode,
            key=rng,
        )
        out = model(batch_1d)
        assert out.shape == (B, T, N)

    def test_invalid_grad_mode_raises(self, rng, batch_1d):
        _B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=3,
            grad_mode="invalid",  # type: ignore[arg-type]
            key=rng,
        )
        with pytest.raises(ValueError, match="Unknown grad_mode"):
            model(batch_1d)

    def test_one_step_gradients_are_finite(self, rng, batch_1d):
        import equinox as eqx

        _B, T, N = batch_1d.input.shape
        model = FourDVarNet1D(
            state_dim=N,
            n_time=T,
            latent_dim=8,
            hidden_dim=16,
            n_solver_steps=3,
            grad_mode="one_step",
            key=rng,
        )

        def loss_fn(model):
            x_hat = model(batch_1d)
            return jnp.mean((x_hat - batch_1d.target) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
