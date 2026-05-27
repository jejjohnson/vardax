"""Tests for vardax._src.priors."""

# (removed: flax dropped in equinox migration)
import jax.numpy as jnp

from vardax import (
    BilinAEPrior1D,
    BilinAEPrior2D,
    BilinAEPrior2DMultivar,
    ConvAEPrior1D,
    L63Prior,
    L96Prior,
    MLPAEPrior1D,
)


class TestBilinAEPrior1D:
    def test_output_shape(self, rng, batch_1d):
        model = BilinAEPrior1D(state_dim=16, latent_dim=8, n_time=5, key=rng)
        out = model(batch_1d.input)
        assert out.shape == batch_1d.input.shape

    def test_encode_decode_shapes(self, rng, batch_1d):
        model = BilinAEPrior1D(state_dim=16, latent_dim=8, n_time=5, key=rng)
        z = model.encode(batch_1d.input)
        assert z.shape == (batch_1d.input.shape[0], 8)
        decoded = model.decode(z)
        assert decoded.shape == batch_1d.input.shape


class TestMLPAEPrior1D:
    def test_output_shape(self, rng, batch_1d):
        model = MLPAEPrior1D(
            state_dim=16, latent_dim=8, hidden_dim=32, n_time=5, key=rng
        )
        out = model(batch_1d.input)
        assert out.shape == batch_1d.input.shape


class TestBilinAEPrior2D:
    def test_output_shape(self, rng, batch_2d):
        _, T, H, W = batch_2d.input.shape
        model = BilinAEPrior2D(latent_dim=16, n_time=T, height=H, width=W, key=rng)
        out = model(batch_2d.input)
        assert out.shape == batch_2d.input.shape


class TestBilinAEPrior2DMultivar:
    def test_output_shape(self, rng, batch_2d_multivar):
        _, T, C, H, W = batch_2d_multivar.input.shape
        model = BilinAEPrior2DMultivar(
            latent_dim=16,
            n_time=T,
            n_channels=C,
            height=H,
            width=W,
            key=rng,
        )
        out = model(batch_2d_multivar.input)
        assert out.shape == batch_2d_multivar.input.shape


class TestL63Prior:
    def test_output_shape(self, rng):
        model = L63Prior(latent_dim=3, hidden_dim=16, state_dim=3, key=rng)
        x = jnp.ones((4, 3))
        out = model(x)
        assert out.shape == (4, 3)


class TestL96Prior:
    def test_output_shape(self, rng):
        model = L96Prior(latent_dim=16, hidden_dim=64, state_dim=40, key=rng)
        x = jnp.ones((4, 40))
        out = model(x)
        assert out.shape == (4, 40)

    def test_custom_state_size(self, rng):
        model = L96Prior(latent_dim=8, hidden_dim=32, state_dim=20, key=rng)
        x = jnp.ones((2, 20))
        out = model(x)
        assert out.shape == (2, 20)


class TestConvAEPrior1D:
    def test_output_shape(self, rng):
        model = ConvAEPrior1D(latent_channels=8, kernel_size=3, n_time=5, key=rng)
        x = jnp.ones((2, 5, 40))
        out = model(x)
        assert out.shape == (2, 5, 40)

    def test_small_spatial_dim(self, rng):
        model = ConvAEPrior1D(latent_channels=4, kernel_size=3, n_time=3, key=rng)
        x = jnp.ones((2, 3, 10))
        out = model(x)
        assert out.shape == (2, 3, 10)
