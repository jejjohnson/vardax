"""Tests for vardax._src.grad_mod."""

import jax.numpy as jnp

from vardax import ConvLSTMGradMod1D, ConvLSTMGradMod2D, LSTMState1D, LSTMState2D


class TestConvLSTMGradMod1D:
    def test_output_shapes(self, rng, batch_1d):
        B, T, N = batch_1d.input.shape
        hidden_dim = 16
        model = ConvLSTMGradMod1D(state_channels=T, hidden_dim=hidden_dim, key=rng)
        lstm = LSTMState1D.zeros(B, hidden_dim, N)
        update, new_lstm = model(batch_1d.input, batch_1d.input, lstm)
        assert update.shape == batch_1d.input.shape
        assert new_lstm.h.shape == (B, hidden_dim, N)
        assert new_lstm.c.shape == (B, hidden_dim, N)

    def test_lstm_state_changes(self, rng, batch_1d):
        B, T, N = batch_1d.input.shape
        hidden_dim = 16
        model = ConvLSTMGradMod1D(state_channels=T, hidden_dim=hidden_dim, key=rng)
        lstm = LSTMState1D.zeros(B, hidden_dim, N)
        _, new_lstm = model(batch_1d.input, batch_1d.input, lstm)
        # Hidden state should have changed from zero
        assert not jnp.allclose(new_lstm.h, jnp.zeros_like(new_lstm.h))


class TestConvLSTMGradMod2D:
    def test_output_shapes(self, rng, batch_2d):
        B, T, H, W = batch_2d.input.shape
        hidden_dim = 8
        model = ConvLSTMGradMod2D(state_channels=T, hidden_dim=hidden_dim, key=rng)
        lstm = LSTMState2D.zeros(B, hidden_dim, H, W)
        update, new_lstm = model(batch_2d.input, batch_2d.input, lstm)
        assert update.shape == batch_2d.input.shape
        assert new_lstm.h.shape == (B, hidden_dim, H, W)
        assert new_lstm.c.shape == (B, hidden_dim, H, W)
