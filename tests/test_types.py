"""Tests for vardax._src._types."""

import jax.numpy as jnp

from vardax import (
    LSTMState1D,
    LSTMState2D,
)


class TestBatch1D:
    def test_fields(self, batch_1d):
        assert hasattr(batch_1d, "input")
        assert hasattr(batch_1d, "mask")
        assert hasattr(batch_1d, "target")

    def test_shapes_consistent(self, batch_1d):
        assert batch_1d.input.shape == batch_1d.mask.shape
        assert batch_1d.input.shape == batch_1d.target.shape

    def test_is_named_tuple(self, batch_1d):
        assert isinstance(batch_1d, tuple)
        assert len(batch_1d) == 3


class TestBatch2D:
    def test_fields(self, batch_2d):
        assert hasattr(batch_2d, "input")
        assert hasattr(batch_2d, "mask")
        assert hasattr(batch_2d, "target")

    def test_shapes_consistent(self, batch_2d):
        assert batch_2d.input.shape == batch_2d.mask.shape
        assert batch_2d.input.shape == batch_2d.target.shape

    def test_ndim(self, batch_2d):
        assert batch_2d.input.ndim == 4


class TestBatch2DMultivar:
    def test_ndim(self, batch_2d_multivar):
        assert batch_2d_multivar.input.ndim == 5

    def test_channel_dim(self, batch_2d_multivar):
        # (B, T, C, H, W) — C is index 2
        assert batch_2d_multivar.input.shape[2] == 2


class TestLSTMState1D:
    def test_zeros(self):
        state = LSTMState1D.zeros(batch_size=2, hidden_dim=16, seq_len=8)
        assert state.h.shape == (2, 16, 8)
        assert state.c.shape == (2, 16, 8)
        assert jnp.all(state.h == 0)
        assert jnp.all(state.c == 0)


class TestLSTMState2D:
    def test_zeros(self):
        state = LSTMState2D.zeros(batch_size=2, hidden_dim=16, height=4, width=4)
        assert state.h.shape == (2, 16, 4, 4)
        assert state.c.shape == (2, 16, 4, 4)
        assert jnp.all(state.h == 0)
        assert jnp.all(state.c == 0)
