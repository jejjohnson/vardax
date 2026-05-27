"""Smoke tests for vardax._src.utils.viz."""

import numpy as np
import pytest

from vardax._src.utils.viz import (
    plot_3d_attractor,
    plot_l96_grid,
    plot_l96_trajectories,
    plot_reconstruction_comparison,
    plot_state_grid,
    plot_trajectories,
)


@pytest.fixture(autouse=True)
def _no_display(monkeypatch):
    """Use a non-interactive matplotlib backend for tests."""
    import matplotlib

    matplotlib.use("Agg")


def _dummy_states(T=50, N=3):
    rng = np.random.default_rng(0)
    return rng.standard_normal((T, N)).astype(np.float32)


def _dummy_time(T=50):
    return np.linspace(0, 1, T)


class TestPlot3dAttractor:
    def test_returns_figure_and_axes(self):
        import matplotlib.figure

        states = _dummy_states()
        fig, _ax = plot_3d_attractor(states)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_dim_indices(self):
        states = _dummy_states(N=4)
        fig, _ax = plot_3d_attractor(states, dim_indices=(0, 2, 3))
        assert fig is not None


class TestPlotStateGrid:
    def test_returns_figure_and_axes(self):
        import matplotlib.figure

        states = _dummy_states()
        time = _dummy_time()
        fig, _ax = plot_state_grid(states, time)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotTrajectories:
    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_returns_figure(self, orientation):
        import matplotlib.figure

        states = _dummy_states()
        time = _dummy_time()
        fig, _ax = plot_trajectories(states, time, orientation=orientation)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_default_orientation(self):
        import matplotlib.figure

        states = _dummy_states()
        time = _dummy_time()
        fig, _ax = plot_trajectories(states, time)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotReconstructionComparison:
    def test_returns_figure_and_axes(self):
        import matplotlib.figure

        rng = np.random.default_rng(0)
        B, T, N = 4, 20, 3
        target = rng.standard_normal((B, T, N)).astype(np.float32)
        masked = target.copy()
        masked[:, ::3, :] = 0.0
        recon = target + 0.01 * rng.standard_normal((B, T, N)).astype(np.float32)
        fig, axes = plot_reconstruction_comparison(target, masked, recon, sample_idx=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(axes) == 3


class TestPlotL96Grid:
    def test_returns_figure_and_axes(self):
        import matplotlib.figure

        states = _dummy_states(T=50, N=40)
        time = _dummy_time(T=50)
        fig, _ax = plot_l96_grid(states, time)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotL96Trajectories:
    def test_returns_figure_and_axes(self):
        import matplotlib.figure

        states = _dummy_states(T=50, N=40)
        time = _dummy_time(T=50)
        fig, _ax = plot_l96_trajectories(states, time)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_n_vars(self):
        import matplotlib.figure

        states = _dummy_states(T=50, N=40)
        time = _dummy_time(T=50)
        fig, _ax = plot_l96_trajectories(states, time, n_vars=3)
        assert isinstance(fig, matplotlib.figure.Figure)
