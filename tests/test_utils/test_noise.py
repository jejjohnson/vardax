"""Tests for vardax._src.utils.noise."""

import numpy as np

from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.patches import extract_patches, trajectory_to_xr_dataset


def _make_patched_ds(seed=0):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((200, 3)).astype(np.float32)
    time_coords = np.arange(200, dtype=float)
    ds = trajectory_to_xr_dataset(states, time_coords)
    return extract_patches(ds, n_patches=10, n_timesteps=20, seed=seed)


class TestAddGaussianNoise:
    def test_output_variable_exists(self):
        ds = _make_patched_ds()
        result = add_gaussian_noise(ds, sigma=0.1)
        assert "obs" in result

    def test_custom_name(self):
        ds = _make_patched_ds()
        result = add_gaussian_noise(ds, sigma=0.1, name="noisy_state")
        assert "noisy_state" in result

    def test_output_shape(self):
        ds = _make_patched_ds()
        result = add_gaussian_noise(ds, sigma=0.1)
        assert result["obs"].shape == ds["state"].shape

    def test_noise_magnitude(self):
        """Std of residuals should be ≈ sigma."""
        ds = _make_patched_ds()
        sigma = 0.5
        result = add_gaussian_noise(ds, sigma=sigma, seed=0)
        residuals = result["obs"].values - ds["state"].values
        measured_std = float(np.std(residuals))
        assert abs(measured_std - sigma) < 0.05

    def test_deterministic_with_seed(self):
        ds = _make_patched_ds()
        r1 = add_gaussian_noise(ds, sigma=0.1, seed=123)
        r2 = add_gaussian_noise(ds, sigma=0.1, seed=123)
        np.testing.assert_array_equal(r1["obs"].values, r2["obs"].values)

    def test_different_seeds_differ(self):
        ds = _make_patched_ds()
        r1 = add_gaussian_noise(ds, sigma=0.1, seed=0)
        r2 = add_gaussian_noise(ds, sigma=0.1, seed=1)
        assert not np.allclose(r1["obs"].values, r2["obs"].values)

    def test_zero_sigma_returns_original(self):
        ds = _make_patched_ds()
        result = add_gaussian_noise(ds, sigma=0.0)
        np.testing.assert_allclose(
            result["obs"].values,
            ds["state"].values.astype(np.float32),
            atol=1e-6,
        )
