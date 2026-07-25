"""Tests for vardax._src.utils.patches."""

import numpy as np
import pytest

from vardax._src.utils.patches import extract_patches, trajectory_to_xr_dataset


def _make_trajectory(n_time: int = 100, n_features: int = 3):
    rng = np.random.default_rng(0)
    states = rng.standard_normal((n_time, n_features)).astype(np.float32)
    time_coords = np.linspace(0.0, 1.0, n_time)
    return states, time_coords


class TestTrajectoryToXrDataset:
    def test_dims(self):
        states, time_coords = _make_trajectory()
        ds = trajectory_to_xr_dataset(states, time_coords)
        assert "state" in ds
        assert ds["state"].dims == ("time", "feature")

    def test_shape(self):
        n_time, n_features = 50, 3
        states, time_coords = _make_trajectory(n_time, n_features)
        ds = trajectory_to_xr_dataset(states, time_coords)
        assert ds["state"].shape == (n_time, n_features)

    def test_default_feature_names(self):
        states, time_coords = _make_trajectory(n_time=10, n_features=3)
        ds = trajectory_to_xr_dataset(states, time_coords)
        assert list(ds["state"].coords["feature"].values) == ["x0", "x1", "x2"]

    def test_custom_feature_names(self):
        states, time_coords = _make_trajectory(n_time=10, n_features=3)
        ds = trajectory_to_xr_dataset(
            states, time_coords, feature_names=["X", "Y", "Z"]
        )
        assert list(ds["state"].coords["feature"].values) == ["X", "Y", "Z"]

    def test_time_coords_preserved(self):
        states, time_coords = _make_trajectory(n_time=10)
        ds = trajectory_to_xr_dataset(states, time_coords)
        np.testing.assert_allclose(ds["state"].coords["time"].values, time_coords)


class TestExtractPatches:
    def test_patch_count(self):
        states, time_coords = _make_trajectory(n_time=200)
        ds = trajectory_to_xr_dataset(states, time_coords)
        result = extract_patches(ds, n_patches=10, n_timesteps=20)
        assert result["state"].sizes["patch"] == 10

    def test_patch_length(self):
        states, time_coords = _make_trajectory(n_time=200)
        ds = trajectory_to_xr_dataset(states, time_coords)
        n_timesteps = 30
        result = extract_patches(ds, n_patches=5, n_timesteps=n_timesteps)
        assert result["state"].sizes["time"] == n_timesteps

    def test_dims(self):
        states, time_coords = _make_trajectory(n_time=100)
        ds = trajectory_to_xr_dataset(states, time_coords)
        result = extract_patches(ds, n_patches=4, n_timesteps=10)
        assert result["state"].dims == ("patch", "time", "feature")

    def test_temporal_ordering_preserved(self):
        """Values within each patch must be in time order."""
        n_time = 100
        # Use linearly increasing values so we can check ordering
        states = np.arange(n_time * 3, dtype=np.float32).reshape(n_time, 3)
        time_coords = np.arange(n_time, dtype=float)
        ds = trajectory_to_xr_dataset(states, time_coords)
        result = extract_patches(ds, n_patches=5, n_timesteps=10, seed=42)
        patch_data = result["state"].values  # (P, T, F)
        for p in range(patch_data.shape[0]):
            # Each feature's values should be strictly increasing
            for f in range(patch_data.shape[2]):
                vals = patch_data[p, :, f]
                assert np.all(np.diff(vals) > 0)

    def test_too_short_raises(self):
        states, time_coords = _make_trajectory(n_time=10)
        ds = trajectory_to_xr_dataset(states, time_coords)
        with pytest.raises(ValueError):
            extract_patches(ds, n_patches=2, n_timesteps=20)

    def test_deterministic_with_seed(self):
        states, time_coords = _make_trajectory(n_time=200)
        ds = trajectory_to_xr_dataset(states, time_coords)
        r1 = extract_patches(ds, n_patches=5, n_timesteps=10, seed=0)
        r2 = extract_patches(ds, n_patches=5, n_timesteps=10, seed=0)
        np.testing.assert_array_equal(r1["state"].values, r2["state"].values)


class TestTimePatches:
    def test_pairs(self):
        import jax.numpy as jnp

        from vardax import time_patches

        out = time_patches(jnp.array([0.0, 0.1, 0.2, 0.3]))
        assert out.shape == (3, 2)
        assert np.allclose(np.asarray(out), [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]])


class TestXarrayIsLazy:
    def test_core_import_does_not_load_xarray(self):
        # xarray lives in the optional [data] extra; the core import path
        # (vardax.__init__ -> priors_dynamical -> utils.patches) must not
        # pull it in (Codex P1 on PR #60).
        import subprocess
        import sys

        code = "import vardax, sys; sys.exit(1 if 'xarray' in sys.modules else 0)"
        result = subprocess.run([sys.executable, "-c", code], check=False)
        assert result.returncode == 0, "plain `import vardax` loaded xarray"
