"""Tests for vardax._src.utils.masks."""

import numpy as np

from vardax._src.utils.masks import feature_mask, random_mask, regular_mask
from vardax._src.utils.patches import extract_patches, trajectory_to_xr_dataset


def _make_patched_ds(n_patches=10, n_timesteps=20, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((200, n_features)).astype(np.float32)
    time_coords = np.arange(200, dtype=float)
    ds = trajectory_to_xr_dataset(states, time_coords)
    return extract_patches(ds, n_patches=n_patches, n_timesteps=n_timesteps, seed=seed)


class TestRandomMask:
    def test_mask_exists(self):
        ds = _make_patched_ds()
        result = random_mask(ds, missing_rate=0.5)
        assert "mask" in result

    def test_mask_shape(self):
        ds = _make_patched_ds()
        result = random_mask(ds, missing_rate=0.5)
        assert result["mask"].shape == ds["state"].shape

    def test_mask_values_binary(self):
        ds = _make_patched_ds()
        result = random_mask(ds, missing_rate=0.5)
        vals = result["mask"].values
        assert set(np.unique(vals)).issubset({0.0, 1.0})

    def test_missing_rate_approx(self):
        ds = _make_patched_ds(n_patches=50, n_timesteps=50)
        missing_rate = 0.3
        result = random_mask(ds, missing_rate=missing_rate, seed=42)
        actual_missing = 1.0 - float(result["mask"].values.mean())
        assert abs(actual_missing - missing_rate) < 0.05

    def test_deterministic_with_seed(self):
        ds = _make_patched_ds()
        r1 = random_mask(ds, seed=0)
        r2 = random_mask(ds, seed=0)
        np.testing.assert_array_equal(r1["mask"].values, r2["mask"].values)


class TestRegularMask:
    def test_mask_exists(self):
        ds = _make_patched_ds()
        result = regular_mask(ds, obs_interval=4)
        assert "mask" in result

    def test_observation_density(self):
        n_timesteps = 20
        obs_interval = 4
        ds = _make_patched_ds(n_timesteps=n_timesteps)
        result = regular_mask(ds, obs_interval=obs_interval)
        # Along time axis, every obs_interval-th step is observed
        time_mask = result["mask"].values[0, :, 0]  # first patch, first feature
        expected_obs = len(range(0, n_timesteps, obs_interval))
        assert int(time_mask.sum()) == expected_obs

    def test_exact_density(self):
        """1/obs_interval of time steps should be observed."""
        ds = _make_patched_ds(n_timesteps=20)
        obs_interval = 5
        result = regular_mask(ds, obs_interval=obs_interval)
        time_slice = result["mask"].values[0, :, 0]
        # positions 0, 5, 10, 15 → 4 observed
        expected = np.zeros(20, dtype=np.float32)
        expected[::obs_interval] = 1.0
        np.testing.assert_array_equal(time_slice, expected)


class TestFeatureMask:
    def test_mask_features_exists(self):
        ds = _make_patched_ds(n_features=3)
        result = feature_mask(ds, observed_dims=["x0", "x2"])
        assert "mask_features" in result

    def test_observed_dims_are_one(self):
        ds = _make_patched_ds(n_features=3)
        result = feature_mask(ds, observed_dims=["x0", "x2"])
        vals = result["mask_features"].values[0, 0, :]  # first patch, first time
        assert vals[0] == 1.0  # x0 observed
        assert vals[1] == 0.0  # x1 not observed
        assert vals[2] == 1.0  # x2 observed

    def test_no_observed_dims(self):
        ds = _make_patched_ds(n_features=3)
        result = feature_mask(ds, observed_dims=[])
        assert float(result["mask_features"].values.sum()) == 0.0

    def test_all_observed_dims(self):
        ds = _make_patched_ds(n_features=3)
        result = feature_mask(ds, observed_dims=["x0", "x1", "x2"])
        assert float(result["mask_features"].values.min()) == 1.0
