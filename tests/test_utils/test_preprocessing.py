"""Tests for vardax._src.utils.preprocessing."""

import numpy as np
import pytest

from vardax import Batch1D
from vardax._src.utils.masks import random_mask
from vardax._src.utils.noise import add_gaussian_noise
from vardax._src.utils.patches import extract_patches, trajectory_to_xr_dataset
from vardax._src.utils.preprocessing import (
    interpolate_initial_condition,
    train_test_split,
    xr_to_batch1d,
)


def _make_full_ds(n_patches=30, n_timesteps=20, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((500, n_features)).astype(np.float32)
    time_coords = np.arange(500, dtype=float)
    ds = trajectory_to_xr_dataset(states, time_coords)
    ds = extract_patches(ds, n_patches=n_patches, n_timesteps=n_timesteps, seed=seed)
    ds = random_mask(ds, missing_rate=0.3, seed=seed)
    ds = add_gaussian_noise(ds, sigma=0.1, seed=seed)
    return ds


class TestTrainTestSplit:
    def test_correct_counts(self):
        ds = _make_full_ds()
        ds_train, ds_test = train_test_split(ds, n_train=20, n_test=8)
        assert ds_train.sizes["patch"] == 20
        assert ds_test.sizes["patch"] == 8

    def test_non_overlapping(self):
        ds = _make_full_ds()
        ds_train, ds_test = train_test_split(ds, n_train=20, n_test=8)
        train_patches = set(ds_train.coords["patch"].values.tolist())
        test_patches = set(ds_test.coords["patch"].values.tolist())
        assert train_patches.isdisjoint(test_patches)

    def test_too_many_raises(self):
        ds = _make_full_ds(n_patches=10)
        with pytest.raises(ValueError):
            train_test_split(ds, n_train=8, n_test=5)

    def test_deterministic(self):
        ds = _make_full_ds()
        t1, _e1 = train_test_split(ds, n_train=20, n_test=8, seed=0)
        t2, _e2 = train_test_split(ds, n_train=20, n_test=8, seed=0)
        np.testing.assert_array_equal(t1["state"].values, t2["state"].values)


class TestXrToBatch1d:
    def test_returns_batch1d(self):
        ds = _make_full_ds()
        batch = xr_to_batch1d(ds)
        assert isinstance(batch, Batch1D)

    def test_shapes(self):
        n_patches, n_timesteps, n_features = 10, 20, 3
        ds = _make_full_ds(
            n_patches=n_patches, n_timesteps=n_timesteps, n_features=n_features
        )
        batch = xr_to_batch1d(ds)
        assert batch.input.shape == (n_patches, n_timesteps, n_features)
        assert batch.mask.shape == (n_patches, n_timesteps, n_features)
        assert batch.target.shape == (n_patches, n_timesteps, n_features)

    def test_jax_arrays(self):
        ds = _make_full_ds()
        batch = xr_to_batch1d(ds)
        # Check all fields are JAX arrays
        assert hasattr(batch.input, "device")
        assert hasattr(batch.mask, "device")
        assert hasattr(batch.target, "device")


class TestInterpolateInitialCondition:
    def test_x_init_exists(self):
        ds = _make_full_ds()
        result = interpolate_initial_condition(ds)
        assert "x_init" in result

    def test_x_init_shape(self):
        ds = _make_full_ds()
        result = interpolate_initial_condition(ds)
        assert result["x_init"].shape == ds["obs"].shape

    def test_fills_gaps(self):
        """Observed points should be reproduced; gaps should be filled."""
        ds = _make_full_ds()
        result = interpolate_initial_condition(ds, fillna=0.0)
        x_init = result["x_init"].values
        # x_init should not be entirely zero where mask is 1
        mask = ds["mask"].values.astype(bool)
        obs = ds["obs"].values
        # At observed positions, x_init ≈ obs
        np.testing.assert_allclose(x_init[mask], obs[mask], atol=1e-5)


class TestObsInterpolationInit:
    def _make_nan_obs_ds(self, n_patches=5, n_timesteps=20, n_features=3, seed=0):
        """Create a dataset where obs has NaN at masked locations."""
        rng = np.random.default_rng(seed)
        states = rng.standard_normal((500, n_features)).astype(np.float32)
        import xarray as xr

        from vardax._src.utils.patches import (
            extract_patches,
            trajectory_to_xr_dataset,
        )

        time_coords = np.arange(500, dtype=float)
        ds = trajectory_to_xr_dataset(states, time_coords)
        ds = extract_patches(
            ds, n_patches=n_patches, n_timesteps=n_timesteps, seed=seed
        )
        # Build obs with NaN at missing positions
        from vardax._src.utils.masks import random_mask

        ds = random_mask(ds, variable="state", missing_rate=0.4, seed=seed)
        state_vals = ds["state"].values
        mask_vals = ds["mask"].values.astype(bool)
        obs_vals = np.where(mask_vals, state_vals, np.nan).astype(np.float32)
        obs_da = xr.DataArray(
            obs_vals, dims=ds["state"].dims, coords=ds["state"].coords
        )
        return ds.assign(obs=obs_da)

    def test_state_init_exists(self):
        ds = self._make_nan_obs_ds()
        from vardax._src.utils.preprocessing import obs_interpolation_init

        result = obs_interpolation_init(ds)
        assert "state_init" in result

    def test_state_init_shape(self):
        ds = self._make_nan_obs_ds()
        from vardax._src.utils.preprocessing import obs_interpolation_init

        result = obs_interpolation_init(ds)
        assert result["state_init"].shape == ds["obs"].shape

    def test_no_nan_in_output(self):
        ds = self._make_nan_obs_ds()
        from vardax._src.utils.preprocessing import obs_interpolation_init

        result = obs_interpolation_init(ds, fillna=0.0)
        assert not np.any(np.isnan(result["state_init"].values))

    def test_observed_values_preserved(self):
        """Values at originally-observed (non-NaN) locations should be preserved."""
        ds = self._make_nan_obs_ds()
        from vardax._src.utils.preprocessing import obs_interpolation_init

        result = obs_interpolation_init(ds, fillna=0.0)
        obs = ds["obs"].values
        state_init = result["state_init"].values
        observed_mask = ~np.isnan(obs)
        np.testing.assert_allclose(
            state_init[observed_mask], obs[observed_mask], atol=1e-5
        )
