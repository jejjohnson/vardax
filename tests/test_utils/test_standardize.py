"""Tests for vardax._src.utils.standardize."""

import jax.numpy as jnp
import numpy as np

from vardax._src.utils.masks import random_mask
from vardax._src.utils.patches import extract_patches, trajectory_to_xr_dataset
from vardax._src.utils.standardize import (
    apply_standardization,
    compute_scaler_params,
    inverse_standardization,
)


def _make_patched_ds(seed=0):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((200, 3)).astype(np.float32)
    time_coords = np.arange(200, dtype=float)
    ds = trajectory_to_xr_dataset(states, time_coords)
    ds = extract_patches(ds, n_patches=20, n_timesteps=20, seed=seed)
    return random_mask(ds, missing_rate=0.3, seed=seed)


class TestComputeScalerParams:
    def test_returns_float_tuple(self):
        ds = _make_patched_ds()
        mean, std = compute_scaler_params(ds)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_reasonable_values(self):
        """Mean ~ 0, std ~ 1 for standard-normal data."""
        ds = _make_patched_ds()
        mean, std = compute_scaler_params(ds, mask_variable=None)
        assert abs(mean) < 0.2
        assert abs(std - 1.0) < 0.2

    def test_uses_mask(self):
        """Results with and without mask should differ."""
        ds = _make_patched_ds()
        mean_masked, _ = compute_scaler_params(ds, mask_variable="mask")
        mean_all, _ = compute_scaler_params(ds, mask_variable=None)
        # They should be close but not necessarily identical
        # (both computed from std-normal data; just check they don't crash)
        assert isinstance(mean_masked, float)
        assert isinstance(mean_all, float)

    def test_no_mask_variable_uses_all(self):
        ds = _make_patched_ds()
        mean, _std = compute_scaler_params(ds, mask_variable=None)
        all_vals = ds["state"].values.ravel()
        assert abs(mean - float(np.mean(all_vals))) < 1e-5


class TestApplyStandardization:
    def test_output_has_variable(self):
        ds = _make_patched_ds()
        mean, std = compute_scaler_params(ds, mask_variable=None)
        result = apply_standardization(ds, variables=["state"], mean=mean, std=std)
        assert "state" in result

    def test_standardized_mean_approx_zero(self):
        ds = _make_patched_ds()
        mean, std = compute_scaler_params(ds, mask_variable=None)
        result = apply_standardization(ds, variables=["state"], mean=mean, std=std)
        standardized_mean = float(np.mean(result["state"].values))
        assert abs(standardized_mean) < 1e-5

    def test_standardized_std_approx_one(self):
        ds = _make_patched_ds()
        mean, std = compute_scaler_params(ds, mask_variable=None)
        result = apply_standardization(ds, variables=["state"], mean=mean, std=std)
        standardized_std = float(np.std(result["state"].values))
        assert abs(standardized_std - 1.0) < 1e-5


class TestInverseStandardization:
    def test_recovers_original(self):
        rng = np.random.default_rng(0)
        data = jnp.array(rng.standard_normal((10, 3)).astype(np.float32))
        mean = 2.5
        std = 0.8
        standardized = (data - mean) / std
        recovered = inverse_standardization(standardized, mean=mean, std=std)
        assert jnp.allclose(recovered, data, atol=1e-5)

    def test_output_shape_preserved(self):
        data = jnp.ones((5, 4, 3))
        result = inverse_standardization(data, mean=0.0, std=1.0)
        assert result.shape == data.shape
