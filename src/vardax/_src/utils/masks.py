"""Observation mask generation for xarray Datasets."""

from __future__ import annotations

import numpy as np
import xarray as xr


def random_mask(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    missing_rate: float = 0.5,
    seed: int = 0,
) -> xr.Dataset:
    """Add a random binary observation mask to the dataset.

    A value of ``1`` means *observed*; ``0`` means *missing*.

    Parameters
    ----------
    ds:
        Dataset containing ``variable``.
    variable:
        Name of the variable to mask.
    missing_rate:
        Fraction of values to mark as missing (in ``[0, 1)``).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with an additional ``"mask"`` coordinate that shares the
        same dimensions as ``variable``.
    """
    rng = np.random.default_rng(seed)
    da = ds[variable]
    mask_values = (rng.random(da.shape) >= missing_rate).astype(np.float32)
    mask_da = xr.DataArray(mask_values, dims=da.dims, coords=da.coords)
    return ds.assign(mask=mask_da)


def regular_mask(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    obs_interval: int = 2,
) -> xr.Dataset:
    """Add a regular (periodic) observation mask to the dataset.

    Every ``obs_interval``-th timestep along the ``time`` dimension is
    marked as observed (``1``); all others are ``0``.

    Parameters
    ----------
    ds:
        Dataset containing ``variable``.
    variable:
        Name of the variable to mask.
    obs_interval:
        Spacing between observed timesteps (every ``obs_interval``-th step
        is observed).

    Returns
    -------
    xr.Dataset
        Dataset with an additional ``"mask"`` coordinate.
    """
    da = ds[variable]
    time_size = da.sizes["time"]
    time_mask = np.zeros(time_size, dtype=np.float32)
    time_mask[::obs_interval] = 1.0

    # Broadcast along all dims
    shape = da.shape
    dims = da.dims
    time_idx = list(dims).index("time")

    # Build broadcastable shape
    broadcast_shape = [1] * len(dims)
    broadcast_shape[time_idx] = time_size
    mask_values = time_mask.reshape(broadcast_shape) * np.ones(shape, dtype=np.float32)

    mask_da = xr.DataArray(mask_values, dims=dims, coords=da.coords)
    return ds.assign(mask=mask_da)


def feature_mask(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    coord: str = "feature",
    observed_dims: list[str],
) -> xr.Dataset:
    """Add a feature-dimension mask to the dataset.

    Parameters
    ----------
    ds:
        Dataset containing ``variable``.
    variable:
        Name of the variable to mask.
    coord:
        Name of the coordinate that enumerates feature dimensions.
    observed_dims:
        Feature names that should be marked as *observed* (``1``); all
        others are ``0``.

    Returns
    -------
    xr.Dataset
        Dataset with an additional ``"mask_features"`` coordinate sharing
        the same dimensions as ``variable``.
    """
    da = ds[variable]
    feature_values = da.coords[coord].values
    feature_mask_1d = np.array(
        [1.0 if f in observed_dims else 0.0 for f in feature_values],
        dtype=np.float32,
    )

    dims = da.dims
    shape = da.shape
    feature_idx = list(dims).index(coord)

    broadcast_shape = [1] * len(dims)
    broadcast_shape[feature_idx] = len(feature_values)
    mask_values = feature_mask_1d.reshape(broadcast_shape) * np.ones(
        shape, dtype=np.float32
    )

    mask_da = xr.DataArray(mask_values, dims=dims, coords=da.coords)
    return ds.assign(mask_features=mask_da)
