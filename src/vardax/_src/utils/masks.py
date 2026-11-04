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
    """Add a random binary observation mask m to the dataset.

        mᵢ = 1 if a uniform draw on [0, 1) is at least ``missing_rate``, else 0

    A value of 1 means observed; 0 means missing.

    Args:
        ds: Dataset containing ``variable``.
        variable: Name of the variable to mask.
        missing_rate: Fraction of values to mark as missing (in [0, 1)).
        seed: Random seed for reproducibility.

    Returns:
        Dataset with an additional ``"mask"`` variable that shares the same
        dimensions as ``variable``.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> state = xr.DataArray(np.zeros((4, 2)), dims=["time", "feature"])
        >>> ds = xr.Dataset({"state": state})
        >>> random_mask(ds, seed=1)["mask"].shape
        (4, 2)

        A zero missing rate observes everything:

        >>> bool((random_mask(ds, missing_rate=0.0)["mask"] == 1.0).all())
        True

        The same seed gives the same mask:

        >>> first = random_mask(ds, seed=3)["mask"]
        >>> bool((first == random_mask(ds, seed=3)["mask"]).all())
        True
    """
    rng = np.random.default_rng(seed)
    data_array = ds[variable]
    # mᵢ = 1 where the draw is at least the missing rate
    draws = rng.random(data_array.shape)
    mask_values = (draws >= missing_rate).astype(np.float32)
    mask_data_array = xr.DataArray(
        mask_values, dims=data_array.dims, coords=data_array.coords
    )
    return ds.assign(mask=mask_data_array)


def regular_mask(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    obs_interval: int = 2,
) -> xr.Dataset:
    """Add a regular (periodic) observation mask m to the dataset.

        mₜ = 1 if t mod ``obs_interval`` = 0, else 0

    Every ``obs_interval``-th step along the ``time`` dimension is marked as
    observed (1); all others are 0.

    Args:
        ds: Dataset containing ``variable``.
        variable: Name of the variable to mask.
        obs_interval: Spacing between observed time steps.

    Returns:
        Dataset with an additional ``"mask"`` variable.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> state = xr.DataArray(np.zeros((4, 2)), dims=["time", "feature"])
        >>> ds = xr.Dataset({"state": state})
        >>> regular_mask(ds)["mask"].values[:, 0].tolist()
        [1.0, 0.0, 1.0, 0.0]

        The mask broadcasts across the other dimensions, wherever ``time`` sits:

        >>> swapped = xr.Dataset({"state": state.transpose("feature", "time")})
        >>> regular_mask(swapped, obs_interval=3)["mask"].values[1].tolist()
        [1.0, 0.0, 0.0, 1.0]

        An interval of one observes every step:

        >>> bool((regular_mask(ds, obs_interval=1)["mask"] == 1.0).all())
        True
    """
    data_array = ds[variable]
    num_times = data_array.sizes["time"]
    time_mask = np.zeros(num_times, dtype=np.float32)
    time_mask[::obs_interval] = 1.0

    shape = data_array.shape
    dims = data_array.dims
    time_axis = list(dims).index("time")

    broadcast_shape = [1] * len(dims)
    broadcast_shape[time_axis] = num_times
    # reshape, not einx: einx cannot size a zero-length axis, and an empty dataset
    # must still get an empty mask
    mask_values = time_mask.reshape(broadcast_shape) * np.ones(shape, dtype=np.float32)

    mask_data_array = xr.DataArray(mask_values, dims=dims, coords=data_array.coords)
    return ds.assign(mask=mask_data_array)


def feature_mask(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    coord: str = "feature",
    observed_dims: list[str],
) -> xr.Dataset:
    """Add a feature-dimension mask to the dataset.

    Args:
        ds: Dataset containing ``variable``.
        variable: Name of the variable to mask.
        coord: Name of the coordinate that enumerates feature dimensions.
        observed_dims: Feature names that should be marked as observed (1); all
            others are 0.

    Returns:
        Dataset with an additional ``"mask_features"`` variable sharing the same
        dimensions as ``variable``.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> state = xr.DataArray(
        ...     np.zeros((3, 2)),
        ...     dims=["time", "feature"],
        ...     coords={"feature": ["x0", "x1"]},
        ... )
        >>> ds = xr.Dataset({"state": state})
        >>> feature_mask(ds, observed_dims=["x1"])["mask_features"].values[0].tolist()
        [0.0, 1.0]

        Naming no feature observes nothing:

        >>> bool((feature_mask(ds, observed_dims=[])["mask_features"] == 0.0).all())
        True
    """
    data_array = ds[variable]
    feature_values = data_array.coords[coord].values
    feature_mask_1d = np.array(
        [1.0 if feature in observed_dims else 0.0 for feature in feature_values],
        dtype=np.float32,
    )

    dims = data_array.dims
    shape = data_array.shape
    feature_axis = list(dims).index(coord)

    broadcast_shape = [1] * len(dims)
    broadcast_shape[feature_axis] = len(feature_values)
    # reshape, not einx: einx cannot size a zero-length axis, and an empty dataset
    # must still get an empty mask
    mask_values = feature_mask_1d.reshape(broadcast_shape) * np.ones(
        shape, dtype=np.float32
    )

    mask_data_array = xr.DataArray(mask_values, dims=dims, coords=data_array.coords)
    return ds.assign(mask_features=mask_data_array)
