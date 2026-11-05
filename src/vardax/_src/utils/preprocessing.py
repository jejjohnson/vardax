"""xarray to JAX conversion, train/test split and gap-filled initial conditions."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import xarray as xr

from vardax._src._types import Batch1D
from vardax._src.errors import ConfigurationError, DataError


def train_test_split(
    ds: xr.Dataset,
    *,
    n_train: int,
    n_test: int,
    seed: int = 0,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Randomly split a patched dataset along the ``patch`` dimension.

    Args:
        ds: Dataset with a ``patch`` dimension.
        n_train: Number of patches for the training set.
        n_test: Number of patches for the test set.
        seed: Random seed for reproducibility.

    Returns:
        ``(ds_train, ds_test)``: the training and test subsets.

    Raises:
        DataError: If ``n_train + n_test`` exceeds the available patches.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> state = xr.DataArray(np.zeros((5, 3, 2)), dims=["patch", "time", "feature"])
        >>> ds = xr.Dataset({"state": state})
        >>> train, test = train_test_split(ds, n_train=3, n_test=2)
        >>> train.sizes["patch"], test.sizes["patch"]
        (3, 2)

        A split may leave patches unused:

        >>> train, test = train_test_split(ds, n_train=1, n_test=1)
        >>> train.sizes["patch"] + test.sizes["patch"]
        2

        Asking for more patches than exist is rejected:

        >>> train_test_split(
        ...     ds, n_train=5, n_test=1
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        DataError: n_train + n_test (6) exceeds available patches (5)
    """
    rng = np.random.default_rng(seed)
    num_patches = ds.sizes["patch"]

    if n_train + n_test > num_patches:
        msg = (
            f"n_train + n_test ({n_train + n_test}) exceeds available "
            f"patches ({num_patches})"
        )
        raise DataError(msg)

    indices = rng.permutation(num_patches)
    train_indices = indices[:n_train]
    test_indices = indices[n_train : n_train + n_test]

    ds_train = ds.isel(patch=sorted(train_indices.tolist()))
    ds_test = ds.isel(patch=sorted(test_indices.tolist()))
    return ds_train, ds_test


def xr_to_batch1d(
    ds: xr.Dataset,
    *,
    state_var: str = "state",
    obs_var: str = "obs",
    mask_var: str = "mask",
) -> Batch1D:
    """Convert an xarray Dataset to a [`Batch1D`][vardax.Batch1D].

    The dataset is expected to have dims ``(patch, time, feature)``; the feature
    dimension is treated as the spatial ``N`` axis of
    [`Batch1D`][vardax.Batch1D].

    Args:
        ds: Dataset with variables ``state_var``, ``obs_var`` and ``mask_var``, each
            with dims ``(patch, time, feature)``.
        state_var: Name of the ground-truth state variable x_gt (``target``).
        obs_var: Name of the observed or noisy variable y (``input``).
        mask_var: Name of the binary mask variable m (``mask``).

    Returns:
        [`Batch1D`][vardax.Batch1D] with ``input``, ``mask`` and ``target`` as JAX
        arrays, with ``B = patch``, ``T = time`` and ``N = feature``.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> dims = ["patch", "time", "feature"]
        >>> values = xr.DataArray(np.zeros((1, 3, 2)), dims=dims)
        >>> ds = xr.Dataset({"state": values, "obs": values, "mask": values})
        >>> batch = xr_to_batch1d(ds)
        >>> batch.input.shape, batch.mask.shape, batch.target.shape
        ((1, 3, 2), (1, 3, 2), (1, 3, 2))

        The variable names can be chosen:

        >>> renamed = ds.rename({"obs": "noisy"})
        >>> xr_to_batch1d(renamed, obs_var="noisy").input.shape
        (1, 3, 2)
    """
    target = jnp.array(ds[state_var].values)
    obs = jnp.array(ds[obs_var].values)
    mask = jnp.array(ds[mask_var].values)
    return Batch1D(input=obs, mask=mask, target=target)


def interpolate_initial_condition(
    ds: xr.Dataset,
    *,
    obs_var: str = "obs",
    mask_var: str = "mask",
    method: str = "linear",
    fillna: float = 0.0,
) -> xr.Dataset:
    """Interpolate observed points to fill gaps and add as ``"x_init"``.

    For each patch and feature, interpolates the observed values linearly across
    the time axis to produce a dense initial-condition estimate. Two or more
    observed times give the interpolant, with ``fillna`` outside them; a single
    observed time is held constant; no observed time gives ``fillna``.

    Args:
        ds: Dataset with ``obs_var`` and ``mask_var``, dims
            ``(patch, time, feature)``.
        obs_var: Name of the (noisy) observation variable y.
        mask_var: Name of the binary observation mask m.
        method: Interpolation method along the time axis. Only ``"linear"`` is
            supported, using ``numpy.interp``.
        fillna: Fill value for leading and trailing gaps that cannot be interpolated.

    Returns:
        Dataset with an additional ``"x_init"`` variable with dims
        ``(patch, time, feature)``.

    Raises:
        ConfigurationError: If ``method`` is not ``"linear"``.

    Examples:
        The gap between two observed times is interpolated:

        >>> import numpy as np
        >>> import xarray as xr
        >>> dims = ["patch", "time", "feature"]
        >>> obs = xr.DataArray(np.array([[[1.0], [0.0], [3.0]]]), dims=dims)
        >>> mask = xr.DataArray(np.array([[[1.0], [0.0], [1.0]]]), dims=dims)
        >>> ds = xr.Dataset({"obs": obs, "mask": mask})
        >>> interpolate_initial_condition(ds)["x_init"].values[0, :, 0].tolist()
        [1.0, 2.0, 3.0]

        A single observed time is held constant:

        >>> single = ds.assign(mask=mask * xr.DataArray([1.0, 0.0, 0.0], dims="time"))
        >>> interpolate_initial_condition(single)["x_init"].values[0, :, 0].tolist()
        [1.0, 1.0, 1.0]

        With nothing observed, every time gets ``fillna``:

        >>> empty = ds.assign(mask=mask * 0.0)
        >>> filled = interpolate_initial_condition(empty, fillna=-1.0)
        >>> filled["x_init"].values[0, :, 0].tolist()
        [-1.0, -1.0, -1.0]
    """
    if method != "linear":
        raise ConfigurationError(f"method must be 'linear'; got {method!r}.")

    obs = np.asarray(ds[obs_var])
    mask = np.asarray(ds[mask_var])

    num_patches, num_times, num_features = obs.shape
    initial_state = np.full_like(obs, fillna)
    all_times = np.arange(num_times, dtype=float)

    for patch in range(num_patches):
        for feature in range(num_features):
            feature_obs = obs[patch, :, feature]
            feature_mask = mask[patch, :, feature].astype(bool)
            observed_times = all_times[feature_mask]
            observed_values = feature_obs[feature_mask]
            if len(observed_times) >= 2:
                initial_state[patch, :, feature] = np.interp(
                    all_times,
                    observed_times,
                    observed_values,
                    left=fillna,
                    right=fillna,
                )
            elif len(observed_times) == 1:
                initial_state[patch, :, feature] = observed_values[0]

    obs_data_array = ds[obs_var]
    initial_data_array = xr.DataArray(
        initial_state.astype(np.float32),
        dims=obs_data_array.dims,
        coords=obs_data_array.coords,
    )
    return ds.assign(x_init=initial_data_array)


def obs_interpolation_init(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    obs_variable: str = "obs",
    method: str = "linear",
    fillna: float = 0.0,
) -> xr.Dataset:
    """Interpolate observations along time and add as ``"state_init"``.

    Uses ``xr.DataArray.interpolate_na`` to fill NaN gaps along the time dimension,
    then fills any remaining NaNs (e.g. leading or trailing edges) with ``fillna``.
    The result is a warm-start initial condition for the 4D-Var solver.

    The time dimension is ``time`` when ``obs_variable`` has one, else its second
    dimension, which matches the layout ``(patch, time, feature)`` that
    ``extract_patches`` produces.

    Args:
        ds: Dataset containing at least ``obs_variable``, which should have NaN at
            unobserved locations.
        variable: Name of the ground-truth state variable (unused for interpolation,
            kept for API symmetry).
        obs_variable: Name of the observation variable y (NaN where unobserved).
        method: Interpolation method passed to ``xr.DataArray.interpolate_na``.
        fillna: Fill value for any remaining NaN after interpolation.

    Returns:
        Dataset with an additional ``"state_init"`` variable of the same shape as
        ``obs_variable``.

    Examples:
        A gap inside the window is interpolated:

        >>> import numpy as np
        >>> import xarray as xr
        >>> dims = ["patch", "time", "feature"]
        >>> obs = np.array([[[1.0], [np.nan], [3.0]]])
        >>> ds = xr.Dataset({"obs": xr.DataArray(obs, dims=dims)})
        >>> obs_interpolation_init(ds)["state_init"].values[0, :, 0].tolist()
        [1.0, 2.0, 3.0]

        A gap at the edge gets ``fillna``:

        >>> edge = np.array([[[np.nan], [2.0], [3.0]]])
        >>> ds = xr.Dataset({"obs": xr.DataArray(edge, dims=dims)})
        >>> filled = obs_interpolation_init(ds, fillna=-1.0)
        >>> filled["state_init"].values[0, :, 0].tolist()
        [-1.0, 2.0, 3.0]
    """
    obs_data_array = ds[obs_variable]
    if "time" in obs_data_array.dims:
        time_dim = "time"
    elif obs_data_array.ndim >= 2:
        time_dim = obs_data_array.dims[1]
    else:
        time_dim = obs_data_array.dims[0]
    interpolated = obs_data_array.interpolate_na(dim=time_dim, method=cast(Any, method))
    state_init = interpolated.fillna(fillna)
    return ds.assign(state_init=state_init.astype(np.float32))
