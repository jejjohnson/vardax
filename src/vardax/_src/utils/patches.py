"""xarray patch extraction utilities for time-series data."""

from __future__ import annotations

from jaxtyping import Array, Float
import numpy as np
import xarray as xr


def trajectory_to_xr_dataset(
    states: Float[Array, "T F"],
    time_coords: Float[Array, T],  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
    *,
    feature_names: list[str] | None = None,
) -> xr.Dataset:
    """Convert a state trajectory to an xarray Dataset.

    Parameters
    ----------
    states:
        State trajectory of shape ``(T, F)``.
    time_coords:
        Time coordinates of shape ``(T,)``.
    feature_names:
        Names for the feature dimensions.  Defaults to
        ``["x0", "x1", ..., "x{F-1}"]``.

    Returns
    -------
    xr.Dataset
        Dataset with a ``"state"`` DataArray of dims ``(time, feature)``.
    """
    states_np = np.asarray(states)
    time_np = np.asarray(time_coords)
    n_features = states_np.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    da = xr.DataArray(
        states_np,
        dims=["time", "feature"],
        coords={
            "time": time_np,
            "feature": feature_names,
        },
    )
    return xr.Dataset({"state": da})


def extract_patches(
    ds: xr.Dataset,
    *,
    n_patches: int,
    n_timesteps: int,
    seed: int = 0,
) -> xr.Dataset:
    """Extract random temporal patches from a trajectory dataset.

    Parameters
    ----------
    ds:
        Dataset with a ``"state"`` DataArray of dims ``(time, feature)``.
    n_patches:
        Number of patches to extract.
    n_timesteps:
        Length (in timesteps) of each patch.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with a ``"state"`` DataArray of dims
        ``(patch, time, feature)``.
    """
    rng = np.random.default_rng(seed)
    state = ds["state"]
    n_time = state.sizes["time"]

    if n_timesteps > n_time:
        msg = f"n_timesteps ({n_timesteps}) > available time steps ({n_time})"
        raise ValueError(msg)

    max_start = n_time - n_timesteps
    starts = rng.integers(0, max_start + 1, size=n_patches)

    patches = np.stack(
        [np.asarray(state.isel(time=slice(s, s + n_timesteps))) for s in starts],
        axis=0,
    )

    patch_time = np.arange(n_timesteps)
    feature_coords = state.coords["feature"].values

    da = xr.DataArray(
        patches,
        dims=["patch", "time", "feature"],
        coords={
            "patch": np.arange(n_patches),
            "time": patch_time,
            "feature": feature_coords,
        },
    )
    return xr.Dataset({"state": da})
