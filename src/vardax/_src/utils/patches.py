"""xarray patch extraction utilities for time-series data.

``xarray`` is an optional dependency (the ``[data]`` extra) and is
imported lazily inside the functions that need it, so that core-path
imports of this module (e.g. ``time_patches`` via the package
initializer) work on a base install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def time_patches(ts: Float[Array, T]) -> Float[Array, "T-1 2"]:  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
    """Overlapping consecutive time pairs for one-step increment losses.

    Reimplements the legacy ``kernex.kmap(kernel_size=(2,), relative=True)``
    sliding window without the ``kernex`` dependency: for a 1-D array of
    times ``[t_0, t_1, ..., t_{T-1}]`` it returns the ``T - 1`` overlapping
    pairs ``[[t_0, t_1], [t_1, t_2], ..., [t_{T-2}, t_{T-1}]]``.

    Args:
        ts: Monotonic time coordinates of shape ``(T,)``.

    Returns:
        Array of shape ``(T - 1, 2)`` of consecutive time pairs.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import time_patches
        >>> time_patches(jnp.arange(4.0))
        Array([[0., 1.],
               [1., 2.],
               [2., 3.]], dtype=float32)
    """
    return jnp.stack([ts[:-1], ts[1:]], axis=-1)


def trajectory_to_xr_dataset(
    states: Float[Array, "T F"],
    time_coords: Float[Array, T],  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
    *,
    feature_names: list[str] | None = None,
) -> xr.Dataset:
    """Convert a state trajectory to an xarray Dataset.

    Args:
        states: State trajectory of shape ``(T, F)``.
        time_coords: Time coordinates of shape ``(T,)``.
        feature_names: Names for the feature dimensions.  Defaults to
            ``["x0", "x1", ..., "x{F-1}"]``.

    Returns:
        Dataset with a ``"state"`` DataArray of dims ``(time, feature)``.
    """
    import xarray as xr

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

    Args:
        ds: Dataset with a ``"state"`` DataArray of dims
            ``(time, feature)``.
        n_patches: Number of patches to extract.
        n_timesteps: Length (in timesteps) of each patch.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with a ``"state"`` DataArray of dims
        ``(patch, time, feature)``.
    """
    import xarray as xr

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
