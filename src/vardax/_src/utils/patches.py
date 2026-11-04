"""Temporal patches: consecutive time pairs, and xarray trajectory patches.

``xarray`` is an optional dependency (the ``[data]`` extra) and is imported lazily
inside the functions that need it, so ``time_patches`` works on a base install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float
import numpy as np

from vardax._src.errors import DataError

if TYPE_CHECKING:
    import xarray as xr


def time_patches(ts: Float[Array, " T"]) -> Float[Array, "T-1 2"]:
    """Overlapping consecutive time pairs for one-step increment losses.

        (tᵢ, tᵢ₊₁),   i = 0, …, T − 2

    Reimplements the legacy ``kernex.kmap(kernel_size=(2,), relative=True)`` sliding
    window without the ``kernex`` dependency.

    Args:
        ts: Monotonic time coordinates t₀, …, t_(T−1).

    Returns:
        The consecutive pairs (tᵢ, tᵢ₊₁), one row per pair.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import time_patches
        >>> pairs = time_patches(jnp.arange(4.0))
        >>> pairs.shape, pairs.tolist()
        ((3, 2), [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

        A single time yields no pairs:

        >>> time_patches(jnp.array([0.0])).shape
        (0, 2)

        The pairs drive the one-step solves of
        [`DynIncrements`][vardax.DynIncrements]:

        >>> time_patches(jnp.linspace(0.0, 1.0, 5)).shape
        (4, 2)
    """
    # jnp, not einx: a single time gives zero pairs, which einx cannot size
    return jnp.stack([ts[:-1], ts[1:]], axis=-1)


def trajectory_to_xr_dataset(
    states: Float[ArrayLike, "T F"],
    time_coords: Float[ArrayLike, " T"],
    *,
    feature_names: list[str] | None = None,
) -> xr.Dataset:
    """Convert a state trajectory to an xarray Dataset.

    Args:
        states: State trajectory x₁, …, x_T.
        time_coords: Time coordinates of the rows of ``states``.
        feature_names: Names for the feature dimensions. Defaults to
            ``["x0", "x1", ..., "x{F-1}"]``.

    Returns:
        Dataset with a ``"state"`` DataArray of dims ``(time, feature)``.

    Examples:
        >>> import numpy as np
        >>> ds = trajectory_to_xr_dataset(np.zeros((4, 2)), np.arange(4.0))
        >>> ds["state"].dims, ds["feature"].values.tolist()
        (('time', 'feature'), ['x0', 'x1'])

        The features can be named:

        >>> named = trajectory_to_xr_dataset(
        ...     np.zeros((4, 2)), np.arange(4.0), feature_names=["u", "v"]
        ... )
        >>> named["feature"].values.tolist()
        ['u', 'v']
    """
    # lazy: xarray is the optional [data] extra
    import xarray as xr

    state_values = np.asarray(states)
    time_values = np.asarray(time_coords)
    num_features = state_values.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(num_features)]

    data_array = xr.DataArray(
        state_values,
        dims=["time", "feature"],
        coords={
            "time": time_values,
            "feature": feature_names,
        },
    )
    return xr.Dataset({"state": data_array})


def extract_patches(
    ds: xr.Dataset,
    *,
    n_patches: int,
    n_timesteps: int,
    seed: int = 0,
) -> xr.Dataset:
    """Extract random temporal patches from a trajectory dataset.

    Args:
        ds: Dataset with a ``"state"`` DataArray of dims ``(time, feature)``.
        n_patches: Number of patches to extract.
        n_timesteps: Length (in time steps) of each patch.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with a ``"state"`` DataArray of dims ``(patch, time, feature)``.

    Raises:
        DataError: If ``n_timesteps`` exceeds the available time steps.

    Examples:
        >>> import numpy as np
        >>> ds = trajectory_to_xr_dataset(np.zeros((10, 2)), np.arange(10.0))
        >>> extract_patches(ds, n_patches=3, n_timesteps=4)["state"].shape
        (3, 4, 2)

        A patch as long as the trajectory is the whole trajectory:

        >>> extract_patches(ds, n_patches=1, n_timesteps=10)["state"].shape
        (1, 10, 2)

        A longer patch is rejected:

        >>> extract_patches(
        ...     ds, n_patches=1, n_timesteps=11
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        DataError: n_timesteps (11) > available time steps (10)
    """
    # lazy: xarray is the optional [data] extra
    import xarray as xr

    rng = np.random.default_rng(seed)
    state = ds["state"]
    num_times = state.sizes["time"]

    if n_timesteps > num_times:
        msg = f"n_timesteps ({n_timesteps}) > available time steps ({num_times})"
        raise DataError(msg)

    max_start = num_times - n_timesteps
    starts = rng.integers(0, max_start + 1, size=n_patches)

    # the leading axis is the patch: stacked along np.stack's default axis 0
    patches = np.stack(
        [
            np.asarray(state.isel(time=slice(start, start + n_timesteps)))
            for start in starts
        ],
    )

    patch_time = np.arange(n_timesteps)
    feature_coords = state.coords["feature"].values

    data_array = xr.DataArray(
        patches,
        dims=["patch", "time", "feature"],
        coords={
            "patch": np.arange(n_patches),
            "time": patch_time,
            "feature": feature_coords,
        },
    )
    return xr.Dataset({"state": data_array})
