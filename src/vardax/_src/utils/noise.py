"""Noise injection utilities for xarray Datasets."""

from __future__ import annotations

import numpy as np
import xarray as xr


def add_gaussian_noise(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    sigma: float = 1.0,
    seed: int = 0,
    name: str = "obs",
) -> xr.Dataset:
    """Add Gaussian noise to a variable and store the result.

        y = x + σ ε,   ε ~ 𝒩(0, I)

    Args:
        ds: Dataset containing ``variable``.
        variable: Name of the source variable x.
        sigma: Standard deviation σ of the additive Gaussian noise.
        seed: Random seed for reproducibility (via ``numpy.random.default_rng``).
        name: Name of the new noisy variable y added to the dataset.

    Returns:
        Dataset with the new ``name`` variable added.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> state = xr.DataArray(np.zeros((4, 2)), dims=["time", "feature"])
        >>> ds = xr.Dataset({"state": state})
        >>> add_gaussian_noise(ds, sigma=0.1)["obs"].shape
        (4, 2)

        Zero noise returns the source values:

        >>> noiseless = add_gaussian_noise(ds, sigma=0.0)
        >>> bool(np.allclose(noiseless["obs"], ds["state"]))
        True

        The new variable takes the given name:

        >>> "noisy" in add_gaussian_noise(ds, name="noisy")
        True
    """
    rng = np.random.default_rng(seed)
    data_array = ds[variable]
    # ε ~ 𝒩(0, I)
    noise = rng.standard_normal(data_array.shape).astype(np.float32)
    # y = x + σ ε
    noisy_values = np.asarray(data_array) + sigma * noise
    noisy_data_array = xr.DataArray(
        noisy_values, dims=data_array.dims, coords=data_array.coords
    )
    return ds.assign({name: noisy_data_array})
