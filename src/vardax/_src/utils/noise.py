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

    Computes ``noisy = variable + σ * ε`` where ``ε ~ N(0, 1)`` and adds
    ``noisy`` as a new variable named ``name`` to the dataset.

    Parameters
    ----------
    ds:
        Dataset containing ``variable``.
    variable:
        Name of the source variable.
    sigma:
        Standard deviation of the additive Gaussian noise.
    seed:
        Random seed for reproducibility (via ``numpy.random.default_rng``).
    name:
        Name of the new noisy variable added to the dataset (default
        ``"obs"``).

    Returns
    -------
    xr.Dataset
        Dataset with the new ``name`` variable added.
    """
    rng = np.random.default_rng(seed)
    da = ds[variable]
    noise = rng.standard_normal(da.shape).astype(np.float32)
    noisy_values = np.asarray(da) + sigma * noise
    noisy_da = xr.DataArray(noisy_values, dims=da.dims, coords=da.coords)
    return ds.assign({name: noisy_da})
