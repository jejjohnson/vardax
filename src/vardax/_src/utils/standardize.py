"""Standardization utilities for xarray Datasets and JAX arrays."""

from __future__ import annotations

from jaxtyping import Array, Float
import numpy as np
import xarray as xr


def compute_scaler_params(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    mask_variable: str | None = "mask",
) -> tuple[float, float]:
    """Compute mean and standard deviation from (optionally masked) data.

    Parameters
    ----------
    ds:
        Dataset containing ``variable`` and optionally ``mask_variable``.
    variable:
        Name of the data variable.
    mask_variable:
        Name of the binary mask variable (``1`` = observed).  When
        ``None`` or absent from the dataset, all values are used.

    Returns
    -------
    mean : float
        Mean of the observed values.
    std : float
        Standard deviation of the observed values.
    """
    values = np.asarray(ds[variable])

    if mask_variable is not None and mask_variable in ds:
        mask = np.asarray(ds[mask_variable]).astype(bool)
        observed = values[mask]
    else:
        observed = values.ravel()

    mean = float(np.mean(observed))
    std = float(np.std(observed))
    if np.isnan(mean) or np.isnan(std):
        msg = (
            f"No observed values found in variable '{variable}' "
            "(mask selects no elements)."
        )
        raise ValueError(msg)
    if std == 0.0:
        msg = (
            f"Standard deviation of variable '{variable}' is zero "
            "(constant data); cannot standardize."
        )
        raise ValueError(msg)
    return mean, std


def apply_standardization(
    ds: xr.Dataset,
    *,
    variables: list[str],
    mean: float,
    std: float,
) -> xr.Dataset:
    """Standardize one or more variables in a dataset.

    Applies ``(x − mean) / std`` to each listed variable in-place (returns
    a new dataset).

    Parameters
    ----------
    ds:
        Source dataset.
    variables:
        List of variable names to standardize.
    mean:
        Mean value to subtract.
    std:
        Standard deviation to divide by.

    Returns
    -------
    xr.Dataset
        Dataset with the listed variables replaced by their standardized
        versions.
    """
    updates = {}
    for var in variables:
        da = ds[var]
        standardized = (np.asarray(da) - mean) / std
        updates[var] = xr.DataArray(
            standardized.astype(np.float32), dims=da.dims, coords=da.coords
        )
    return ds.assign(updates)


def inverse_standardization(
    data: Float[Array, ...],
    *,
    mean: float,
    std: float,
) -> Float[Array, ...]:
    """Reverse a standardization transform on a JAX array.

    Parameters
    ----------
    data:
        Standardized data.
    mean:
        Mean that was subtracted during standardization.
    std:
        Standard deviation that was divided during standardization.

    Returns
    -------
    Float[Array, "..."]
        Array in the original scale.
    """
    return data * std + mean
