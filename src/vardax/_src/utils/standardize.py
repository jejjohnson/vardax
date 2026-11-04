"""Standardization utilities for xarray Datasets and JAX arrays."""

from __future__ import annotations

from jaxtyping import Array, Float
import numpy as np
import xarray as xr

from vardax._src.errors import DataError


def compute_scaler_params(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    mask_variable: str | None = "mask",
) -> tuple[float, float]:
    """Compute mean μ and standard deviation σ from (optionally masked) data.

    Args:
        ds: Dataset containing ``variable`` and optionally ``mask_variable``.
        variable: Name of the data variable.
        mask_variable: Name of the binary mask variable (1 = observed). When ``None``
            or absent from the dataset, all values are used.

    Returns:
        ``(mean, std)``: mean μ and standard deviation σ of the observed values.

    Raises:
        DataError: If the mask selects no values, or the values are constant.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> values = xr.DataArray(np.array([1.0, 3.0, 100.0]), dims="time")
        >>> ds = xr.Dataset({"state": values})
        >>> compute_scaler_params(ds)[0] > 30.0
        True

        A mask restricts the statistics to the observed values:

        >>> mask = xr.DataArray(np.array([1.0, 1.0, 0.0]), dims="time")
        >>> compute_scaler_params(ds.assign(mask=mask))
        (2.0, 1.0)

        Constant data cannot be standardized:

        >>> constant = xr.Dataset({"state": xr.ones_like(values)})
        >>> compute_scaler_params(constant)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        DataError: Standard deviation of variable 'state' is zero ...
    """
    values = np.asarray(ds[variable])

    if mask_variable is not None and mask_variable in ds:
        mask = np.asarray(ds[mask_variable]).astype(bool)
        observed = values[mask]
    else:
        # ravel, not einx: einx cannot size a zero-length axis, and an empty dataset
        # must still reach the DataError below
        observed = values.ravel()

    # μ = (1 / N) ∑ᵢ xᵢ
    mean = float(np.mean(observed))
    # σ = √((1 / N) ∑ᵢ (xᵢ − μ)²)
    std = float(np.std(observed))
    if np.isnan(mean) or np.isnan(std):
        msg = (
            f"No observed values found in variable '{variable}' "
            "(mask selects no elements)."
        )
        raise DataError(msg)
    if std == 0.0:
        msg = (
            f"Standard deviation of variable '{variable}' is zero "
            "(constant data); cannot standardize."
        )
        raise DataError(msg)
    return mean, std


def apply_standardization(
    ds: xr.Dataset,
    *,
    variables: list[str],
    mean: float,
    std: float,
) -> xr.Dataset:
    """Standardize one or more variables in a dataset, returning a new dataset.

        x_std = (x − μ) / σ

    Args:
        ds: Source dataset.
        variables: Names of the variables to standardize.
        mean: Mean μ to subtract.
        std: Standard deviation σ to divide by.

    Returns:
        Dataset with the listed variables replaced by their standardized versions.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> values = xr.DataArray(np.array([1.0, 3.0]), dims="time")
        >>> ds = xr.Dataset({"state": values, "other": values})
        >>> result = apply_standardization(ds, variables=["state"], mean=2.0, std=1.0)
        >>> result["state"].values.tolist()
        [-1.0, 1.0]

        Variables not listed are left as they were:

        >>> result["other"].values.tolist()
        [1.0, 3.0]
    """
    updates = {}
    for name in variables:
        data_array = ds[name]
        # x_std = (x − μ) / σ
        standardized = (np.asarray(data_array) - mean) / std
        updates[name] = xr.DataArray(
            standardized.astype(np.float32),
            dims=data_array.dims,
            coords=data_array.coords,
        )
    return ds.assign(updates)


def inverse_standardization(
    data: Float[Array, ...],
    *,
    mean: float,
    std: float,
) -> Float[Array, ...]:
    """Reverse a standardization transform on a JAX array.

        x = x_std σ + μ

    Args:
        data: Standardized data x_std.
        mean: Mean μ that was subtracted during standardization.
        std: Standard deviation σ that was divided during standardization.

    Returns:
        Array x in the original scale.

    Examples:
        >>> import jax.numpy as jnp
        >>> restored = inverse_standardization(
        ...     jnp.array([-1.0, 1.0]), mean=2.0, std=1.0
        ... )
        >>> bool(jnp.allclose(restored, jnp.array([1.0, 3.0])))
        True

        The shape is preserved:

        >>> inverse_standardization(jnp.zeros((2, 3)), mean=0.0, std=1.0).shape
        (2, 3)
    """
    # x = x_std σ + μ
    return data * std + mean
