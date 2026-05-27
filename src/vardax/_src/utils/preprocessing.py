"""xarray → JAX conversion and train/test split utilities."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import xarray as xr

from vardax._src._types import Batch1D


def train_test_split(
    ds: xr.Dataset,
    *,
    n_train: int,
    n_test: int,
    seed: int = 0,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Randomly split a patched dataset along the ``patch`` dimension.

    Parameters
    ----------
    ds:
        Dataset with a ``patch`` dimension.
    n_train:
        Number of patches for the training set.
    n_test:
        Number of patches for the test set.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    ds_train : xr.Dataset
        Training subset.
    ds_test : xr.Dataset
        Test subset.
    """
    rng = np.random.default_rng(seed)
    n_patches = ds.sizes["patch"]

    if n_train + n_test > n_patches:
        msg = (
            f"n_train + n_test ({n_train + n_test}) exceeds available "
            f"patches ({n_patches})"
        )
        raise ValueError(msg)

    indices = rng.permutation(n_patches)
    train_idx = indices[:n_train]
    test_idx = indices[n_train : n_train + n_test]

    ds_train = ds.isel(patch=sorted(train_idx.tolist()))
    ds_test = ds.isel(patch=sorted(test_idx.tolist()))
    return ds_train, ds_test


def xr_to_batch1d(
    ds: xr.Dataset,
    *,
    state_var: str = "state",
    obs_var: str = "obs",
    mask_var: str = "mask",
) -> Batch1D:
    """Convert an xarray Dataset to a :class:`~vardax.Batch1D`.

    The dataset is expected to have dims ``(patch, time, feature)``; the
    feature dimension is treated as the spatial ``N`` axis of
    :class:`~vardax.Batch1D`.

    Parameters
    ----------
    ds:
        Dataset with variables ``state_var``, ``obs_var``, and
        ``mask_var``, each of shape ``(patch, time, feature)``.
    state_var:
        Name of the ground-truth state variable (→ ``target``).
    obs_var:
        Name of the observed / noisy variable (→ ``input``).
    mask_var:
        Name of the binary mask variable (→ ``mask``).

    Returns
    -------
    Batch1D
        Named tuple with ``input``, ``mask``, and ``target`` as JAX
        arrays of shape ``(B, T, N)`` where ``B = patch``, ``T = time``,
        and ``N = feature``.
    """
    target = jnp.array(ds[state_var].values)
    inp = jnp.array(ds[obs_var].values)
    mask = jnp.array(ds[mask_var].values)
    return Batch1D(input=inp, mask=mask, target=target)


def interpolate_initial_condition(
    ds: xr.Dataset,
    *,
    obs_var: str = "obs",
    mask_var: str = "mask",
    method: str = "linear",
    fillna: float = 0.0,
) -> xr.Dataset:
    """Interpolate observed points to fill gaps and add as ``"x_init"``.

    For each patch, linearly interpolates the observed values across the
    time axis to produce a dense initial-condition estimate.

    Parameters
    ----------
    ds:
        Dataset with ``obs_var`` and ``mask_var``, dims
        ``(patch, time, feature)``.
    obs_var:
        Name of the (noisy) observation variable.
    mask_var:
        Name of the binary observation mask.
    method:
        Interpolation method passed to
        :func:`numpy.interp` along the time axis (``"linear"`` only).
    fillna:
        Fill value for leading/trailing gaps that cannot be interpolated.

    Returns
    -------
    xr.Dataset
        Dataset with an additional ``"x_init"`` variable of shape
        ``(patch, time, feature)``.
    """
    obs = np.asarray(ds[obs_var])  # (P, T, F)
    mask = np.asarray(ds[mask_var])  # (P, T, F)

    P, T, F = obs.shape
    x_init = np.full_like(obs, fillna)
    t_all = np.arange(T, dtype=float)

    for p in range(P):
        for f in range(F):
            obs_pf = obs[p, :, f]
            mask_pf = mask[p, :, f].astype(bool)
            t_obs = t_all[mask_pf]
            y_obs = obs_pf[mask_pf]
            if len(t_obs) >= 2:
                x_init[p, :, f] = np.interp(
                    t_all, t_obs, y_obs, left=fillna, right=fillna
                )
            elif len(t_obs) == 1:
                x_init[p, :, f] = y_obs[0]

    obs_da = ds[obs_var]
    x_init_da = xr.DataArray(
        x_init.astype(np.float32),
        dims=obs_da.dims,
        coords=obs_da.coords,
    )
    return ds.assign(x_init=x_init_da)


def obs_interpolation_init(
    ds: xr.Dataset,
    *,
    variable: str = "state",
    obs_variable: str = "obs",
    method: str = "linear",
    fillna: float = 0.0,
) -> xr.Dataset:
    """Interpolate observations along time and add as ``"state_init"``.

    Uses :meth:`xr.DataArray.interpolate_na` to fill NaN gaps along the
    time dimension, then fills any remaining NaNs (e.g. leading/trailing
    edges) with ``fillna``.  The result is stored as a new ``"state_init"``
    variable — a warm-start initial condition for the 4DVar solver.

    Parameters
    ----------
    ds:
        Dataset containing at least ``obs_variable``, which should have NaN
        at unobserved locations.
    variable:
        Name of the ground-truth state variable (unused for interpolation,
        kept for API symmetry).
    obs_variable:
        Name of the observation variable (NaN where unobserved).
    method:
        Interpolation method passed to
        :meth:`xr.DataArray.interpolate_na` (default ``"linear"``).
    fillna:
        Fill value for any remaining NaN after interpolation (default
        ``0.0``).

    Returns
    -------
    xr.Dataset
        Dataset with an additional ``"state_init"`` variable of the same
        shape as ``obs_variable``.

    Notes
    -----
    The time dimension is assumed to be the *second* dimension of
    ``obs_variable`` (i.e. ``dims[1]`` for shape
    ``(patch, time, feature)``), which matches the standard vardax
    dataset layout produced by :func:`extract_patches`.
    """
    obs_da = ds[obs_variable]
    if "time" in obs_da.dims:
        time_dim = "time"
    elif obs_da.ndim >= 2:
        time_dim = obs_da.dims[1]
    else:
        time_dim = obs_da.dims[0]
    state_init = obs_da.interpolate_na(dim=time_dim, method=cast(Any, method)).fillna(
        fillna
    )
    return ds.assign(state_init=state_init.astype(np.float32))
