r"""Family constructors for `LinearBasis`, delegating the matrix math to geonnax.

Each constructor builds the geometry (grid points, domain box, …),
calls the corresponding pure function in ``geonnax.basis`` to obtain
the $(N, M)$ design matrix $\Phi$, attaches a diagonal prior variance,
and returns a `LinearBasis`.

Families:

- :func:`rbf_basis` — placeable radial bumps (Gaussian or compactly
  supported Wendland kernels). MASSH ``GAUSS*``-style local atoms.
- :func:`fourier_basis` — Laplacian (Dirichlet) eigenfunctions on a
  box; the prior variance can be a spectral density evaluated at the
  eigenvalues (the HSGP construction).
- :func:`wavelet_basis` — orthonormal Haar/Daubechies DWT basis.
- :func:`eof_basis` — data-driven EOFs with explained-variance prior.

Reference: MASSH `mapping/src/basis.py` (`set_basis`, `operg`,
families `GAUSS*` / `BM` / `MIOST` / `WAVELET3D`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import geonnax.basis as _gnx
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as _np

from .linear import LinearBasis, linear_basis


def _grid_points(
    grid_coords: Sequence[Float[Array, " _"]],
) -> tuple[_np.ndarray, tuple[int, ...]]:
    """Turn per-axis coordinate arrays into ``(N, d)`` points + grid shape."""
    if len(grid_coords) < 1:
        raise ValueError("grid_coords must contain at least one axis array.")
    axes = [_np.asarray(a, dtype=float) for a in grid_coords]
    for i, ax in enumerate(axes):
        if ax.ndim != 1 or ax.size < 1:
            raise ValueError(
                f"grid_coords[{i}] must be a non-empty 1-D axis array; "
                f"got shape {ax.shape}."
            )
    mesh = _np.meshgrid(*axes, indexing="ij")
    pts = _np.stack([m.ravel() for m in mesh], axis=1)
    return pts, tuple(ax.size for ax in axes)


def rbf_basis(
    grid_coords: Sequence[Float[Array, " _"]],
    centers: Float[Array, "M d"],
    *,
    widths: float | Float[Array, " M"],
    kernel: Literal["gaussian", "wendland_c2", "wendland_c4"] = "gaussian",
    variance: float | Float[Array, " M"] = 1.0,
) -> LinearBasis:
    r"""Placeable radial-basis reduced basis (Gaussian or Wendland kernels).

    Column $a$ of $\Phi$ is $\varphi_a(g) = K(\lVert g - c_a \rVert /
    \ell_a)$ evaluated on the flattened grid. ``kernel="gaussian"``
    reproduces the MASSH stationary Gaussian-RBF basis; the compactly
    supported ``"wendland_c2"`` / ``"wendland_c4"`` kernels give columns
    that are exactly zero past their width.

    Args:
        grid_coords: Per-axis 1-D coordinate arrays, e.g. ``(y_axis,
            x_axis)`` for a 2-D grid. Any dimension ``d >= 1``.
        centers: ``(M, d)`` array of atom centre locations.
        widths: Scalar or per-atom ``(M,)`` width $\ell_a$ (Gaussian
            length scale, or Wendland support radius); must be positive.
        kernel: ``"gaussian"``, ``"wendland_c2"``, or ``"wendland_c4"``.
        variance: Scalar or per-coefficient prior variance.

    Returns:
        `LinearBasis` with ``output_shape`` the grid shape and
        ``nbasis == M``.
    """
    pts, grid_shape = _grid_points(grid_coords)
    ndim = pts.shape[1]
    cen = _np.asarray(centers, dtype=float)
    if cen.ndim != 2 or cen.shape[1] != ndim:
        raise ValueError(
            f"centers must be (M, {ndim}) to match {ndim}-D grid_coords; "
            f"got {cen.shape}."
        )
    m = cen.shape[0]
    width_arr = _np.broadcast_to(_np.asarray(widths, dtype=float), (m,))
    if _np.any(width_arr <= 0):
        raise ValueError(
            f"widths must be strictly positive; got min={float(width_arr.min())}."
        )
    phi = _gnx.rbf_basis(
        jnp.asarray(pts),
        jnp.asarray(cen),
        jnp.asarray(width_arr),
        kernel=kernel,
    )
    return linear_basis(phi, variance=variance, output_shape=grid_shape)


def fourier_basis(
    grid_coords: Sequence[Float[Array, " _"]],
    num_modes: int | tuple[int, ...],
    *,
    boundary_scale: float = 1.25,
    variance: float | Float[Array, " M"] = 1.0,
    spectral_density: Callable[[Float[Array, " M"]], Float[Array, " M"]] | None = None,
) -> LinearBasis:
    r"""Laplacian-eigenfunction (Dirichlet/HSGP) reduced basis on a box.

    Tensor-product sine eigenfunctions of $-\Delta$ on the (recentred,
    slightly enlarged) grid bounding box. Passing ``spectral_density``
    evaluates it at the Laplacian eigenvalues $\lambda_j$ and scales it
    by ``variance`` — supply a kernel's spectral density to obtain the
    Hilbert-space GP (HSGP) approximation of that kernel's prior with
    amplitude ``variance``.

    Args:
        grid_coords: Per-axis 1-D coordinate arrays; any dimension
            ``d >= 1``. Each axis must span a non-degenerate interval.
        num_modes: Number of 1-D modes per axis; an ``int`` is broadcast
            to all axes. Total ``nbasis`` is the product.
        boundary_scale: Domain half-widths are the grid half-spans
            times this factor (``>= 1``). The Dirichlet eigenfunctions
            vanish on the box boundary, so a value ``> 1`` keeps the
            basis expressive at the grid edges.
        variance: Scalar or per-coefficient ``(M,)`` prior variance;
            with ``spectral_density`` it acts as a scalar amplitude.
        spectral_density: Optional callable mapping the eigenvalues
            ``(M,)`` to per-mode prior power ``(M,)`` (HSGP prior).

    Returns:
        `LinearBasis` with ``output_shape`` the grid shape.
    """
    if boundary_scale < 1.0:
        raise ValueError(f"boundary_scale must be >= 1, got {boundary_scale}.")
    pts, grid_shape = _grid_points(grid_coords)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    half_span = (hi - lo) / 2.0
    if _np.any(half_span <= 0):
        raise ValueError(
            "each grid_coords axis must span a non-degenerate interval "
            f"(got half-spans {tuple(half_span.tolist())})."
        )
    centered = pts - (lo + hi) / 2.0
    ndim = pts.shape[1]
    half_widths = tuple(float(h) * boundary_scale for h in half_span)
    modes = (
        (int(num_modes),) * ndim
        if isinstance(num_modes, int)
        else tuple(int(k) for k in num_modes)
    )
    phi, eigvals = _gnx.fourier_basis(jnp.asarray(centered), modes, half_widths)
    var = (
        variance * spectral_density(eigvals)
        if spectral_density is not None
        else variance
    )
    return linear_basis(phi, variance=var, output_shape=grid_shape)


def wavelet_basis(
    grid_shape: int | tuple[int, ...],
    *,
    wavelet: str = "haar",
    levels: int | None = None,
    variance: float | Float[Array, " M"] = 1.0,
) -> LinearBasis:
    r"""Orthonormal wavelet (DWT) reduced basis on a periodic grid.

    Haar / Daubechies synthesis basis: complete ($M = N$) and
    orthonormal ($\Phi^\top \Phi = I$), so it is a multiscale
    *re-parameterisation* of the grid — per-coefficient ``variance``
    is the natural place to encode scale-dependent prior power.

    Args:
        grid_shape: ``n`` or ``(n,)`` for a 1-D grid, ``(ny, nx)`` for
            2-D. Every extent must be a power of two ``>= 2``.
        wavelet: ``"haar"``, ``"db2"``, or ``"db4"``.
        levels: Decomposition levels; defaults to the full cascade.
        variance: Scalar or per-coefficient prior variance.

    Returns:
        `LinearBasis` with ``output_shape == grid_shape`` and
        ``nbasis == prod(grid_shape)``.
    """
    shape = (grid_shape,) if isinstance(grid_shape, int) else tuple(grid_shape)
    if len(shape) == 1:
        phi = _gnx.wavelet_basis_1d(int(shape[0]), wavelet=wavelet, levels=levels)
    elif len(shape) == 2:
        phi = _gnx.wavelet_basis_2d(
            int(shape[0]), int(shape[1]), wavelet=wavelet, levels=levels
        )
    else:
        raise ValueError(
            f"wavelet_basis supports 1-D or 2-D grids; got {len(shape)}-D "
            f"grid_shape {shape}."
        )
    return linear_basis(phi, variance=variance, output_shape=shape)


def eof_basis(
    data: Float[Array, "T N"],
    n_modes: int,
    *,
    center: bool = True,
    variance: float | Float[Array, " M"] | str = "explained",
    output_shape: tuple[int, ...] | None = None,
) -> LinearBasis:
    r"""Data-driven EOF (PCA) reduced basis from a sample matrix.

    The leading ``n_modes`` empirical orthogonal functions of ``data``
    become the columns of $\Phi$. With ``variance="explained"`` the
    prior variance of mode $a$ is its sample variance
    $\sigma_a^2 / (T - 1)$, so the background term matches the
    climatology of the training samples.

    Args:
        data: ``(T, N)`` sample matrix — ``T`` snapshots over the
            flattened grid.
        n_modes: Number of leading EOFs to keep.
        center: Subtract the per-point sample mean before the SVD.
        variance: ``"explained"`` (default), or a scalar / ``(M,)``
            array overriding the prior variance.
        output_shape: Grid shape to reshape increments to; must satisfy
            ``prod(output_shape) == N``. Defaults to ``(N,)``.

    Returns:
        `LinearBasis` with orthonormal columns and ``nbasis == n_modes``.
    """
    data_arr = jnp.asarray(data)
    phi, svals = _gnx.eof_basis(data_arr, n_modes, center=center)
    if isinstance(variance, str):
        if variance != "explained":
            raise ValueError(
                f'variance must be "explained", a scalar, or an array; '
                f"got {variance!r}."
            )
        n_samples = int(data_arr.shape[0])
        if n_samples < 2:
            raise ValueError(
                'variance="explained" needs at least 2 samples (T >= 2); '
                f"got T={n_samples}."
            )
        var: float | Float[Array, " M"] = svals**2 / (n_samples - 1)
    else:
        var = variance
    return linear_basis(phi, variance=var, output_shape=output_shape)
