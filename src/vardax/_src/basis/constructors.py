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
from jaxtyping import Array, ArrayLike, Float
import numpy as _np

from vardax._src.errors import (
    ConfigurationError,
    DataError,
    GridError,
    ShapeError,
    UnsupportedInputError,
)

from .linear import LinearBasis, linear_basis


def _grid_points(
    grid_coords: Sequence[Float[ArrayLike, " _"]],
) -> tuple[_np.ndarray, tuple[int, ...]]:
    """Turn per-axis coordinate arrays into ``(N, d)`` points + grid shape."""
    if len(grid_coords) < 1:
        raise GridError("grid_coords must contain at least one axis array.")
    axes = [_np.asarray(axis_values, dtype=float) for axis_values in grid_coords]
    for axis_number, axis in enumerate(axes):
        if axis.ndim != 1 or axis.size < 1:
            raise GridError(
                f"grid_coords[{axis_number}] must be a non-empty 1-D axis array; "
                f"got shape {axis.shape}."
            )
    mesh = _np.meshgrid(*axes, indexing="ij")
    points = _np.stack([grid_axis.ravel() for grid_axis in mesh], axis=1)
    return points, tuple(axis.size for axis in axes)


def rbf_basis(
    grid_coords: Sequence[Float[ArrayLike, " _"]],
    centers: Float[ArrayLike, "M d"],
    *,
    widths: float | Float[ArrayLike, " M"],
    kernel: Literal["gaussian", "wendland_c2", "wendland_c4"] = "gaussian",
    variance: float | Float[ArrayLike, " M"] = 1.0,
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
    points, grid_shape = _grid_points(grid_coords)
    ndim = points.shape[1]
    centers_array = _np.asarray(centers, dtype=float)
    if centers_array.ndim != 2 or centers_array.shape[1] != ndim:
        raise ShapeError(
            f"centers must be (M, {ndim}) to match {ndim}-D grid_coords; "
            f"got {centers_array.shape}."
        )
    n_atoms = centers_array.shape[0]
    widths_array = _np.broadcast_to(_np.asarray(widths, dtype=float), (n_atoms,))
    if _np.any(widths_array <= 0):
        raise ConfigurationError(
            f"widths must be strictly positive; got min={float(widths_array.min())}."
        )
    # Φ[:, a] = K(‖g − c_a‖ / ℓ_a)
    design_matrix = _gnx.rbf_basis(
        jnp.asarray(points),
        jnp.asarray(centers_array),
        jnp.asarray(widths_array),
        kernel=kernel,
    )
    return linear_basis(design_matrix, variance=variance, output_shape=grid_shape)


def fourier_basis(
    grid_coords: Sequence[Float[ArrayLike, " _"]],
    num_modes: int | tuple[int, ...],
    *,
    boundary_scale: float = 1.25,
    variance: float | Float[ArrayLike, " M"] = 1.0,
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
        raise ConfigurationError(f"boundary_scale must be >= 1, got {boundary_scale}.")
    points, grid_shape = _grid_points(grid_coords)
    lower, upper = points.min(axis=0), points.max(axis=0)
    half_span = (upper - lower) / 2.0
    if _np.any(half_span <= 0):
        raise GridError(
            "each grid_coords axis must span a non-degenerate interval "
            f"(got half-spans {tuple(half_span.tolist())})."
        )
    centered = points - (lower + upper) / 2.0
    ndim = points.shape[1]
    half_widths = tuple(float(span) * boundary_scale for span in half_span)
    modes = (
        (int(num_modes),) * ndim
        if isinstance(num_modes, int)
        else tuple(int(axis_modes) for axis_modes in num_modes)
    )
    # Φ: Dirichlet eigenfunctions of −Δ on the box, eigenvalues λ
    design_matrix, eigvals = _gnx.fourier_basis(
        jnp.asarray(centered), modes, half_widths
    )
    # Q = variance · S(λ)
    prior_variance = (
        variance * spectral_density(eigvals)
        if spectral_density is not None
        else variance
    )
    return linear_basis(design_matrix, variance=prior_variance, output_shape=grid_shape)


def wavelet_basis(
    grid_shape: int | tuple[int, ...],
    *,
    wavelet: str = "haar",
    levels: int | None = None,
    variance: float | Float[ArrayLike, " M"] = 1.0,
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
        design_matrix = _gnx.wavelet_basis_1d(
            int(shape[0]), wavelet=wavelet, levels=levels
        )
    elif len(shape) == 2:
        design_matrix = _gnx.wavelet_basis_2d(
            int(shape[0]), int(shape[1]), wavelet=wavelet, levels=levels
        )
    else:
        raise UnsupportedInputError(
            f"wavelet_basis supports 1-D or 2-D grids; got {len(shape)}-D "
            f"grid_shape {shape}."
        )
    return linear_basis(design_matrix, variance=variance, output_shape=shape)


def eof_basis(
    data: Float[ArrayLike, "T N"],
    n_modes: int,
    *,
    center: bool = True,
    variance: float | Float[ArrayLike, " M"] | str = "explained",
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
    data_array = jnp.asarray(data)
    # Φ: leading EOFs of the data, singular values σ
    design_matrix, singular_values = _gnx.eof_basis(data_array, n_modes, center=center)
    if isinstance(variance, str):
        if variance != "explained":
            raise ConfigurationError(
                f'variance must be "explained", a scalar, or an array; '
                f"got {variance!r}."
            )
        n_samples = int(data_array.shape[0])
        if n_samples < 2:
            raise DataError(
                'variance="explained" needs at least 2 samples (T >= 2); '
                f"got T={n_samples}."
            )
        # σ_a² / (T − 1)
        prior_variance: float | Float[Array, " M"] = singular_values**2 / (
            n_samples - 1
        )
    else:
        prior_variance = variance
    return linear_basis(
        design_matrix, variance=prior_variance, output_shape=output_shape
    )
