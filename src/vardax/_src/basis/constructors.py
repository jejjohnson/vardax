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

    Column $a$ of $\Phi$ is
    $\varphi_a(g) = K(\lVert g - c_a \rVert / \ell_a)$
    evaluated on the flattened grid. ``kernel="gaussian"`` reproduces the
    MASSH stationary Gaussian-RBF basis; the compactly supported
    ``"wendland_c2"`` / ``"wendland_c4"`` kernels give columns
    that are exactly zero past their width.

    Args:
        grid_coords: Per-axis 1-D coordinate arrays, e.g. ``(y_axis,
            x_axis)`` for a 2-D grid. Any dimension ``d >= 1``.
        centers: Atom centre locations $c_a$, one row per atom with $d$
            columns.
        widths: Scalar or per-atom width $\ell_a$ (Gaussian
            length scale, or Wendland support radius); must be positive.
        kernel: ``"gaussian"``, ``"wendland_c2"``, or ``"wendland_c4"``.
        variance: Scalar or per-coefficient prior variance.

    Algorithm:
        1. Flatten the grid axes into points $g$.
        2. Evaluate $K(\lVert g - c_a \rVert / \ell_a)$ for every atom.
        3. Wrap the design matrix with the diagonal prior ``variance``.

    Returns:
        `LinearBasis` with ``output_shape`` the grid shape and
        ``nbasis == M``.

    Raises:
        GridError: If an axis of ``grid_coords`` is empty or not 1-D.
        ShapeError: If ``centers`` does not match the grid dimension.
        ConfigurationError: If a width is not strictly positive.

    Examples:
        Three Gaussian bumps on a 1-D grid of five points:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> x = jnp.linspace(0.0, 1.0, 5)
        >>> basis = vdx.rbf_basis([x], jnp.array([[0.0], [0.5], [1.0]]), widths=0.3)
        >>> basis.nbasis, basis.output_shape
        (3, (5,))

        A 2-D grid gives increments on the grid shape:

        >>> y = jnp.linspace(0.0, 1.0, 3)
        >>> b2 = vdx.rbf_basis([y, x], jnp.array([[0.5, 0.5]]), widths=0.4)
        >>> b2.operg(0.0, jnp.ones(1)).shape
        (3, 5)

        A non-positive width is rejected:

        >>> vdx.rbf_basis(
        ...     [x], jnp.zeros((1, 1)), widths=0.0
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: widths must be strictly positive; got min=0.0.
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
        variance: Scalar or per-coefficient prior variance;
            with ``spectral_density`` it acts as a scalar amplitude.
        spectral_density: Optional callable mapping the eigenvalues
            $\lambda_j$ to per-mode prior power (HSGP prior).

    Algorithm:
        1. Flatten the grid, recentre it, and enlarge the half-spans by
           ``boundary_scale``.
        2. Build the tensor-product Dirichlet eigenfunctions $\Phi$ and
           eigenvalues $\lambda$ with ``num_modes`` modes per axis.
        3. Set the prior variance to ``variance`` or, with a spectral
           density, to ``variance * spectral_density(lambda)``.

    Returns:
        `LinearBasis` with ``output_shape`` the grid shape.

    Raises:
        ConfigurationError: If ``boundary_scale < 1``.
        GridError: If an axis is empty, not 1-D, or has zero span.

    Examples:
        Four sine modes on a 1-D grid:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> x = jnp.linspace(0.0, 1.0, 6)
        >>> basis = vdx.fourier_basis([x], 4)
        >>> basis.nbasis, basis.output_shape
        (4, (6,))

        A spectral density turns the prior into an HSGP prior with decaying
        variance:

        >>> hsgp = vdx.fourier_basis(
        ...     [x], 4, spectral_density=lambda lam: 1.0 / (1.0 + lam)
        ... )
        >>> bool(hsgp.variance[0] > hsgp.variance[-1])
        True

        A shrinking boundary scale is rejected:

        >>> vdx.fourier_basis(
        ...     [x], 4, boundary_scale=0.5
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: boundary_scale must be >= 1, got 0.5.
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

    Algorithm:
        1. Build the 1-D or 2-D synthesis matrix $\Phi$ for the chosen
           wavelet and number of levels.
        2. Wrap it with the diagonal prior ``variance``.

    Returns:
        `LinearBasis` with ``output_shape == grid_shape`` and
        ``nbasis == prod(grid_shape)``.

    Raises:
        UnsupportedInputError: If ``grid_shape`` is neither 1-D nor 2-D.

    Examples:
        A Haar basis on 8 points is complete and orthonormal:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> basis = vdx.wavelet_basis(8)
        >>> basis.nbasis
        8
        >>> bool(
        ...     jnp.allclose(
        ...         basis.design_matrix.T @ basis.design_matrix, jnp.eye(8), atol=1e-5
        ...     )
        ... )
        True

        A 2-D grid flattens to ``ny * nx`` coefficients:

        >>> vdx.wavelet_basis((4, 4)).nbasis
        16

        A 3-D grid is rejected:

        >>> vdx.wavelet_basis((2, 2, 2))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        UnsupportedInputError: wavelet_basis supports 1-D or 2-D grids; ...
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
        data: Sample matrix, ``T`` snapshots over the flattened grid.
        n_modes: Number of leading EOFs to keep.
        center: Subtract the per-point sample mean before the SVD.
        variance: ``"explained"`` (default), or a scalar or
            per-coefficient array overriding the prior variance.
        output_shape: Grid shape to reshape increments to; must satisfy
            ``prod(output_shape) == N``. Defaults to ``(N,)``.

    Algorithm:
        1. Optionally subtract the per-point sample mean of ``data``.
        2. Take the SVD and keep the leading ``n_modes`` right singular
           vectors as $\Phi$ with singular values $\sigma_a$.
        3. Set the prior variance to $\sigma_a^2 / (T - 1)$ when
           ``variance="explained"``, otherwise use ``variance``.

    Returns:
        `LinearBasis` with orthonormal columns and ``nbasis == n_modes``.

    Raises:
        ConfigurationError: If ``variance`` is a string other than
            ``"explained"``.
        DataError: If ``variance="explained"`` and ``data`` has fewer than
            two samples.

    Examples:
        Two leading EOFs of ten snapshots over four grid points:

        >>> import jax
        >>> import vardax as vdx
        >>> data = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
        >>> basis = vdx.eof_basis(data, 2)
        >>> basis.design_matrix.shape, basis.variance.shape
        ((4, 2), (2,))

        A single snapshot cannot give an explained variance:

        >>> vdx.eof_basis(data[:1], 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        DataError: variance="explained" needs at least 2 samples (T >= 2); got T=1.
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
