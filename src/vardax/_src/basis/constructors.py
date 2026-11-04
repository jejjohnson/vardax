"""Family constructors for `LinearBasis`, one per family of design matrices Φ.

Each builds the geometry, takes Φ from the matching pure function of
``geonnax.basis``, attaches the diagonal prior Σ_z and returns a `LinearBasis`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import einx
import geonnax.basis as gnx_basis
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float
import numpy as np

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
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Turn per-axis coordinates into the stacked points s_n and the grid shape."""
    if len(grid_coords) < 1:
        raise GridError("grid_coords must contain at least one axis array.")
    axes = [np.asarray(axis_values, dtype=float) for axis_values in grid_coords]
    for axis_number, axis in enumerate(axes):
        if axis.ndim != 1 or axis.size < 1:
            raise GridError(
                f"grid_coords[{axis_number}] must be a non-empty 1-D axis array; "
                f"got shape {axis.shape}."
            )
    mesh = np.meshgrid(*axes, indexing="ij")
    stacked_mesh = np.stack(mesh)
    # one row s_n per grid point, with the first axis varying slowest
    points = einx.id("d s... -> (s...) d", stacked_mesh)
    return points, tuple(axis.size for axis in axes)


def rbf_basis(
    grid_coords: Sequence[Float[ArrayLike, " _"]],
    centers: Float[ArrayLike, "M d"],
    *,
    widths: float | Float[ArrayLike, " M"],
    kernel: Literal["gaussian", "wendland_c2", "wendland_c4"] = "gaussian",
    variance: float | Float[ArrayLike, " M"] = 1.0,
) -> LinearBasis:
    """Radial-basis reduced basis of Gaussian or Wendland atoms.

        [Φ]_nd = φ_d(s_n) = k(‖s_n − c_d‖ / ℓ_d)
        Σ_z = diag(σ²)

    ``kernel="gaussian"`` reproduces the MASSH stationary Gaussian-RBF basis; the
    compactly supported ``"wendland_c2"`` and ``"wendland_c4"`` give columns that
    are exactly zero past their width.

    Algorithm:
        1. Flatten the grid axes into the points s_n.
        2. Evaluate k(‖s_n − c_d‖ / ℓ_d) for every atom d.
        3. Wrap the design matrix Φ with the diagonal prior Σ_z.

    Args:
        grid_coords: Per-axis 1-D coordinate arrays, e.g. ``(y_axis, x_axis)`` for
            a 2-D grid, in any dimension D_s ≥ 1.
        centers: Atom centres c_d, one row per atom with D_s columns.
        widths: Scalar or per-atom width ℓ_d, a Gaussian lengthscale or a Wendland
            support radius; must be positive.
        kernel: Radial profile k: ``"gaussian"``, ``"wendland_c2"`` or
            ``"wendland_c4"``.
        variance: Scalar or per-coefficient prior variance σ².

    Returns:
        A `LinearBasis` with ``output_shape`` the grid shape and one coefficient
        per atom.

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
    num_dims = points.shape[1]
    centers_array = np.asarray(centers, dtype=float)
    if centers_array.ndim != 2 or centers_array.shape[1] != num_dims:
        raise ShapeError(
            f"centers must be (M, {num_dims}) to match {num_dims}-D grid_coords; "
            f"got {centers_array.shape}."
        )
    num_atoms = centers_array.shape[0]
    widths_input = np.asarray(widths, dtype=float)
    widths_array = np.broadcast_to(widths_input, (num_atoms,))
    if np.any(widths_array <= 0):
        raise ConfigurationError(
            f"widths must be strictly positive; got min={float(widths_array.min())}."
        )
    # [Φ]_nd = k(‖s_n − c_d‖ / ℓ_d)
    design_matrix = gnx_basis.rbf_basis(
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
    """Laplacian-eigenfunction reduced basis on a box, Dirichlet or HSGP.

        −∇²_s φ_d = γ_d φ_d,   [Φ]_nd = φ_d(s_n)
        Σ_z = diag(σ²),   σ_d² = σ_k² S(γ_d)

    The φ_d are tensor-product sine eigenfunctions on the recentred, slightly
    enlarged grid bounding box, with eigenvalues γ_d. Passing a kernel's spectral
    density as S gives the Hilbert-space GP (HSGP) approximation of that kernel's
    prior with amplitude σ_k²; without one, σ² is ``variance``.

    Algorithm:
        1. Flatten the grid, recentre it, and enlarge the half-spans by
           ``boundary_scale``.
        2. Build the Dirichlet eigenfunctions Φ and eigenvalues γ with
           ``num_modes`` modes per axis.
        3. Set the prior variance to ``variance`` or, with a spectral density, to
           σ_k² S(γ).

    Args:
        grid_coords: Per-axis 1-D coordinate arrays in any dimension D_s ≥ 1. Each
            axis must span a non-degenerate interval.
        num_modes: Number of 1-D modes per axis; an ``int`` is broadcast to all
            axes. ``nbasis`` is the product.
        boundary_scale: Factor ≥ 1 from the grid half-spans to the domain
            half-widths. The eigenfunctions vanish on the box boundary, so a value
            above 1 keeps the basis expressive at the grid edges.
        variance: Scalar or per-coefficient prior variance σ²; with
            ``spectral_density`` it is the scalar amplitude σ_k².
        spectral_density: Optional callable S from the eigenvalues γ_d to the
            per-mode prior power (HSGP prior).

    Returns:
        A `LinearBasis` with ``output_shape`` the grid shape.

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
        ...     [x], 4, spectral_density=lambda gamma: 1.0 / (1.0 + gamma)
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
    # s_min = minₙ s_n
    lower = einx.min("[n] d", points)
    # s_max = maxₙ s_n
    upper = einx.max("[n] d", points)
    # half-span = (s_max − s_min) / 2
    half_span = (upper - lower) / 2.0
    if np.any(half_span <= 0):
        raise GridError(
            "each grid_coords axis must span a non-degenerate interval "
            f"(got half-spans {tuple(half_span.tolist())})."
        )
    # s_center = (s_min + s_max) / 2
    box_center = (lower + upper) / 2.0
    # s_n − s_center
    centered = points - box_center
    num_dims = points.shape[1]
    # half-width = half-span · boundary_scale
    half_widths = tuple(float(span) * boundary_scale for span in half_span)
    modes_per_axis = (
        (int(num_modes),) * num_dims
        if isinstance(num_modes, int)
        else tuple(int(axis_modes) for axis_modes in num_modes)
    )
    # −∇²_s φ_d = γ_d φ_d on the box, with φ_d = 0 on its boundary
    design_matrix, eigenvalues = gnx_basis.fourier_basis(
        jnp.asarray(centered), modes_per_axis, half_widths
    )
    # σ_d² = σ_k² S(γ_d)
    prior_variance = (
        variance * spectral_density(eigenvalues)
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
    """Orthonormal wavelet (DWT) reduced basis on a periodic grid.

        ΦᵀΦ = I,   Σ_z = diag(σ²)

    The Haar or Daubechies synthesis basis is complete, one coefficient per grid
    point, so it is a multiscale re-parameterisation of the grid, and the
    per-coefficient σ² is the natural place to encode scale-dependent prior power.

    Algorithm:
        1. Build the 1-D or 2-D synthesis matrix Φ for the chosen wavelet and
           number of levels.
        2. Wrap it with the diagonal prior Σ_z.

    Args:
        grid_shape: ``n`` or ``(n,)`` for a 1-D grid, ``(ny, nx)`` for 2-D. Every
            extent must be a power of two ≥ 2.
        wavelet: ``"haar"``, ``"db2"`` or ``"db4"``.
        levels: Decomposition levels; defaults to the full cascade.
        variance: Scalar or per-coefficient prior variance σ².

    Returns:
        A `LinearBasis` with ``output_shape == grid_shape`` and
        ``nbasis == prod(grid_shape)``.

    Raises:
        UnsupportedInputError: If ``grid_shape`` is neither 1-D nor 2-D.

    Examples:
        A Haar basis on 8 points is complete and orthonormal:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> basis = vdx.wavelet_basis(8)
        >>> basis.nbasis
        8
        >>> phi = basis.design_matrix
        >>> gram = einx.dot("n row, n col -> row col", phi, phi)
        >>> bool(jnp.allclose(gram, jnp.eye(8), atol=1e-5))
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
        # Φ: 1-D synthesis matrix
        design_matrix = gnx_basis.wavelet_basis_1d(
            int(shape[0]), wavelet=wavelet, levels=levels
        )
    elif len(shape) == 2:
        # Φ: 2-D synthesis matrix
        design_matrix = gnx_basis.wavelet_basis_2d(
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
    """Data-driven EOF (PCA) reduced basis from a sample matrix.

        X̃ = U diag(σ_svd) Φᵀ
        Σ_z = diag(σ²),   σ_d² = σ_svd,d² / (T − 1)

    X̃ holds the ``data`` snapshots as rows, as anomalies about the per-point sample
    mean when ``center`` is set, and the leading ``n_modes`` empirical orthogonal
    functions are the columns of Φ. With ``variance="explained"`` the prior variance
    of each mode is its sample variance, so the background term matches the
    climatology of the training samples.

    Algorithm:
        1. Optionally subtract the per-point sample mean of ``data``.
        2. Take the SVD and keep the leading ``n_modes`` right singular vectors as
           Φ, with singular values σ_svd.
        3. Set σ² to σ_svd² / (T − 1) when ``variance="explained"``, otherwise use
           ``variance``.

    Args:
        data: Sample matrix, ``T`` snapshots over the flattened grid.
        n_modes: Number of leading EOFs to keep.
        center: Subtract the per-point sample mean before the SVD.
        variance: ``"explained"`` (default), or a scalar or per-coefficient array
            overriding the prior variance σ².
        output_shape: Grid shape to reshape increments to; must satisfy
            ``prod(output_shape) == N``. Defaults to ``(N,)``.

    Returns:
        A `LinearBasis` with orthonormal columns and ``nbasis == n_modes``.

    Raises:
        ConfigurationError: If ``variance`` is a string other than ``"explained"``.
        DataError: If ``variance="explained"`` and ``data`` has fewer than two
            samples.

    Examples:
        Two leading EOFs of ten snapshots over four grid points:

        >>> import jax
        >>> import vardax as vdx
        >>> data = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
        >>> basis = vdx.eof_basis(data, 2)
        >>> basis.design_matrix.shape, basis.variance.shape
        ((4, 2), (2,))

        A fixed prior variance replaces the explained one:

        >>> import jax.numpy as jnp
        >>> fixed = vdx.eof_basis(data, 2, variance=0.5)
        >>> bool(jnp.allclose(fixed.variance, 0.5))
        True

        A single snapshot cannot give an explained variance:

        >>> vdx.eof_basis(data[:1], 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        DataError: variance="explained" needs at least 2 samples (T >= 2); got T=1.
    """
    data_array = jnp.asarray(data)
    # X̃ = U diag(σ_svd) Φᵀ, keeping the leading modes
    design_matrix, singular_values = gnx_basis.eof_basis(
        data_array, n_modes, center=center
    )
    if isinstance(variance, str):
        if variance != "explained":
            raise ConfigurationError(
                f'variance must be "explained", a scalar, or an array; '
                f"got {variance!r}."
            )
        num_samples = int(data_array.shape[0])
        if num_samples < 2:
            raise DataError(
                'variance="explained" needs at least 2 samples (T >= 2); '
                f"got T={num_samples}."
            )
        # σ_d² = σ_svd,d² / (T − 1)
        prior_variance: float | Float[Array, " M"] = singular_values**2 / (
            num_samples - 1
        )
    else:
        prior_variance = variance
    return linear_basis(
        design_matrix, variance=prior_variance, output_shape=output_shape
    )
