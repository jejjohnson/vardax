"""Generic linear reduced basis: any precomputed design matrix Φ with a diagonal Σ_z.

The basis family (RBF, Fourier, wavelet, EOF, …) lives in the constructor that builds
Φ; see ``vardax._src.basis.constructors``.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float
import numpy as np

from vardax._src.errors import ConfigurationError, ShapeError


def _as_variance(variance: float | Float[ArrayLike, " M"], nbasis: int) -> np.ndarray:
    """Broadcast σ² to one entry per coefficient and require strict positivity."""
    variance_input = np.asarray(variance, dtype=float)
    variance_array = np.broadcast_to(variance_input, (nbasis,))
    if not np.all(np.isfinite(variance_array)):
        raise ConfigurationError("variance must be finite.")
    if np.any(variance_array <= 0):
        raise ConfigurationError(
            "variance must be strictly positive; "
            f"got min={float(variance_array.min())}."
        )
    return variance_array


class LinearBasis(eqx.Module):
    """Generic linear reduced basis with a diagonal prior over the coefficients.

        δx = Φ z,   Σ_z = diag(σ²)
        𝒥_b(z) = ½ zᵀ Σ_z⁻¹ z

    Build it with [`linear_basis`][vardax.linear_basis] or a family constructor
    ([`rbf_basis`][vardax.rbf_basis], [`fourier_basis`][vardax.fourier_basis],
    [`wavelet_basis`][vardax.wavelet_basis], [`eof_basis`][vardax.eof_basis]).
    It satisfies [`ReducedBasis`][vardax.ReducedBasis] and is stationary, so
    ``operg`` ignores its ``t`` argument.

    Attributes:
        design_matrix: Design matrix Φ from the coefficients to the flattened state.
        variance: Prior variances σ², the diagonal of Σ_z.
        output_shape: Grid shape of the increment, with ``prod(output_shape) == N``.

    Examples:
        A hand-built design matrix with a non-unit prior variance:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> basis = vdx.linear_basis(
        ...     jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), variance=2.0
        ... )
        >>> basis.operg(0.0, jnp.array([1.0, 2.0])).shape
        (3,)
        >>> bool(jnp.allclose(basis.prior_inv(jnp.ones(2)), 0.5))
        True

        An ``output_shape`` reshapes the increment onto a grid:

        >>> gridded = vdx.linear_basis(jnp.eye(4), output_shape=(2, 2))
        >>> gridded.operg(0.0, jnp.ones(4)).shape
        (2, 2)

        It conforms to the protocol:

        >>> isinstance(gridded, vdx.ReducedBasis)
        True

    References:
        MASSH ``mapping/src/basis.py`` (``set_basis``, ``operg``, families
        ``GAUSS*``, ``BM``, ``MIOST``, ``WAVELET3D``).
    """

    design_matrix: Float[Array, "N M"]
    variance: Float[Array, " M"]
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def operg(
        self,
        t: float,
        coefficients: Float[Array, " M"],
        state: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        """Map coefficients to the increment on the grid: δx = Φ z.

        Args:
            t: Time t; ignored because the basis is stationary.
            coefficients: Coefficient vector z.
            state: Current state x; ignored by this basis.

        Returns:
            The state increment δx, reshaped onto ``output_shape``.
        """
        # δx = Φ z
        increment = einx.dot("n m, m -> n", self.design_matrix, coefficients)
        return einx.id("(s...) -> s...", increment, s=self.output_shape)

    def prior_inv(self, coefficients: Float[Array, " M"]) -> Float[Array, " M"]:
        """Apply the prior precision: Σ_z⁻¹ z = z / σ².

        Args:
            coefficients: Coefficient vector z.

        Returns:
            The vector Σ_z⁻¹ z.
        """
        # Σ_z⁻¹ z = z / σ²
        return coefficients / self.variance

    @property
    def nbasis(self) -> int:
        """Number of coefficients, the length of z."""
        return int(self.design_matrix.shape[1])


def linear_basis(
    design_matrix: Float[ArrayLike, "N M"],
    *,
    variance: float | Float[ArrayLike, " M"] = 1.0,
    output_shape: tuple[int, ...] | None = None,
) -> LinearBasis:
    """Build a `LinearBasis` from an arbitrary precomputed design matrix.

        δx = Φ z,   Σ_z = diag(σ²)

    The escape hatch for basis families without a dedicated constructor: any design
    matrix (spherical harmonics, Slepian, Gabor frames, hand-rolled, …) becomes a
    `ReducedBasis`.

    Args:
        design_matrix: Design matrix Φ from the coefficients to the flattened state.
        variance: Scalar or per-coefficient prior variance σ².
        output_shape: Grid shape the increment is reshaped to; must satisfy
            ``prod(output_shape) == N``. Defaults to ``(N,)``.

    Returns:
        The configured `LinearBasis`.

    Raises:
        ShapeError: If ``design_matrix`` is not 2-D or ``output_shape`` does not
            multiply to the number of rows.
        ConfigurationError: If ``variance`` is not finite and positive.

    Examples:
        Wrap a matrix; the default grid shape is ``(N,)``:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> basis = vdx.linear_basis(jnp.ones((3, 2)))
        >>> basis.nbasis, basis.output_shape
        (2, (3,))

        The prior variance follows the dtype of Φ:

        >>> vdx.linear_basis(jnp.ones((3, 2))).variance.dtype == jnp.ones(1).dtype
        True

        A mismatched grid shape is rejected:

        >>> vdx.linear_basis(
        ...     jnp.ones((3, 2)), output_shape=(2, 2)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapeError: prod(output_shape)=4 does not match design_matrix rows N=3.
    """
    design_matrix = jnp.asarray(design_matrix)
    if design_matrix.ndim != 2:
        raise ShapeError(
            f"design_matrix must be 2-D (N, M); got shape {design_matrix.shape}."
        )
    state_size = int(design_matrix.shape[0])
    num_coefficients = int(design_matrix.shape[1])
    shape = (
        (state_size,)
        if output_shape is None
        else tuple(int(extent) for extent in output_shape)
    )
    if math.prod(shape) != state_size:
        raise ShapeError(
            f"prod(output_shape)={math.prod(shape)} does not match design_matrix rows "
            f"N={state_size}."
        )
    positive_variance = _as_variance(variance, num_coefficients)
    # the prior variance follows Φ's dtype, so x64 callers keep float64
    variance_dtype = jnp.result_type(design_matrix, float)
    prior_variance = jnp.asarray(positive_variance, dtype=variance_dtype)
    return LinearBasis(
        design_matrix=design_matrix,
        variance=prior_variance,
        output_shape=shape,
    )
