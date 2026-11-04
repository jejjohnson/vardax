"""A separable space-time basis for the model error of weak-constraint 4D-Var.

The model error of a window has C P unknowns instead of T N, and its implied
covariance Σ_η is never formed.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vardax._src.errors import ShapeError


class ModelErrorBasis(eqx.Module):
    """Separable space-time basis of the model error, with a Kronecker prior.

        η(s_n, t_k) = ∑_c ∑_d χ_c(t_k) W_cd φ_d(s_n),   c = 1, …, C,   d = 1, …, P
        η = χ W Φᵀ,   W = Σ_time^½ ω Σ_space^½,   ω ~ 𝒩(0, I)
        Σ_η = (χ ⊗ Φ) (Σ_time ⊗ Σ_space) (χ ⊗ Φ)ᵀ

    The control is the whitened matrix ω, W holds the coefficients and t_k is the
    time stamp of model-error step k, so the model error after step k is
    η_k = Φ Wᵀ χ_k. Σ_time and Σ_space are the diagonal prior covariances of the
    temporal and spatial modes, and Σ_η is implied, never formed.

    Attributes:
        temporal: Temporal modes χ: column c is mode c at the T model-error steps of
            the window (e.g. a constant, or Legendre polynomials in time).
        spatial: Spatial modes Φ: column d is mode φ_d on the N grid points.
        temporal_variance: Diagonal of Σ_time.
        spatial_variance: Diagonal of Σ_space.

    Examples:
        A constant bias over a three-step window on four cells is one coefficient,
        C = P = 1, and W = 2 · 1 · 0.5 = 1:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> bias = vdx.ModelErrorBasis(
        ...     temporal=jnp.ones((3, 1)),
        ...     spatial=jnp.ones((4, 1)),
        ...     temporal_variance=jnp.ones(1),
        ...     spatial_variance=jnp.full(1, 0.25),
        ... )
        >>> bias.coefficient_shape
        (1, 1)
        >>> errors = bias.errors(jnp.full((1, 1), 2.0))
        >>> errors.shape
        (3, 4)
        >>> bool(jnp.allclose(errors, 1.0))
        True

        ``pattern`` starts from W itself, so it skips the prior scaling:

        >>> bool(jnp.allclose(bias.pattern(jnp.ones((1, 1))), errors))
        True

        A window of another length is rejected:

        >>> bias.check_steps(5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapeError: model_err_basis.temporal has 3 rows but the window has 5 ...
    """

    temporal: Float[Array, "T C"]
    spatial: Float[Array, "N P"]
    temporal_variance: Float[Array, " C"]
    spatial_variance: Float[Array, " P"]

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        """``(C, P)``, the shape of ω and W."""
        return (self.temporal.shape[1], self.spatial.shape[1])

    def coefficients(self, whitened: Float[Array, "C P"]) -> Float[Array, "C P"]:
        """Scale the whitened control: W = Σ_time^½ ω Σ_space^½.

        Args:
            whitened: The whitened coefficients ω.

        Returns:
            The coefficients W.
        """
        # diag(Σ_time)^½
        temporal_scale = jnp.sqrt(self.temporal_variance)
        # diag(Σ_space)^½
        spatial_scale = jnp.sqrt(self.spatial_variance)
        # (Σ_time ⊗ Σ_space)^½ vec_r ω, elementwise because both factors are diagonal
        scale = einx.multiply("c, p -> c p", temporal_scale, spatial_scale)
        # W = Σ_time^½ ω Σ_space^½
        return scale * whitened

    def errors(self, whitened: Float[Array, "C P"]) -> Float[Array, "T N"]:
        """Model errors from the whitened control: η = χ W Φᵀ, one row per step.

        Args:
            whitened: The whitened coefficients ω.

        Returns:
            The model errors η_k of the window.
        """
        # W = Σ_time^½ ω Σ_space^½
        coefficients = self.coefficients(whitened)
        # η = χ W Φᵀ
        return self.pattern(coefficients)

    def pattern(self, coefficients: Float[Array, "C P"]) -> Float[Array, "T N"]:
        """Model errors from the coefficients themselves: η = χ W Φᵀ.

        Args:
            coefficients: The coefficients W, as ``WeakFourDVar`` returns.

        Returns:
            The model errors η_k of the window.
        """
        # η_kn = ∑_c ∑_d χ_kc W_cd φ_d(s_n)
        return einx.dot(
            "t c, c p, n p -> t n", self.temporal, coefficients, self.spatial
        )

    def check_steps(self, n_steps: int) -> None:
        """Raise unless χ has one row per model-error step.

        Args:
            n_steps: Model-error steps T of the window (its length − 1).

        Raises:
            ShapeError: If ``temporal`` has a different number of rows.
        """
        if self.temporal.shape[0] != n_steps:
            raise ShapeError(
                f"model_err_basis.temporal has {self.temporal.shape[0]} rows but "
                f"the window has {n_steps} model-error steps (its length - 1)."
            )
