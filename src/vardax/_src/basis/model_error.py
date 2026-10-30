r"""A space-time basis for the model error of weak-constraint 4D-Var.

$$
\eta(x_n, t_k) = \sum_{c=1}^{C} \sum_{j=1}^{P} \chi_c(t_k) W_{cj} \varphi_j(x_n),
\qquad
\mathrm{vec}_r W = (\Lambda_t \otimes \Lambda_s)^{1/2} \omega,
\quad \omega \sim \mathcal{N}(0, I),
$$

so the model error of a window has $C P$ unknowns instead of $T N$, and its
implied covariance $Q = (\chi \otimes \Phi)(\Lambda_t \otimes \Lambda_s)
(\chi \otimes \Phi)^\top$ is never formed.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vardax._src.errors import ShapeError


class ModelErrorBasis(eqx.Module):
    r"""Separable space-time basis of the model error, with a Kronecker prior.

    The control is the whitened $\omega \in \mathbb{R}^{C \times P}$; the
    coefficients are $W = \Lambda_t^{1/2} \omega \Lambda_s^{1/2}$ and the
    model error after step $k$ is $\eta_k = \Phi W^\top \chi_k$, i.e.
    $\eta = \chi W \Phi^\top$ over the window.

    Attributes:
        temporal: $\chi$, shape ``(T, C)``: column $c$ is mode $c$ at the
            $T$ model-error steps of the window (e.g. a constant, or
            Legendre polynomials in time).
        spatial: $\Phi$, shape ``(N, P)``: column $j$ is spatial mode $j$.
        temporal_variance: $\mathrm{diag}\, \Lambda_t$, shape ``(C,)``.
        spatial_variance: $\mathrm{diag}\, \Lambda_s$, shape ``(P,)``.

    Examples:
        A constant bias over a three-step window on four cells is one
        coefficient, $C = P = 1$:

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
        >>> bias.errors(jnp.full((1, 1), 2.0))  # W = 2 · 1 · 0.5 = 1
        Array([[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.]], dtype=float32)
    """

    temporal: Float[Array, "T C"]
    spatial: Float[Array, "N P"]
    temporal_variance: Float[Array, " C"]
    spatial_variance: Float[Array, " P"]

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        """``(C, P)``, the shape of $\\omega$ and $W$."""
        return (self.temporal.shape[1], self.spatial.shape[1])

    def coefficients(self, whitened: Float[Array, "C P"]) -> Float[Array, "C P"]:
        """$W = \\Lambda_t^{1/2} \\omega \\Lambda_s^{1/2}$.

        Args:
            whitened: The whitened coefficients $\\omega$.

        Returns:
            The coefficients $W$.
        """
        # (Λ_t ⊗ Λ_s)^{1/2} vec_r ω, elementwise for diagonal Λ
        scale = einx.multiply(
            "c, p -> c p",
            jnp.sqrt(self.temporal_variance),
            jnp.sqrt(self.spatial_variance),
        )
        return scale * whitened

    def errors(self, whitened: Float[Array, "C P"]) -> Float[Array, "T N"]:
        """$\\eta = \\chi W \\Phi^\\top$, one row per model-error step.

        Args:
            whitened: The whitened coefficients $\\omega$.

        Returns:
            The model errors $\\eta_k$ of the window.
        """
        coefficients = self.coefficients(whitened)
        # η_kn = Σ_c Σ_j χ_kc W_cj φ_nj
        return einx.dot(
            "t c, c p, n p -> t n", self.temporal, coefficients, self.spatial
        )

    def check_steps(self, n_steps: int) -> None:
        """Raise unless $\\chi$ has one row per model-error step.

        Args:
            n_steps: Model-error steps $T$ of the window (its length − 1).

        Raises:
            ShapeError: If ``temporal`` has a different number of rows.
        """
        if self.temporal.shape[0] != n_steps:
            raise ShapeError(
                f"model_err_basis.temporal has {self.temporal.shape[0]} rows but "
                f"the window has {n_steps} model-error steps (its length - 1)."
            )
