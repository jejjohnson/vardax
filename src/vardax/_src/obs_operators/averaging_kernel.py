"""Averaging-kernel observation operator, for retrieval-derived L2 products."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx


class AveragingKernel(eqx.Module):
    """Averaging-kernel observation operator of an L2 retrieval.

        h(x) = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_x)

    An L2 retrieval (TROPOMI CH₄, EMIT CH₄, OCO CO₂, MOPITT CO, …) is not a direct
    sample of the mixing-ratio profile x: it smooths through the averaging kernel J_x̂
    and falls back to its retrieval prior μ_x where the signal is weak. Skipping the
    kernel is the most common cause of bias in operational satellite inversions. It
    satisfies ``pipekit_cycle.ObservationOperator``.

    Algorithm:
        1. Blend the state with the retrieval prior: z = w ⊙ x + (1 − w) ⊙ μ_x.
        2. Apply the kernel: h(x) = J_x̂ z.
        3. ``linearize`` returns J_h = J_x̂ diag(w) without evaluating anything,
           because the map is affine in x.

    Attributes:
        kernel_matrix: Averaging kernel J_x̂, as a lineax operator: a
            ``MatrixLinearOperator`` for a dense kernel, or a structured operator
            (Kronecker for a separable kernel, low rank for a retrieval with few
            degrees of freedom).
        prior_profile: Retrieval prior μ_x, the same shape as the state.
        pressure_weights: Weights w, often pressure weights.

    Examples:
        With w = 1 the prior drops out and the operator is just J_x̂ x:

        >>> import jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> matrix = jnp.array([[0.5, 0.5], [0.0, 1.0]])
        >>> kernel = lx.MatrixLinearOperator(matrix)
        >>> op = vdx.AveragingKernel(
        ...     kernel_matrix=kernel,
        ...     prior_profile=jnp.zeros(2),
        ...     pressure_weights=jnp.ones(2),
        ... )
        >>> x = jnp.array([1.0, 3.0])
        >>> bool(jnp.allclose(op(x), matrix @ x))
        True

        Where w = 0 the output falls back to the retrieval prior μ_x:

        >>> prior = jnp.array([2.0, 2.0])
        >>> fallback = vdx.AveragingKernel(
        ...     kernel_matrix=kernel,
        ...     prior_profile=prior,
        ...     pressure_weights=jnp.zeros(2),
        ... )
        >>> bool(jnp.allclose(fallback(jnp.array([9.0, 9.0])), matrix @ prior))
        True

        The tangent-linear operator is J_x̂ diag(w):

        >>> bool(jnp.allclose(op.linearize(jnp.zeros(2)).as_matrix(), matrix))
        True
    """

    kernel_matrix: lx.AbstractLinearOperator
    prior_profile: Float[Array, " N"]
    pressure_weights: Float[Array, " N"]

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        """Predict the retrieved profile: h(x) = J_x̂ (w ⊙ x + (1 − w) ⊙ μ_x).

        Args:
            x: State x.

        Returns:
            The predicted L2 retrieval.
        """
        # w ⊙ x
        weighted_state = self.pressure_weights * x
        # 1 − w
        prior_weights = 1.0 - self.pressure_weights
        # (1 − w) ⊙ μ_x
        weighted_prior = prior_weights * self.prior_profile
        # z = w ⊙ x + (1 − w) ⊙ μ_x
        blended_profile = weighted_state + weighted_prior
        # h(x) = J_x̂ z
        return self.kernel_matrix.mv(blended_profile)

    def linearize(self, x: Float[Array, " N"]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``x``: J_h = J_x̂ diag(w).

        Args:
            x: State at which to linearise; unused because the map is affine.

        Returns:
            The operator J_x̂ diag(w).
        """
        # diag(w)
        weights_operator = lx.DiagonalLinearOperator(self.pressure_weights)
        # J_h = J_x̂ diag(w)
        return self.kernel_matrix @ weights_operator


def averaging_kernel_from_l2(
    kernel_matrix: Float[Array, "N N"],
    prior_profile: Float[Array, " N"],
    pressure_weights: Float[Array, " N"],
) -> AveragingKernel:
    """Build an `AveragingKernel` from a dense averaging-kernel matrix.

    The matrix is wrapped in ``lineax.MatrixLinearOperator``, so it has the same
    interface as a structured kernel.

    Args:
        kernel_matrix: Dense averaging kernel J_x̂.
        prior_profile: Retrieval prior μ_x.
        pressure_weights: Weights w.

    Returns:
        The configured ``AveragingKernel``.

    Examples:
        A dense identity kernel with w = 1 returns the state:

        >>> import jax.numpy as jnp
        >>> op = averaging_kernel_from_l2(jnp.eye(2), jnp.zeros(2), jnp.ones(2))
        >>> bool(jnp.allclose(op(jnp.array([1.0, 3.0])), jnp.array([1.0, 3.0])))
        True

        With w = 0 it returns the prior, and the kernel is a lineax operator:

        >>> op = averaging_kernel_from_l2(jnp.eye(2), jnp.full(2, 2.0), jnp.zeros(2))
        >>> type(op.kernel_matrix).__name__
        'MatrixLinearOperator'
        >>> op(jnp.array([9.0, 9.0])).tolist()
        [2.0, 2.0]
    """
    return AveragingKernel(
        kernel_matrix=lx.MatrixLinearOperator(jnp.asarray(kernel_matrix)),
        prior_profile=jnp.asarray(prior_profile),
        pressure_weights=jnp.asarray(pressure_weights),
    )
