r"""Averaging-kernel observation operator (Decision D9).

For RTM-derived L2 satellite products (TROPOMI CH₄, EMIT CH₄,
OCO CO₂, MOPITT CO, …), the L2 retrieval is not a direct sample of
the mixing-ratio profile $x$ — it smooths through a kernel
matrix $A$ and falls back to a retrieval prior $x_a$
where the signal is weak:

$$
\hat y = A \, (h \odot x + (1 - h) \odot x_a).
$$

Tangent-linear: $H'(x) = A \cdot \mathrm{diag}(h)$.

Skipping the averaging kernel is the most common cause of bias in
operational satellite inversions; vardax exposes it as a first-class
day-one operator (not a future feature).

Satisfies ``pipekit_cycle.ObservationOperator``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx


class AveragingKernel(eqx.Module):
    """RTM L2-style averaging-kernel obs operator.

    Attributes:
        A: Averaging kernel matrix, stored as an
            ``lineax.AbstractLinearOperator``. May be ``MatrixLinearOperator``
            for dense kernels or a structured operator (Kronecker for
            separable kernels, LowRank for retrievals with low DOF).
        x_a: Retrieval prior, same shape as the model state.
        h: Weighting vector (often pressure-weighted).
    """

    A: lx.AbstractLinearOperator
    x_a: Float[Array, N]  # ty:ignore[unresolved-reference]
    h: Float[Array, N]  # ty:ignore[unresolved-reference]

    def __call__(self, x: Float[Array, N]) -> Float[Array, N]:  # ty:ignore[unresolved-reference]
        inner = self.h * x + (1.0 - self.h) * self.x_a
        return self.A.mv(inner)

    def linearize(self, x: Float[Array, N]) -> lx.AbstractLinearOperator:  # ty:ignore[unresolved-reference]
        r"""Tangent-linear operator at ``x``: $A \cdot \mathrm{diag}(h)$."""
        diag_h = lx.DiagonalLinearOperator(self.h)
        return self.A @ diag_h


def averaging_kernel_from_l2(
    A_matrix: Float[Array, "N N"],
    x_a: Float[Array, N],  # ty:ignore[unresolved-reference]
    h: Float[Array, N],  # ty:ignore[unresolved-reference]
) -> AveragingKernel:
    """Convenience constructor from a dense averaging-kernel matrix.

    Wraps the dense matrix in ``lineax.MatrixLinearOperator`` so it
    satisfies the same interface as structured kernels.

    Args:
        A_matrix: Dense averaging-kernel matrix of shape ``(N, N)``.
        x_a: Retrieval prior of shape ``(N,)``.
        h: Weighting vector of shape ``(N,)``.

    Returns:
        Configured ``AveragingKernel`` instance.
    """
    A = lx.MatrixLinearOperator(jnp.asarray(A_matrix))
    return AveragingKernel(A=A, x_a=jnp.asarray(x_a), h=jnp.asarray(h))
