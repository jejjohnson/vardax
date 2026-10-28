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
    r"""RTM L2-style averaging-kernel obs operator.

    $$H(x) = A \, (h \odot x + (1 - h) \odot x_a)$$

    Algorithm:
        1. Blend the state with the retrieval prior:
           $z = h \odot x + (1 - h) \odot x_a$.
        2. Apply the kernel: $\hat y = A z$.
        3. ``linearize`` returns $A \, \mathrm{diag}(h)$ without evaluating
           anything, because the map is affine in $x$.

    Attributes:
        A: Averaging kernel matrix, stored as an
            ``lineax.AbstractLinearOperator``. May be ``MatrixLinearOperator``
            for dense kernels or a structured operator (Kronecker for
            separable kernels, LowRank for retrievals with low DOF).
        x_a: Retrieval prior, same shape as the model state.
        h: Weighting vector (often pressure-weighted).

    Examples:
        With $h = 1$ the prior drops out and the operator is just $A x$:

        >>> import jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> A = jnp.array([[0.5, 0.5], [0.0, 1.0]])
        >>> kernel = lx.MatrixLinearOperator(A)
        >>> op = vdx.AveragingKernel(A=kernel, x_a=jnp.zeros(2), h=jnp.ones(2))
        >>> bool(jnp.allclose(op(jnp.array([1.0, 3.0])), A @ jnp.array([1.0, 3.0])))
        True

        Where $h = 0$ the output falls back to the retrieval prior $x_a$:

        >>> op0 = vdx.AveragingKernel(
        ...     A=kernel, x_a=jnp.array([2.0, 2.0]), h=jnp.zeros(2)
        ... )
        >>> bool(jnp.allclose(op0(jnp.array([9.0, 9.0])), A @ jnp.array([2.0, 2.0])))
        True

        The tangent-linear operator is $A \, \mathrm{diag}(h)$:

        >>> bool(jnp.allclose(op.linearize(jnp.zeros(2)).as_matrix(), A))
        True
    """

    A: lx.AbstractLinearOperator
    x_a: Float[Array, " N"]
    h: Float[Array, " N"]

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        inner = self.h * x + (1.0 - self.h) * self.x_a
        return self.A.mv(inner)

    def linearize(self, x: Float[Array, " N"]) -> lx.AbstractLinearOperator:
        r"""Tangent-linear operator at ``x``: $A \cdot \mathrm{diag}(h)$.

        Args:
            x: State at which to linearise; unused because the map is
                affine.

        Returns:
            The operator $A \, \mathrm{diag}(h)$.
        """
        diag_h = lx.DiagonalLinearOperator(self.h)
        return self.A @ diag_h


def averaging_kernel_from_l2(
    A_matrix: Float[Array, "N N"],
    x_a: Float[Array, " N"],
    h: Float[Array, " N"],
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
