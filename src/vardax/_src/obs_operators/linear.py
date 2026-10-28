"""Linear-projection observation operator.

``H(x) = H_mat @ x`` where ``H_mat`` is supplied as a
``lineax.AbstractLinearOperator`` so structure (sparse, low-rank,
Kronecker) can be exploited without materialising the dense matrix.

Use cases: Lagrangian footprint matrices, spectral filtering, spatial
interpolation projections.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx


class LinearObs(eqx.Module):
    r"""$H(x) = H x$ for a linear operator $H$ (``H_mat``).

    Examples:
        >>> import jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> H = lx.MatrixLinearOperator(jnp.eye(3)[:2])
        >>> obs = vdx.LinearObs(H_mat=H)
        >>> obs(jnp.array([1.0, 2.0, 3.0]))
        Array([1., 2.], dtype=float32)
        >>> obs.linearize(jnp.zeros(3)) is H
        True

        A structured operator is applied without a dense matrix:

        >>> diag = vdx.LinearObs(H_mat=lx.DiagonalLinearOperator(jnp.array([2.0, 3.0])))
        >>> diag(jnp.ones(2))
        Array([2., 3.], dtype=float32)

        The tangent-linear operator of a linear map does not depend on the
        linearisation point, so the cost of linearising is zero:

        >>> diag.linearize(jnp.ones(2)) is diag.linearize(jnp.zeros(2))
        True
    """

    H_mat: lx.AbstractLinearOperator

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        return self.H_mat.mv(x)

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``x``: just ``H_mat`` (linear).

        Args:
            x: State at which to linearise; unused.

        Returns:
            The stored operator ``H_mat``.
        """
        return self.H_mat
