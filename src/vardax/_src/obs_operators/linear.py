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
    """``H(x) = H_mat @ x``."""

    H_mat: lx.AbstractLinearOperator

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        return self.H_mat.mv(x)

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``x``: just ``H_mat`` (linear)."""
        return self.H_mat
