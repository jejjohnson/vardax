"""Masked-identity observation operator.

The default operator for "we observe the state with gaps" use cases:
SSH altimetry along-track gaps, SST with cloud masks, sparse in-situ.

Satisfies ``pipekit_cycle.ObservationOperator``.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx


class MaskedIdentity(eqx.Module):
    """:math:`H(x) = m \\odot x` with optional element-wise mask.

    Stateless: no parameters. The mask can be supplied at call time
    (per-batch) rather than baked in.
    """

    def __call__(
        self,
        x: Float[Array, ...],
        mask: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        if mask is None:
            return x
        return x * mask

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``x``: identity (since :math:`H` is linear)."""
        return lx.JacobianLinearOperator(lambda y, _args=None: y, x)
