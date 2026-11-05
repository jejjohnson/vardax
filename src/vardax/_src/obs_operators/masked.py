"""Masked-identity observation operator, for a state observed with gaps."""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, Float
import lineax as lx


class MaskedIdentity(eqx.Module):
    """Identity observation operator with an optional element-wise mask m.

        h(x) = m ⊙ x

    The default operator when the state is observed with gaps: along-track SSH
    altimetry, SST under cloud masks, sparse in-situ data. It has no parameters, as
    the mask arrives at call time, per batch. It satisfies
    ``pipekit_cycle.ObservationOperator``.

    Examples:
        A mask zeroes the entries it drops:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> op = vdx.MaskedIdentity()
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> op(x, mask=jnp.array([1.0, 0.0, 1.0])).tolist()
        [1.0, 0.0, 3.0]
        >>> type(op.linearize(x)).__name__
        'IdentityLinearOperator'

        Without a mask the operator is the identity:

        >>> bool(jnp.all(op(x) == x))
        True

        A per-batch mask broadcasts over the batch axis:

        >>> xb = jnp.ones((2, 3))
        >>> op(xb, mask=jnp.array([1.0, 0.0, 1.0])).tolist()[1]
        [1.0, 0.0, 1.0]
    """

    def __call__(
        self,
        x: Float[Array, ...],
        mask: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        """Apply the mask to the state: h(x) = m ⊙ x.

        Args:
            x: State x.
            mask: Mask m, broadcastable to ``x``. ``None`` leaves ``x`` unchanged.

        Returns:
            The predicted observations.
        """
        if mask is None:
            return x
        # h(x) = m ⊙ x
        return x * mask

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``x``: the identity, J_h = I.

        Args:
            x: State x at which to linearise.

        Returns:
            The identity operator on the shape of ``x``.
        """
        return lx.IdentityLinearOperator(jax.eval_shape(lambda: x))
