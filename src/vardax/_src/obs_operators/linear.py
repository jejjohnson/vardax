"""Linear observation operator on a lineax operator, so structure is never densified."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx
import pipekit_cycle as _pc


class LinearObs(eqx.Module, _pc.LinearObs):
    """Linear observation operator H held as a lineax operator.

        h(x) = H x

    It is a ``pipekit_cycle.LinearObs`` that stays a pytree, so its operator can be
    traced and differentiated, and a sparse, low-rank or Kronecker H is applied
    without forming the dense matrix: a Lagrangian footprint, a spectral filter or a
    spatial interpolation. A plain ``pipekit_cycle.LinearObs`` also works in every
    vardax model.

    Attributes:
        operator: The linear map H.

    Examples:
        A dense H selects the first two entries:

        >>> import jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> matrix_operator = lx.MatrixLinearOperator(jnp.eye(3)[:2])
        >>> obs = vdx.LinearObs(operator=matrix_operator)
        >>> obs(jnp.array([1.0, 2.0, 3.0])).tolist()
        [1.0, 2.0]
        >>> obs.linearize(jnp.zeros(3)) is matrix_operator
        True

        A structured operator is applied without a dense matrix:

        >>> diag = vdx.LinearObs(
        ...     operator=lx.DiagonalLinearOperator(jnp.array([2.0, 3.0]))
        ... )
        >>> diag(jnp.ones(2)).tolist()
        [2.0, 3.0]

        The tangent-linear operator does not depend on the linearisation point:

        >>> diag.linearize(jnp.ones(2)) is diag.linearize(jnp.zeros(2))
        True
    """

    operator: lx.AbstractLinearOperator

    @property
    def H(self) -> lx.AbstractLinearOperator:
        """``pipekit_cycle.LinearObs``'s name for the operator."""
        return self.operator

    def _apply(self, state: Float[Array, ...]) -> Float[Array, ...]:
        """Forward map h(x) = H x, which ``pipekit_cycle``'s ``__call__`` runs."""
        # h(x) = H x
        return self.operator.mv(state)

    def linearize(  # ty: ignore[invalid-method-override]
        self, state: Float[Array, ...]
    ) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``state``: J_h = H, since h is linear.

        Args:
            state: State at which to linearise; unused.

        Returns:
            The stored operator H.
        """
        return self.operator
