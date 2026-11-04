"""Linear-projection observation operator.

``H(x) = operator @ x`` where ``operator`` is supplied as a
``lineax.AbstractLinearOperator`` so structure (sparse, low-rank,
Kronecker) can be exploited without materialising the dense matrix.

Use cases: Lagrangian footprint matrices, spectral filtering, spatial
interpolation projections.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx
import pipekit_cycle as _pc


class LinearObs(eqx.Module, _pc.LinearObs):
    r"""$H(x) = H x$ for a linear operator $H$ (``operator``).

    It is a ``pipekit_cycle.LinearObs`` that holds a lineax operator and stays
    a pytree, so its operator can be traced and differentiated. A plain
    ``pipekit_cycle.LinearObs`` also works in every vardax model.

    Examples:
        >>> import jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> H = lx.MatrixLinearOperator(jnp.eye(3)[:2])
        >>> obs = vdx.LinearObs(operator=H)
        >>> obs(jnp.array([1.0, 2.0, 3.0]))
        Array([1., 2.], dtype=float32)
        >>> obs.linearize(jnp.zeros(3)) is H
        True

        A structured operator is applied without a dense matrix:

        >>> diag = vdx.LinearObs(
        ...     operator=lx.DiagonalLinearOperator(jnp.array([2.0, 3.0]))
        ... )
        >>> diag(jnp.ones(2))
        Array([2., 3.], dtype=float32)

        The tangent-linear operator of a linear map does not depend on the
        linearisation point, so the cost of linearising is zero:

        >>> diag.linearize(jnp.ones(2)) is diag.linearize(jnp.zeros(2))
        True
    """

    operator: lx.AbstractLinearOperator

    @property
    def H(self) -> lx.AbstractLinearOperator:
        """``pipekit_cycle.LinearObs``'s name for the operator."""
        return self.operator

    def _apply(self, state: Float[Array, ...]) -> Float[Array, ...]:
        return self.operator.mv(state)

    def linearize(  # ty: ignore[invalid-method-override]
        self, state: Float[Array, ...]
    ) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``state``: just ``operator`` (linear).

        Args:
            state: State at which to linearise; unused.

        Returns:
            The stored operator ``operator``.
        """
        return self.operator
