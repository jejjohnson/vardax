"""The tangent-linear operator of any observation operator, as a lineax operator."""

from __future__ import annotations

from typing import Any

from jaxtyping import Array, Float
import lineax as lx


def obs_tangent(obs_op: Any, state: Float[Array, ...]) -> lx.AbstractLinearOperator:
    r"""$H'(x)$ as a lineax operator, whatever ``obs_op`` returns from ``linearize``.

    ``pipekit_cycle``'s observation operators are carrier-agnostic: ``IdentityObs``
    and ``LinearObs`` linearise to themselves, ``CallableObs`` only when given a
    ``linearize_fn``, and ``CompositeObs`` to a composition of those. Anything that
    does not give a lineax operator takes its tangent by autodiff of ``obs_op``.

    Args:
        obs_op: The observation operator.
        state: The state at which to linearise.

    Returns:
        The tangent-linear operator at ``state``.
    """
    try:
        tangent = obs_op.linearize(state)
    except NotImplementedError:
        tangent = None
    if isinstance(tangent, lx.AbstractLinearOperator):
        return tangent
    return lx.JacobianLinearOperator(lambda x, _args=None: obs_op(x), state)
