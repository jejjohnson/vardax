"""The tangent-linear operator of any observation operator, as a lineax operator."""

from __future__ import annotations

from typing import Any

from jaxtyping import Array, Float
import lineax as lx


def obs_tangent(obs_op: Any, state: Float[Array, ...]) -> lx.AbstractLinearOperator:
    """Tangent-linear operator of h at ``state``, whatever ``linearize`` returns.

        J_h(x) = ∂h/∂x

    ``pipekit_cycle``'s observation operators are carrier-agnostic: ``IdentityObs``
    and ``LinearObs`` linearise to themselves, ``CallableObs`` only when given a
    ``linearize_fn``, and ``CompositeObs`` to a composition of those. Anything that
    does not give a lineax operator takes its tangent by autodiff of ``obs_op``.

    Args:
        obs_op: The observation operator h.
        state: The state x at which to linearise.

    Returns:
        The tangent-linear operator J_h(x).

    Examples:
        An operator that linearises to a lineax operator keeps it:

        >>> import jax.numpy as jnp, pipekit_cycle, vardax as vdx
        >>> state = jnp.array([1.0, 2.0])
        >>> type(obs_tangent(vdx.MaskedIdentity(), state)).__name__
        'IdentityLinearOperator'

        A ``pipekit_cycle.CallableObs`` with no ``linearize_fn`` goes through autodiff:

        >>> obs_op = pipekit_cycle.CallableObs(lambda x: x**2)
        >>> tangent = obs_tangent(obs_op, state)
        >>> bool(jnp.allclose(tangent.mv(jnp.ones(2)), jnp.array([2.0, 4.0])))
        True

        So does an operator that linearises to something else:

        >>> identity = pipekit_cycle.IdentityObs()
        >>> type(obs_tangent(identity, state)).__name__
        'JacobianLinearOperator'
    """
    try:
        tangent = obs_op.linearize(state)
    except NotImplementedError:
        tangent = None
    if isinstance(tangent, lx.AbstractLinearOperator):
        return tangent
    # J_h(x), a JVP and VJP of h
    return lx.JacobianLinearOperator(lambda x, _args=None: obs_op(x), state)
