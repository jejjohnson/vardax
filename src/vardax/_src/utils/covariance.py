"""Covariance operators as the models store them."""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
import lineax as lx

_COMPOSITES = (lx.MulLinearOperator, gaussx.Kronecker, gaussx.BlockDiag)


def as_covariance(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Tag a covariance operator positive semidefinite.

    A covariance is PSD by definition, but lineax drops the tag when an
    operator is scaled (``σ² I``) and never sets it on a
    ``DiagonalLinearOperator``, and ``lx.CG`` refuses an untagged operator.

    Args:
        operator: A covariance operator.

    Returns:
        ``operator`` unchanged if it is already PSD, else ``operator``
        wrapped in a PSD ``TaggedLinearOperator``.
    """
    # gaussx.solve hands each part to the solver, and lx.CG refuses an untagged
    # one; the parts of a covariance's product or block form are covariances
    if isinstance(operator, lx.TaggedLinearOperator) and isinstance(
        operator.operator, _COMPOSITES
    ):
        operator = operator.operator
    if isinstance(operator, lx.MulLinearOperator):
        operator = as_covariance(operator.operator) * operator.scalar
    elif isinstance(operator, gaussx.Kronecker):
        operator = gaussx.Kronecker(*(as_covariance(op) for op in operator.operators))
    elif isinstance(operator, gaussx.BlockDiag):
        operator = gaussx.BlockDiag(*(as_covariance(op) for op in operator.operators))
    if lx.is_positive_semidefinite(operator):
        return operator
    return lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)


def inverse_operator(
    operator: lx.AbstractLinearOperator, solver: lx.AbstractLinearSolver
) -> lx.AbstractLinearOperator:
    """Lazy inverse of a PSD operator, keeping gaussx's ``InverseOperator``.

    ``gaussx.inv`` does not look through the PSD tag added here, so a tagged
    structured operator becomes a generic ``InverseOperator``. Its matvec is
    still ``gaussx.solve``, which is structured, and ``gaussx.diag`` of it is
    closed form when the operator is diagonal.

    Args:
        operator: Symmetric positive-definite operator A.
        solver: Linear solver for operators without a closed-form inverse.

    Returns:
        A⁻¹ as a lazy operator, tagged by way of A.
    """
    tagged = lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
    return gaussx.inv(tagged, solver=solver)


def guarded_solve(
    cov_op: lx.AbstractLinearOperator,
    vector: Float[Array, "*shape"],
    solver: lx.AbstractLinearSolver,
) -> tuple[Float[Array, "*shape"], Bool[Array, ""]]:
    r"""$C^{-1} v$, without raising on a vector no solver can handle.

    A line search tries points whose rollout has blown up, and it can only
    reject a trial that returns. ``lx.linear_solve`` raises on such a vector,
    and ``throw=False`` would not help: its JVP and transpose rules solve with
    ``throw=True``, and a minimiser evaluates trial points through them. So a
    non-finite or overflowing vector never reaches the solver.

    Args:
        cov_op: The covariance $C$.
        vector: The vector $v$.
        solver: The solver for $C$.

    Returns:
        $C^{-1} v$, NaN throughout when $v$ was unusable, and whether it was
        usable. The tangent of the unusable branch is zero.
    """
    # |v| ≤ max^¼ keeps vᵀ v and vᵀ C⁻¹ v representable for a reasonable C
    limit = jnp.finfo(vector.dtype).max ** 0.25
    usable = jnp.all(jnp.abs(vector) <= limit)
    safe_vector = jnp.where(usable, vector, jnp.zeros_like(vector))
    # structural rules first, so a diagonal, Kronecker or block-diagonal C skips
    # solver; anything else is lx.linear_solve with solver
    solved = gaussx.solve(cov_op, safe_vector, solver=solver)
    return jnp.where(usable, solved, jnp.full_like(solved, jnp.nan)), usable


def half_weighted_norm(
    cov_op: lx.AbstractLinearOperator,
    vector: Float[Array, "*shape"],
    solver: lx.AbstractLinearSolver,
) -> Float[Array, ""]:
    r"""$\tfrac12 v^\top C^{-1} v$, and $+\infty$ at an unusable trial point.

    Args:
        cov_op: The covariance $C$.
        vector: The vector $v$.
        solver: The solver for $C$.

    Returns:
        The weighted norm. A line search rejects $+\infty$ and backtracks.
    """
    solved, usable = guarded_solve(cov_op, vector, solver)
    safe_vector = jnp.where(usable, vector, jnp.zeros_like(vector))
    safe_solved = jnp.where(usable, solved, jnp.zeros_like(solved))
    # ½ vᵀ C⁻¹ v
    norm = 0.5 * jnp.sum(safe_vector * safe_solved)
    return jnp.where(usable, norm, jnp.full_like(norm, jnp.inf))
