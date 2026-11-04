"""Covariance operators as the models store them, and guarded solves against them.

Σ is a covariance and v a vector of its size.
"""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
import lineax as lx

_COMPOSITES = (lx.MulLinearOperator, gaussx.Kronecker, gaussx.BlockDiag)


def as_covariance(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Tag a covariance operator Σ positive semidefinite.

    A covariance is PSD by definition, but lineax drops the tag when an operator
    is scaled (σ² I) and never sets it on a ``DiagonalLinearOperator``, and
    ``lx.CG`` refuses an untagged operator.

    Args:
        operator: A covariance operator Σ.

    Returns:
        ``operator`` unchanged if it is already PSD, else ``operator`` wrapped in a
        PSD ``TaggedLinearOperator``.

    Examples:
        A diagonal operator comes back tagged:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> diagonal = lx.DiagonalLinearOperator(jnp.ones(3))
        >>> lx.is_positive_semidefinite(diagonal)
        False
        >>> covariance = as_covariance(diagonal)
        >>> lx.is_positive_semidefinite(covariance)
        True

        An operator that is already PSD is returned as it is:

        >>> as_covariance(covariance) is covariance
        True

        The parts of a Kronecker product are tagged too, for ``gaussx.solve``:

        >>> import gaussx
        >>> product = as_covariance(gaussx.Kronecker(diagonal, diagonal))
        >>> all(lx.is_positive_semidefinite(part) for part in product.operators)
        True
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
    """Lazy inverse A⁻¹ of a PSD operator, keeping gaussx's ``InverseOperator``.

    ``gaussx.inv`` does not look through the PSD tag added here, so a tagged
    structured operator becomes a generic ``InverseOperator``. Its matvec is still
    ``gaussx.solve``, which is structured, and ``gaussx.diag`` of it is closed form
    when the operator is diagonal.

    Args:
        operator: Symmetric positive-definite operator A.
        solver: Linear solver for operators without a closed-form inverse.

    Returns:
        A⁻¹ as a lazy operator, tagged by way of A.

    Examples:
        The matvec of a diagonal operator's inverse divides by the diagonal:

        >>> import gaussx
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> diagonal = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> inverse = inverse_operator(diagonal, lx.Cholesky())
        >>> solved = inverse.mv(jnp.array([2.0, 8.0]))
        >>> bool(jnp.allclose(solved, jnp.array([1.0, 2.0])))
        True

        Its diagonal is available without a solve:

        >>> bool(jnp.allclose(gaussx.diag(inverse), jnp.array([0.5, 0.25])))
        True

        A dense operator goes through the solver:

        >>> dense = lx.MatrixLinearOperator(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
        >>> solved = inverse_operator(dense, lx.Cholesky()).mv(jnp.array([2.0, 8.0]))
        >>> bool(jnp.allclose(solved, jnp.array([1.0, 2.0])))
        True
    """
    tagged = lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
    return gaussx.inv(tagged, solver=solver)


def guarded_solve(
    cov_op: lx.AbstractLinearOperator,
    vector: Float[Array, "*shape"],
    solver: lx.AbstractLinearSolver,
) -> tuple[Float[Array, "*shape"], Bool[Array, ""]]:
    """Solve Σ⁻¹ v, without raising on a vector no solver can handle.

    A line search tries points whose rollout has blown up, and it can only reject
    a trial that returns. ``lx.linear_solve`` raises on such a vector, and
    ``throw=False`` would not help: its JVP and transpose rules solve with
    ``throw=True``, and a minimiser evaluates trial points through them. So a
    non-finite or overflowing vector never reaches the solver.

    Algorithm:
        1. Call v usable when every |vᵢ| ≤ max^¼, with max the largest finite
           value of its dtype.
        2. Replace an unusable v by zeros, and solve Σ⁻¹ v.
        3. Replace the solution of an unusable v by NaN.

    Args:
        cov_op: The covariance Σ.
        vector: The vector v.
        solver: The solver for Σ.

    Returns:
        Σ⁻¹ v, NaN throughout when v was unusable, and whether it was usable. The
        tangent of the unusable branch is zero.

    Examples:
        A usable vector is solved:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = as_covariance(lx.DiagonalLinearOperator(jnp.array([2.0, 4.0])))
        >>> solver = lx.CG(atol=1e-6, rtol=1e-6)
        >>> solved, usable = guarded_solve(covariance, jnp.array([2.0, 8.0]), solver)
        >>> bool(usable), bool(jnp.allclose(solved, jnp.array([1.0, 2.0])))
        (True, True)

        A non-finite vector returns NaN instead of raising:

        >>> solved, usable = guarded_solve(
        ...     covariance, jnp.array([jnp.inf, 0.0]), solver
        ... )
        >>> bool(usable), bool(jnp.all(jnp.isnan(solved)))
        (False, True)

        So does a finite vector above the limit:

        >>> huge = jnp.full(2, jnp.finfo(jnp.zeros(1).dtype).max)
        >>> bool(guarded_solve(covariance, huge, solver)[1])
        False
    """
    # |vᵢ| ≤ max^¼ keeps vᵀ v and vᵀ Σ⁻¹ v representable for a reasonable Σ
    limit = jnp.finfo(vector.dtype).max ** 0.25
    usable = jnp.all(jnp.abs(vector) <= limit)
    safe_vector = jnp.where(usable, vector, jnp.zeros_like(vector))
    # structural rules first, so a diagonal, Kronecker or block-diagonal Σ skips
    # solver; anything else is lx.linear_solve with solver
    # Σ⁻¹ v
    solved = gaussx.solve(cov_op, safe_vector, solver=solver)
    guarded = jnp.where(usable, solved, jnp.full_like(solved, jnp.nan))
    return guarded, usable


def half_weighted_norm(
    cov_op: lx.AbstractLinearOperator,
    vector: Float[Array, "*shape"],
    solver: lx.AbstractLinearSolver,
) -> Float[Array, ""]:
    """Half the squared norm of v weighted by Σ⁻¹, +∞ at an unusable trial point.

        ½ ‖v‖²_Σ⁻¹ = ½ vᵀ Σ⁻¹ v

    Args:
        cov_op: The covariance Σ.
        vector: The vector v.
        solver: The solver for Σ.

    Returns:
        The weighted norm. A line search rejects +∞ and backtracks.

    Examples:
        With Σ = diag(2, 4) and v = (2, 4), ½ vᵀ Σ⁻¹ v = ½ (2 + 4) = 3:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = as_covariance(lx.DiagonalLinearOperator(jnp.array([2.0, 4.0])))
        >>> solver = lx.CG(atol=1e-6, rtol=1e-6)
        >>> norm = half_weighted_norm(covariance, jnp.array([2.0, 4.0]), solver)
        >>> bool(jnp.allclose(norm, 3.0))
        True

        An unusable trial point costs +∞:

        >>> trial = jnp.array([jnp.nan, 0.0])
        >>> bool(jnp.isposinf(half_weighted_norm(covariance, trial, solver)))
        True

        The vector may have any shape; the norm is still a scalar:

        >>> covariance = as_covariance(lx.DiagonalLinearOperator(jnp.ones(4)))
        >>> half_weighted_norm(covariance, jnp.ones(4), solver).shape
        ()
    """
    solved, usable = guarded_solve(cov_op, vector, solver)
    # the NaNs of the unusable branch must not reach the sum or its tangent
    safe_vector = jnp.where(usable, vector, jnp.zeros_like(vector))
    safe_solved = jnp.where(usable, solved, jnp.zeros_like(solved))
    # vᵀ Σ⁻¹ v
    weighted_inner_product = jnp.sum(safe_vector * safe_solved)
    # ½ vᵀ Σ⁻¹ v
    norm = 0.5 * weighted_inner_product
    return jnp.where(usable, norm, jnp.full_like(norm, jnp.inf))
