"""Covariance operators as the models store them."""

from __future__ import annotations

import lineax as lx


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
    if lx.is_positive_semidefinite(operator):
        return operator
    return lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
