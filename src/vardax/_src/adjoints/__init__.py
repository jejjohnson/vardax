"""Adjoint composition for the FourDVarNet inner solver.

vardax does not own adjoint code: gradients through the learned inner solver are
composed by choosing an ``optimistix.AbstractAdjoint`` for the model's
``solver_adjoint``, and this package adds
[`KStepAdjoint`][vardax.adjoints.KStepAdjoint],
[`OneStepAdjoint`][vardax.OneStepAdjoint] and
[`to_optimistix_adjoint`][vardax.adjoints.to_optimistix_adjoint], and re-exports the
optimistix adjoints for one-stop import.
"""

from __future__ import annotations

from optimistix import (
    AbstractAdjoint,
    ImplicitAdjoint,
    RecursiveCheckpointAdjoint,
)

from .k_step import KStepAdjoint
from .mapping import to_optimistix_adjoint
from .one_step import OneStepAdjoint

__all__ = [
    "AbstractAdjoint",
    "ImplicitAdjoint",
    "KStepAdjoint",
    "OneStepAdjoint",
    "RecursiveCheckpointAdjoint",
    "to_optimistix_adjoint",
]
