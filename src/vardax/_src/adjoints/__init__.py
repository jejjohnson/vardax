"""Adjoint composition for the FourDVarNet inner solver (Decision D15).

vardax does not own adjoint code. Gradients through the inner learned
solver are composed by selecting an ``optimistix.AbstractAdjoint`` on
the model:

.. code-block:: python

    import optimistix as optx
    from vardax.adjoints import OneStepAdjoint

    model = FourDVarNet1D(
        ...,
        solver_adjoint=OneStepAdjoint(),  # O(1) memory, Bolte 2023
    )
    model = FourDVarNet1D(
        ...,
        solver_adjoint=optx.RecursiveCheckpointAdjoint(),  # default
    )
    model = FourDVarNet1D(
        ...,
        solver_adjoint=optx.ImplicitAdjoint(),  # IFT at fixed point
    )

The choice replaces the v0.3 ``grad_mode: Literal["unrolled",
"one_step", "implicit"]`` enum (dropped in this epic, Decision D15).

This module ships:

- :class:`OneStepAdjoint` — Bolte, Pauwels & Vaiter (NeurIPS 2023);
  the only vardax-owned adjoint. Subclasses
  ``optimistix.AbstractAdjoint`` and targets upstream contribution.
- Re-exports ``optimistix.RecursiveCheckpointAdjoint`` and
  ``optimistix.ImplicitAdjoint`` for one-stop import.
"""

from __future__ import annotations

from optimistix import (
    AbstractAdjoint,
    ImplicitAdjoint,
    RecursiveCheckpointAdjoint,
)

from .one_step import OneStepAdjoint

__all__ = [
    "AbstractAdjoint",
    "ImplicitAdjoint",
    "OneStepAdjoint",
    "RecursiveCheckpointAdjoint",
]
