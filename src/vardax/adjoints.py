"""Public ``vardax.adjoints`` API surface.

Re-exports from ``vardax._src.adjoints``. See that module's docstring
for the design rationale.
"""

from vardax._src.adjoints import (
    AbstractAdjoint,
    ImplicitAdjoint,
    KStepAdjoint,
    OneStepAdjoint,
    RecursiveCheckpointAdjoint,
    to_optimistix_adjoint,
)

__all__ = [
    "AbstractAdjoint",
    "ImplicitAdjoint",
    "KStepAdjoint",
    "OneStepAdjoint",
    "RecursiveCheckpointAdjoint",
    "to_optimistix_adjoint",
]
