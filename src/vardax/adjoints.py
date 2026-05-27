"""Public ``vardax.adjoints`` API surface.

Re-exports from ``vardax._src.adjoints``. See that module's docstring
for the design rationale (Decision D15).
"""

from vardax._src.adjoints import (
    AbstractAdjoint,
    ImplicitAdjoint,
    OneStepAdjoint,
    RecursiveCheckpointAdjoint,
)

__all__ = [
    "AbstractAdjoint",
    "ImplicitAdjoint",
    "OneStepAdjoint",
    "RecursiveCheckpointAdjoint",
]
