"""Adjoint composition for the FourDVarNet inner solver (Decision D15).

vardax does not own adjoint code. Gradients through the inner learned
solver are composed by selecting an ``optimistix.AbstractAdjoint`` on
the model:

```python
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
```

The choice replaces the v0.3 ``grad_mode: Literal["unrolled",
"one_step", "implicit"]`` enum (dropped in this epic, Decision D15).

This module ships:

- [`KStepAdjoint`][vardax.adjoints.KStepAdjoint] — warmup under
  ``stop_gradient``, then ``k`` differentiable steps (Bolte, Pauwels &
  Vaiter, NeurIPS 2023, generalised). The vardax-owned adjoint;
  targets upstream contribution.
- [`OneStepAdjoint`][vardax.OneStepAdjoint] — the ``k=1`` alias.
- [`to_optimistix_adjoint`][vardax.adjoints.to_optimistix_adjoint] —
  interpreter for the shared ``pipekit_cycle.adjoints`` spec
  vocabulary (``TruncatedAdjoint(k)`` maps here; the dynamics-layer
  counterpart is ``pipekit_jax.to_diffrax_adjoint``).
- Re-exports ``optimistix.RecursiveCheckpointAdjoint`` and
  ``optimistix.ImplicitAdjoint`` for one-stop import.
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
