r"""Observation operator family.

All operators satisfy ``pipekit_cycle.ObservationOperator`` directly
(Decision D8): they implement ``__call__(state) -> obs`` and
``linearize(state) -> AbstractLinearOperator``. The tangent-linear
operator is used by incremental 4DVar (Epic 4) and posterior covariance
computations (Epic 5).

Decision D9 makes ``AveragingKernel`` and ``MultiInstrumentFusion``
first-class day-one operators (not deferred to a later epic) since
they're required for any RTM-derived L2 satellite product.

Public surface:

- [`MaskedIdentity`][vardax.MaskedIdentity] — $H(x) = m \odot x$
- [`LinearObs`][vardax.LinearObs] — $H(x) = H_\text{mat} \cdot x$
- [`AveragingKernel`][vardax.AveragingKernel] —
  $H(x) = A(h \cdot x + (1-h) x_a)$
- [`MultiInstrumentFusion`][vardax.MultiInstrumentFusion] —
  per-instrument composition at the likelihood level
- [`InstrumentRegistry`][vardax.InstrumentRegistry] —
  ``{instrument_id: InstrumentSpec}``
- [`InstrumentSpec`][vardax.InstrumentSpec] —
  ``(obs_op, mask, R_op, id)`` tuple
"""

from __future__ import annotations

from .averaging_kernel import AveragingKernel
from .linear import LinearObs
from .masked import MaskedIdentity
from .multi_instrument import (
    InstrumentRegistry,
    InstrumentSpec,
    MultiInstrumentFusion,
)

__all__ = [
    "AveragingKernel",
    "InstrumentRegistry",
    "InstrumentSpec",
    "LinearObs",
    "MaskedIdentity",
    "MultiInstrumentFusion",
]
