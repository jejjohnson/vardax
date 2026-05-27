"""Observation operator family.

All operators satisfy ``pipekit_cycle.ObservationOperator`` directly
(Decision D8): they implement ``__call__(state) -> obs`` and
``linearize(state) -> AbstractLinearOperator``. The tangent-linear
operator is used by incremental 4DVar (Epic 4) and posterior covariance
computations (Epic 5).

Decision D9 makes ``AveragingKernel`` and ``MultiInstrumentFusion``
first-class day-one operators (not deferred to a later epic) since
they're required for any RTM-derived L2 satellite product.

Public surface:

- :class:`MaskedIdentity`     — :math:`H(x) = m \\odot x`
- :class:`LinearObs`          — :math:`H(x) = H_\\text{mat} \\cdot x`
- :class:`AveragingKernel`    — :math:`H(x) = A(h \\cdot x + (1-h) x_a)`
- :class:`MultiInstrumentFusion` — per-instrument composition at the
  likelihood level
- :class:`InstrumentRegistry` — :math:`\\{instrument\\_id : InstrumentSpec\\}`
- :class:`InstrumentSpec`     — :math:`(obs\\_op, mask, R\\_op, id)` tuple
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
