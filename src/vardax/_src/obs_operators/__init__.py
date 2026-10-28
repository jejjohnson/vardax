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
- [`InterpObs`][vardax.InterpObs] — sparse bilinear grid-to-points
  interpolation
- [`LinearObs`][vardax.LinearObs] — $H(x) = H_\text{mat} \cdot x$
- [`CallableObs`][vardax.CallableObs] — $H(x) = f(x)$ for any
  differentiable JAX function, linearised by autodiff
- [`AveragingKernel`][vardax.AveragingKernel] —
  $H(x) = A(h \cdot x + (1-h) x_a)$
- [`MultiInstrumentFusion`][vardax.MultiInstrumentFusion] —
  per-instrument composition at the likelihood level
- [`InstrumentRegistry`][vardax.InstrumentRegistry] —
  ``{instrument_id: InstrumentSpec}``
- [`InstrumentSpec`][vardax.InstrumentSpec] —
  ``(obs_op, mask, obs_cov_op, id)`` tuple
- [`GridPointObs`][vardax.GridPointObs] — gridded state or trajectory at
  scattered, time-stamped points ([`PointObsBatch`][vardax.PointObsBatch])
- [`FieldPointObs`][vardax.FieldPointObs] — a callable field at scattered
  points
- [`point_obs_cost`][vardax.point_obs_cost] — the masked,
  $\sigma$-weighted cost of a point batch
"""

from __future__ import annotations

from .averaging_kernel import AveragingKernel
from .callable import CallableObs
from .interp_obs import InterpObs, interp_obs_from_coords
from .linear import LinearObs
from .masked import MaskedIdentity
from .multi_instrument import (
    InstrumentRegistry,
    InstrumentSpec,
    MultiInstrumentFusion,
)
from .point import FieldPointObs, GridPointObs, point_obs_cost

__all__ = [
    "AveragingKernel",
    "CallableObs",
    "FieldPointObs",
    "GridPointObs",
    "InstrumentRegistry",
    "InstrumentSpec",
    "InterpObs",
    "LinearObs",
    "MaskedIdentity",
    "MultiInstrumentFusion",
    "interp_obs_from_coords",
    "point_obs_cost",
]
