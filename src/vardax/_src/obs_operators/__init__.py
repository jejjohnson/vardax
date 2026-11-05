"""Observation operators h: grids, point batches, averaging kernels and instruments.

``MaskedIdentity``, ``LinearObs``, ``CallableObs``, ``AveragingKernel`` and
``InterpObs`` map a state to observations, ``GridPointObs``, ``FieldPointObs`` and
``point_obs_cost`` handle scattered points, and ``MultiInstrumentFusion`` composes
instruments.
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
