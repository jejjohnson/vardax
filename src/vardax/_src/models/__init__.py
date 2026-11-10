"""Classical analysis methods, siblings that each implement ``AnalysisStep``.

In the linear-Gaussian limit they all agree with ``OptimalInterpolation``, which
``tests/test_linear_gaussian_agreement.py`` enforces.
"""

from __future__ import annotations

from .dual_threedvar import DualThreeDVar
from .incremental_fourdvar import IncrementalConfig, IncrementalFourDVar
from .optimal_interpolation import OptimalInterpolation
from .reduced_basis_fourdvar import ReducedBasisFourDVar
from .spacetime_field import SpaceTimeFieldResult, SpaceTimeFieldVar
from .strong_fourdvar import StrongFourDVar
from .threedvar import ThreeDVar
from .weak_fourdvar import WeakFourDVar

__all__ = [
    "DualThreeDVar",
    "IncrementalConfig",
    "IncrementalFourDVar",
    "OptimalInterpolation",
    "ReducedBasisFourDVar",
    "SpaceTimeFieldResult",
    "SpaceTimeFieldVar",
    "StrongFourDVar",
    "ThreeDVar",
    "WeakFourDVar",
]
