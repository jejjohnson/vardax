"""Classical and learned DA analysis methods (Decisions D14, D11, D16).

Each class is a sibling — no parent/child relationship between methods.
All implement ``pipekit_cycle.AnalysisStep`` via ``.as_analysis_step()``.

Classical methods:

- [`OptimalInterpolation`][vardax.OptimalInterpolation] — BLUE / OI,
  closed-form linear-Gaussian
- [`ThreeDVar`][vardax.ThreeDVar] — 3DVar, nonlinear H, single time
- [`StrongFourDVar`][vardax.StrongFourDVar] — strong-constraint 4DVar,
  control = x_0
- [`WeakFourDVar`][vardax.WeakFourDVar] — weak-constraint 4DVar,
  control = (x_0, η)

Learned methods live alongside in ``vardax._src.model`` (the
``FourDVarNet*`` classes). They satisfy the same protocol contract.

The Decision D14 invariant — all methods agree with
``OptimalInterpolation`` in the linear-Gaussian limit — is enforced
by ``tests/test_linear_gaussian_agreement.py``.
"""

from __future__ import annotations

from .incremental_4dvar import Incremental4DVar
from .incremental_fourdvar import IncrementalConfig, IncrementalFourDVar
from .optimal_interpolation import OptimalInterpolation
from .strong_fourdvar import StrongFourDVar
from .threedvar import ThreeDVar
from .weak_fourdvar import WeakFourDVar

__all__ = [
    "Incremental4DVar",
    "IncrementalConfig",
    "IncrementalFourDVar",
    "OptimalInterpolation",
    "StrongFourDVar",
    "ThreeDVar",
    "WeakFourDVar",
]
