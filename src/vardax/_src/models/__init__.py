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
- [`IncrementalFourDVar`][vardax.IncrementalFourDVar] — strong 4DVar by
  Gauss–Newton outer / CG inner iterations
- [`ReducedBasisFourDVar`][vardax.ReducedBasisFourDVar] — strong 4DVar
  with the control in a reduced basis, control = X with x_0 = x_b + ΦX
- [`SpaceTimeFieldVar`][vardax.SpaceTimeFieldVar] — neural-field 4DVar,
  control = the weights of one space-time field (no rollout, no B)

Learned methods live alongside in ``vardax._src.model`` (the
``FourDVarNet*`` classes). They satisfy the same protocol contract.

The Decision D14 invariant — all methods agree with
``OptimalInterpolation`` in the linear-Gaussian limit — is enforced
by ``tests/test_linear_gaussian_agreement.py``.
"""

from __future__ import annotations

from .incremental_fourdvar import IncrementalConfig, IncrementalFourDVar
from .optimal_interpolation import OptimalInterpolation
from .reduced_basis_fourdvar import ReducedBasisFourDVar
from .spacetime_field import SpaceTimeFieldResult, SpaceTimeFieldVar
from .strong_fourdvar import StrongFourDVar
from .threedvar import ThreeDVar
from .weak_fourdvar import WeakFourDVar

__all__ = [
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
