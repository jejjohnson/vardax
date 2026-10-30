"""vardax — Modular variational data assimilation with learned components.

All public symbols are re-exported from the private ``_src`` subpackage so
that user code imports from the top-level namespace:

```python
import vardax

model = vardax.FourDVarNet1D(...)
```
"""

# Bind public submodules into the ``vardax`` namespace so the
# advertised ``vdx.cycle.VarDACycle(...)`` / ``vdx.amortized.X`` /
# ``vdx.adjoints.X`` access pattern works after a plain ``import vardax``.
from vardax import adjoints, amortized, cycle, examples  # noqa: F401
from vardax._src._types import (
    Batch1D,
    Batch2D,
    Batch2DMultivar,
    LSTMState1D,
    LSTMState2D,
    PointObsBatch,
)
from vardax._src.adjoints import (
    ImplicitAdjoint,
    KStepAdjoint,
    OneStepAdjoint,
    RecursiveCheckpointAdjoint,
    to_optimistix_adjoint,
)
from vardax._src.amortized import (
    AmortizedConfig,
    AmortizedPosterior,
    ConditionalFlowHead,
    IdentityObsEncoder,
    MLPObsEncoder,
    RegressionHead,
    ScoreDiffusionHead,
)
from vardax._src.basis import (
    CompositeBasis,
    LinearBasis,
    ModelErrorBasis,
    composite_basis,
    eof_basis,
    fourier_basis,
    linear_basis,
    rbf_basis,
    wavelet_basis,
)
from vardax._src.costs import (
    ResidualPenalty,
    background_cost,
    decomposed_loss,
    obs_cost_1d,
    obs_cost_2d,
    prior_cost,
    strong_variational_cost,
    variational_cost,
    variational_cost_grad,
)
from vardax._src.cycle import VarDACycle, VarSmootherCycle
from vardax._src.errors import (
    ConfigurationError,
    DataError,
    GridError,
    InputTypeError,
    MissingInputError,
    ObservationError,
    ShapeError,
    UnsupportedInputError,
    VardaxError,
)
from vardax._src.grad_mod import (
    ConvLSTMGradMod1D,
    ConvLSTMGradMod2D,
)
from vardax._src.minimisers import (
    MinimiseResult,
    OptaxMinimiser,
    OptimistixMinimiser,
)
from vardax._src.model import (
    FourDVarNet1D,
    FourDVarNet2D,
)
from vardax._src.models import (
    IncrementalConfig,
    IncrementalFourDVar,
    KalmanSmoother,
    OptimalInterpolation,
    ReducedBasisFourDVar,
    SpaceTimeFieldResult,
    SpaceTimeFieldVar,
    StrongFourDVar,
    ThreeDVar,
    WeakFourDVar,
)
from vardax._src.obs_operators import (
    AveragingKernel,
    CallableObs,
    FieldPointObs,
    GridPointObs,
    InstrumentRegistry,
    InstrumentSpec,
    InterpObs,
    LinearObs,
    MaskedIdentity,
    MultiInstrumentFusion,
    interp_obs_from_coords,
    point_obs_cost,
)
from vardax._src.parameterization import (
    LinearParameterization,
    NeuralFieldParameterization,
)
from vardax._src.posterior import (
    EnsembleCovariance,
    GaussianMarkLikelihood,
    GaussNewtonHessian,
    LaplaceCovariance,
    Posterior,
)
from vardax._src.priors import (
    BilinAEPrior1D,
    BilinAEPrior2D,
    BilinAEPrior2DMultivar,
    ConvAEPrior1D,
    IdentityPrior,
    L63Prior,
    L96Prior,
    MLPAEPrior1D,
)
from vardax._src.priors_dynamical import (
    DynamicalPrior,
    DynIncrements,
    DynTrajectory,
)
from vardax._src.protocols import (
    AnalysisStep,
    CostFunction,
    ForwardModel,
    GradModulator,
    Minimiser,
    ObservationOperator,
    PosteriorAdapter,
    Prior,
    ReducedBasis,
    StateParameterization,
    SupportsForwardAdjoint,
    TemporalPrior,
)
from vardax._src.solver import (
    SolverState1D,
    SolverState2D,
    fp_solver_step_1d,
    init_solver_state_1d,
    init_solver_state_2d,
    one_step_solve_4dvarnet_1d,
    one_step_solve_4dvarnet_2d,
    solve_4dvarnet_1d,
    solve_4dvarnet_1d_fixedpoint,
    solve_4dvarnet_2d,
    solver_step_1d,
    solver_step_2d,
)
from vardax._src.training import (
    amortized_nll_loss_fn,
    amortized_train_step,
    eval_step,
    reconstruction_loss,
    train_loss_fn,
    train_step,
)
from vardax._src.utils.dynamical_systems import (
    Lorenz63,
    Lorenz96,
    simulate_lorenz63,
    simulate_lorenz96,
)
from vardax._src.utils.forward import SoftBoundedForward
from vardax._src.utils.patches import time_patches
from vardax._src.utils.validation import (
    assert_adjoint_calibrated,
    assert_posterior_agreement,
    simulation_based_calibration,
)
from vardax._src.utils.viz import (
    plot_l96_grid,
    plot_l96_trajectories,
    plot_reconstruction_comparison,
)

__all__ = [
    # Types
    "Batch1D",
    "Batch2D",
    "Batch2DMultivar",
    "LSTMState1D",
    "LSTMState2D",
    "PointObsBatch",
    # Protocols (pipekit-cycle + vardax-specific)
    "AnalysisStep",
    "CostFunction",
    "ForwardModel",
    "GradModulator",
    "Minimiser",
    "ObservationOperator",
    "PosteriorAdapter",
    "Prior",
    "ReducedBasis",
    "StateParameterization",
    "SupportsForwardAdjoint",
    "TemporalPrior",
    # State parameterizations (issue #84)
    "LinearParameterization",
    "NeuralFieldParameterization",
    # Reduced bases (issue #49)
    "CompositeBasis",
    "LinearBasis",
    "ModelErrorBasis",
    "composite_basis",
    "eof_basis",
    "fourier_basis",
    "linear_basis",
    "rbf_basis",
    "wavelet_basis",
    # Observation operators (Decision D9)
    "AveragingKernel",
    "CallableObs",
    "InstrumentRegistry",
    "InstrumentSpec",
    "InterpObs",
    "LinearObs",
    "MaskedIdentity",
    "MultiInstrumentFusion",
    "interp_obs_from_coords",
    "FieldPointObs",
    "GridPointObs",
    "point_obs_cost",
    # Posterior adapters (Decision D10)
    "EnsembleCovariance",
    "GaussNewtonHessian",
    "GaussianMarkLikelihood",
    "LaplaceCovariance",
    "Posterior",
    # Costs
    "ResidualPenalty",
    "background_cost",
    "decomposed_loss",
    "obs_cost_1d",
    "obs_cost_2d",
    "prior_cost",
    "strong_variational_cost",
    "variational_cost",
    "variational_cost_grad",
    # Priors
    "BilinAEPrior1D",
    "BilinAEPrior2D",
    "BilinAEPrior2DMultivar",
    "ConvAEPrior1D",
    "DynIncrements",
    "DynTrajectory",
    "DynamicalPrior",
    "IdentityPrior",
    "L63Prior",
    "L96Prior",
    "MLPAEPrior1D",
    # Gradient modulators
    "ConvLSTMGradMod1D",
    "ConvLSTMGradMod2D",
    # Adjoints (Decision D15 — replaces v0.1 grad_mode enum)
    "ImplicitAdjoint",
    "KStepAdjoint",
    "OneStepAdjoint",
    "to_optimistix_adjoint",
    "RecursiveCheckpointAdjoint",
    # Minimisers (issue #85)
    "MinimiseResult",
    "OptaxMinimiser",
    "OptimistixMinimiser",
    # Solver
    "SolverState1D",
    "SolverState2D",
    "fp_solver_step_1d",
    "init_solver_state_1d",
    "init_solver_state_2d",
    "one_step_solve_4dvarnet_1d",
    "one_step_solve_4dvarnet_2d",
    "solve_4dvarnet_1d",
    "solve_4dvarnet_1d_fixedpoint",
    "solve_4dvarnet_2d",
    "solver_step_1d",
    "solver_step_2d",
    # Model
    "FourDVarNet1D",
    "FourDVarNet2D",
    # Classical DA methods (Decision D14)
    "IncrementalConfig",
    "IncrementalFourDVar",
    "KalmanSmoother",
    "OptimalInterpolation",
    "ReducedBasisFourDVar",
    "SpaceTimeFieldResult",
    "SpaceTimeFieldVar",
    "StrongFourDVar",
    "ThreeDVar",
    "WeakFourDVar",
    # Amortized inference (Epic 8 / Decision D12)
    "AmortizedConfig",
    "AmortizedPosterior",
    "ConditionalFlowHead",
    "IdentityObsEncoder",
    "MLPObsEncoder",
    "RegressionHead",
    "ScoreDiffusionHead",
    # pipekit cycle integration (Epic 7 / Decision D8)
    "VarDACycle",
    "VarSmootherCycle",
    # Six-step validation gates (Decision D12)
    "assert_adjoint_calibrated",
    "assert_posterior_agreement",
    "simulation_based_calibration",
    # Training
    "amortized_nll_loss_fn",
    "amortized_train_step",
    "eval_step",
    "reconstruction_loss",
    "train_loss_fn",
    "train_step",
    # Dynamical systems
    "Lorenz63",
    "Lorenz96",
    "simulate_lorenz63",
    "simulate_lorenz96",
    "SoftBoundedForward",
    "time_patches",
    # Errors
    "VardaxError",
    "ConfigurationError",
    "DataError",
    "GridError",
    "InputTypeError",
    "MissingInputError",
    "ObservationError",
    "ShapeError",
    "UnsupportedInputError",
    # Visualization
    "plot_l96_grid",
    "plot_l96_trajectories",
    "plot_reconstruction_comparison",
]


# Deprecated names, resolved lazily so that only code that still uses
# them sees the warning. Each maps to its replacement; the object is the
# same, so ``isinstance`` checks keep working.
_DEPRECATED_ALIASES = {
    "Incremental4DVar": (
        "ReducedBasisFourDVar",
        (
            "it minimises the full nonlinear cost over basis coefficients and "
            "is not Gauss-Newton; the Gauss-Newton method is IncrementalFourDVar"
        ),
    ),
}


def __getattr__(name: str):
    """Resolve a deprecated public name to its replacement, with a warning."""
    if name not in _DEPRECATED_ALIASES:
        raise AttributeError(f"module 'vardax' has no attribute {name!r}")
    import warnings

    new_name, reason = _DEPRECATED_ALIASES[name]
    warnings.warn(
        f"vardax.{name} is deprecated and will be removed in the next minor "
        f"release; use vardax.{new_name} ({reason}).",
        DeprecationWarning,
        stacklevel=2,
    )
    return globals()[new_name]
