"""Public ``vardax.amortized`` API surface.

Re-exports from ``vardax._src.amortized`` so namespaced access like
``vdx.amortized.AmortizedPosterior(...)`` works.
"""

from vardax._src.amortized import (
    AmortizedConfig,
    AmortizedPosterior,
    ConditionalFlowHead,
    IdentityObsEncoder,
    MLPObsEncoder,
    RegressionHead,
    ScoreDiffusionHead,
)

__all__ = [
    "AmortizedConfig",
    "AmortizedPosterior",
    "ConditionalFlowHead",
    "IdentityObsEncoder",
    "MLPObsEncoder",
    "RegressionHead",
    "ScoreDiffusionHead",
]
