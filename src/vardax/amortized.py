"""Public ``vardax.amortized`` API surface (Epic 8 / Decision D12).

Re-exports from ``vardax._src.amortized`` so namespaced access like
``vdx.amortized.AmortizedPosterior(...)`` works for users following
the design docs.
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
