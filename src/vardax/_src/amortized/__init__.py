"""Amortized inference: a posterior q(x | y; ν) that maps observations to states.

Training is by simulation: draw x ∼ p(x), simulate y | x, and fit ν to recover x
from y. Inference per event is then one forward pass. The head is chosen by
``AmortizedConfig.head_type``: ``"regression"`` is a diagonal Gaussian head,
``"flow"`` a conditional normalising flow (stub, needs ``gauss_flows``) and
``"score"`` a score-based diffusion head (stub, pending a diffrax reverse SDE).
The validation gates of the six-step cycle live in ``vardax.utils.validation``.
"""

from .config import AmortizedConfig
from .encoder import IdentityObsEncoder, MLPObsEncoder
from .heads import ConditionalFlowHead, RegressionHead, ScoreDiffusionHead
from .posterior import AmortizedPosterior

__all__ = [
    "AmortizedConfig",
    "AmortizedPosterior",
    "ConditionalFlowHead",
    "IdentityObsEncoder",
    "MLPObsEncoder",
    "RegressionHead",
    "ScoreDiffusionHead",
]
