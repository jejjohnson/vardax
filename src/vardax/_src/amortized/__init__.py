r"""Amortized inference (Epic 8 / Decision D12).

Learns a conditional density $q_\phi(x \mid y)$ that maps
observations directly to posteriors. Trained by simulation: sample
$x \sim p(x)$, simulate $y \mid x$, train
$q_\phi$ to recover $x$ from $y$. Per-event inference
is a single forward pass — the cost amortises over many events.

Three head choices via ``AmortizedConfig.head_type``:

- ``"regression"`` — Gaussian head ``(μ_φ(y), Σ_φ(y))``. Cheapest.
- ``"flow"`` — Conditional normalising flow. Exact density, exact
  samples. **Stub** — requires ``gauss_flows``.
- ``"score"`` — Score-based diffusion. Sampling-only, handles
  multimodal posteriors. **Stub** — pending diffrax reverse-SDE.

Validation gates (six-step cycle, Decision D12) live in
``vardax.utils.validation``.
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
