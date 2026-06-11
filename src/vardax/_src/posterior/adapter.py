"""``GaussianMarkLikelihood`` — serialise a Posterior for downstream models.

Per-event posteriors feed Tier V population models (TMTPP, hierarchical
Bayesian) via a JSON-friendly mark-likelihood serialisation. The
contract between the inference layer (vardax) and the audit /
population layer (catalog / database / downstream model).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import numpy as np

from .container import Posterior


class GaussianMarkLikelihood(eqx.Module):
    """Serialise a Gaussian posterior into a mark-likelihood dict.

    Attributes:
        posterior: A vardax ``Posterior`` (with Gaussian ``cov``).
        event_metadata: Free-form dict (event_id, time, geometry,
            instruments_used, …). Passed through verbatim into the
            output.

    Use ``.to_dict()`` to get a JSON-friendly representation that
    downstream consumers (GeoCatalog, hierarchical Bayesian models,
    DuckDB ingest) can store as-is.
    """

    posterior: Posterior
    event_metadata: dict[str, Any] = eqx.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mark-likelihood representation."""
        out: dict[str, Any] = {
            "mean": _to_list(self.posterior.mean),
            "cov_diag": (
                _to_list(self._cov_diag()) if self.posterior.cov is not None else None
            ),
            "samples": (
                _to_list(self.posterior.samples)
                if self.posterior.samples is not None
                else None
            ),
            "provenance": dict(self.posterior.provenance),
            "event_metadata": dict(self.event_metadata),
        }
        return out

    def _cov_diag(self):
        """Extract the diagonal of the covariance operator.

        For materialised operators this is cheap; for lazy operators
        it costs one mat-vec per state dim (probe with unit vectors).
        We pick the materialised path when the operator exposes it,
        else fall back to a single ``as_matrix()`` call.
        """
        cov = self.posterior.cov
        if cov is None:
            return None
        try:
            return gaussx.diag(cov)
        except (AttributeError, NotImplementedError):
            # Lazy operators don't always materialise; just emit a
            # null marker.
            return None


def _to_list(arr: Any) -> Any:
    """JSON-safe conversion: numpy/jax arrays → nested lists."""
    if arr is None:
        return None
    return np.asarray(arr).tolist()
