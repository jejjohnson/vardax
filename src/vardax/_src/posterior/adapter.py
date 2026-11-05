"""``GaussianMarkLikelihood``: a Posterior as a JSON-friendly mark likelihood.

It is the contract between the inference layer and the population models,
catalogues and databases downstream, which read μ_x|y and diag(Σ_x|y) per event.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
from jaxtyping import Array, Float
import lineax as lx
import numpy as np

from .container import Posterior

# gaussx.diag_inv: a dense Cholesky up to this size, then Hutchinson
_DIAG_INV_DENSE_LIMIT = 2048
# its exact structured paths at any size
_DIAG_INV_STRUCTURED = (
    lx.DiagonalLinearOperator,
    gaussx.BlockDiag,
    gaussx.BlockTriDiag,
    gaussx.Kronecker,
    gaussx.KroneckerSum,
    gaussx.LowRankUpdate,
)


class GaussianMarkLikelihood(eqx.Module):
    """Serialise a Gaussian posterior into a mark-likelihood dict.

        p(x ∣ y) ≈ 𝒩(x; μ_x|y, Σ_x|y)  ↦  {μ_x|y, diag(Σ_x|y), x⁽ᵉ⁾, provenance}

    A JSON-friendly summary of a [`Posterior`][vardax.Posterior]: its mean, the
    diagonal of its covariance, any samples, the provenance and the event metadata.
    ``.to_dict()`` returns a form that downstream consumers (GeoCatalog, hierarchical
    Bayesian models, DuckDB ingest) can store as it is.

    Attributes:
        posterior: The ``Posterior`` to serialise, with a Gaussian ``cov``.
        event_metadata: Free-form dict (event_id, time, geometry,
            instruments_used, …), passed through verbatim into the output.

    Examples:
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> post = vdx.Posterior(mean=jnp.zeros(3))
        >>> mark = vdx.GaussianMarkLikelihood(
        ...     posterior=post, event_metadata={"event_id": "e1"}
        ... )
        >>> sorted(mark.to_dict())
        ['cov_diag', 'event_metadata', 'mean', 'provenance', 'samples']

        Without a covariance the diagonal is ``None``:

        >>> mark.to_dict()["cov_diag"] is None
        True

        Composed with an ensemble posterior, the diagonal and samples are
        serialised as nested lists:

        >>> analyses = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 8.0]])
        >>> ens = vdx.EnsembleCovariance(n_members=3)(analyses, None, None)
        >>> out = vdx.GaussianMarkLikelihood(posterior=ens).to_dict()
        >>> len(out["cov_diag"]), len(out["samples"])
        (2, 3)
    """

    posterior: Posterior
    event_metadata: dict[str, Any] = eqx.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mark-likelihood representation.

        Returns:
            A dict with keys ``mean``, ``cov_diag``, ``samples``,
            ``provenance`` and ``event_metadata``; arrays become nested
            lists and missing parts become ``None``.
        """
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

    def _cov_diag(self) -> Float[Array, ...] | None:
        """diag(Σ_x|y), from Λ_x|y when that is exact, else from Σ_x|y.

        Returns:
            The marginal variances, or ``None`` without a covariance or when a
            lazy operator cannot give its diagonal.
        """
        covariance = self.posterior.cov
        if covariance is None:
            return None
        precision = self.posterior.precision
        try:
            if precision is not None and _has_exact_diag_inv(precision):
                # diag(Σ_x|y) = diag(Λ_x|y⁻¹), without forming the inverse
                return gaussx.diag_inv(precision)
            # diag(Σ_x|y)
            return gaussx.diag(covariance)
        except (AttributeError, NotImplementedError):
            # a lazy operator does not always materialise: emit a null marker
            return None


def _has_exact_diag_inv(precision: lx.AbstractLinearOperator) -> bool:
    """Whether ``gaussx.diag_inv`` is exact for Λ_x|y: structured, or small enough.

    Above ``_DIAG_INV_DENSE_LIMIT`` an unstructured operator would get a
    stochastic estimate, which an adapter must not switch to silently.
    """
    while isinstance(precision, lx.TaggedLinearOperator):
        precision = precision.operator
    return (
        isinstance(precision, _DIAG_INV_STRUCTURED)
        or precision.in_size() <= _DIAG_INV_DENSE_LIMIT
    )


def _to_list(arr: Any) -> Any:
    """Convert a NumPy or JAX array to nested lists, keeping ``None``."""
    if arr is None:
        return None
    return np.asarray(arr).tolist()
