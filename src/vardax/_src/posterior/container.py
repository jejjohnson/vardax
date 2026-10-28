"""``Posterior`` container — the standard output of every analysis.

Carries the posterior mean, covariance (as an ``AbstractLinearOperator``
when a Gaussian / Laplace approximation is used), optional ensemble
samples, and provenance metadata for downstream auditing.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx


class Posterior(eqx.Module):
    r"""Posterior container — Decision D10.

    A plain container for $p(x \mid y) \approx \mathcal{N}(\text{mean}, \text{cov})$,
    plus optional samples and provenance.

    Attributes:
        mean: Posterior mean / MAP estimate.
        cov: Posterior covariance as ``lineax.AbstractLinearOperator``,
            or ``None`` for amortized / sampling-only heads. Stored
            lazily — never materialised into a dense matrix unless the
            user asks.
        samples: Optional posterior samples (from a flow head, ensemble,
            or score-based diffusion). ``None`` for analytical adapters.
        provenance: Free-form dict carrying audit info — at minimum
            ``{forward_model_id, n_iter, J_star, converged,
            vardax_version}``. Consumed by
            [`GaussianMarkLikelihood`][vardax.GaussianMarkLikelihood]
            when serialising for downstream population models.

    Examples:
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> post = vdx.Posterior(mean=jnp.zeros(4))
        >>> post.mean.shape
        (4,)
        >>> post.cov is None
        True
        >>> post.provenance
        {}

        A Gaussian posterior carries a lazy covariance operator and audit
        metadata:

        >>> import jax, lineax as lx
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> gauss = vdx.Posterior(
        ...     mean=jnp.zeros(2), cov=eye, provenance={"adapter": "demo"}
        ... )
        >>> gauss.cov.mv(jnp.ones(2)).shape, gauss.provenance["adapter"]
        ((2,), 'demo')
    """

    mean: Float[Array, ...]
    cov: lx.AbstractLinearOperator | None = None
    samples: Float[Array, ...] | None = None
    provenance: dict[str, Any] = eqx.field(default_factory=dict)
