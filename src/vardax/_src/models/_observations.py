"""Observations as the analysis models read them."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def fill_gaps(
    observations: Float[Array, "*obs"], mask: Float[Array, "*obs"]
) -> Float[Array, "*obs"]:
    """Zero the observations where ``mask`` is zero.

    Gappy fields usually hold ``NaN`` at missing cells, and ``0 · NaN`` is
    ``NaN``, so a multiplicative mask would let one gap poison the whole
    residual. A ``NaN`` at an observed cell is kept.

    Args:
        observations: Observations, with any value at masked-out cells.
        mask: Observation mask, zero where a cell is missing.

    Returns:
        ``observations`` with every masked-out cell set to zero.
    """
    return jnp.where(mask != 0, observations, 0.0)
