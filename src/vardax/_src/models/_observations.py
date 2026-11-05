"""Observations as the analysis models read them."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def fill_gaps(
    observations: Float[Array, "*obs"], mask: Float[Array, "*obs"]
) -> Float[Array, "*obs"]:
    """Zero the observations y where the mask m is zero.

        yₙ ← yₙ if mₙ ≠ 0, else 0

    Gappy fields usually hold ``NaN`` at missing cells, and ``0 · NaN`` is ``NaN``, so
    a multiplicative mask would let one gap poison the whole residual. A ``NaN`` at an
    observed cell is kept.

    Args:
        observations: Observations y, with any value at masked-out cells.
        mask: Observation mask m, zero where a cell is missing.

    Returns:
        ``observations`` with every masked-out cell set to zero.

    Examples:
        A gap holding ``NaN`` becomes zero:

        >>> import jax.numpy as jnp
        >>> gappy = jnp.array([1.0, jnp.nan, 3.0])
        >>> fill_gaps(gappy, jnp.array([1.0, 0.0, 1.0])).tolist()
        [1.0, 0.0, 3.0]

        A ``NaN`` at an observed cell stays, so it is not silently dropped:

        >>> bool(jnp.isnan(fill_gaps(gappy, jnp.ones(3))[1]))
        True

        The dtype of the observations is kept, and an empty input stays empty:

        >>> fill_gaps(gappy, jnp.ones(3)).dtype == gappy.dtype
        True
        >>> fill_gaps(jnp.zeros((0, 2)), jnp.zeros((0, 2))).shape
        (0, 2)
    """
    # 0 · NaN is NaN, so select rather than multiply
    return jnp.where(mask != 0, observations, 0.0)
