r"""Concatenated reduced basis.

Stacks several `ReducedBasis` components into a single control vector
$X = [X_1, \ldots, X_C]$; ``operg`` sums the per-component increments,
``prior_inv`` is block-diagonal.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class CompositeBasis(eqx.Module):
    """Concatenate several reduced bases into one control vector.

    Satisfies `vardax.protocols.ReducedBasis`. ``nbasis`` is the sum of
    component sizes; ``operg`` slices ``X``, applies each component,
    sums the per-grid increments; ``prior_inv`` slices ``X`` and
    concatenates the per-component blocks. Components may mix families
    (e.g. a broad `rbf_basis` layer plus a `wavelet_basis` detail
    layer), provided their ``operg`` outputs share one grid shape.
    """

    components: tuple[Any, ...]
    splits: tuple[int, ...] = eqx.field(static=True)

    def operg(
        self,
        t: float,
        X: Float[Array, " M"],
        state: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        parts = self._split(X)
        out = self.components[0].operg(t, parts[0], state)
        for comp, part in zip(self.components[1:], parts[1:], strict=True):
            out = out + comp.operg(t, part, state)
        return out

    def prior_inv(self, X: Float[Array, " M"]) -> Float[Array, " M"]:
        parts = self._split(X)
        blocks = [c.prior_inv(p) for c, p in zip(self.components, parts, strict=True)]
        return jnp.concatenate(blocks)

    @property
    def nbasis(self) -> int:
        return int(sum(c.nbasis for c in self.components))

    def _split(self, X: Float[Array, " M"]) -> list[Float[Array, ...]]:
        parts, start = [], 0
        for cut in self.splits:
            parts.append(X[start:cut])
            start = cut
        parts.append(X[start:])
        return parts


def composite_basis(*components: Any) -> CompositeBasis:
    """Build a `CompositeBasis` from a sequence of `ReducedBasis` components."""
    if not components:
        raise ValueError("composite_basis requires at least one component.")
    cuts, total = [], 0
    for c in components[:-1]:
        total += c.nbasis
        cuts.append(total)
    return CompositeBasis(components=tuple(components), splits=tuple(cuts))
