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

from vardax._src.errors import ConfigurationError


class CompositeBasis(eqx.Module):
    r"""Concatenate several reduced bases into one control vector.

    Satisfies `vardax.protocols.ReducedBasis`. ``nbasis`` is the sum of
    component sizes; ``operg`` slices ``X``, applies each component,
    sums the per-grid increments; ``prior_inv`` slices ``X`` and
    concatenates the per-component blocks. Components may mix families
    (e.g. a broad `rbf_basis` layer plus a `wavelet_basis` detail
    layer), provided their ``operg`` outputs share one grid shape.

    $$\delta x = \sum_c \Phi_c X_c$$

    $$Q^{-1} = \mathrm{blockdiag}(Q_1^{-1}, \ldots, Q_C^{-1})$$

    Algorithm:
        1. Split $X$ at the cumulative component sizes (``splits``).
        2. ``operg``: apply each component to its slice and sum the grid
           increments.
        3. ``prior_inv``: apply each component's ``prior_inv`` to its slice
           and concatenate the blocks.

    Examples:
        Two linear bases on a 3-point grid, stacked into one control vector:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> b1 = vdx.linear_basis(jnp.eye(3)[:, :2], variance=2.0)
        >>> b2 = vdx.linear_basis(jnp.ones((3, 1)), variance=0.5)
        >>> basis = vdx.composite_basis(b1, b2)
        >>> basis.nbasis
        3
        >>> basis.operg(0.0, jnp.array([1.0, 2.0, 3.0])).shape
        (3,)

        The prior is block-diagonal, so ``prior_inv`` divides each block by
        its own variance:

        >>> bool(jnp.allclose(basis.prior_inv(jnp.ones(3)), jnp.array([0.5, 0.5, 2.0])))
        True
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
    """Build a `CompositeBasis` from a sequence of `ReducedBasis` components.

    Args:
        *components: Reduced bases to concatenate, in control-vector order.
            Their ``operg`` outputs must share one grid shape.

    Returns:
        The composite basis, with ``nbasis`` the sum of component sizes.

    Raises:
        ConfigurationError: If no component is given.

    Examples:
        Composing two bases and checking the control-vector size:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> a = vdx.linear_basis(jnp.eye(2))
        >>> b = vdx.linear_basis(jnp.ones((2, 1)))
        >>> vdx.composite_basis(a, b).nbasis
        3

        An empty call is rejected:

        >>> vdx.composite_basis()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: composite_basis requires at least one component.
    """
    if not components:
        raise ConfigurationError("composite_basis requires at least one component.")
    cuts, total = [], 0
    for c in components[:-1]:
        total += c.nbasis
        cuts.append(total)
    return CompositeBasis(components=tuple(components), splits=tuple(cuts))
