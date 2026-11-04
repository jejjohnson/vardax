"""Concatenated reduced basis: several bases stacked into one control vector z.

``operg`` sums the per-component increments and ``prior_inv`` is block-diagonal.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vardax._src.errors import ConfigurationError


class CompositeBasis(eqx.Module):
    """Concatenate several reduced bases into one control vector.

        z = [z_1, …, z_C]
        δx = ∑_c Φ_c z_c
        Σ_z⁻¹ = blockdiag(Σ_z,1⁻¹, …, Σ_z,C⁻¹)

    Satisfies [`ReducedBasis`][vardax.ReducedBasis], with ``nbasis`` the sum of the
    component sizes. Components may mix families (e.g. a broad `rbf_basis` layer
    plus a `wavelet_basis` detail layer), provided their ``operg`` outputs share one
    grid shape.

    Algorithm:
        1. Split z at the cumulative component sizes (``splits``).
        2. ``operg``: apply each component to its slice and sum the grid increments.
        3. ``prior_inv``: apply each component's ``prior_inv`` to its slice and
           concatenate the blocks.

    Attributes:
        components: The reduced bases, in control-vector order.
        splits: Cumulative sizes at which z is cut between the components.

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

        The prior is block-diagonal, so ``prior_inv`` divides each block by its own
        variance:

        >>> bool(jnp.allclose(basis.prior_inv(jnp.ones(3)), jnp.array([0.5, 0.5, 2.0])))
        True

        A single component is the component itself:

        >>> alone = vdx.composite_basis(b1)
        >>> z = jnp.array([1.0, 2.0])
        >>> bool(jnp.allclose(alone.operg(0.0, z), b1.operg(0.0, z)))
        True
    """

    components: tuple[Any, ...]
    splits: tuple[int, ...] = eqx.field(static=True)

    def operg(
        self,
        t: float,
        coefficients: Float[Array, " M"],
        state: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        """Map coefficients to the summed increment: δx = ∑_c Φ_c z_c.

        Args:
            t: Time t, passed to each component.
            coefficients: Coefficient vector z.
            state: Current state x, passed to each component.

        Returns:
            The state increment δx on the components' grid.
        """
        parts = self._split(coefficients)
        # Φ_1 z_1
        increment = self.components[0].operg(t, parts[0], state)
        for component, part in zip(self.components[1:], parts[1:], strict=True):
            # δx ← δx + Φ_c z_c
            increment = increment + component.operg(t, part, state)
        return increment

    def prior_inv(self, coefficients: Float[Array, " M"]) -> Float[Array, " M"]:
        """Apply the block-diagonal prior precision: Σ_z⁻¹ z.

        Args:
            coefficients: Coefficient vector z.

        Returns:
            The vector Σ_z⁻¹ z.
        """
        parts = self._split(coefficients)
        # Σ_z,c⁻¹ z_c
        blocks = [
            component.prior_inv(part)
            for component, part in zip(self.components, parts, strict=True)
        ]
        # Σ_z⁻¹ z = [Σ_z,1⁻¹ z_1, …, Σ_z,C⁻¹ z_C]
        return jnp.concatenate(blocks)

    @property
    def nbasis(self) -> int:
        """Number of coefficients, the sum of the component sizes."""
        return int(sum(component.nbasis for component in self.components))

    def _split(self, coefficients: Float[Array, " M"]) -> list[Float[Array, ...]]:
        """Cut z into the per-component slices z_c."""
        parts, start = [], 0
        for cut in self.splits:
            parts.append(coefficients[start:cut])
            start = cut
        parts.append(coefficients[start:])
        return parts


def composite_basis(*components: Any) -> CompositeBasis:
    """Build a `CompositeBasis` from a sequence of `ReducedBasis` components.

        δx = ∑_c Φ_c z_c

    Args:
        *components: Reduced bases to concatenate, in control-vector order. Their
            ``operg`` outputs must share one grid shape.

    Returns:
        The composite basis, with ``nbasis`` the sum of the component sizes.

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

        The cuts are the cumulative sizes of all but the last component:

        >>> vdx.composite_basis(a, b, b).splits
        (2, 3)

        An empty call is rejected:

        >>> vdx.composite_basis()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: composite_basis requires at least one component.
    """
    if not components:
        raise ConfigurationError("composite_basis requires at least one component.")
    cuts, total = [], 0
    for component in components[:-1]:
        total += component.nbasis
        cuts.append(total)
    return CompositeBasis(components=tuple(components), splits=tuple(cuts))
