"""Flatten a state and lay flat values back onto a shape, both with einx."""

import einx
from jaxtyping import Array, Float


def flatten(values: Float[Array, "*shape"]) -> Float[Array, " size"]:
    """Flatten ``values`` to one axis, in row-major order.

    Examples:
        >>> import jax.numpy as jnp
        >>> flatten(jnp.ones((2, 3))).shape
        (6,)

        A scalar becomes a length-one vector:

        >>> flatten(jnp.asarray(1.0)).shape
        (1,)
    """
    return einx.id("s... -> (s...)", values)


def as_shape(values: Float[Array, "*any"], shape: tuple[int, ...]) -> Array:
    """Lay ``values`` out on ``shape``; the element count must match.

    Parameterisations and bases may return a flat vector or any shape with
    the state's size, so models put their output onto the background's
    shape with this.

    Examples:
        >>> import jax.numpy as jnp
        >>> as_shape(jnp.arange(6.0), (2, 3)).shape
        (2, 3)

        Already-shaped input is re-laid in row-major order:

        >>> as_shape(jnp.ones((3, 2)), (6,)).shape
        (6,)
    """
    flat = flatten(values)
    return einx.id("(s...) -> s...", flat, s=shape)
