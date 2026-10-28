"""Observation encoders for ``AmortizedPosterior``.

The encoder maps a single sample's ``(input, mask)`` to a context
vector consumed by the head. Vardax ships two reference encoders:

- ``IdentityObsEncoder`` — concatenates ``input`` and ``mask`` and
  flattens. No learned parameters; useful as a baseline.
- ``MLPObsEncoder`` — small MLP, trainable. Sensible default for
  small-to-moderate state sizes.

Users are encouraged to provide custom encoders (Conv, attention, ...)
for higher-dimensional inputs.
"""

from __future__ import annotations

import einx
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from vardax._src.utils.shapes import flatten


class IdentityObsEncoder(eqx.Module):
    r"""Concatenate ``input`` and ``mask`` into a flat context vector.

    Zero parameters, the encoder is purely structural:
    $c = [\mathrm{vec}(y); \mathrm{vec}(m)]$. The output dimension is
    ``input.size + mask.size``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> enc = vdx.IdentityObsEncoder()
        >>> enc(jnp.ones((3, 4)), jnp.zeros((3, 4))).shape
        (24,)

        The first half is the data and the second half the mask:

        >>> ctx = enc(jnp.ones((2, 2)), jnp.zeros((2, 2)))
        >>> ctx.tolist()
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        It feeds a [`RegressionHead`][vardax.RegressionHead] whose context size
        is twice the field size:

        >>> import jax
        >>> head = vdx.RegressionHead(
        ...     context_dim=24, state_shape=(3, 4), key=jax.random.PRNGKey(0)
        ... )
        >>> head.map_estimate(enc(jnp.ones((3, 4)), jnp.ones((3, 4)))).shape
        (3, 4)
    """

    def __call__(
        self,
        input: Float[Array, ...],
        mask: Float[Array, ...],
    ) -> Float[Array, " D"]:
        return einx.id("a, b -> (a + b)", flatten(input), flatten(mask))


class MLPObsEncoder(eqx.Module):
    r"""Two-layer MLP encoder from flat ``(input, mask)`` to context.

    Computes $c = \mathrm{MLP}([\mathrm{vec}(y); \mathrm{vec}(m)])$.

    Args:
        input_size: Flattened size of the input field (``T * N`` for
            ``Batch1D``, ``T * H * W`` for ``Batch2D``).
        context_dim: Dimension of the context vector $c$.
        hidden_dim: Width of the MLP hidden layers.
        depth: Number of hidden layers of the MLP.
        key: PRNG key for the MLP weights.

    Attributes:
        mlp: ``eqx.nn.MLP`` taking ``2 * input_size`` features and
            emitting ``context_dim``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> enc = vdx.MLPObsEncoder(
        ...     input_size=12, context_dim=6, key=jax.random.PRNGKey(0)
        ... )
        >>> enc(jnp.ones((3, 4)), jnp.ones((3, 4))).shape
        (6,)

        Any field shape with the right size is flattened first:

        >>> enc(jnp.ones((12,)), jnp.ones((2, 6))).shape
        (6,)

        It is the encoder of an [`AmortizedPosterior`][vardax.AmortizedPosterior],
        where it is vmapped over the batch:

        >>> head = vdx.RegressionHead(
        ...     context_dim=6, state_shape=(3, 4), key=jax.random.PRNGKey(1)
        ... )
        >>> posterior = vdx.AmortizedPosterior(enc, head, vdx.AmortizedConfig())
        >>> ones = jnp.ones((2, 3, 4))
        >>> posterior(vdx.Batch1D(input=ones, mask=ones)).shape
        (2, 3, 4)
    """

    mlp: eqx.nn.MLP
    input_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        context_dim: int,
        *,
        hidden_dim: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
    ) -> None:
        self.input_size = input_size
        self.mlp = eqx.nn.MLP(
            in_size=2 * input_size,
            out_size=context_dim,
            width_size=hidden_dim,
            depth=depth,
            key=key,
        )

    def __call__(
        self,
        input: Float[Array, ...],
        mask: Float[Array, ...],
    ) -> Float[Array, " context_dim"]:
        # features = [vec(y); vec(m)]
        features = einx.id("a, b -> (a + b)", flatten(input), flatten(mask))
        return self.mlp(features)
