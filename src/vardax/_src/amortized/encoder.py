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

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class IdentityObsEncoder(eqx.Module):
    """Concatenate ``input`` and ``mask`` into a flat context vector.

    Zero parameters — the encoder is purely structural. The output
    dimension is ``input.size + mask.size``.
    """

    def __call__(
        self,
        input: Float[Array, ...],
        mask: Float[Array, ...],
    ) -> Float[Array, " D"]:
        return jnp.concatenate([input.ravel(), mask.ravel()])


class MLPObsEncoder(eqx.Module):
    """Two-layer MLP encoder from flat ``(input, mask)`` to context.

    Attributes:
        mlp: ``eqx.nn.MLP`` taking ``2 * input_size`` features and
            emitting ``context_dim``.
        input_size: Flattened size of the input field (``T * N`` for
            ``Batch1D``, ``T * H * W`` for ``Batch2D``).
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
        x = jnp.concatenate([input.ravel(), mask.ravel()])
        return self.mlp(x)
