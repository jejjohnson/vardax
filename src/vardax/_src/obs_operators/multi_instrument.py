r"""Multi-instrument observation-operator fusion (Decision D9).

Operational satellite work combines multiple instruments. Each has
its own $H_i$, mask $m_i$, error covariance $R_i$,
and possibly its own averaging kernel. ``MultiInstrumentFusion``
composes per-instrument operators at the **likelihood level** — no
pre-regridding to a common grid, no assumption of shared coordinate
systems.

The fused observation cost:

$$
J_\text{obs}(x) = \sum_{i \in \mathcal{I}} \alpha_i \cdot
                  \tfrac{1}{2} \|m_i \odot (y_i - H_i(x))\|^2_{R_i^{-1}}.
$$

This module ships:

- [`InstrumentSpec`][vardax.InstrumentSpec] — ``(obs_op, mask, R_op, id)``
  tuple
- [`InstrumentRegistry`][vardax.InstrumentRegistry] —
  ``{instrument_id: InstrumentSpec}``
- [`MultiInstrumentFusion`][vardax.MultiInstrumentFusion] — composes the
  registry; returns per-instrument ``dict[str, Array]`` of predicted
  observations. For strict-protocol contexts
  (``pipekit_cycle.ObservationOperator``), the
  ``.to_observation_operator()`` adapter flattens to a single output
  with a block-diagonal linear operator.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx


class InstrumentSpec(eqx.Module):
    """Per-instrument ``(obs_op, mask, R_op, id)`` bundle.

    Attributes:
        obs_op: ``ObservationOperator``-conforming operator (typically
            [`AveragingKernel`][vardax.AveragingKernel]).
        mask: Quality mask of shape compatible with the instrument's
            observation space. ``1`` for valid pixels, ``0`` for
            flagged/dropped.
        R_op: Observation-error covariance as a
            ``lineax.AbstractLinearOperator``. Often a
            ``DiagonalLinearOperator`` keyed on the per-pixel
            retrieval uncertainty.
        instrument_id: Identifier (e.g. ``"TROPOMI"``, ``"EMIT"``,
            ``"GHGSat"``).
    """

    obs_op: Any  # ObservationOperator (we don't import the Protocol to avoid a cycle)
    mask: Float[Array, ...]
    R_op: lx.AbstractLinearOperator
    instrument_id: str = eqx.field(static=True)


class InstrumentRegistry(eqx.Module):
    """Keyed lookup of ``InstrumentSpec`` by ``instrument_id``.

    Attributes:
        entries: ``dict[instrument_id, InstrumentSpec]``.
    """

    entries: dict[str, InstrumentSpec] = eqx.field(default_factory=dict)


class MultiInstrumentFusion(eqx.Module):
    r"""Compose per-instrument operators at the likelihood level.

    ``__call__`` returns ``dict[instrument_id, predicted_obs]`` — one
    array per instrument. The cost function consumes the dict and
    sums per-instrument terms with their respective $R_i^{-1}$.
    There is no shared coordinate system; each instrument keeps its
    native footprint and resolution.

    For strict ``pipekit_cycle.ObservationOperator`` contexts where a
    single observation vector + single linear operator are required,
    call ``.to_observation_operator()`` for a flattened wrapper.

    Attributes:
        registry: Per-instrument
            [`InstrumentRegistry`][vardax.InstrumentRegistry].
        weights: Optional ``{instrument_id: alpha}`` mapping. ``None``
            ⇒ uniform $\alpha_i = 1$.
    """

    registry: InstrumentRegistry
    weights: dict[str, float] | None = None

    def __call__(self, x: Float[Array, ...]) -> dict[str, Float[Array, ...]]:
        return {
            inst_id: spec.obs_op(x) for inst_id, spec in self.registry.entries.items()
        }

    def linearize(self, x: Float[Array, ...]) -> dict[str, lx.AbstractLinearOperator]:
        """Per-instrument tangent-linear operators.

        Returns ``{instrument_id: H_i'(x)}``. The fused tangent linear
        is the block-diagonal stack of these — assembled lazily by
        the cost function or via ``to_observation_operator()``.
        """
        return {
            inst_id: spec.obs_op.linearize(x)
            for inst_id, spec in self.registry.entries.items()
        }

    def to_observation_operator(self) -> _FlattenedMultiInstrument:
        """Adapt to the strict ``pipekit_cycle.ObservationOperator`` protocol.

        Returns a wrapper that concatenates per-instrument outputs and
        exposes a block-diagonal linear operator. Use this when the
        consumer requires a single ``(state) -> Array`` signature.
        """
        return _FlattenedMultiInstrument(fusion=self)


class _FlattenedMultiInstrument(eqx.Module):
    """``ObservationOperator``-compliant flattening of ``MultiInstrumentFusion``.

    Concatenates per-instrument observations along axis 0; the
    tangent-linear becomes a vertical stack of the per-instrument
    Jacobians.
    """

    fusion: MultiInstrumentFusion

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        per_inst = self.fusion(x)
        return jnp.concatenate([per_inst[k].ravel() for k in sorted(per_inst)], axis=0)

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        # Stack the per-instrument Jacobians as a block via JacobianLinearOperator
        # on the flattened forward map.
        return lx.JacobianLinearOperator(self.__call__, x)
