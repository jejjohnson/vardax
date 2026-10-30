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

- [`InstrumentSpec`][vardax.InstrumentSpec] — ``(obs_op, mask, obs_cov_op, id)``
  tuple
- [`InstrumentRegistry`][vardax.InstrumentRegistry] —
  ``{instrument_id: InstrumentSpec}``
- [`MultiInstrumentFusion`][vardax.MultiInstrumentFusion] — composes the
  registry; returns per-instrument ``dict[str, Array]`` of predicted
  observations. For strict-protocol contexts
  (``pipekit_cycle.ObservationOperator``), the
  ``.to_observation_operator()`` adapter flattens to a single output,
  and ``.obs_cov_op()`` gives the matching block-diagonal $R_i / \alpha_i$.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src.models._observations import fill_gaps
from vardax._src.utils.covariance import as_covariance


class InstrumentSpec(eqx.Module):
    """Per-instrument ``(obs_op, mask, obs_cov_op, id)`` bundle.

    A plain container: it only groups one instrument's operator, quality
    mask, error covariance and name.

    Attributes:
        obs_op: ``ObservationOperator``-conforming operator (typically
            [`AveragingKernel`][vardax.AveragingKernel]).
        mask: Quality mask of shape compatible with the instrument's
            observation space. ``1`` for valid pixels, ``0`` for
            flagged/dropped.
        obs_cov_op: Observation-error covariance as a
            ``lineax.AbstractLinearOperator``. Often a
            ``DiagonalLinearOperator`` keyed on the per-pixel
            retrieval uncertainty.
        instrument_id: Identifier (e.g. ``"TROPOMI"``, ``"EMIT"``,
            ``"GHGSat"``).

    Examples:
        Bundle a masked-identity instrument with unit error covariance:

        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> spec = vdx.InstrumentSpec(
        ...     obs_op=vdx.MaskedIdentity(),
        ...     mask=jnp.ones(3),
        ...     obs_cov_op=lx.IdentityLinearOperator(
        ...         jax.ShapeDtypeStruct((3,), jnp.float32)
        ...     ),
        ...     instrument_id="TROPOMI",
        ... )
        >>> spec.instrument_id
        'TROPOMI'
    """

    obs_op: Any  # ObservationOperator (we don't import the Protocol to avoid a cycle)
    mask: Float[Array, ...]
    obs_cov_op: lx.AbstractLinearOperator
    instrument_id: str = eqx.field(static=True)


class InstrumentRegistry(eqx.Module):
    """Keyed lookup of ``InstrumentSpec`` by ``instrument_id``.

    Attributes:
        entries: ``dict[instrument_id, InstrumentSpec]``.

    Examples:
        An empty registry by default, then one keyed by instrument id:

        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> vdx.InstrumentRegistry().entries
        {}
        >>> spec = vdx.InstrumentSpec(
        ...     obs_op=vdx.MaskedIdentity(),
        ...     mask=jnp.ones(2),
        ...     obs_cov_op=lx.IdentityLinearOperator(
        ...         jax.ShapeDtypeStruct((2,), jnp.float32)
        ...     ),
        ...     instrument_id="EMIT",
        ... )
        >>> list(vdx.InstrumentRegistry(entries={"EMIT": spec}).entries)
        ['EMIT']
    """

    entries: dict[str, InstrumentSpec] = eqx.field(default_factory=dict)


class MultiInstrumentFusion(eqx.Module):
    r"""Compose per-instrument operators at the likelihood level.

    ``__call__`` returns ``dict[instrument_id, predicted_obs]`` — one
    array per instrument. ``cost`` sums the per-instrument terms
    $\alpha_i \tfrac12 \|m_i \odot (y_i - H_i(x))\|^2_{R_i^{-1}}$.
    There is no shared coordinate system; each instrument keeps its
    native footprint and resolution.

    For strict ``pipekit_cycle.ObservationOperator`` contexts where a
    single observation vector + single linear operator are required,
    call ``.to_observation_operator()`` for a flattened wrapper, and
    pass ``.obs_cov_op()`` as the model's $R$: its blocks are
    $R_i / \alpha_i$, so the model's observation term is the same
    weighted sum.

    Attributes:
        registry: Per-instrument
            [`InstrumentRegistry`][vardax.InstrumentRegistry].
        weights: Optional ``{instrument_id: alpha}`` mapping, read by
            ``cost`` and ``obs_cov_op``. ``None``, or an instrument
            missing from the mapping, means $\alpha_i = 1$.

    Examples:
        Two instruments observing the same state, one an identity and one a
        linear projection:

        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> a = vdx.InstrumentSpec(vdx.MaskedIdentity(), jnp.ones(2), eye, "a")
        >>> H = lx.MatrixLinearOperator(jnp.array([[1.0, 1.0]]))
        >>> one = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1,), jnp.float32))
        >>> b = vdx.InstrumentSpec(vdx.LinearObs(operator=H), jnp.ones(1), one, "b")
        >>> fusion = vdx.MultiInstrumentFusion(
        ...     vdx.InstrumentRegistry(entries={"a": a, "b": b})
        ... )
        >>> preds = fusion(jnp.array([1.0, 2.0]))
        >>> {k: v.shape for k, v in preds.items()}
        {'a': (2,), 'b': (1,)}

        The flattened adapter concatenates the outputs in sorted id order:

        >>> flat = fusion.to_observation_operator()
        >>> flat(jnp.array([1.0, 2.0])).shape
        (3,)

        A weight of 2 on instrument ``b`` doubles its term in the cost and
        halves its block of the flattened $R$:

        >>> weighted = vdx.MultiInstrumentFusion(fusion.registry, weights={"b": 2.0})
        >>> obs = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([4.0])}
        >>> float(weighted.cost(jnp.zeros(2), obs))  # ½ (1 + 4) + 2 · ½ 16
        18.5
        >>> weighted.obs_cov_op().as_matrix().diagonal().tolist()
        [1.0, 1.0, 0.5]
    """

    registry: InstrumentRegistry
    weights: dict[str, float] | None = None

    def __call__(self, x: Float[Array, ...]) -> dict[str, Float[Array, ...]]:
        """Apply every instrument operator to the same state.

        Args:
            x: State shared by all instruments.

        Returns:
            Predicted observations keyed by instrument id.
        """
        return {
            inst_id: spec.obs_op(x) for inst_id, spec in self.registry.entries.items()
        }

    def linearize(self, x: Float[Array, ...]) -> dict[str, lx.AbstractLinearOperator]:
        """Per-instrument tangent-linear operators.

        Returns ``{instrument_id: H_i'(x)}``. The fused tangent linear
        is the block-diagonal stack of these — assembled lazily by
        the cost function or via ``to_observation_operator()``.

        Args:
            x: State at which to linearise.
        """
        return {
            inst_id: spec.obs_op.linearize(x)
            for inst_id, spec in self.registry.entries.items()
        }

    def weight(self, instrument_id: str) -> float:
        """The weight $\alpha_i$ of one instrument, ``1`` unless given.

        Args:
            instrument_id: Key of the instrument in the registry.

        Returns:
            ``weights[instrument_id]``, or ``1.0``.
        """
        if self.weights is None:
            return 1.0
        return self.weights.get(instrument_id, 1.0)

    def cost(
        self,
        x: Float[Array, ...],
        observations: dict[str, Float[Array, ...]],
        solver: lx.AbstractLinearSolver | None = None,
    ) -> Float[Array, ""]:
        r"""Fused observation cost of the state ``x``.

        $$J_\text{obs}(x) = \sum_i \alpha_i \tfrac12
        \|m_i \odot (y_i - H_i(x))\|^2_{R_i^{-1}}$$

        Masked-out cells may hold ``NaN``; they are ignored.

        Args:
            x: State shared by all instruments.
            observations: Observations $y_i$ keyed by instrument id, one
                for every instrument in the registry.
            solver: Linear solver for $R_i^{-1} r_i$. Default ``None``:
                ``lx.AutoLinearSolver(well_posed=True)``, a direct solve
                (closed form for a diagonal $R_i$).

        Returns:
            The scalar cost.
        """
        solver = solver or lx.AutoLinearSolver(well_posed=True)
        total = jnp.zeros(())
        for inst_id, spec in self.registry.entries.items():
            observed = fill_gaps(observations[inst_id], spec.mask)
            # rᵢ = mᵢ ⊙ (yᵢ − Hᵢ(x))
            residual = spec.mask * (observed - spec.obs_op(x))
            # Rᵢ⁻¹ rᵢ
            weighted = lx.linear_solve(
                as_covariance(spec.obs_cov_op), residual, solver=solver
            ).value
            # αᵢ ½ rᵢᵀ Rᵢ⁻¹ rᵢ
            total = total + self.weight(inst_id) * 0.5 * jnp.sum(residual * weighted)
        return total

    def obs_cov_op(self) -> lx.AbstractLinearOperator:
        r"""Block-diagonal $R$ of the flattened observations, blocks $R_i / \alpha_i$.

        Blocks follow the sorted instrument order of
        ``to_observation_operator()``, so a model given both minimises the
        weighted fused cost.

        Returns:
            A lazy PSD operator on the concatenated observation vector.
        """
        ids = sorted(self.registry.entries)
        covs = [self.registry.entries[k].obs_cov_op for k in ids]
        shapes = [cov.in_structure().shape for cov in covs]
        sizes = [int(jnp.prod(jnp.array(shape))) for shape in shapes]
        dtype = covs[0].in_structure().dtype if covs else jnp.float32
        flat = jax.ShapeDtypeStruct((sum(sizes),), dtype)

        def covariance_mv(vector: Float[Array, " M"]) -> Float[Array, " M"]:
            blocks = []
            offset = 0
            for inst_id, cov, shape, size in zip(ids, covs, shapes, sizes, strict=True):
                segment = vector[offset : offset + size].reshape(shape)
                # Rᵢ vᵢ / αᵢ
                block = cov.mv(segment) / self.weight(inst_id)
                blocks.append(jnp.ravel(block))
                offset += size
            return jnp.concatenate(blocks)

        return lx.FunctionLinearOperator(
            covariance_mv, flat, lx.positive_semidefinite_tag
        )

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
        # ravel, not einx: an instrument may have no observations
        flat_per_inst = [jnp.ravel(per_inst[k]) for k in sorted(per_inst)]
        return jnp.concatenate(flat_per_inst)

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        # Stack the per-instrument Jacobians as a block via JacobianLinearOperator
        # on the flattened forward map.
        return lx.JacobianLinearOperator(self.__call__, x)
