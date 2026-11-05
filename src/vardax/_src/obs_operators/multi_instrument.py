"""Multi-instrument fusion: each instrument keeps its own h, mask and Σ_ε.

Operators compose at the likelihood level, with no regridding to a common grid and no
shared coordinate system.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src.models._observations import fill_gaps
from vardax._src.obs_operators.tangent import obs_tangent
from vardax._src.utils.covariance import as_covariance


class InstrumentSpec(eqx.Module):
    """One instrument's operator hᵢ, quality mask mᵢ, covariance Σ_ε,i and name.

    A plain container, grouping what ``MultiInstrumentFusion`` needs per instrument.

    Attributes:
        obs_op: Observation operator hᵢ, an ``ObservationOperator`` (typically an
            [`AveragingKernel`][vardax.AveragingKernel]).
        mask: Quality mask mᵢ on the instrument's observation space: ``1`` for a valid
            pixel, ``0`` for a flagged or dropped one.
        obs_cov_op: Observation-error covariance Σ_ε,i as a lineax operator, often a
            ``DiagonalLinearOperator`` of the per-pixel retrieval uncertainty.
        instrument_id: Identifier (e.g. ``"TROPOMI"``, ``"EMIT"``, ``"GHGSat"``).

    Examples:
        Bundle a masked-identity instrument with unit error covariance:

        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> spec = vdx.InstrumentSpec(
        ...     obs_op=vdx.MaskedIdentity(),
        ...     mask=jnp.ones(3),
        ...     obs_cov_op=lx.IdentityLinearOperator(
        ...         jax.eval_shape(lambda: jnp.ones(3))
        ...     ),
        ...     instrument_id="TROPOMI",
        ... )
        >>> spec.instrument_id
        'TROPOMI'

        The operator is applied as it is, whatever the mask holds:

        >>> spec.obs_op(jnp.ones(3)).shape
        (3,)
    """

    # Any, as importing the ObservationOperator protocol would make a cycle
    obs_op: Any
    mask: Float[Array, ...]
    obs_cov_op: lx.AbstractLinearOperator
    instrument_id: str = eqx.field(static=True)


class InstrumentRegistry(eqx.Module):
    """Keyed lookup of ``InstrumentSpec`` by ``instrument_id``.

    Attributes:
        entries: The ``InstrumentSpec`` of each instrument, by ``instrument_id``.

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
        ...         jax.eval_shape(lambda: jnp.ones(2))
        ...     ),
        ...     instrument_id="EMIT",
        ... )
        >>> list(vdx.InstrumentRegistry(entries={"EMIT": spec}).entries)
        ['EMIT']
    """

    entries: dict[str, InstrumentSpec] = eqx.field(default_factory=dict)


class MultiInstrumentFusion(eqx.Module):
    """Compose per-instrument operators at the likelihood level.

        𝒥_o(x) = ∑ᵢ αᵢ ½ ‖mᵢ ⊙ (yᵢ − hᵢ(x))‖²_Σ̃_ε,i⁻¹
        Σ̃_ε,i = Mᵢ Σ_ε,i Mᵢ + (I − Mᵢ),   Mᵢ = diag(mᵢ)

    Operational satellite work combines instruments, each with its own hᵢ, mask mᵢ,
    error covariance Σ_ε,i and possibly its own averaging kernel. There is no shared
    coordinate system: each instrument keeps its native footprint and resolution, and
    ``__call__`` returns one array per instrument.

    Where a single observation vector is required
    (``pipekit_cycle.ObservationOperator``), call ``.to_observation_operator()`` for a
    flattened wrapper and pass ``.obs_cov_op()`` as the model's Σ_ε: its blocks are
    Σ_ε,i / αᵢ, so the model's observation term is the same weighted sum.

    Attributes:
        registry: Per-instrument [`InstrumentRegistry`][vardax.InstrumentRegistry].
        weights: Optional weights αᵢ by instrument id, read by ``cost`` and
            ``obs_cov_op``. ``None``, or an instrument missing from the mapping,
            means αᵢ = 1.

    Examples:
        Two instruments observing the same state, one an identity and one a linear
        projection:

        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> eye = lx.IdentityLinearOperator(jax.eval_shape(lambda: jnp.ones(2)))
        >>> a = vdx.InstrumentSpec(vdx.MaskedIdentity(), jnp.ones(2), eye, "a")
        >>> matrix_operator = lx.MatrixLinearOperator(jnp.array([[1.0, 1.0]]))
        >>> one = lx.IdentityLinearOperator(jax.eval_shape(lambda: jnp.ones(1)))
        >>> b = vdx.InstrumentSpec(
        ...     vdx.LinearObs(operator=matrix_operator), jnp.ones(1), one, "b"
        ... )
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

        A weight of 2 on instrument ``b`` doubles its term in the cost, here
        ½ (1 + 4) + 2 · ½ · 16, and halves its block of the flattened Σ_ε:

        >>> weighted = vdx.MultiInstrumentFusion(fusion.registry, weights={"b": 2.0})
        >>> obs = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([4.0])}
        >>> float(weighted.cost(jnp.zeros(2), obs))
        18.5
        >>> weighted.obs_cov_op().as_matrix().diagonal().tolist()
        [1.0, 1.0, 0.5]
    """

    registry: InstrumentRegistry
    weights: dict[str, float] | None = None

    def __call__(self, x: Float[Array, ...]) -> dict[str, Float[Array, ...]]:
        """Apply every instrument operator to the same state: hᵢ(x) for each i.

        Args:
            x: State x shared by all instruments.

        Returns:
            Predicted observations keyed by instrument id.
        """
        return {
            instrument_id: spec.obs_op(x)
            for instrument_id, spec in self.registry.entries.items()
        }

    def linearize(self, x: Float[Array, ...]) -> dict[str, lx.AbstractLinearOperator]:
        """Per-instrument tangent-linear operators J_hᵢ(x), keyed by instrument id.

        The fused tangent-linear operator is the stack of these.

        Args:
            x: State x at which to linearise.

        Returns:
            The operator J_hᵢ(x) of each instrument, by instrument id.
        """
        return {
            instrument_id: obs_tangent(spec.obs_op, x)
            for instrument_id, spec in self.registry.entries.items()
        }

    def weight(self, instrument_id: str) -> float:
        """The weight αᵢ of one instrument, ``1`` unless given.

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
        """Fused observation cost of the state ``x``.

            𝒥_o(x) = ∑ᵢ αᵢ ½ ‖mᵢ ⊙ (yᵢ − hᵢ(x))‖²_Σ̃_ε,i⁻¹

        Masked-out cells drop out of the likelihood through Σ̃_ε,i even when Σ_ε,i is
        correlated. They may hold ``NaN``; they are ignored.

        Args:
            x: State x shared by all instruments.
            observations: Observations yᵢ keyed by instrument id, one for every
                instrument in the registry.
            solver: Linear solver for Σ̃_ε,i⁻¹ rᵢ. Default ``None``: ``gaussx.solve``'s
                structural dispatch, a direct solve (closed form for a diagonal Σ_ε,i).

        Returns:
            The scalar cost 𝒥_o.
        """
        total = jnp.zeros(())
        for instrument_id, spec in self.registry.entries.items():
            observed = fill_gaps(observations[instrument_id], spec.mask)
            # hᵢ(x)
            predicted_obs = spec.obs_op(x)
            # yᵢ − hᵢ(x)
            innovation = observed - predicted_obs
            # rᵢ = mᵢ ⊙ (yᵢ − hᵢ(x))
            residual = spec.mask * innovation
            # Σ̃_ε,i = Mᵢ Σ_ε,i Mᵢ + (I − Mᵢ), so masked cells drop out
            masked_covariance = gaussx.masked_obs_covariance(
                as_covariance(spec.obs_cov_op), spec.mask
            )
            # ravel, not einx: an instrument may have no observations
            flat_residual = jnp.ravel(residual)
            # Σ̃_ε,i⁻¹ rᵢ on flat vectors, as gaussx's direct solve treats the columns of
            # a 2-D right-hand side as separate systems
            whitened = gaussx.solve(
                _flat_covariance(masked_covariance), flat_residual, solver=solver
            )
            # rᵢᵀ Σ̃_ε,i⁻¹ rᵢ; jnp, not einx: an instrument may have no observations
            squared_norm = jnp.sum(flat_residual * whitened)
            # 𝒥_o += αᵢ ½ rᵢᵀ Σ̃_ε,i⁻¹ rᵢ
            total = total + self.weight(instrument_id) * 0.5 * squared_norm
        return total

    def obs_cov_op(self) -> lx.AbstractLinearOperator:
        """Block-diagonal Σ_ε of the flattened observations, with blocks Σ_ε,i / αᵢ.

        Blocks follow the sorted instrument order of ``to_observation_operator()``, so
        a model given both minimises the weighted fused cost.

        Returns:
            A `gaussx.BlockDiag` on the concatenated observation vector, so solves and
            factors work block by block.
        """
        instrument_ids = sorted(self.registry.entries)
        if not instrument_ids:
            # BlockDiag needs at least one block
            return lx.FunctionLinearOperator(
                lambda vector: vector,
                jax.ShapeDtypeStruct((0,), jnp.float32),
                lx.positive_semidefinite_tag,
            )
        # Σ_ε,i / αᵢ
        blocks = [
            gaussx.scaled_operator(
                _flat_covariance(self.registry.entries[instrument_id].obs_cov_op),
                1.0 / self.weight(instrument_id),
                tags=lx.positive_semidefinite_tag,
            )
            for instrument_id in instrument_ids
        ]
        # blockdiag(Σ_ε,1 / α₁, …, Σ_ε,k / α_k)
        return gaussx.BlockDiag(*blocks)

    def to_observation_operator(self) -> _FlattenedMultiInstrument:
        """Adapt to the strict ``pipekit_cycle.ObservationOperator`` protocol.

        Use this when the consumer requires a single ``(state) -> Array`` signature.

        Returns:
            A wrapper that concatenates the per-instrument outputs.
        """
        return _FlattenedMultiInstrument(fusion=self)


def _flat_covariance(cov: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """``cov`` on flattened vectors, as a `gaussx.BlockDiag` block must act.

    Args:
        cov: A covariance Σ_ε,i on an instrument's observation shape.

    Returns:
        The same covariance acting on 1-D vectors: ``cov`` itself when it already does,
        a diagonal when it is diagonal, else a lazy reshape.
    """
    structure = cov.in_structure()
    if len(structure.shape) == 1:
        return cov
    if lx.is_diagonal(cov):
        return lx.DiagonalLinearOperator(lx.diagonal(cov))
    flat = jax.ShapeDtypeStruct((cov.in_size(),), structure.dtype)

    def flat_mv(vector: Float[Array, " M"]) -> Float[Array, " M"]:
        """Apply Σ_ε,i to a flat vector."""
        # reshape and ravel, not einx: an instrument may have no observations
        return jnp.ravel(cov.mv(vector.reshape(structure.shape)))

    return lx.FunctionLinearOperator(flat_mv, flat, lx.positive_semidefinite_tag)


class _FlattenedMultiInstrument(eqx.Module):
    """``ObservationOperator`` flattening of ``MultiInstrumentFusion``.

    Concatenates the per-instrument observations in sorted id order.
    """

    fusion: MultiInstrumentFusion

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Concatenate the flattened hᵢ(x) in sorted instrument order."""
        per_instrument = self.fusion(x)
        # ravel, not einx: an instrument may have no observations
        flat_per_instrument = [
            jnp.ravel(per_instrument[key]) for key in sorted(per_instrument)
        ]
        # h(x) = (h₁(x), …, h_k(x))
        return jnp.concatenate(flat_per_instrument)

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        """Stacked tangent-linear operator, from autodiff of the flattened map."""
        # lineax calls fn(x, args), and h takes only x
        return lx.JacobianLinearOperator(lambda state, _args=None: self(state), x)
