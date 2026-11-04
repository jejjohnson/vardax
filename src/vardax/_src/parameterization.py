"""State parameterizations: a control θ decoded to the state, x = g_dec(θ).

A model given ``parameterization=`` minimises 𝒥(θ) = ℛ(θ) + 𝒥_o(g_dec(θ)), with the
penalty ℛ(θ) in place of the background term;
[`LinearParameterization`][vardax.LinearParameterization] adapts a reduced basis and
[`NeuralFieldParameterization`][vardax.NeuralFieldParameterization] makes the state a
coordinate network on the grid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from vardax._src.errors import ConfigurationError, ShapeError
from vardax._src.utils.shapes import as_shape


class LinearParameterization(eqx.Module):
    """Adapter from a [`ReducedBasis`][vardax.ReducedBasis] to a parameterization.

        x = g_dec(z) = μ_x + Φ(t) z
        ℛ(z) = ½ zᵀ Σ_z⁻¹ z

    The control θ is the coefficient vector z, and t = 0 when no time is given, so a
    model built with ``parameterization=LinearParameterization(basis, background)``
    minimises the same cost as ``ReducedBasisFourDVar(basis=basis)`` with
    ``prior_mean`` equal to μ_x.

    Algorithm:
        1. ``decode`` applies the basis to the coefficients and adds μ_x.
        2. ``penalty`` applies Σ_z⁻¹ through the basis.
        3. ``init_control`` returns the zero coefficient vector.

    Attributes:
        basis: The `ReducedBasis`, with Φ, Σ_z⁻¹ and ``nbasis``.
        background: Prior mean μ_x, the state the increment is added to.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> basis = vardax.linear_basis(jnp.eye(4)[:, :2], variance=2.0)
        >>> param = vardax.LinearParameterization(basis, background=jnp.ones(4))
        >>> z = param.init_control()
        >>> z.shape
        (2,)
        >>> decoded = param.decode(jnp.array([1.0, -1.0]))
        >>> bool(jnp.allclose(decoded, jnp.array([2.0, 0.0, 1.0, 1.0])))
        True
        >>> float(param.penalty(jnp.array([1.0, -1.0])))  # ½ (1 + 1) / 2
        0.5

        The initial control decodes to the background, whatever the basis.

        >>> bool(jnp.allclose(param.decode(z), param.background))
        True

        It satisfies the [`StateParameterization`][vardax.StateParameterization]
        protocol.

        >>> isinstance(param, vardax.StateParameterization)
        True
    """

    basis: Any
    background: Float[Array, ...]

    def decode(self, theta: Float[Array, " M"], t: float | None = None) -> Array:
        """Decode the coefficients: x = μ_x + Φ(t) z, shaped like ``background``.

        Args:
            theta: Coefficient vector z.
            t: Time t at which the basis is evaluated; ``None`` means 0.

        Returns:
            The decoded state x.
        """
        basis_time = 0.0 if t is None else t
        # δx = Φ(t) z
        increment = self.basis.operg(basis_time, theta)
        # x = μ_x + δx
        return self.background + as_shape(increment, self.background.shape)

    def penalty(self, theta: Float[Array, " M"]) -> Float[Array, ""]:
        """Evaluate the background term: ℛ(z) = ½ zᵀ Σ_z⁻¹ z.

        Args:
            theta: Coefficient vector z.

        Returns:
            The scalar penalty ℛ(z).
        """
        # Σ_z⁻¹ z
        precision_times_coefficients = self.basis.prior_inv(theta)
        # ℛ(z) = ½ zᵀ Σ_z⁻¹ z
        return 0.5 * jnp.sum(theta * precision_times_coefficients)

    def init_control(self, *, key: PRNGKeyArray | None = None) -> Array:
        """Return the zero coefficient vector, which decodes to μ_x.

        Args:
            key: Unused; the start is deterministic.

        Returns:
            The zero coefficient vector z = 0.
        """
        del key
        return jnp.zeros((self.basis.nbasis,), dtype=self.background.dtype)


class NeuralFieldParameterization(eqx.Module):
    """A coordinate network evaluated on the grid as the state.

        x = g_dec(θ) = [x(s_n; θ)]ₙ,   n = 1, …, N
        ℛ(θ) = ½ β ‖θ‖²

    The control θ is the array half of the field, ``eqx.filter(field,
    eqx.is_inexact_array)``, and the rest of the module (activations, sizes) is
    fixed. The state has shape ``(N,)`` for a scalar field and ``(N, C)`` for a
    ``C``-channel one. With a time t the field is evaluated at (s_n, t), so a
    space-time field takes ``D + 1`` inputs. There is no Σ_x: the network's spectral
    bias acts as the prior, and the penalty is weight decay.

    Geonnax and equinox keep fixed arrays, such as random Fourier frequencies or
    precomputed tables, as ordinary leaves read through ``jax.lax.stop_gradient``.
    The default filter would make them controls and the weight decay would shrink
    them to zero. Exclude them with ``filter_spec``, e.g.
    ``eqx.tree_at(lambda f: f.B, jax.tree.map(eqx.is_inexact_array, field), False)``;
    they then stay fixed and outside the penalty.

    Algorithm:
        1. ``init_control`` extracts the array leaves of ``field`` selected by
           ``filter_spec`` as θ.
        2. ``decode`` recombines θ with the fixed part of ``field`` and evaluates it
           at every grid coordinate.
        3. ``penalty`` returns the weight decay of θ.

    Attributes:
        field: Single-example network ``field(s) -> (C,)``, for example a
            `geonnax.SIREN`. Its current parameters are the initial control.
        coords: Grid coordinates s_n.
        weight_decay: Weight β ≥ 0 of the penalty.
        filter_spec: Which leaves of ``field`` are the control: an ``equinox``
            filter (a callable on leaves, or a pytree of bools shaped like
            ``field``), used for both halves of the partition. Default
            ``eqx.is_inexact_array``: every floating array.

    Raises:
        ShapeError: If ``coords`` is not two-dimensional.
        ConfigurationError: If ``weight_decay`` is negative.

    Examples:
        >>> import einx, jax, jax.numpy as jnp, geonnax, vardax
        >>> angle = 2 * jnp.pi * jnp.arange(40) / 40
        >>> coords = einx.id("n, n -> n (1 + 1)", jnp.cos(angle), jnp.sin(angle))
        >>> siren = geonnax.SIREN.init(2, 16, 1, depth=3, key=jax.random.PRNGKey(0))
        >>> param = vardax.NeuralFieldParameterization(siren, coords, 1e-4)
        >>> theta = param.init_control()
        >>> param.decode(theta).shape
        (40,)
        >>> bool(param.penalty(theta) > 0)
        True

        With a time, the field takes ``D + 1`` inputs.

        >>> st = geonnax.SIREN.init(3, 16, 1, depth=3, key=jax.random.PRNGKey(1))
        >>> st_param = vardax.NeuralFieldParameterization(st, coords, 0.0)
        >>> st_param.decode(st_param.init_control(), t=0.5).shape
        (40,)

        Negative weight decay is rejected.

        >>> vardax.NeuralFieldParameterization(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     siren, coords, -1.0
        ... )
        Traceback (most recent call last):
        ...
        ConfigurationError: weight_decay must be >= 0, got -1.0.
    """

    field: eqx.Module
    coords: Float[Array, "N D"]
    weight_decay: float = eqx.field(static=True)
    filter_spec: Any = eqx.field(static=True)

    def __init__(
        self,
        field: eqx.Module,
        coords: Float[Array, "N D"],
        weight_decay: float,
        *,
        filter_spec: Any = eqx.is_inexact_array,
    ) -> None:
        coords = jnp.asarray(coords)
        if coords.ndim != 2:
            raise ShapeError(f"coords must be 2-D (N, D); got shape {coords.shape}.")
        if weight_decay < 0:
            raise ConfigurationError(f"weight_decay must be >= 0, got {weight_decay}.")
        self.field = field
        self.coords = coords
        self.weight_decay = weight_decay
        self.filter_spec = filter_spec

    def as_field(self, theta: PyTree[Array]) -> Callable[[Array], Array]:
        """Return the field with parameters θ, as a callable ``s -> (C,)``.

        Args:
            theta: Control θ, the array half of the field.

        Returns:
            The field with θ combined into the fixed part.
        """
        static = eqx.filter(self.field, self.filter_spec, inverse=True)
        return eqx.combine(theta, static)

    def decode(self, theta: PyTree[Array], t: float | None = None) -> Array:
        """Evaluate the field on the grid: x = g_dec(θ) = [x(s_n; θ)]ₙ.

        Args:
            theta: Control θ, the array half of the field.
            t: Time t appended to every coordinate; ``None`` for a time-independent
                field.

        Returns:
            The field values x at the grid coordinates.
        """
        field = self.as_field(theta)
        coords = self.coords
        if t is not None:
            time = jnp.full((coords.shape[0],), t, dtype=coords.dtype)
            # (s_n, t)
            coords = einx.id("n d, n -> n (d + 1)", coords, time)
        # x_n = x(s_n; θ)
        values = jax.vmap(field)(coords)
        # a scalar field's state has no channel axis
        if values.ndim == 2 and values.shape[1] == 1:
            return values[:, 0]
        return values

    def penalty(self, theta: PyTree[Array]) -> Float[Array, ""]:
        """Evaluate the weight decay: ℛ(θ) = ½ β ‖θ‖².

        Args:
            theta: Control θ, the array half of the field.

        Returns:
            The scalar penalty ℛ(θ).
        """
        leaves = jtu.tree_leaves(theta)
        # ‖θ‖² = ∑ᵢ θᵢ²
        squared_norm = sum(
            (jnp.sum(jnp.square(leaf)) for leaf in leaves), jnp.zeros(())
        )
        # ℛ(θ) = ½ β ‖θ‖²
        return 0.5 * self.weight_decay * squared_norm

    def init_control(self, *, key: PRNGKeyArray | None = None) -> PyTree[Array]:
        """Return the field's current parameters as the starting control θ.

        Args:
            key: Unused; the start is the field's own parameters.

        Returns:
            The array half of the field selected by ``filter_spec``.
        """
        del key
        return eqx.filter(self.field, self.filter_spec)


__all__ = ["LinearParameterization", "NeuralFieldParameterization"]
