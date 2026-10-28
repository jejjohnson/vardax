r"""State parameterizations: control $\theta$ to state $x = \psi(\theta)$.

A [`StateParameterization`][vardax.StateParameterization] replaces the
raw-array control of the variational models by any pytree $\theta$, mapped
to the state by ``decode`` and regularised by ``penalty``. For strong
4D-Var with a parameterized initial state $x_0 = \psi(\theta)$ the cost is

$$
J(\theta) = \mathrm{pen}(\theta)
  + \tfrac12 \sum_t \big\lVert y_t - \mathcal H_t\big(\mathcal M^t(\psi(\theta))
  \big)\big\rVert^2_{R_t^{-1}},
$$

with the penalty in place of the background term
$\tfrac12 \lVert x_0 - x_b \rVert^2_{B^{-1}}$.

- [`LinearParameterization`][vardax.LinearParameterization] adapts any
  [`ReducedBasis`][vardax.ReducedBasis]: $\psi(X) = x_b + \Phi X$,
  $\mathrm{pen}(X) = \tfrac12 X^\top Q^{-1} X$.
- [`NeuralFieldParameterization`][vardax.NeuralFieldParameterization]
  makes the state a coordinate network evaluated on the grid,
  $\psi(\theta) = [f_\theta(s_j)]_j$, with weight decay
  $\mathrm{pen}(\theta) = \tfrac{\alpha}{2}\lVert\theta\rVert^2$; there is
  no $B$, the network's spectral bias acts as the prior.
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
    r"""Adapter from a [`ReducedBasis`][vardax.ReducedBasis] to a parameterization.

    $\psi(X) = x_b + \Phi(t) X$ (with $t = 0$ when not given) and
    $\mathrm{pen}(X) = \tfrac12 X^\top Q^{-1} X$, so a model built with
    ``parameterization=LinearParameterization(basis, x_b)`` minimises the
    same cost as ``ReducedBasisFourDVar(basis=basis)`` with ``prior_mean=x_b``.

    Algorithm:
        1. ``decode`` applies the basis to the coefficients and adds
           $x_b$.
        2. ``penalty`` applies $Q^{-1}$ through the basis.
        3. ``init_control`` returns the zero coefficient vector.

    Attributes:
        basis: The `ReducedBasis` ($\Phi$, $Q^{-1}$, ``nbasis``).
        x_b: Background state the increment is added to.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> basis = vardax.linear_basis(jnp.eye(4)[:, :2], variance=2.0)
        >>> param = vardax.LinearParameterization(basis, x_b=jnp.ones(4))
        >>> X = param.init_control()
        >>> X.shape
        (2,)
        >>> param.decode(jnp.array([1.0, -1.0]))
        Array([2., 0., 1., 1.], dtype=float32)
        >>> float(param.penalty(jnp.array([1.0, -1.0])))  # ½ (1 + 1) / 2
        0.5

        The initial control decodes to the background, whatever the basis.

        >>> bool(jnp.allclose(param.decode(X), param.x_b))
        True

        It satisfies the [`StateParameterization`][vardax.StateParameterization]
        protocol.

        >>> isinstance(param, vardax.StateParameterization)
        True
    """

    basis: Any
    x_b: Float[Array, ...]

    def decode(self, theta: Float[Array, " M"], t: float | None = None) -> Array:
        """Return the state $x_b + \\Phi(t) X$, shaped like ``x_b``.

        Args:
            theta: Coefficient vector $X$.
            t: Time at which the basis is evaluated; ``None`` means ``0``.

        Returns:
            The decoded state.
        """
        increment = self.basis.operg(0.0 if t is None else t, theta)
        # x = x_b + Φ(t) X
        return self.x_b + as_shape(increment, self.x_b.shape)

    def penalty(self, theta: Float[Array, " M"]) -> Float[Array, ""]:
        """Return $\\tfrac12 X^\\top Q^{-1} X$.

        Args:
            theta: Coefficient vector $X$.

        Returns:
            The scalar penalty.
        """
        return 0.5 * jnp.sum(theta * self.basis.prior_inv(theta))

    def init_control(self, *, key: PRNGKeyArray | None = None) -> Array:
        """Return the zero coefficient vector (the background).

        Args:
            key: Unused; the start is deterministic.

        Returns:
            The zero coefficient vector $X = 0$.
        """
        del key
        return jnp.zeros((self.basis.nbasis,), dtype=self.x_b.dtype)


class NeuralFieldParameterization(eqx.Module):
    r"""A coordinate network evaluated on the grid as the state.

    The control is the array half of the field,
    $\theta = $ ``eqx.filter(field, eqx.is_inexact_array)``; the rest of
    the module (activations, sizes) is fixed. The state is the field at
    the grid coordinates,
    $\psi(\theta) = [f_\theta(s_j)]_{j=1}^{N}$, of shape ``(N,)`` for a
    scalar field and ``(N, C)`` for a ``C``-channel one. With a time ``t``
    the field is evaluated at $(s_j, t)$, so a space-time field takes
    ``D + 1`` inputs. The penalty is weight decay,
    $\mathrm{pen}(\theta) = \tfrac{\alpha}{2} \sum_i \theta_i^2$.

    Algorithm:
        1. ``init_control`` extracts the array leaves of ``field``
           selected by ``filter_spec`` as $\theta$.
        2. ``decode`` recombines $\theta$ with the fixed part of ``field``
           and evaluates it at every grid coordinate.
        3. ``penalty`` returns the weight decay of $\theta$.

    Raises:
        ShapeError: If ``coords`` is not two-dimensional.
        ConfigurationError: If ``weight_decay`` is negative.

    Attributes:
        field: Single-example network ``field(s) -> (C,)``, for example a
            `geonnax.SIREN`. Its current parameters are the initial control.
        coords: Grid coordinates $s_j$, shape ``(N, D)``.
        weight_decay: $\alpha \ge 0$.
        filter_spec: Which leaves of ``field`` are the control: an
            ``equinox`` filter (a callable on leaves, or a pytree of bools
            shaped like ``field``), used for both halves of the partition.
            Default ``eqx.is_inexact_array``: every floating array.

    Frozen arrays: geonnax (and equinox) keep fixed arrays — random Fourier
    frequencies, precomputed tables — as ordinary leaves read through
    ``jax.lax.stop_gradient``. The default filter would make them controls,
    and the weight decay would shrink them to zero. Exclude them with
    ``filter_spec``, e.g.
    ``eqx.tree_at(lambda f: f.B, jax.tree.map(eqx.is_inexact_array, field), False)``;
    they then stay fixed and outside the penalty.

    Examples:
        >>> import jax, jax.numpy as jnp, geonnax, vardax
        >>> angle = 2 * jnp.pi * jnp.arange(40) / 40
        >>> coords = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
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
        """The field with parameters ``theta``, as a callable ``s -> (C,)``.

        Args:
            theta: Control, the array half of the field.

        Returns:
            The field with ``theta`` combined into the fixed part.
        """
        static = eqx.filter(self.field, self.filter_spec, inverse=True)
        return eqx.combine(theta, static)

    def decode(self, theta: PyTree[Array], t: float | None = None) -> Array:
        """Evaluate the field with parameters ``theta`` on the grid.

        Args:
            theta: Control, the array half of the field.
            t: Time appended to every coordinate; ``None`` for a
                time-independent field.

        Returns:
            The field values at the grid coordinates.
        """
        field = self.as_field(theta)
        coords = self.coords
        if t is not None:
            time = jnp.full((coords.shape[0],), t, dtype=coords.dtype)
            coords = einx.id("n d, n -> n (d + 1)", coords, time)
        values = jax.vmap(field)(coords)
        if values.ndim == 2 and values.shape[1] == 1:
            return values[:, 0]
        return values

    def penalty(self, theta: PyTree[Array]) -> Float[Array, ""]:
        """Return $\\tfrac{\\alpha}{2} \\lVert \\theta \\rVert^2$.

        Args:
            theta: Control, the array half of the field.

        Returns:
            The scalar weight-decay penalty.
        """
        leaves = jtu.tree_leaves(theta)
        sq = sum((jnp.sum(jnp.square(leaf)) for leaf in leaves), jnp.zeros(()))
        return 0.5 * self.weight_decay * sq

    def init_control(self, *, key: PRNGKeyArray | None = None) -> PyTree[Array]:
        """Return the field's current parameters.

        Args:
            key: Unused; the start is the field's own parameters.

        Returns:
            The array half of the field selected by ``filter_spec``.
        """
        del key
        return eqx.filter(self.field, self.filter_spec)


__all__ = ["LinearParameterization", "NeuralFieldParameterization"]
