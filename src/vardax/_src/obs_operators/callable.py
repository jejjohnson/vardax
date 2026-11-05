"""Observation operator from any differentiable JAX function, linearised by autodiff."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx
import pipekit_cycle as _pc


class CallableObs(eqx.Module, _pc.CallableObs):
    """Observation operator h from any differentiable JAX function.

        h(x) = ``fn``(x)
        J_h(x) = ∂h/∂x

    It is a ``pipekit_cycle.CallableObs`` that linearises itself by autodiff and stays
    a pytree. A plain ``pipekit_cycle.CallableObs`` also works in every vardax model,
    which then takes its tangent by autodiff too.

    This is the operator for parameter estimation: the input x is a source, forcing
    or parameter vector and h is a downstream model run, so the 3D-Var cost

        𝒥(x) = ½ ‖x − μ_x‖²_Σ_x⁻¹ + ½ ‖y − h(x)‖²_Σ_ε⁻¹

    is [`ThreeDVar`][vardax.ThreeDVar] with ``obs_op=CallableObs(fn=h)``, or
    [`OptimalInterpolation`][vardax.OptimalInterpolation] when h is linear.

    ``linearize(x)`` is a matrix-free ``lineax.JacobianLinearOperator`` at x: its
    ``mv`` is a JVP of h and its ``transpose().mv`` a VJP, so nothing of size D_y × D_x
    is ever formed. The operator is an Equinox module, so it works under
    ``eqx.filter_jit`` and ``jax.vmap``; when ``fn`` is itself an Equinox module (an
    ``eqx.nn.MLP`` emulator, say), its parameters are pytree leaves and
    ``eqx.filter_grad`` differentiates through them.

    Prefer [`LinearObs`][vardax.LinearObs] when you already hold an explicit or
    structured matrix you want to reuse; use this for a model run, a simulator or an
    emulator.

    Pass ``jac="bwd"`` for a model that defines only a reverse-mode rule (a
    ``jax.custom_vjp`` function): forward mode raises on it as soon as an analysis or
    posterior linearises the operator. Matrix-free, lineax then applies J_h v by
    transposing the VJP, which fails when the backward pass contains a loop JAX cannot
    transpose, as a ``diffrax`` solve's does (``RecursiveCheckpointAdjoint`` runs a
    ``while_loop``). For those, also pass ``materialise=True``: ``linearize`` then
    forms the Jacobian once, with one VJP per observation (``jax.jacrev``), and both
    J_h and J_hᵀ are dense products. That suits a small control, the usual case for
    source and parameter estimation.

    Attributes:
        fn: The observation map h, from a state of any shape to an observation of any
            shape.
        jac: How ``linearize`` differentiates ``fn``: ``None`` (lineax's default,
            forward mode), ``"fwd"`` or ``"bwd"`` (reverse mode).
        materialise: Form the Jacobian in ``linearize`` instead of keeping it
            matrix-free, with ``jax.jacrev`` for ``jac="bwd"`` and ``jax.jacfwd``
            otherwise. Default ``False``.

    Examples:
        A linear h on a ``(1, 2)`` state with three observations:
        ``OptimalInterpolation`` with Σ_x = Σ_ε = I gives the closed form
        x⋆ = (I + Hᵀ H)⁻¹ Hᵀ y, which is (1, 1) for this y.

        >>> import einx, jax, jax.numpy as jnp, lineax as lx, vardax as vdx
        >>> matrix = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        >>> obs = vdx.CallableObs(
        ...     fn=lambda x: einx.dot("t [s], o [s] -> t o", x, matrix)
        ... )
        >>> obs(jnp.ones((1, 2))).shape
        (1, 3)
        >>> state_struct = jax.eval_shape(lambda: jnp.zeros((1, 2)))
        >>> obs_struct = jax.eval_shape(lambda: jnp.zeros((1, 3)))
        >>> oi = vdx.OptimalInterpolation(
        ...     obs_op=obs,
        ...     prior_mean=jnp.zeros((1, 2)),
        ...     prior_cov_op=lx.IdentityLinearOperator(state_struct),
        ...     obs_cov_op=lx.IdentityLinearOperator(obs_struct),
        ... )
        >>> y = jnp.array([[[1.0, 2.0, 3.0]]])
        >>> batch = vdx.Batch1D(input=y, mask=jnp.ones((1, 1, 3)))
        >>> bool(jnp.allclose(oi(batch), 1.0, atol=1e-4))
        True

        A nonlinear map is linearised by autodiff, and ``materialise=True`` forms the
        dense Jacobian:

        >>> sq = vdx.CallableObs(fn=lambda x: x**2, materialise=True)
        >>> jac = sq.linearize(jnp.array([1.0, 2.0, 3.0])).as_matrix()
        >>> bool(jnp.allclose(jac, jnp.diag(jnp.array([2.0, 4.0, 6.0]))))
        True

        Without ``materialise`` the tangent stays matrix-free:

        >>> tangent = vdx.CallableObs(fn=lambda x: x**2).linearize(jnp.ones(2))
        >>> type(tangent).__name__
        'JacobianLinearOperator'
    """

    fn: Callable[[Float[Array, ...]], Float[Array, ...]]
    jac: Literal["fwd", "bwd"] | None = eqx.field(static=True, default=None)
    materialise: bool = eqx.field(static=True, default=False)

    @property
    def name(self) -> str:
        """``pipekit_cycle.CallableObs``'s name, from ``fn``."""
        return getattr(self.fn, "__name__", "CallableObs")

    def _apply(self, state: Float[Array, ...]) -> Float[Array, ...]:
        """Forward map h(x), which ``pipekit_cycle``'s ``__call__`` runs."""
        # h(x)
        return self.fn(state)

    def linearize(self, state: Float[Array, ...]) -> lx.AbstractLinearOperator:
        """Tangent-linear operator at ``state``: J_h(x) = ∂h/∂x.

        Args:
            state: State x at which to linearise.

        Returns:
            A matrix-free ``JacobianLinearOperator``, or its dense materialisation
            when ``materialise=True``.
        """
        # J_h(x), matrix-free
        tangent = lx.JacobianLinearOperator(
            lambda z, _: self.fn(z), state, jac=self.jac
        )
        if self.materialise:
            # dense J_h: one VJP per output (jac="bwd") or JVP per input
            return lx.materialise(tangent)
        return tangent
