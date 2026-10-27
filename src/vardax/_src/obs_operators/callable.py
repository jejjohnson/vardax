r"""Observation operator from any differentiable JAX function.

$H(x) = f(x)$, with the tangent-linear operator $H'(x) = \partial f / \partial x$
taken by autodiff. This is the operator for *parameter estimation*: the
control is a source, forcing or parameter vector $q$, and the observation
operator is a downstream model run $\mathcal{G}(q)$, so the 3DVar cost

$$
J(q) = \tfrac{1}{2} (q - q_b)^\top B^{-1} (q - q_b)
     + \tfrac{1}{2} (y - \mathcal{G}(q))^\top R^{-1} (y - \mathcal{G}(q))
$$

is [`ThreeDVar`][vardax.ThreeDVar] with ``obs_op=CallableObs(fn=G)``, or
[`OptimalInterpolation`][vardax.OptimalInterpolation] when $\mathcal{G}$ is
linear.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx


class CallableObs(eqx.Module):
    r"""$H(x) = f(x)$ for any differentiable JAX function $f$.

    ``linearize(x)`` is a matrix-free ``lineax.JacobianLinearOperator`` at
    ``x``: its ``mv`` is a JVP of $f$ and its ``transpose().mv`` a VJP, so
    nothing of size $\dim y \times \dim x$ is ever formed. The operator is
    an Equinox module, so it works under ``eqx.filter_jit`` and
    ``jax.vmap``; when ``fn`` is itself an Equinox module (an
    ``eqx.nn.MLP`` emulator, say), its parameters are pytree leaves and
    ``eqx.filter_grad`` differentiates through them.

    Prefer [`LinearObs`][vardax.LinearObs] when you already hold an explicit
    or structured matrix you want to reuse; use this for a model run, a
    simulator or an emulator.

    Attributes:
        fn: The observation map $f$, from a state of any shape to an
            observation of any shape.

    Examples:
        A linear $f(x) = F x$ on a ``(1, N)`` state with $M \ne N$
        observations: ``OptimalInterpolation`` with $B = R = I$ gives the
        closed-form BLUE, $x^* = (I + F^\top F)^{-1} F^\top y$.

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax as vdx
        >>> F = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        >>> obs = vdx.CallableObs(fn=lambda x: x @ F.T)
        >>> obs(jnp.ones((1, 2))).shape
        (1, 3)
        >>> eye_x = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 2), jnp.float32))
        >>> eye_y = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float32))
        >>> oi = vdx.OptimalInterpolation(
        ...     obs_op=obs,
        ...     prior_mean=jnp.zeros((1, 2)),
        ...     prior_cov_op=eye_x,
        ...     obs_cov_op=eye_y,
        ... )
        >>> y = jnp.array([1.0, 2.0, 3.0])
        >>> batch = vdx.Batch1D(input=y[None, None], mask=jnp.ones((1, 1, 3)))
        >>> blue = jnp.linalg.solve(jnp.eye(2) + F.T @ F, F.T @ y)
        >>> bool(jnp.allclose(oi(batch)[0, 0], blue, atol=1e-4))
        True
    """

    fn: Callable[[Float[Array, ...]], Float[Array, ...]]

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Forward map $f(x)$."""
        return self.fn(x)

    def linearize(self, x: Float[Array, ...]) -> lx.AbstractLinearOperator:
        r"""Tangent-linear operator $H'(x) = \partial f / \partial x$ at ``x``."""
        return lx.JacobianLinearOperator(lambda z, _: self.fn(z), x)
