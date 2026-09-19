"""Forward-model adapters shared by the variational models and tutorials."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class SoftBoundedForward(eqx.Module):
    r"""``ForwardModel`` adapter that softly confines states to a box.

    Explicit Runge–Kutta steps of dynamics with quadratic terms (Lorenz-63,
    Lorenz-96, …) overflow, or drive an adaptive step-size controller past
    its step cap, when the state is far outside the attractor. A
    line-search minimiser such as the BFGS inside
    [`StrongFourDVar`][vardax.StrongFourDVar] will try such states, and one
    non-finite cost is enough to poison the whole solve. This adapter maps
    the state through

    $$
    s(x) = \operatorname{sign}(x)\,\begin{cases}
      |x| & |x| \le b \\
      b + b \tanh\!\big((|x| - b)/b\big) & |x| > b
    \end{cases}
    $$

    on the way into and out of every step, so that

    - states inside the box $|x_i| \le b$ are untouched (choose ``bound``
      so the attractor sits well inside it — the adapter is then exactly
      the wrapped model wherever the analysis can end up);
    - states outside are confined to $|x_i| < 2b$, keeping every trial
      cost finite;
    - the map is smooth with a non-zero gradient, unlike a hard clip,
      which would zero the gradient and break quasi-Newton curvature
      updates.

    Attributes:
        forward: The wrapped ``pipekit_cycle.ForwardModel``.
        bound: Half-width $b$ of the box on which the adapter is the
            identity.

    Examples:
        >>> import jax.numpy as jnp
        >>> import vardax
        >>> class Double:
        ...     dt = 1.0
        ...     state_signature = None
        ...
        ...     def step(self, x, dt):
        ...         return 2.0 * x
        >>> fwd = vardax.SoftBoundedForward(Double(), bound=10.0)
        >>> fwd.step(jnp.array([1.0, -3.0]), 1.0)  # inside the box: untouched
        Array([ 2., -6.], dtype=float32)
        >>> bool(jnp.all(jnp.abs(fwd.step(jnp.array([1e6, -1e6]), 1.0)) < 20.0))
        True
    """

    forward: Any
    bound: float = eqx.field(static=True)

    @property
    def dt(self) -> float:
        return self.forward.dt

    @property
    def state_signature(self) -> Any:
        return getattr(self.forward, "state_signature", None)

    def _confine(self, state: Array) -> Array:
        b = self.bound
        a = jnp.abs(state)
        # The in-box branch returns ``state`` itself (not ``sign * abs``) so the
        # gradient is exactly one everywhere inside, including at zero, where
        # ``jnp.sign`` would otherwise zero it.
        outside = jnp.sign(state) * (b + b * jnp.tanh((a - b) / b))
        return jnp.where(a <= b, state, outside)

    def step(self, state: Array, dt: float) -> Array:
        """Advance ``state`` by ``dt`` with confinement before and after."""
        return self._confine(self.forward.step(self._confine(state), dt))
