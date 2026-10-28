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
      b + b\,\dfrac{u}{1 + u}, \quad u = (|x| - b)/b & |x| > b
    \end{cases}
    $$

    on the way into and out of every step, so that

    - states inside the box $|x_i| \le b$ are untouched (choose ``bound``
      so the attractor sits well inside it — the adapter is then exactly
      the wrapped model wherever the analysis can end up);
    - states outside are confined to $|x_i| < 2b$, keeping every trial
      cost finite;
    - the map is continuously differentiable (slope one at the box edge
      from both sides) and its gradient outside, $1/(1+u)^2$, stays
      representable in single precision for trial states up to about
      $10^{19}$ times the bound, unlike a hard clip (zero gradient) or a
      ``tanh`` saturation (gradient underflows to zero a few bounds out);
      a rejected trial therefore still carries curvature information if
      the minimiser asks for it.

    Algorithm:
        1. Confine the incoming state with $s$.
        2. Advance it with the wrapped model's ``step``.
        3. Confine the result with $s$ again.

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

        The step length and signature are forwarded from the wrapped model:

        >>> fwd.dt, fwd.state_signature
        (1.0, None)

        Wrapped around a diffrax-backed model it keeps a far-off state finite:

        >>> def decay(t, y, args):
        ...     return -y
        >>> inner = vardax.DynTrajectory(model=decay).as_forward_model(0.1)
        >>> safe = vardax.SoftBoundedForward(inner, bound=5.0)
        >>> bool(jnp.all(jnp.isfinite(safe.step(jnp.array([1e9]), 0.1))))
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
        half_width = self.bound
        abs_state = jnp.abs(state)
        # The in-box branch returns ``state`` itself (not ``sign * abs``) so the
        # gradient is exactly one everywhere inside, including at zero, where
        # ``jnp.sign`` would otherwise zero it. Outside, the algebraic
        # saturation u / (1 + u) keeps the derivative 1 / (1 + u)^2 finite and
        # non-zero far beyond any state a line search will try (tanh would
        # underflow to a zero gradient a few bounds out).
        # u = max(|x| − b, 0) / b
        excess = jnp.maximum(abs_state - half_width, 0.0) / half_width
        # s(x) = sign(x) (b + b u / (1 + u))
        outside = jnp.sign(state) * (half_width + half_width * excess / (1.0 + excess))
        return jnp.where(abs_state <= half_width, state, outside)

    def step(self, state: Array, dt: float) -> Array:
        r"""Advance ``state`` by ``dt`` with confinement before and after.

        Args:
            state: State $x$ to advance.
            dt: Step length $\Delta t$.

        Returns:
            $s(\Phi_{\Delta t}(s(x)))$, with $\Phi$ the wrapped model's step.
        """
        return self._confine(self.forward.step(self._confine(state), dt))
