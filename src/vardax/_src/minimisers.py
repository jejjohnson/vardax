r"""Minimisers implementing the [`Minimiser`][vardax.Minimiser] protocol.

The classical analysis models ([`ThreeDVar`][vardax.ThreeDVar],
[`StrongFourDVar`][vardax.StrongFourDVar],
[`WeakFourDVar`][vardax.WeakFourDVar],
[`Incremental4DVar`][vardax.Incremental4DVar]) hand their cost
$J(\theta)$ to ``optimistix.minimise`` with a quasi-Newton solver by
default. That is the right tool for a state-sized control, but not for a
control made of $10^5$ network weights, where BFGS's dense inverse-Hessian
estimate is not viable. Two minimisers share one calling convention,
``minimiser(cost_fn, x0, batch) -> x_star`` with ``cost_fn(x, batch)``
returning a scalar and ``x0`` any pytree:

- [`OptimistixMinimiser`][vardax.OptimistixMinimiser] wraps an
  ``optimistix.AbstractMinimiser`` (default ``optx.BFGS(rtol=1e-6,
  atol=1e-6)``, the models' default).
- [`OptaxMinimiser`][vardax.OptaxMinimiser] runs a first-order ``optax``
  optimiser (Adam, say) in a ``jax.lax.while_loop`` with early stopping,
  optionally followed by a second optimiser (L-BFGS, say) to polish.

Differentiating *through* the minimum uses an ``optimistix`` adjoint. The
default, ``optimistix.ImplicitAdjoint``, applies the implicit function
theorem at $\theta^\star$, which assumes stationarity,
$\nabla J(\theta^\star) = 0$. A first-order optimiser stopped by a step
budget or by patience does not guarantee it, so `OptaxMinimiser` reports
the final gradient norm $\lVert \nabla J(\theta^\star) \rVert_2$ and warns
when it exceeds ``warn_grad_norm``.
"""

from __future__ import annotations

from collections.abc import Callable
import functools as ft
from typing import Any
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PyTree
import optax
import optimistix as optx


class MinimiseResult(eqx.Module):
    r"""Outcome of a minimisation.

    Attributes:
        value: The minimiser $\theta^\star$, a pytree shaped like ``x0``.
        cost: $J(\theta^\star)$, a scalar.
        grad_norm: $\lVert \nabla J(\theta^\star) \rVert_2$ over all
            leaves of $\theta$, a scalar. The implicit adjoint is exact
            only when this is (close to) zero.
        num_steps: Number of optimiser steps taken (both phases, for
            `OptaxMinimiser` with a polish), an integer scalar.
    """

    value: PyTree[Array]
    cost: Float[Array, ""]
    grad_norm: Float[Array, ""]
    num_steps: Int[Array, ""]


def _tree_l2_norm(tree: PyTree[Array]) -> Float[Array, ""]:
    leaves = jtu.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _close_over(
    cost_fn: Callable[[Any, Any], Float[Array, ""]], x0: PyTree, batch: Any
) -> Callable[[Any, Any], Float[Array, ""]]:
    """Closure-convert ``cost_fn`` so arrays it closes over become leaves.

    The implicit adjoint differentiates with respect to the leaves of its
    inputs, so a cost that closes over model arrays (``prior_mean``, an
    operator) must expose them, as ``optimistix.minimise`` does.
    """
    fn = eqx.filter_closure_convert(cost_fn, x0, batch)
    f_struct = fn.out_struct  # ty:ignore[unresolved-attribute]
    if not (
        isinstance(f_struct, jax.ShapeDtypeStruct)
        and f_struct.shape == ()
        and jnp.issubdtype(f_struct.dtype, jnp.floating)
    ):
        raise ValueError("cost_fn must return a single floating-point scalar.")
    return fn


def _value_and_grad_norm(fn, x, batch):
    value, grad = jax.value_and_grad(fn)(x, batch)
    return value, _tree_l2_norm(grad)


class OptimistixMinimiser(eqx.Module):
    r"""An ``optimistix`` minimiser behind the `Minimiser` protocol.

    ``OptimistixMinimiser()`` is the models' default made explicit:
    ``optx.BFGS(rtol=1e-6, atol=1e-6)`` with ``max_steps=200`` and
    ``throw=False``, so a model built with it returns exactly what the
    default model returns.

    Attributes:
        solver: The ``optimistix.AbstractMinimiser``.
        max_steps: Iteration cap.
        throw: Passed to ``optimistix.minimise``: raise on failure
            (non-convergence within ``max_steps``) instead of returning
            the last iterate.

    Examples:
        A quadratic $J(x) = \tfrac12 \lVert x - m \rVert^2$ has its
        minimum at $m$.

        >>> import jax.numpy as jnp, vardax
        >>> m = jnp.array([1.0, -2.0, 0.5])
        >>> minimiser = vardax.OptimistixMinimiser()
        >>> x = minimiser(lambda x, m: 0.5 * jnp.sum((x - m) ** 2), jnp.zeros(3), m)
        >>> bool(jnp.allclose(x, m, atol=1e-5))
        True
    """

    solver: optx.AbstractMinimiser = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    throw: bool = eqx.field(static=True)

    def __init__(
        self,
        solver: optx.AbstractMinimiser | None = None,
        *,
        max_steps: int = 200,
        throw: bool = False,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}.")
        self.solver = solver or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.max_steps = max_steps
        self.throw = throw

    def run(
        self,
        cost_fn: Callable[[Any, Any], Float[Array, ""]],
        x0: PyTree[Array],
        batch: Any = None,
        *,
        adjoint: optx.AbstractAdjoint | None = None,
    ) -> MinimiseResult:
        """Minimise ``cost_fn(x, batch)`` from ``x0``, with diagnostics.

        Args:
            cost_fn: Scalar cost ``cost_fn(x, batch)``.
            x0: Initial guess, any pytree of arrays.
            batch: Passed as the second argument of ``cost_fn``.
            adjoint: How to differentiate through the minimum. Default
                ``optimistix.ImplicitAdjoint()``.

        Returns:
            A [`MinimiseResult`][vardax.MinimiseResult].
        """
        solution = optx.minimise(
            fn=cost_fn,
            solver=self.solver,
            y0=x0,
            args=batch,
            max_steps=self.max_steps,
            adjoint=adjoint or optx.ImplicitAdjoint(),
            throw=self.throw,
        )
        value = solution.value
        cost, grad_norm = _value_and_grad_norm(
            cost_fn, jax.lax.stop_gradient(value), batch
        )
        return MinimiseResult(
            value=value,
            cost=jax.lax.stop_gradient(cost),
            grad_norm=jax.lax.stop_gradient(grad_norm),
            num_steps=solution.stats["num_steps"],
        )

    def __call__(
        self,
        cost_fn: Callable[[Any, Any], Float[Array, ""]],
        x0: PyTree[Array],
        batch: Any = None,
    ) -> PyTree[Array]:
        """Return ``argmin_x cost_fn(x, batch)`` starting from ``x0``."""
        return optx.minimise(
            fn=cost_fn,
            solver=self.solver,
            y0=x0,
            args=batch,
            max_steps=self.max_steps,
            throw=self.throw,
        ).value


class OptaxMinimiser(eqx.Module):
    r"""First-order minimisation with an ``optax`` optimiser.

    Iterates $\theta_{k+1} = \theta_k + u_k$, with $u_k$ the ``optimizer``
    update from $\nabla J(\theta_k)$, inside a ``jax.lax.while_loop`` (so
    the whole minimisation is ``jit``-able), and stops at the first of:

    - ``max_steps`` steps;
    - a relative cost change $|J_k - J_{k-1}| \le \mathrm{tol}\,|J_{k-1}|$;
    - ``patience`` consecutive steps without a new lowest cost.

    It returns the iterate with the lowest cost seen. With ``polish``, a
    second loop of at most ``polish_steps`` steps of that optimiser (for
    example ``optax.lbfgs()``) starts from there, with the same stopping
    rules. The optimisers are called with the extra arguments
    ``value``, ``grad`` and ``value_fn``, so line-search optimisers such as
    ``optax.lbfgs`` work unchanged.

    After the loop it computes $\lVert \nabla J(\theta^\star) \rVert_2$
    and warns when it exceeds ``warn_grad_norm``: the implicit adjoint
    used to differentiate through the minimum assumes
    $\nabla J(\theta^\star) = 0$, which a first-order optimiser stopped
    early does not guarantee.

    Attributes:
        optimizer: The ``optax.GradientTransformation`` of the main loop.
        max_steps: Step cap of the main loop.
        tol: Relative cost-change tolerance; ``0`` disables the test.
        patience: Steps without improvement before stopping; ``None``
            disables the test.
        polish: Optional second optimiser, run after the main loop.
        polish_steps: Step cap of the polish loop.
        warn_grad_norm: Warn when the final gradient norm exceeds this
            (an absolute threshold, in the units of $\nabla J$); ``None``
            disables the warning.

    Examples:
        Adam on $J(x) = \tfrac12 \lVert x - m \rVert^2$, polished by
        L-BFGS, reaches the closed-form minimum $x^\star = m$.

        >>> import jax.numpy as jnp, optax, vardax
        >>> m = jnp.array([1.0, -2.0, 0.5])
        >>> minimiser = vardax.OptaxMinimiser(
        ...     optimizer=optax.adam(1e-1), max_steps=500, polish=optax.lbfgs()
        ... )
        >>> result = minimiser.run(
        ...     lambda x, m: 0.5 * jnp.sum((x - m) ** 2), jnp.zeros(3), m
        ... )
        >>> bool(jnp.allclose(result.value, m, atol=1e-4))
        True
        >>> bool(result.grad_norm < 1e-4)
        True

    References:
        Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic
        optimization. ICLR.
    """

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    polish: optax.GradientTransformation | None = eqx.field(static=True)
    polish_steps: int = eqx.field(static=True)
    warn_grad_norm: float | None = eqx.field(static=True)

    def __init__(
        self,
        optimizer: optax.GradientTransformation | None = None,
        *,
        max_steps: int = 5000,
        tol: float = 1e-6,
        patience: int | None = 200,
        polish: optax.GradientTransformation | None = None,
        polish_steps: int = 200,
        warn_grad_norm: float | None = 1e-3,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}.")
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}.")
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be >= 1 or None, got {patience}.")
        if polish_steps < 1:
            raise ValueError(f"polish_steps must be >= 1, got {polish_steps}.")
        if warn_grad_norm is not None and warn_grad_norm < 0:
            raise ValueError(
                f"warn_grad_norm must be >= 0 or None, got {warn_grad_norm}."
            )
        self.optimizer = optimizer if optimizer is not None else optax.adam(1e-3)
        self.max_steps = max_steps
        self.tol = tol
        self.patience = patience
        self.polish = polish
        self.polish_steps = polish_steps
        self.warn_grad_norm = warn_grad_norm

    def run(
        self,
        cost_fn: Callable[[Any, Any], Float[Array, ""]],
        x0: PyTree[Array],
        batch: Any = None,
        *,
        adjoint: optx.AbstractAdjoint | None = None,
    ) -> MinimiseResult:
        """Minimise ``cost_fn(x, batch)`` from ``x0``, with diagnostics.

        Args:
            cost_fn: Scalar cost ``cost_fn(x, batch)``.
            x0: Initial guess, any pytree of arrays (for a network, the
                array half of ``eqx.partition(model, eqx.is_inexact_array)``).
            batch: Passed as the second argument of ``cost_fn``.
            adjoint: How to differentiate through the minimum. Default
                ``optimistix.ImplicitAdjoint()``; the loop is then a
                ``jax.lax.while_loop``. ``optimistix.RecursiveCheckpointAdjoint``
                backpropagates through the iterates instead; it cannot
                differentiate an optimiser with an inner ``while_loop``
                (the line search of ``optax.lbfgs``).

        Returns:
            A [`MinimiseResult`][vardax.MinimiseResult].
        """
        x0 = jtu.tree_map(jnp.asarray, x0)
        fn = _close_over(cost_fn, x0, batch)
        inputs = (fn, self, x0, batch)
        value, (cost, grad_norm, num_steps) = (adjoint or optx.ImplicitAdjoint()).apply(
            _optax_primal, _optax_rewrite, inputs, frozenset()
        )
        if self.warn_grad_norm is not None:
            jax.debug.callback(
                ft.partial(_warn_if_not_stationary, threshold=self.warn_grad_norm),
                grad_norm,
            )
        return MinimiseResult(
            value=value, cost=cost, grad_norm=grad_norm, num_steps=num_steps
        )

    def __call__(
        self,
        cost_fn: Callable[[Any, Any], Float[Array, ""]],
        x0: PyTree[Array],
        batch: Any = None,
    ) -> PyTree[Array]:
        """Return ``argmin_x cost_fn(x, batch)`` starting from ``x0``."""
        return self.run(cost_fn, x0, batch).value


def _warn_if_not_stationary(grad_norm: Any, *, threshold: float) -> None:
    worst = float(jnp.max(jnp.asarray(grad_norm)))
    if worst > threshold:
        warnings.warn(
            f"OptaxMinimiser stopped with gradient norm {worst:.3e} > "
            f"warn_grad_norm={threshold:.1e}; the implicit adjoint assumes "
            "a stationary point, so gradients through this minimum are "
            "approximate. Raise max_steps / polish, or loosen warn_grad_norm.",
            UserWarning,
            stacklevel=2,
        )


def _optax_loop(fn, batch, x0, optimizer, max_steps, tol, patience, while_loop):
    """One early-stopped optimisation loop; returns (best x, steps taken)."""
    optimizer = optax.with_extra_args_support(optimizer)

    def value_fn(x):
        return fn(x, batch)

    f_dtype = fn.out_struct.dtype
    inf = jnp.asarray(jnp.inf, dtype=f_dtype)
    no_patience = patience is None

    def cond(carry):
        step, _, _, _, _, _, since_best, converged = carry
        keep = (step < max_steps) & ~converged
        if not no_patience:
            keep = keep & (since_best < patience)
        return keep

    def body(carry):
        step, x, opt_state, f_prev, best_x, best_f, since_best, _ = carry
        f, g = jax.value_and_grad(fn)(x, batch)
        updates, opt_state = optimizer.update(
            g, opt_state, x, value=f, grad=g, value_fn=value_fn
        )
        x_new = optax.apply_updates(x, updates)
        improved = f < best_f
        best_x = jtu.tree_map(lambda a, b: jnp.where(improved, a, b), x, best_x)
        best_f = jnp.where(improved, f, best_f)
        since_best = jnp.where(improved, 0, since_best + 1)
        # f_prev is +inf before the first evaluation: never converged there.
        converged = (
            jnp.isfinite(f_prev) & (jnp.abs(f - f_prev) <= tol * jnp.abs(f_prev))
            if tol > 0
            else jnp.asarray(False)
        )
        return (step + 1, x_new, opt_state, f, best_x, best_f, since_best, converged)

    init = (
        jnp.asarray(0, dtype=jnp.int32),
        x0,
        optimizer.init(x0),
        inf,
        x0,
        inf,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )
    step, x_last, _, _, best_x, best_f, _, _ = while_loop(
        cond, body, init, max_steps=max_steps
    )
    # The last update was never evaluated: keep it if it is the best.
    f_last = fn(x_last, batch)
    take_last = f_last <= best_f
    x_star = jtu.tree_map(lambda a, b: jnp.where(take_last, a, b), x_last, best_x)
    return x_star, step


def _optax_primal(inputs):
    fn, minimiser, x0, batch, while_loop = inputs
    x, steps = _optax_loop(
        fn,
        batch,
        x0,
        minimiser.optimizer,
        minimiser.max_steps,
        minimiser.tol,
        minimiser.patience,
        while_loop,
    )
    if minimiser.polish is not None:
        x, polish_steps = _optax_loop(
            fn,
            batch,
            x,
            minimiser.polish,
            minimiser.polish_steps,
            minimiser.tol,
            minimiser.patience,
            while_loop,
        )
        steps = steps + polish_steps
    cost, grad_norm = _value_and_grad_norm(fn, x, batch)
    return x, (cost, grad_norm, steps)


def _optax_rewrite(root, _residual, inputs):
    """Stationarity condition $\\nabla J(\\theta^\\star) = 0$ for the IFT."""
    fn, _, _, batch, *_ = inputs
    return jax.grad(fn)(root, batch)


def minimise(
    minimiser: Any,
    cost_fn: Callable[[Any, Any], Float[Array, ""]],
    x0: PyTree[Array],
    *,
    adjoint: optx.AbstractAdjoint,
    max_steps: int,
) -> PyTree[Array]:
    """Run a model's ``minimiser`` on ``cost_fn(x, None)`` from ``x0``.

    An ``optimistix.AbstractMinimiser`` goes through ``optimistix.minimise``
    with the model's ``max_steps`` and ``throw=False``, exactly as the
    models always have. `OptaxMinimiser` and `OptimistixMinimiser` keep
    their own step caps and receive the model's ``adjoint``; any other
    [`Minimiser`][vardax.Minimiser] is called as
    ``minimiser(cost_fn, x0, None)``.
    """
    if isinstance(minimiser, optx.AbstractMinimiser):
        return optx.minimise(
            fn=cost_fn,
            solver=minimiser,
            y0=x0,
            args=None,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=False,
        ).value
    if isinstance(minimiser, OptaxMinimiser | OptimistixMinimiser):
        return minimiser.run(cost_fn, x0, None, adjoint=adjoint).value
    return minimiser(cost_fn, x0, None)


__all__ = [
    "MinimiseResult",
    "OptaxMinimiser",
    "OptimistixMinimiser",
    "minimise",
]
