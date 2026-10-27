r"""Minimisers implementing the [`Minimiser`][vardax.Minimiser] protocol.

The classical analysis models ([`ThreeDVar`][vardax.ThreeDVar],
[`StrongFourDVar`][vardax.StrongFourDVar],
[`WeakFourDVar`][vardax.WeakFourDVar],
[`ReducedBasisFourDVar`][vardax.ReducedBasisFourDVar]) hand their cost
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
budget or by patience does not guarantee it, so both minimisers report
the final gradient norm $\lVert \nabla J(\theta^\star) \rVert_2$ in
[`MinimiseResult`][vardax.MinimiseResult]; check it before trusting
gradients through the minimum.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
        cost, grad_norm = _cost_and_grad_norm(cost_fn, value, batch)
        return MinimiseResult(
            value=value,
            cost=cost,
            grad_norm=grad_norm,
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
    update from $g_k = \nabla J(\theta_k)$, inside a ``jax.lax.while_loop``
    (so the whole minimisation is ``jit``-able), and stops at the first of:

    - ``max_steps`` steps;
    - a small gradient, $\lVert g_k \rVert_2 \le \mathrm{gtol}$;
    - ``step_patience`` consecutive small steps,
      $\lVert u_k \rVert_2 \le \mathrm{atol} +
      \mathrm{rtol}\,\lVert \theta_k \rVert_2$ (one small step is not
      enough: a learning-rate schedule starting at zero gives a zero first
      update);
    - ``patience`` consecutive steps whose cost exceeds the lowest cost
      seen by more than float round-off.

    There is deliberately no relative cost-change test: with a large
    irreducible misfit, $J_k - J_{k-1}$ rounds to zero in float32 long
    before $\theta$ converges. For the same reason a step counts as an
    improvement when $J_k \le J_\text{best} + 8\,\varepsilon\,|J_\text{best}|$
    ($\varepsilon$ the machine epsilon of $J$), so a cost that is flat to
    round-off does not exhaust the patience. At a small gradient the loop
    stops *before* applying the update, so stored momentum cannot carry a
    stationary iterate away. It returns the best iterate seen: the lowest
    cost, with ties within round-off broken by the smaller gradient norm.

    With ``polish``, a second loop of at most ``polish_steps`` steps of that
    optimiser (for example ``optax.lbfgs()``) starts from there, with its
    own, tight, tolerances (``polish_gtol``, ``polish_rtol``,
    ``polish_atol``, ``polish_patience``; by default it stops only on
    ``polish_steps`` or ``step_patience`` exactly zero steps). The optimisers are called
    with the extra arguments ``value``, ``grad`` and ``value_fn``, so
    line-search optimisers such as ``optax.lbfgs`` work unchanged.

    The implicit adjoint used to differentiate through the minimum assumes
    $\nabla J(\theta^\star) = 0$, which a first-order optimiser stopped
    early does not guarantee: check
    [`MinimiseResult.grad_norm`][vardax.MinimiseResult] from `run`.

    Attributes:
        optimizer: The ``optax.GradientTransformation`` of the main loop.
        max_steps: Step cap of the main loop.
        gtol: Gradient-norm tolerance of the main loop; ``0`` disables it.
        rtol: Relative step tolerance of the main loop.
        atol: Absolute step tolerance of the main loop; with ``rtol = atol
            = 0`` only exactly zero steps stop the loop.
        step_patience: Consecutive small steps needed to stop (both loops).
        patience: Steps without improvement before stopping; ``None``
            disables the test.
        polish: Optional second optimiser, run after the main loop.
        polish_steps: Step cap of the polish loop.
        polish_gtol: As ``gtol``, for the polish loop.
        polish_rtol: As ``rtol``, for the polish loop.
        polish_atol: As ``atol``, for the polish loop.
        polish_patience: As ``patience``, for the polish loop.

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
        >>> bool(jnp.allclose(result.value, m, atol=1e-5))
        True
        >>> bool(result.grad_norm < 1e-5)
        True

    References:
        Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic
        optimization. ICLR.
    """

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    gtol: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    step_patience: int = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    polish: optax.GradientTransformation | None = eqx.field(static=True)
    polish_steps: int = eqx.field(static=True)
    polish_gtol: float = eqx.field(static=True)
    polish_rtol: float = eqx.field(static=True)
    polish_atol: float = eqx.field(static=True)
    polish_patience: int | None = eqx.field(static=True)

    def __init__(
        self,
        optimizer: optax.GradientTransformation | None = None,
        *,
        max_steps: int = 5000,
        gtol: float = 1e-6,
        rtol: float = 0.0,
        atol: float = 0.0,
        step_patience: int = 3,
        patience: int | None = 200,
        polish: optax.GradientTransformation | None = None,
        polish_steps: int = 200,
        polish_gtol: float = 0.0,
        polish_rtol: float = 0.0,
        polish_atol: float = 0.0,
        polish_patience: int | None = None,
    ) -> None:
        for name, steps in (
            ("max_steps", max_steps),
            ("polish_steps", polish_steps),
            ("step_patience", step_patience),
        ):
            if steps < 1:
                raise ValueError(f"{name} must be >= 1, got {steps}.")
        for name, value in (
            ("gtol", gtol),
            ("rtol", rtol),
            ("atol", atol),
            ("polish_gtol", polish_gtol),
            ("polish_rtol", polish_rtol),
            ("polish_atol", polish_atol),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
        for name, value in (
            ("patience", patience),
            ("polish_patience", polish_patience),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {value}.")
        self.optimizer = optimizer if optimizer is not None else optax.adam(1e-3)
        self.max_steps = max_steps
        self.gtol = gtol
        self.rtol = rtol
        self.atol = atol
        self.step_patience = step_patience
        self.patience = patience
        self.polish = polish
        self.polish_steps = polish_steps
        self.polish_gtol = polish_gtol
        self.polish_rtol = polish_rtol
        self.polish_atol = polish_atol
        self.polish_patience = polish_patience

    def run(
        self,
        cost_fn: Callable[[Any, Any], Float[Array, ""]],
        x0: PyTree[Array],
        batch: Any = None,
        *,
        adjoint: optx.AbstractAdjoint | None = None,
    ) -> MinimiseResult:
        """Minimise ``cost_fn(x, batch)`` from ``x0``, with diagnostics.

        ``result.cost`` is differentiable: by the envelope theorem
        $\\mathrm{d}J^\\star/\\mathrm{d}\\phi = \\partial J/\\partial\\phi$ at
        $\\theta^\\star$ for any parameter $\\phi$ of the cost.

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
        value, num_steps = (adjoint or optx.ImplicitAdjoint()).apply(
            _optax_primal, _optax_rewrite, inputs, frozenset()
        )
        cost, grad_norm = _cost_and_grad_norm(cost_fn, value, batch)
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


def _cost_and_grad_norm(cost_fn, value, batch):
    """$J(\\theta^\\star)$ (differentiable to all orders) and $\\lVert g \\rVert$.

    The cost is evaluated at ``value`` itself, which carries the adjoint's
    derivative of $\\theta^\\star$: the first derivative is then the
    envelope theorem's ($\\partial J/\\partial\\theta = 0$ at the minimum),
    and higher derivatives keep the implicit dependence of $\\theta^\\star$
    on the parameters. The gradient norm is a diagnostic and is detached.
    """
    cost = cost_fn(value, batch)
    x_star = jax.lax.stop_gradient(value)
    grad = jax.grad(cost_fn)(x_star, batch)
    return cost, jax.lax.stop_gradient(_tree_l2_norm(grad))


def _optax_loop(fn, batch, x0, optimizer, max_steps, tols, patience, while_loop):
    """One early-stopped optimisation loop; returns (best iterate, steps taken)."""
    gtol, rtol, atol, step_patience = tols
    optimizer = optax.with_extra_args_support(optimizer)

    def value_fn(x):
        return fn(x, batch)

    f_dtype = fn.out_struct.dtype
    inf = jnp.asarray(jnp.inf, dtype=f_dtype)
    eps = jnp.finfo(f_dtype).eps

    def slack(ref):
        return jnp.where(jnp.isfinite(ref), 8 * eps * jnp.abs(ref), 0.0)

    def better(f, g_norm, best_f, best_g):
        # Lower cost beyond round-off, or a tie within round-off broken by
        # the gradient norm (closer to stationary).
        s = slack(best_f)
        return (f < best_f - s) | ((f <= best_f + s) & (g_norm <= best_g))

    def select(take, a, b):
        return jtu.tree_map(lambda u, v: jnp.where(take, u, v), a, b)

    def cond(carry):
        step, _, _, _, _, _, _, since_best, _, converged = carry
        keep = (step < max_steps) & ~converged
        if patience is not None:
            keep = keep & (since_best < patience)
        return keep

    def body(carry):
        (step, x, opt_state, best_x, best_f, best_g, min_f, since_best, n_small, _) = (
            carry
        )
        f, g = jax.value_and_grad(fn)(x, batch)
        g_norm = _tree_l2_norm(g)
        take = better(f, g_norm, best_f, best_g)
        best_x = select(take, x, best_x)
        best_f = jnp.where(take, f, best_f)
        best_g = jnp.where(take, g_norm, best_g)
        # Patience: a step counts unless its cost is worse than the lowest
        # seen by more than round-off.
        since_best = jnp.where(f <= min_f + slack(min_f), 0, since_best + 1)
        min_f = jnp.minimum(f, min_f)
        small_grad = g_norm <= gtol
        updates, opt_state = optimizer.update(
            g, opt_state, x, value=f, grad=g, value_fn=value_fn
        )
        small_step = _tree_l2_norm(updates) <= atol + rtol * _tree_l2_norm(x)
        n_small = jnp.where(small_step, n_small + 1, 0)
        # A stationary iterate is final: do not apply the (momentum) update.
        x_new = select(small_grad, x, optax.apply_updates(x, updates))
        converged = small_grad | (n_small >= step_patience)
        return (
            step + 1,
            x_new,
            opt_state,
            best_x,
            best_f,
            best_g,
            min_f,
            since_best,
            n_small,
            converged,
        )

    zero = jnp.asarray(0, dtype=jnp.int32)
    init = (
        zero,
        x0,
        optimizer.init(x0),
        x0,
        inf,
        inf,
        inf,
        zero,
        zero,
        jnp.asarray(False),
    )
    step, x_last, _, best_x, best_f, best_g, *_ = while_loop(
        cond, body, init, max_steps=max_steps
    )
    # The last iterate may not have been evaluated: keep it if it is better.
    f_last, g_last = jax.value_and_grad(fn)(x_last, batch)
    take_last = better(f_last, _tree_l2_norm(g_last), best_f, best_g)
    return select(take_last, x_last, best_x), step


def _optax_primal(inputs):
    fn, minimiser, x0, batch, while_loop = inputs
    x, steps = _optax_loop(
        fn,
        batch,
        x0,
        minimiser.optimizer,
        minimiser.max_steps,
        (minimiser.gtol, minimiser.rtol, minimiser.atol, minimiser.step_patience),
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
            (
                minimiser.polish_gtol,
                minimiser.polish_rtol,
                minimiser.polish_atol,
                minimiser.step_patience,
            ),
            minimiser.polish_patience,
            while_loop,
        )
        steps = steps + polish_steps
    return x, steps


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


def minimise_with_result(
    minimiser: Any,
    cost_fn: Callable[[Any, Any], Float[Array, ""]],
    x0: PyTree[Array],
    *,
    adjoint: optx.AbstractAdjoint,
    max_steps: int,
) -> MinimiseResult:
    r"""``minimise``, returning its diagnostics too.

    The same dispatch and the same iterate, plus $J(\theta^\star)$,
    $\lVert \nabla J(\theta^\star) \rVert_2$ and the step count. A
    custom [`Minimiser`][vardax.Minimiser] does not report its steps, so
    ``num_steps`` is ``-1`` for it. Costs one extra cost and gradient
    evaluation at the minimum, which is why the models' ``__call__`` uses
    ``minimise``.
    """
    if isinstance(minimiser, optx.AbstractMinimiser):
        solution = optx.minimise(
            fn=cost_fn,
            solver=minimiser,
            y0=x0,
            args=None,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=False,
        )
        cost, grad_norm = _cost_and_grad_norm(cost_fn, solution.value, None)
        return MinimiseResult(
            value=solution.value,
            cost=cost,
            grad_norm=grad_norm,
            num_steps=solution.stats["num_steps"],
        )
    if isinstance(minimiser, OptaxMinimiser | OptimistixMinimiser):
        return minimiser.run(cost_fn, x0, None, adjoint=adjoint)
    value = minimiser(cost_fn, x0, None)
    cost, grad_norm = _cost_and_grad_norm(cost_fn, value, None)
    return MinimiseResult(
        value=value, cost=cost, grad_norm=grad_norm, num_steps=jnp.asarray(-1)
    )


__all__ = [
    "MinimiseResult",
    "OptaxMinimiser",
    "OptimistixMinimiser",
    "minimise",
    "minimise_with_result",
]
