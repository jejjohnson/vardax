"""Minimisers behind the `Minimiser` protocol: ``optimistix`` and first-order ``optax``.

Both share one calling convention, ``minimiser(cost_fn, x0, batch)``, and report
the final gradient norm so a caller can check the implicit adjoint's assumption.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, no_type_check

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Float, Int, PyTree
import optax
import optimistix as optx

from vardax._src.errors import ConfigurationError, ShapeError


class MinimiseResult(eqx.Module):
    """Outcome of a minimisation: the optimum θ⋆ and its diagnostics.

    Differentiating *through* the minimum uses an ``optimistix`` adjoint. The
    default, ``optimistix.ImplicitAdjoint``, applies the implicit function
    theorem at θ⋆, which assumes stationarity, ∇_θ𝒥(θ⋆) = 0. A first-order
    optimiser stopped by a step budget or by patience does not guarantee it, so
    both minimisers report the final gradient norm ‖∇_θ𝒥(θ⋆)‖₂ here; check it
    before trusting gradients through the minimum.

    Attributes:
        value: The minimiser θ⋆, with the structure of ``x0``.
        cost: The cost 𝒥(θ⋆).
        grad_norm: ‖∇_θ𝒥(θ⋆)‖₂ over all leaves of θ. The implicit adjoint is
            exact only when this is (close to) zero.
        num_steps: Number of optimiser steps taken (both phases, for
            `OptaxMinimiser` with a polish).
        converged: Whether the minimiser met its own convergence test:
            optimistix's ``result == successful``, or for `OptaxMinimiser` a
            small gradient or ``step_patience`` small steps (in the polish
            phase, when there is one); stopping on ``patience`` or
            ``max_steps`` is ``False``. ``None`` for a custom
            [`Minimiser`][vardax.Minimiser], which reports no status.

    Examples:
        The result carries the minimum and its diagnostics.

        >>> import jax.numpy as jnp, vardax
        >>> m = jnp.array([1.0, -2.0])
        >>> result = vardax.OptimistixMinimiser().run(
        ...     lambda x, m: 0.5 * jnp.sum((x - m) ** 2), jnp.zeros(2), m
        ... )
        >>> type(result).__name__
        'MinimiseResult'
        >>> bool(jnp.allclose(result.value, m, atol=1e-5))
        True

        At a minimum the gradient norm is near zero and the run converged.

        >>> bool(result.grad_norm < 1e-4), bool(result.converged)
        (True, True)

        The cost and step count are scalar arrays.

        >>> result.cost.shape, result.num_steps.shape
        ((), ())
    """

    value: PyTree[Array]
    cost: Float[Array, ""]
    grad_norm: Float[Array, ""]
    num_steps: Int[Array, ""]
    converged: Bool[Array, ""] | None = None


def _tree_l2_norm(tree: PyTree[Array]) -> Float[Array, ""]:
    """Euclidean norm ‖t‖₂ of a pytree, over all its leaves."""
    leaves = jtu.tree_leaves(tree)
    # ‖t‖₂ = sqrt(∑ over leaves of ∑ tⱼ²)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _close_over(
    cost_fn: Callable[[Any, Any], Float[Array, ""]], initial_guess: PyTree, batch: Any
) -> Callable[[Any, Any], Float[Array, ""]]:
    """Closure-convert ``cost_fn`` so arrays it closes over become leaves.

    The implicit adjoint differentiates with respect to the leaves of its
    inputs, so a cost that closes over model arrays (``prior_mean``, an
    operator) must expose them, as ``optimistix.minimise`` does.

    Raises:
        ShapeError: If ``cost_fn`` does not return a floating-point scalar.
    """
    closed_cost_fn = eqx.filter_closure_convert(cost_fn, initial_guess, batch)
    cost_struct = closed_cost_fn.out_struct  # ty:ignore[missing-attribute]
    if not (
        isinstance(cost_struct, jax.ShapeDtypeStruct)
        and cost_struct.shape == ()
        and jnp.issubdtype(cost_struct.dtype, jnp.floating)
    ):
        raise ShapeError("cost_fn must return a single floating-point scalar.")
    return closed_cost_fn


class OptimistixMinimiser(eqx.Module):
    """An ``optimistix`` minimiser behind the `Minimiser` protocol.

        θ⋆ = argmin_θ 𝒥(θ; batch)

    ``OptimistixMinimiser()`` is the models' default made explicit:
    ``optx.BFGS(rtol=1e-6, atol=1e-6)`` with ``max_steps=200`` and
    ``throw=False``, so a model built with it returns exactly what the default
    model returns. A quasi-Newton solver suits a state-sized control, but not
    one of 10⁵ network weights, where BFGS's dense inverse-Hessian estimate is
    not viable; use [`OptaxMinimiser`][vardax.OptaxMinimiser] there.

    Algorithm:
        1. Hand ``cost_fn(params, batch)``, the solver and ``x0`` to
           ``optimistix.minimise``, capped at ``max_steps`` iterations.
        2. Return the minimiser θ⋆; `run` also reports the cost 𝒥(θ⋆), the
           gradient norm ‖∇_θ𝒥(θ⋆)‖₂ and the step count at the solution.

    Raises:
        ConfigurationError: If ``max_steps`` is below 1.

    Attributes:
        solver: The ``optimistix.AbstractMinimiser``.
        max_steps: Iteration cap.
        throw: Passed to ``optimistix.minimise``: raise on failure
            (non-convergence within ``max_steps``) instead of returning the
            last iterate.

    Examples:
        A quadratic 𝒥(θ) = ½ ‖θ − m‖² has its minimum at m.

        >>> import jax.numpy as jnp, vardax
        >>> m = jnp.array([1.0, -2.0, 0.5])
        >>> minimiser = vardax.OptimistixMinimiser()
        >>> x = minimiser(lambda x, m: 0.5 * jnp.sum((x - m) ** 2), jnp.zeros(3), m)
        >>> bool(jnp.allclose(x, m, atol=1e-5))
        True

        The minimiser can return diagnostics, and any pytree is a valid
        control.

        >>> result = minimiser.run(
        ...     lambda p, m: 0.5 * jnp.sum((p["a"] - m) ** 2) + 0.5 * p["b"] ** 2,
        ...     {"a": jnp.zeros(3), "b": jnp.ones(())},
        ...     m,
        ... )
        >>> sorted(result.value)
        ['a', 'b']

        An invalid iteration cap is rejected.

        >>> vardax.OptimistixMinimiser(max_steps=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: max_steps must be >= 1, got 0.
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
            raise ConfigurationError(f"max_steps must be >= 1, got {max_steps}.")
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
        """Minimise 𝒥(θ; batch) from θ⁽⁰⁾ = ``x0``, with diagnostics.

        Args:
            cost_fn: Scalar cost 𝒥, called as ``cost_fn(params, batch)``.
            x0: Initial guess θ⁽⁰⁾, any pytree of arrays.
            batch: Passed as the second argument of ``cost_fn``.
            adjoint: How to differentiate through the minimum. Default
                ``optimistix.ImplicitAdjoint()``.

        Returns:
            A [`MinimiseResult`][vardax.MinimiseResult].

        Examples:
            >>> import jax.numpy as jnp, vardax
            >>> result = vardax.OptimistixMinimiser().run(
            ...     lambda x, m: 0.5 * jnp.sum((x - m) ** 2), jnp.zeros(2), jnp.ones(2)
            ... )
            >>> bool(jnp.allclose(result.value, 1.0, atol=1e-5))
            True
            >>> bool(result.converged)
            True
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
        # 𝒥(θ⋆) and ‖∇_θ𝒥(θ⋆)‖₂
        cost, grad_norm = _cost_and_grad_norm(cost_fn, value, batch)
        return MinimiseResult(
            value=value,
            cost=cost,
            grad_norm=grad_norm,
            num_steps=solution.stats["num_steps"],
            converged=solution.result == optx.RESULTS.successful,
        )

    def __call__(
        self,
        cost_fn: Callable[[Any, Any], Float[Array, ""]],
        x0: PyTree[Array],
        batch: Any = None,
    ) -> PyTree[Array]:
        """Return θ⋆ = argmin_θ 𝒥(θ; batch) starting from θ⁽⁰⁾ = ``x0``.

        Args:
            cost_fn: Scalar cost 𝒥, called as ``cost_fn(params, batch)``.
            x0: Initial guess θ⁽⁰⁾, any pytree of arrays.
            batch: Passed as the second argument of ``cost_fn``.

        Returns:
            The minimiser θ⋆, with the structure of ``x0``.
        """
        # θ⋆ = argmin_θ 𝒥(θ; batch)
        return optx.minimise(
            fn=cost_fn,
            solver=self.solver,
            y0=x0,
            args=batch,
            max_steps=self.max_steps,
            throw=self.throw,
        ).value


class OptaxMinimiser(eqx.Module):
    """First-order minimisation with an ``optax`` optimiser.

        θ⁽ⁱ⁺¹⁾ = θ⁽ⁱ⁾ + Δθ⁽ⁱ⁾,   ∇_θ𝒥(θ⁽ⁱ⁾) = ∂𝒥/∂θ at θ⁽ⁱ⁾

    with Δθ⁽ⁱ⁾ the ``optimizer`` update from ∇_θ𝒥(θ⁽ⁱ⁾). The loop is a
    ``jax.lax.while_loop`` (so the whole minimisation is ``jit``-able) and
    stops at the first of:

    - ``max_steps`` steps;
    - a small gradient, ‖∇_θ𝒥(θ⁽ⁱ⁾)‖₂ ≤ gtol;
    - ``step_patience`` consecutive small steps,
      ‖Δθ⁽ⁱ⁾‖₂ ≤ atol + rtol ‖θ⁽ⁱ⁾‖₂
      (one small step is not enough: a learning-rate schedule starting at
      zero gives a zero first update);
    - ``patience`` consecutive steps whose cost exceeds the lowest cost
      seen by more than float round-off.

    There is deliberately no relative cost-change test: with a large
    irreducible misfit, 𝒥⁽ⁱ⁾ − 𝒥⁽ⁱ⁻¹⁾ rounds to zero in float32 long before θ
    converges. For the same reason a step counts as an improvement when
    𝒥⁽ⁱ⁾ ≤ 𝒥_best + 8 ε_mach |𝒥_best| (ε_mach the machine epsilon of 𝒥), so a
    cost that is flat to round-off does not exhaust the patience. At a small
    gradient the loop stops *before* applying the update, so stored momentum
    cannot carry a stationary iterate away. It returns the best iterate seen:
    the lowest cost, with ties within round-off broken by the smaller gradient
    norm.

    With ``polish``, a second loop of at most ``polish_steps`` steps of that
    optimiser (for example ``optax.lbfgs()``) starts from there, with its own,
    tight, tolerances (``polish_gtol``, ``polish_rtol``, ``polish_atol``,
    ``polish_patience``; by default it stops only on ``polish_steps`` or
    ``step_patience`` exactly zero steps). The optimisers are called with the
    extra arguments ``value``, ``grad`` and ``value_fn``, so line-search
    optimisers such as ``optax.lbfgs`` work unchanged.

    First-order methods suit a control of 10⁵ network weights, where the dense
    inverse-Hessian estimate of BFGS is not viable. The implicit adjoint used
    to differentiate through the minimum assumes ∇_θ𝒥(θ⋆) = 0, which a
    first-order optimiser stopped early does not guarantee: check
    [`MinimiseResult.grad_norm`][vardax.MinimiseResult] from `run`.

    Algorithm:
        1. Evaluate the cost 𝒥(θ⁽ⁱ⁾) and gradient ∇_θ𝒥(θ⁽ⁱ⁾) at the iterate.
        2. Stop if the gradient norm is below ``gtol``; otherwise apply the
           ``optimizer`` update Δθ⁽ⁱ⁾.
        3. Track the best iterate, and count consecutive small steps and
           steps without improvement.
        4. Repeat until a stopping rule above fires; keep the best iterate.
        5. With ``polish``, repeat steps 1-4 from the best iterate with the
           polish optimiser and tolerances.

    Raises:
        ConfigurationError: If a step cap or ``step_patience`` is below 1,
            a tolerance is negative, or a patience is neither ``None``
            nor at least 1.

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
        Adam on 𝒥(θ) = ½ ‖θ − m‖², polished by L-BFGS, reaches the
        closed-form minimum θ⋆ = m.

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

        Calling the minimiser returns only the minimiser, and any pytree
        is a valid control.

        >>> adam = vardax.OptaxMinimiser(optax.adam(1e-1), max_steps=300)
        >>> x = adam(
        ...     lambda p, m: 0.5 * jnp.sum((p["a"] - m) ** 2), {"a": jnp.zeros(3)}, m
        ... )
        >>> bool(jnp.allclose(x["a"], m, atol=1e-2))
        True

        An invalid tolerance is rejected.

        >>> vardax.OptaxMinimiser(gtol=-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: gtol must be >= 0, got -1.0.

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
                raise ConfigurationError(f"{name} must be >= 1, got {steps}.")
        for name, value in (
            ("gtol", gtol),
            ("rtol", rtol),
            ("atol", atol),
            ("polish_gtol", polish_gtol),
            ("polish_rtol", polish_rtol),
            ("polish_atol", polish_atol),
        ):
            if value < 0:
                raise ConfigurationError(f"{name} must be >= 0, got {value}.")
        for name, value in (
            ("patience", patience),
            ("polish_patience", polish_patience),
        ):
            if value is not None and value < 1:
                raise ConfigurationError(f"{name} must be >= 1 or None, got {value}.")
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
        """Minimise 𝒥(θ; batch) from θ⁽⁰⁾ = ``x0``, with diagnostics.

        ``result.cost`` is differentiable: by the envelope theorem
        d𝒥⋆/dω = ∂𝒥/∂ω at θ⋆ for any parameter ω of the cost other than θ.

        Args:
            cost_fn: Scalar cost 𝒥, called as ``cost_fn(params, batch)``.
            x0: Initial guess θ⁽⁰⁾, any pytree of arrays (for a network, the
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

        Examples:
            >>> import jax.numpy as jnp, optax, vardax
            >>> minimiser = vardax.OptaxMinimiser(optax.adam(1e-1), max_steps=200)
            >>> result = minimiser.run(
            ...     lambda x, m: 0.5 * jnp.sum((x - m) ** 2), jnp.zeros(2), jnp.ones(2)
            ... )
            >>> result.num_steps > 0
            Array(True, dtype=bool)
        """
        x0 = jtu.tree_map(jnp.asarray, x0)
        closed_cost_fn = _close_over(cost_fn, x0, batch)
        inputs = (closed_cost_fn, self, x0, batch)
        # θ⋆ from the loop; its adjoint solves ∇_θ𝒥(θ⋆) = 0 (stationarity)
        value, (num_steps, converged) = (adjoint or optx.ImplicitAdjoint()).apply(
            _optax_primal, _optax_rewrite, inputs, frozenset()
        )
        # 𝒥(θ⋆) and ‖∇_θ𝒥(θ⋆)‖₂
        cost, grad_norm = _cost_and_grad_norm(cost_fn, value, batch)
        return MinimiseResult(
            value=value,
            cost=cost,
            grad_norm=grad_norm,
            num_steps=num_steps,
            converged=converged,
        )

    def __call__(
        self,
        cost_fn: Callable[[Any, Any], Float[Array, ""]],
        x0: PyTree[Array],
        batch: Any = None,
    ) -> PyTree[Array]:
        """Return θ⋆ = argmin_θ 𝒥(θ; batch) starting from θ⁽⁰⁾ = ``x0``.

        Args:
            cost_fn: Scalar cost 𝒥, called as ``cost_fn(params, batch)``.
            x0: Initial guess θ⁽⁰⁾, any pytree of arrays.
            batch: Passed as the second argument of ``cost_fn``.

        Returns:
            The minimiser θ⋆, with the structure of ``x0``.
        """
        # θ⋆ from the same loop as `run`, without the diagnostics
        return self.run(cost_fn, x0, batch).value


def _cost_and_grad_norm(
    cost_fn: Callable[[Any, Any], Any], value: PyTree[Array], batch: Any
) -> tuple[Any, Float[Array, ""]]:
    """Return 𝒥(θ⋆), differentiable to all orders, and ‖∇_θ𝒥(θ⋆)‖₂.

    The cost is evaluated at ``value`` itself, which carries the adjoint's
    derivative of θ⋆: the first derivative is then the envelope theorem's
    (∂𝒥/∂θ = 0 at the minimum), and higher derivatives keep the implicit
    dependence of θ⋆ on the parameters. The gradient norm is a diagnostic and is
    detached.
    """
    # 𝒥(θ⋆), at the adjoint-carrying value
    cost = cost_fn(value, batch)
    minimum = jax.lax.stop_gradient(value)
    # ∇_θ𝒥(θ⋆), at the detached minimum
    grad = jax.grad(cost_fn)(minimum, batch)
    return cost, jax.lax.stop_gradient(_tree_l2_norm(grad))


def _optax_loop(
    cost_fn, batch, initial_guess, optimizer, max_steps, tols, patience, while_loop
):
    """Run one early-stopped loop; return (best iterate, steps, converged)."""
    gtol, rtol, atol, step_patience = tols
    optimizer = optax.with_extra_args_support(optimizer)

    def value_fn(iterate):
        return cost_fn(iterate, batch)

    cost_dtype = cost_fn.out_struct.dtype
    inf = jnp.asarray(jnp.inf, dtype=cost_dtype)
    eps_mach = jnp.finfo(cost_dtype).eps

    def slack(reference_cost):
        # 8 ε_mach |𝒥_ref|, and 0 while 𝒥_ref is still infinite
        return jnp.where(
            jnp.isfinite(reference_cost), 8 * eps_mach * jnp.abs(reference_cost), 0.0
        )

    def better(cost, grad_norm, best_cost, best_grad_norm):
        # Lower cost beyond round-off, or a tie within round-off broken by
        # the gradient norm (closer to stationary).
        tolerance = slack(best_cost)
        # 𝒥 < 𝒥_best − 8 ε_mach |𝒥_best|, or a tie with a smaller ‖∇_θ𝒥‖₂
        return (cost < best_cost - tolerance) | (
            (cost <= best_cost + tolerance) & (grad_norm <= best_grad_norm)
        )

    def select(take, if_true, if_false):
        return jtu.tree_map(
            lambda new, old: jnp.where(take, new, old), if_true, if_false
        )

    def cond(carry):
        step, _, _, _, _, _, _, since_best, _, converged = carry
        # continue while i < N_iter, not converged and within the patience
        keep = (step < max_steps) & ~converged
        if patience is not None:
            keep = keep & (since_best < patience)
        return keep

    def body(carry):
        (
            step,
            iterate,
            opt_state,
            best_iterate,
            best_cost,
            best_grad_norm,
            min_cost,
            since_best,
            num_small_steps,
            _,
        ) = carry
        # 𝒥(θ⁽ⁱ⁾) and ∇_θ𝒥(θ⁽ⁱ⁾)
        cost, gradient = jax.value_and_grad(cost_fn)(iterate, batch)
        grad_norm = _tree_l2_norm(gradient)
        take = better(cost, grad_norm, best_cost, best_grad_norm)
        best_iterate = select(take, iterate, best_iterate)
        best_cost = jnp.where(take, cost, best_cost)
        best_grad_norm = jnp.where(take, grad_norm, best_grad_norm)
        # Patience: a step counts unless its cost is worse than the lowest
        # seen by more than round-off.
        since_best = jnp.where(cost <= min_cost + slack(min_cost), 0, since_best + 1)
        min_cost = jnp.minimum(cost, min_cost)
        # ‖∇_θ𝒥(θ⁽ⁱ⁾)‖₂ ≤ gtol
        small_grad = grad_norm <= gtol
        # Δθ⁽ⁱ⁾ from the optimiser, given 𝒥 and ∇_θ𝒥 for line searches
        updates, opt_state = optimizer.update(
            gradient, opt_state, iterate, value=cost, grad=gradient, value_fn=value_fn
        )
        # ‖Δθ⁽ⁱ⁾‖₂ ≤ atol + rtol ‖θ⁽ⁱ⁾‖₂
        small_step = _tree_l2_norm(updates) <= atol + rtol * _tree_l2_norm(iterate)
        num_small_steps = jnp.where(small_step, num_small_steps + 1, 0)
        # A stationary iterate is final: do not apply the (momentum) update.
        # θ⁽ⁱ⁺¹⁾ = θ⁽ⁱ⁾ + Δθ⁽ⁱ⁾
        next_iterate = select(
            small_grad, iterate, optax.apply_updates(iterate, updates)
        )
        converged = small_grad | (num_small_steps >= step_patience)
        return (
            step + 1,
            next_iterate,
            opt_state,
            best_iterate,
            best_cost,
            best_grad_norm,
            min_cost,
            since_best,
            num_small_steps,
            converged,
        )

    zero = jnp.asarray(0, dtype=jnp.int32)
    init = (
        zero,
        initial_guess,
        optimizer.init(initial_guess),
        initial_guess,
        inf,
        inf,
        inf,
        zero,
        zero,
        jnp.asarray(False),
    )
    step, last_iterate, _, best_iterate, best_cost, best_grad_norm, *_, converged = (
        while_loop(cond, body, init, max_steps=max_steps)
    )
    # The last iterate may not have been evaluated: keep it if it is better.
    # 𝒥(θ⁽ⁱ⁾) and ∇_θ𝒥(θ⁽ⁱ⁾) at the last iterate
    last_cost, last_gradient = jax.value_and_grad(cost_fn)(last_iterate, batch)
    take_last = better(
        last_cost, _tree_l2_norm(last_gradient), best_cost, best_grad_norm
    )
    return select(take_last, last_iterate, best_iterate), step, converged


# optimistix's implicit adjoint requires global functions with no closure;
# no_type_check keeps the jaxtyping test hook from wrapping this one
@no_type_check
def _optax_primal(inputs):
    """Run the main loop, then the optional polish loop, from ``initial_guess``."""
    cost_fn, minimiser, initial_guess, batch, while_loop = inputs
    iterate, steps, converged = _optax_loop(
        cost_fn,
        batch,
        initial_guess,
        minimiser.optimizer,
        minimiser.max_steps,
        (minimiser.gtol, minimiser.rtol, minimiser.atol, minimiser.step_patience),
        minimiser.patience,
        while_loop,
    )
    if minimiser.polish is not None:
        iterate, polish_steps, converged = _optax_loop(
            cost_fn,
            batch,
            iterate,
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
        # N_iter = main steps + polish steps
        steps = steps + polish_steps
    return iterate, (steps, converged)


# global and unwrapped for the implicit adjoint, as _optax_primal
@no_type_check
def _optax_rewrite(root, _residual, inputs):
    """Stationarity condition ∇_θ𝒥(θ⋆) = 0 for the implicit function theorem."""
    cost_fn, _, _, batch, *_ = inputs
    # ∇_θ𝒥(θ⋆), the residual whose root is θ⋆
    return jax.grad(cost_fn)(root, batch)


def minimise(
    minimiser: Any,
    cost_fn: Callable[[Any, Any], Float[Array, ""]],
    x0: PyTree[Array],
    *,
    adjoint: optx.AbstractAdjoint,
    max_steps: int,
) -> PyTree[Array]:
    """Run a model's ``minimiser`` on 𝒥(θ) = ``cost_fn(θ, None)`` from θ⁽⁰⁾.

    An ``optimistix.AbstractMinimiser`` goes through ``optimistix.minimise``
    with the model's ``max_steps`` and ``throw=False``. `OptaxMinimiser` and
    `OptimistixMinimiser` keep their own step caps and receive the model's
    ``adjoint``; any other [`Minimiser`][vardax.Minimiser] is called as
    ``minimiser(cost_fn, x0, None)``.

    Args:
        minimiser: An ``optimistix`` solver or a [`Minimiser`][vardax.Minimiser].
        cost_fn: Scalar cost 𝒥, called as ``cost_fn(params, None)``.
        x0: Initial guess θ⁽⁰⁾, any pytree of arrays.
        adjoint: How to differentiate through the minimum.
        max_steps: Iteration cap for an ``optimistix`` solver; the other
            minimisers keep their own.

    Returns:
        The minimiser θ⋆, with the structure of ``x0``.

    Examples:
        An ``optimistix`` solver and a vardax minimiser reach the same minimum
        of 𝒥(θ) = ½ ‖θ − m‖²:

        >>> import jax.numpy as jnp, optimistix as optx, vardax
        >>> from vardax._src.minimisers import minimise
        >>> m = jnp.array([1.0, -2.0])
        >>> cost_fn = lambda x, _: 0.5 * jnp.sum((x - m) ** 2)
        >>> theta = minimise(
        ...     optx.BFGS(rtol=1e-6, atol=1e-6),
        ...     cost_fn,
        ...     jnp.zeros(2),
        ...     adjoint=optx.ImplicitAdjoint(),
        ...     max_steps=50,
        ... )
        >>> bool(jnp.allclose(theta, m, atol=1e-5))
        True
        >>> theta = minimise(
        ...     vardax.OptimistixMinimiser(),
        ...     cost_fn,
        ...     jnp.zeros(2),
        ...     adjoint=optx.ImplicitAdjoint(),
        ...     max_steps=50,
        ... )
        >>> bool(jnp.allclose(theta, m, atol=1e-5))
        True
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
    """`minimise`, returning its diagnostics too.

    The same dispatch and the same iterate θ⋆, plus 𝒥(θ⋆), ‖∇_θ𝒥(θ⋆)‖₂ and the
    step count. A custom [`Minimiser`][vardax.Minimiser] does not report its
    steps, so ``num_steps`` is ``-1`` for it. Costs one extra cost and gradient
    evaluation at the minimum, which is why the models' ``__call__`` uses
    `minimise`.

    Args:
        minimiser: An ``optimistix`` solver or a [`Minimiser`][vardax.Minimiser].
        cost_fn: Scalar cost 𝒥, called as ``cost_fn(params, None)``.
        x0: Initial guess θ⁽⁰⁾, any pytree of arrays.
        adjoint: How to differentiate through the minimum.
        max_steps: Iteration cap for an ``optimistix`` solver; the other
            minimisers keep their own.

    Returns:
        A [`MinimiseResult`][vardax.MinimiseResult].

    Examples:
        A custom minimiser reports ``num_steps = -1`` and no status:

        >>> import jax.numpy as jnp, optimistix as optx, vardax
        >>> from vardax._src.minimisers import minimise_with_result
        >>> m = jnp.array([1.0, -2.0])
        >>> cost_fn = lambda x, _: 0.5 * jnp.sum((x - m) ** 2)
        >>> result = minimise_with_result(
        ...     lambda cost_fn, x0, batch: m,
        ...     cost_fn,
        ...     jnp.zeros(2),
        ...     adjoint=optx.ImplicitAdjoint(),
        ...     max_steps=50,
        ... )
        >>> int(result.num_steps), result.converged is None
        (-1, True)

        An ``optimistix`` solver reports its own status:

        >>> result = minimise_with_result(
        ...     optx.BFGS(rtol=1e-6, atol=1e-6),
        ...     cost_fn,
        ...     jnp.zeros(2),
        ...     adjoint=optx.ImplicitAdjoint(),
        ...     max_steps=50,
        ... )
        >>> bool(result.converged), bool(result.grad_norm < 1e-4)
        (True, True)
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
        # 𝒥(θ⋆) and ‖∇_θ𝒥(θ⋆)‖₂
        cost, grad_norm = _cost_and_grad_norm(cost_fn, solution.value, None)
        return MinimiseResult(
            value=solution.value,
            cost=cost,
            grad_norm=grad_norm,
            num_steps=solution.stats["num_steps"],
            converged=solution.result == optx.RESULTS.successful,
        )
    if isinstance(minimiser, OptaxMinimiser | OptimistixMinimiser):
        return minimiser.run(cost_fn, x0, None, adjoint=adjoint)
    value = minimiser(cost_fn, x0, None)
    # 𝒥(θ⋆) and ‖∇_θ𝒥(θ⋆)‖₂
    cost, grad_norm = _cost_and_grad_norm(cost_fn, value, None)
    return MinimiseResult(
        value=value, cost=cost, grad_norm=grad_norm, num_steps=jnp.asarray(-1)
    )


def least_squares(
    solver: optx.AbstractLeastSquaresSolver,
    residual_fn: Callable[[Any, Any], PyTree[Array]],
    y0: PyTree[Array],
    *,
    adjoint: optx.AbstractAdjoint,
    max_steps: int,
    jac: str = "fwd",
) -> optx.Solution:
    """Run a least-squares solver on 𝒥(y) = ½ ‖r(y)‖², r = ``residual_fn(y, None)``.

    The analysis models call it when the cost is given as whitened residuals, so
    a Gauss–Newton solver such as ``argminax.solve.GaussNewtonCG`` sees the
    residual and its Jacobian rather than the scalar cost. The solve never raises
    (``throw=False``): check ``result`` before trusting the value.

    Args:
        solver: An ``optimistix.AbstractLeastSquaresSolver``.
        residual_fn: The whitened residual r, called as ``residual_fn(y, None)``;
            any pytree of arrays.
        y0: Initial guess y⁽⁰⁾.
        adjoint: How to differentiate through the solution.
        max_steps: optimistix's step cap. A Gauss–Newton step linearises at the
            accepted iterate and proposes the next one, so ``max_steps`` steps
            apply ``max_steps − 1`` increments.
        jac: ``"fwd"`` linearises r by ``jax.linearize``; ``"bwd"`` materialises
            its Jacobian by ``jax.jacrev``, for an r with only a reverse-mode rule.

    Returns:
        The ``optimistix.Solution``.

    Examples:
        A linear residual is solved by one Gauss–Newton increment, so two steps:

        >>> import jax.numpy as jnp, optimistix as optx
        >>> from argminax.solve import GaussNewtonCG
        >>> from vardax._src.minimisers import least_squares
        >>> design = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        >>> data = jnp.array([1.0, 2.0, 2.0])
        >>> solution = least_squares(
        ...     GaussNewtonCG(rtol=0.0, atol=0.0, cg_rtol=1e-6, cg_atol=1e-6),
        ...     lambda x, _: design @ x - data,
        ...     jnp.zeros(2),
        ...     adjoint=optx.ImplicitAdjoint(),
        ...     max_steps=2,
        ... )
        >>> exact = jnp.linalg.lstsq(design, data)[0]
        >>> bool(jnp.allclose(solution.value, exact, atol=1e-5))
        True
    """
    return optx.least_squares(
        fn=residual_fn,
        solver=solver,
        y0=y0,
        args=None,
        options={"jac": jac},
        max_steps=max_steps,
        adjoint=adjoint,
        throw=False,
    )


def least_squares_with_result(
    solver: optx.AbstractLeastSquaresSolver,
    residual_fn: Callable[[Any, Any], PyTree[Array]],
    y0: PyTree[Array],
    *,
    adjoint: optx.AbstractAdjoint,
    max_steps: int,
) -> MinimiseResult:
    """`least_squares`, returning a [`MinimiseResult`][vardax.MinimiseResult].

    The cost is 𝒥(y⋆) = ½ ‖r(y⋆)‖² and the gradient norm ‖∇_y 𝒥(y⋆)‖₂ =
    ‖J_rᵀ r‖₂, both over the solver's own variable y.

    Args:
        solver: An ``optimistix.AbstractLeastSquaresSolver``.
        residual_fn: The whitened residual r, called as ``residual_fn(y, None)``.
        y0: Initial guess y⁽⁰⁾.
        adjoint: How to differentiate through the solution.
        max_steps: optimistix's step cap.

    Returns:
        A [`MinimiseResult`][vardax.MinimiseResult].

    Examples:
        >>> import jax.numpy as jnp, optimistix as optx
        >>> from argminax.solve import GaussNewtonCG
        >>> from vardax._src.minimisers import least_squares_with_result
        >>> result = least_squares_with_result(
        ...     GaussNewtonCG(rtol=1e-6, atol=1e-6),
        ...     lambda x, _: x - jnp.array([1.0, -2.0]),
        ...     jnp.zeros(2),
        ...     adjoint=optx.ImplicitAdjoint(),
        ...     max_steps=10,
        ... )
        >>> bool(result.converged), bool(result.grad_norm < 1e-5)
        (True, True)
    """
    solution = least_squares(
        solver, residual_fn, y0, adjoint=adjoint, max_steps=max_steps
    )

    def cost_fn(y: PyTree[Array], args: Any) -> Float[Array, ""]:
        leaves = jtu.tree_leaves(residual_fn(y, args))
        # ‖r(y)‖², over every leaf of r
        squared_norm = jnp.sum(jnp.stack([jnp.sum(leaf**2) for leaf in leaves]))
        # 𝒥(y) = ½ ‖r(y)‖²
        return 0.5 * squared_norm

    # 𝒥(y⋆) and ‖∇_y𝒥(y⋆)‖₂
    cost, grad_norm = _cost_and_grad_norm(cost_fn, solution.value, None)
    return MinimiseResult(
        value=solution.value,
        cost=cost,
        grad_norm=grad_norm,
        num_steps=solution.stats["num_steps"],
        converged=solution.result == optx.RESULTS.successful,
    )


__all__ = [
    "MinimiseResult",
    "OptaxMinimiser",
    "OptimistixMinimiser",
    "least_squares",
    "least_squares_with_result",
    "minimise",
    "minimise_with_result",
]
