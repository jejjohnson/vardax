"""One-step differentiation of iterative algorithms (Bolte et al., 2023).

`OneStepAdjoint` is the ``k=1`` alias of `KStepAdjoint`; like it, the class only selects
a strategy, and the dispatch lives in ``vardax._src.solver``.
"""

from __future__ import annotations

from typing import Any

from .k_step import KStepAdjoint


class OneStepAdjoint(KStepAdjoint):
    """One-step differentiation: every iteration but the last under ``stop_gradient``.

        dx^(N_iter) / dθ ≈ ∂ g_step(x^(N_iter − 1); θ) / ∂θ

    g_step is one iteration of the inner solver, j counts its iterations in x^(j), and
    the iterate x^(N_iter − 1) is held fixed. It gives O(1) memory, matching
    ``ImplicitAdjoint``, without requiring strict convergence, and is exact at the
    fixed point of the inner iteration. Use it as the ``solver_adjoint`` of
    [`FourDVarNet1D`][vardax.FourDVarNet1D] or
    [`FourDVarNet2D`][vardax.FourDVarNet2D].

    Algorithm:
        1. Run N_iter − 1 solver iterations under ``jax.lax.stop_gradient``.
        2. Run the last iteration differentiably.
        3. Backpropagate through that single step.

    Attributes:
        k: Number of trailing iterations that propagate gradients, here 1.

    Examples:
        >>> import vardax
        >>> adjoint = vardax.OneStepAdjoint()
        >>> adjoint
        OneStepAdjoint()
        >>> adjoint.k
        1

        It is [`KStepAdjoint`][vardax.KStepAdjoint] with ``k=1``.

        >>> isinstance(adjoint, vardax.KStepAdjoint)
        True

        It selects the strategy of a model's inner solver.

        >>> import jax
        >>> model = vardax.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=2,
        ...     n_solver_steps=2,
        ...     solver_adjoint=adjoint,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> model.solver_adjoint
        OneStepAdjoint()

    References:
        Bolte, J., Pauwels, E. & Vaiter, S. (2023). One-step differentiation of
        iterative algorithms. NeurIPS 36.
        [arXiv:2305.13768](https://arxiv.org/abs/2305.13768).
    """

    k: int = 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Return ``OneStepAdjoint()``."""
        return "OneStepAdjoint()"


def _is_one_step_adjoint(adjoint: Any) -> bool:
    """Type-narrowing helper for solver dispatch."""
    return isinstance(adjoint, OneStepAdjoint)
