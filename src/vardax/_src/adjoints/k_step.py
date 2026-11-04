"""K-step differentiation of iterative algorithms, generalising one-step.

`KStepAdjoint` only selects a strategy for the FourDVarNet inner solver; the dispatch
lives in ``vardax._src.solver``, and in the pipekit vocabulary it is
``pipekit_cycle.adjoints.TruncatedAdjoint(k)`` at the inner-solve layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jaxtyping import Array, PyTree
import optimistix as optx

from vardax._src.errors import ConfigurationError


class KStepAdjoint(optx.AbstractAdjoint):
    """K-step differentiation: warmup under ``stop_gradient``, then ``k`` live steps.

        x^(j+1) = g_step(x^(j); θ),   j = 0, …, N_iter − 1
        dx^(N_iter) / dθ ≈ ∂ g_stepᵏ(x^(N_iter − k); θ) / ∂θ

    g_step is one iteration of the inner solver, and the iterate x^(N_iter − k) after
    the warmup is held fixed, so the gradient flows only through the last k steps.
    Memory for the backward pass is O(k) whatever the total iteration count, and at a
    converged fixed point the estimator is exact for any k, since the warmup
    iterations contribute nothing. For unconverged, truncated solvers, the 4DVarNet
    regime, a larger k trades memory for a less biased gradient. Use it as the
    ``solver_adjoint`` of [`FourDVarNet1D`][vardax.FourDVarNet1D] or
    [`FourDVarNet2D`][vardax.FourDVarNet2D].

    Algorithm:
        1. Run the first N_iter − k solver iterations under
           ``jax.lax.stop_gradient``.
        2. Run the last k iterations differentiably.
        3. Backpropagate through those k iterations only.

    Attributes:
        k: Number of trailing solver iterations that propagate gradients. Must be at
            least 1; values larger than ``n_solver_steps`` are clipped to a fully
            differentiable solve.

    Raises:
        ConfigurationError: If ``k`` is below 1.

    Examples:
        >>> import vardax
        >>> vardax.KStepAdjoint(k=3)
        KStepAdjoint(k=3)

        ``k=1`` is the one-step estimator, and `OneStepAdjoint` is a subclass.

        >>> isinstance(vardax.OneStepAdjoint(), vardax.KStepAdjoint)
        True
        >>> vardax.KStepAdjoint().k
        1

        A non-positive ``k`` is rejected.

        >>> vardax.KStepAdjoint(k=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: KStepAdjoint needs k >= 1; got k=0.

    References:
        Bolte, J., Pauwels, E. & Vaiter, S. (2023). One-step differentiation of
        iterative algorithms. NeurIPS 36.
        [arXiv:2305.13768](https://arxiv.org/abs/2305.13768).
    """

    k: int = 1

    def __check_init__(self) -> None:
        if self.k < 1:
            raise ConfigurationError(f"KStepAdjoint needs k >= 1; got k={self.k}.")

    def apply(
        self,
        primal_fn: Callable,
        rewrite_fn: Callable,
        inputs: PyTree,
        tags: frozenset[object],
    ) -> PyTree[Array]:
        """Raise: the learned solver dispatches on the adjoint's type instead.

        ``optimistix.minimise`` is not the use site for the learned solver, which
        selects its path in ``vardax._src.solver`` by ``isinstance`` on the adjoint.

        Args:
            primal_fn: The primal solve (unused).
            rewrite_fn: The fixed-point rewrite (unused).
            inputs: The solve inputs (unused).
            tags: Optimistix tags (unused).

        Raises:
            NotImplementedError: Always; select the adjoint through
                ``FourDVarNet*(solver_adjoint=KStepAdjoint(k=...))``.
        """
        raise NotImplementedError(
            "KStepAdjoint is currently a marker / strategy selector for the "
            "FourDVarNet inner solver. Generic apply() support for "
            "optimistix.minimise is planned as part of the upstream "
            "contribution. Use it via FourDVarNet*(solver_adjoint=KStepAdjoint(k=...))."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Return ``KStepAdjoint(k=...)``."""
        return f"KStepAdjoint(k={self.k})"

    def get_config(self) -> dict[str, Any]:
        """Return the fields needed to rebuild the adjoint."""
        return {
            "k": self.k,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KStepAdjoint:
        """Rebuild an adjoint from ``get_config``."""
        return cls(**config)
