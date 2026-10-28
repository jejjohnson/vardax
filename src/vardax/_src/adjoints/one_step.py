"""One-step differentiation of iterative algorithms (Bolte et al., 2023).

vardax-owned ``optimistix.AbstractAdjoint`` subclass implementing the
one-step differentiation strategy from

    Bolte, Pauwels & Vaiter (2023). One-step differentiation of
    iterative algorithms. NeurIPS 36. arXiv:2305.13768

The strategy: run ``K - 1`` iterations of the inner solver with
``jax.lax.stop_gradient`` applied to the iterate, then one
differentiable step. Memory is O(1) (matching ``ImplicitAdjoint``)
without requiring strict convergence to a fixed point.

The plan (Decision D6, D15) is to contribute this upstream to
``optimistix`` once stable. Until then, it lives in vardax. The class
is implemented as a marker / strategy selector for the
``FourDVarNet`` inner solver — the actual one-step dispatch lives in
``vardax._src.solver.one_step_solve_4dvarnet_*``. The model's
``__call__`` uses ``isinstance(model.solver_adjoint, OneStepAdjoint)``
to pick the path.

Because optimistix's ``minimise`` is not the use site for vardax's
custom learned solver, the ``apply()`` method here is a no-op stub
(raises ``NotImplementedError`` if hit). This will fill in when the
adjoint is generalised for upstream contribution.
"""

from __future__ import annotations

from typing import Any

from .k_step import KStepAdjoint


class OneStepAdjoint(KStepAdjoint):
    r"""One-step differentiation (Bolte et al., 2023).

    Run ``K - 1`` warmup iterations of the inner solver with
    ``jax.lax.stop_gradient``, then one differentiable step. Gives
    O(1) memory and is exact at the fixed point of the inner
    iteration.

    $$ \frac{d z^\star}{d\phi} \approx \partial_\phi T(\bar z_{K-1}; \phi) $$

    where $\bar z_{K-1}$ is the iterate after the warmup, held fixed.

    Algorithm:
        1. Run $K - 1$ solver iterations under ``jax.lax.stop_gradient``.
        2. Run the last iteration differentiably.
        3. Backpropagate through that single step.

    Use as the ``solver_adjoint`` argument to
    [`FourDVarNet1D`][vardax.FourDVarNet1D] /
    [`FourDVarNet2D`][vardax.FourDVarNet2D]:

    ```python
    from vardax.adjoints import OneStepAdjoint

    model = FourDVarNet1D(
        state_dim=N, n_time=T, ...,
        solver_adjoint=OneStepAdjoint(),
        key=key,
    )
    ```

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

    References:
        Bolte, J., Pauwels, E. & Vaiter, S. (2023). One-step
        differentiation of iterative algorithms. NeurIPS 36.
        [arXiv:2305.13768](https://arxiv.org/abs/2305.13768).
    """

    k: int = 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "OneStepAdjoint()"


def _is_one_step_adjoint(adjoint: Any) -> bool:
    """Type-narrowing helper for solver dispatch."""
    return isinstance(adjoint, OneStepAdjoint)
