r"""K-step differentiation of iterative algorithms.

Generalises the one-step strategy of Bolte, Pauwels & Vaiter (2023):
run the first ``n_steps - k`` iterations of the inner solver under
``jax.lax.stop_gradient``, then ``k`` differentiable steps. Memory for
the backward pass is $O(k)$ regardless of the total iteration count,
and at a converged fixed point the estimator is exact for any ``k``
(the warmup iterations contribute nothing). For unconverged, truncated
solvers — the 4DVarNet regime — larger ``k`` trades memory for a less
biased gradient.

``KStepAdjoint(k=1)`` is exactly
[`OneStepAdjoint`][vardax.adjoints.OneStepAdjoint], which remains as
the familiar alias. In the shared pipekit vocabulary this strategy is
``pipekit_cycle.adjoints.TruncatedAdjoint(k)`` applied at the
inner-solve layer — see
[`to_optimistix_adjoint`][vardax.adjoints.to_optimistix_adjoint].
"""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Array, PyTree
import optimistix as optx


class KStepAdjoint(optx.AbstractAdjoint):
    """K-step differentiation: warmup under ``stop_gradient``, then ``k`` live steps.

    Use as the ``solver_adjoint`` argument to
    [`FourDVarNet1D`][vardax.FourDVarNet1D] /
    [`FourDVarNet2D`][vardax.FourDVarNet2D]:

    ```python
    from vardax.adjoints import KStepAdjoint

    model = FourDVarNet1D(
        state_dim=N, n_time=T, ...,
        solver_adjoint=KStepAdjoint(k=3),
        key=key,
    )
    ```

    Attributes:
        k: Number of trailing solver iterations that propagate
            gradients. Must be at least 1; values larger than
            ``n_solver_steps`` are clipped to a fully differentiable
            solve.

    References:
        Bolte, J., Pauwels, E. & Vaiter, S. (2023). One-step
        differentiation of iterative algorithms. NeurIPS 36.
        [arXiv:2305.13768](https://arxiv.org/abs/2305.13768).
    """

    k: int = 1

    def __check_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"KStepAdjoint needs k >= 1; got k={self.k}.")

    def apply(
        self,
        primal_fn: Callable,
        rewrite_fn: Callable,
        inputs: PyTree,
        tags: frozenset[object],
    ) -> PyTree[Array]:
        """Not used by vardax's custom learned solver — dispatch happens
        in ``vardax._src.solver`` via ``isinstance`` on the adjoint type.

        Implementing this method for upstream ``optimistix.minimise``
        compatibility is tracked under the planned upstream
        contribution (Decision D6).
        """
        raise NotImplementedError(
            "KStepAdjoint is currently a marker / strategy selector for the "
            "FourDVarNet inner solver. Generic apply() support for "
            "optimistix.minimise is planned as part of the upstream "
            "contribution. Use it via FourDVarNet*(solver_adjoint=KStepAdjoint(k=...))."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"KStepAdjoint(k={self.k})"
