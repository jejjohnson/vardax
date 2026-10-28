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
from typing import Any

from jaxtyping import Array, PyTree
import optimistix as optx

from vardax._src.errors import ConfigurationError


class KStepAdjoint(optx.AbstractAdjoint):
    r"""K-step differentiation: warmup under ``stop_gradient``, then ``k`` live steps.

    For a solver iteration $z_{j+1} = T(z_j)$ run $n$ times, the gradient
    flows only through the last $k$ steps,

    $$ \frac{d z_n}{d\phi} \approx \frac{\partial T^{k}(\bar z_{n-k})}{\partial\phi} $$

    where $\bar z_{n-k}$ is the iterate after the warmup, held fixed.

    Algorithm:
        1. Run the first $n - k$ solver iterations under
           ``jax.lax.stop_gradient``.
        2. Run the last $k$ iterations differentiably.
        3. Backpropagate through those $k$ iterations only, at memory cost
           $O(k)$.

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

    Raises:
        ConfigurationError: If ``k`` is below 1.

    Attributes:
        k: Number of trailing solver iterations that propagate
            gradients. Must be at least 1; values larger than
            ``n_solver_steps`` are clipped to a fully differentiable
            solve.

    Examples:
        >>> import vardax
        >>> vardax.KStepAdjoint(k=3)
        KStepAdjoint(k=3)

        ``k=1`` is the one-step estimator, and `OneStepAdjoint` is a
        subclass.

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
        Bolte, J., Pauwels, E. & Vaiter, S. (2023). One-step
        differentiation of iterative algorithms. NeurIPS 36.
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
        """Not used by vardax's custom learned solver — dispatch happens
        in ``vardax._src.solver`` via ``isinstance`` on the adjoint type.

        Implementing this method for upstream ``optimistix.minimise``
        compatibility is tracked under the planned upstream
        contribution (Decision D6).

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
        return f"KStepAdjoint(k={self.k})"

    def get_config(self) -> dict[str, Any]:
        return {
            "k": self.k,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KStepAdjoint:
        return cls(**config)
