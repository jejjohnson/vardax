r"""Gauss-Newton Hessian posterior (Decision D10).

The posterior covariance at the MAP as the inverse Gauss–Newton Hessian of
the 4DVar cost,

$$
P^* = \Big(B^{-1} + \sum_t (H'_t M'_{t\leftarrow 0})^\top R_t^{-1}
             (H'_t M'_{t\leftarrow 0})\Big)^{-1},
$$

built matrix-free from the model's tangent-linear and adjoint rollouts and
inverted lazily with a Krylov solver. It is the same operator as
[`LaplaceCovariance`][vardax.LaplaceCovariance]; this adapter exposes the
Krylov budget as ``n_krylov``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx

from ._precision import gauss_newton_covariance
from .container import Posterior


class GaussNewtonHessian(eqx.Module):
    r"""Gauss-Newton Hessian inversion at MAP.

    Computes $(B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t))^{-1}$
    for 4DVar models and $(B^{-1} + H'^\top R^{-1} H')^{-1}$ for
    single-time ones, like
    [`LaplaceCovariance`][vardax.LaplaceCovariance], with CG capped at
    ``n_krylov`` iterations per covariance mat-vec.

    Attributes:
        prior_cov_op: $B$. Default ``None``: the model's ``prior_cov_op``.
            Unused for a linear control, whose prior is the basis's $Q$.
        obs_cov_op: $R$ at one time. Default ``None``: the model's
            ``obs_cov_op``.
        n_krylov: Maximum CG iterations per covariance mat-vec.
        tol: CG absolute and relative tolerance.

    Examples:
        Linear dynamics $x_{t+1} = a x_t$, a fully observed two-state
        window, $B = R = I$: the observations at $t = 0, 1$ both inform
        $x_0$, so $P^* = 1 / (1 + 1 + a^2)$.

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Decay:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> model = vardax.StrongFourDVar(
        ...     forward=Decay(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(2),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 2, 2)), mask=jnp.ones((1, 2, 2)))
        >>> adapter = vardax.GaussNewtonHessian(prior_cov_op=eye, obs_cov_op=eye)
        >>> post = adapter(model(batch)[0], model, batch)
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(2)), 1 / 2.25, atol=1e-5))
        True
    """

    prior_cov_op: lx.AbstractLinearOperator | None = None
    obs_cov_op: lx.AbstractLinearOperator | None = None
    n_krylov: int = eqx.field(static=True, default=50)
    tol: float = eqx.field(static=True, default=1e-6)

    def __call__(
        self,
        analysis: Float[Array, ...] | tuple[Array, Array],
        model: Any,
        batch: Any,
    ) -> Posterior:
        r"""Build the Gauss–Newton posterior at ``analysis``.

        Args:
            analysis: MAP of one window: $x_0^*$, or $(x_0^*, \eta^*)$ for
                ``WeakFourDVar``.
            model: The analysis model or its ``.as_analysis_step()`` wrapper.
            batch: The single window the analysis was computed from.

        Returns:
            [`Posterior`][vardax.Posterior] with a lazy covariance.

        Raises:
            AttributeError: If the model exposes no ``obs_op``.
            ValueError: For a point batch, a multi-sample batch, a
                nonlinear ``parameterization=``, or a parameterized
                ``WeakFourDVar``. A ``LinearParameterization`` (or
                ``basis=``) control gives $\Phi P_\theta \Phi^\top$.
        """
        solver = lx.CG(atol=self.tol, rtol=self.tol, max_steps=self.n_krylov)
        mean, covariance, structure = gauss_newton_covariance(
            analysis,
            model,
            batch,
            self.prior_cov_op,
            self.obs_cov_op,
            solver,
        )
        return Posterior(
            mean=mean,
            cov=covariance,
            samples=None,
            provenance={
                "adapter": "GaussNewtonHessian",
                "n_krylov": self.n_krylov,
                "structure": structure,
            },
        )
