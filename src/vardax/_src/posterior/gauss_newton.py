"""Gauss–Newton Hessian posterior: Λ_x|y at the analysis, inverted by capped CG.

It is the same operator as [`LaplaceCovariance`][vardax.LaplaceCovariance]; this
adapter exposes the Krylov budget as ``n_krylov``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx

from ._precision import gauss_newton_covariance
from .container import Posterior


class GaussNewtonHessian(eqx.Module):
    """Gauss–Newton posterior at the analysis, with a Krylov budget.

        Σ_x|y ≈ Λ_x|y⁻¹,   Λ_x|y = ∑ₜ Jₜᵀ Λ̃_ε,t Jₜ + Λ_x,   Jₜ = Mₜ J_h,t J_f,t

    J_h,t is the Jacobian of h at xₜ, J_f,t = ∂xₜ/∂x₀ that of the rollout, Mₜ = diag(mₜ)
    the mask and Λ̃_ε,t = Σ̃_ε,t⁻¹ the masked observation precision; a single-time model
    has one term with J_f,t = I. It matches
    [`LaplaceCovariance`][vardax.LaplaceCovariance], with CG capped at ``n_krylov``
    iterations per covariance mat-vec.

    Algorithm:
        1. Linearise h and, for 4D-Var, the rollout of f at the analysis.
        2. Assemble the precision Λ_x|y matrix-free.
        3. Return its inverse as a lazy operator applied by CG with at most
           ``n_krylov`` steps.

    Attributes:
        prior_cov_op: Prior covariance Σ_x. Default ``None``: the model's
            ``prior_cov_op``. Unused for a linear control, whose prior is the
            basis's Σ_z.
        obs_cov_op: Observation-error covariance Σ_ε at one time. Default ``None``:
            the model's ``obs_cov_op``.
        n_krylov: Maximum CG iterations per covariance mat-vec.
        tol: CG absolute and relative tolerance.

    Examples:
        Linear dynamics xₜ₊₁ = a xₜ, a fully observed two-state window and
        Σ_x = Σ_ε = I: the observations at t = 0, 1 both inform x₀, so
        Σ_x|y = 1 / (1 + 1 + a²).

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

        The Krylov budget is recorded in the provenance:

        >>> post.provenance["n_krylov"]
        50

        A point batch is not supported yet:

        >>> pts = vardax.PointObsBatch(
        ...     coords=jnp.zeros((1, 1)),
        ...     values=jnp.zeros(1),
        ...     sigma=jnp.ones(1),
        ...     valid=jnp.ones(1, bool),
        ... )
        >>> adapter(model(batch)[0], model, pts)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        UnsupportedInputError: The posterior adapters do not support ...
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
        """Build the Gauss–Newton posterior at ``analysis``.

        Args:
            analysis: Analysis of one window: x₀⋆, or (x₀⋆, η⋆) for ``WeakFourDVar``.
            model: The analysis model or its ``.as_analysis_step()`` wrapper.
            batch: The single window the analysis was computed from.

        Returns:
            The [`Posterior`][vardax.Posterior], with a lazy covariance.

        Raises:
            AttributeError: If the model exposes no ``obs_op``.
            UnsupportedInputError: For a 4D-Var model given a point batch or a batch
                that is not a ``Batch1D``, for a nonlinear parameterization, or for a
                ``WeakFourDVar`` with a ``parameterization=``, a ``model_err_basis=``
                or an analysis that is not ``(x0, eta)``.
            ShapeError: For a 4D-Var model given several windows, or a single-time
                model given several windows with missing entries.
            MissingInputError: If neither the adapter nor the model has a covariance
                operator.
        """
        solver = lx.CG(atol=self.tol, rtol=self.tol, max_steps=self.n_krylov)
        # Σ_x|y = Λ_x|y⁻¹, or Φ Σ_z|y Φᵀ for a linear control
        mean, covariance, structure, coefficients, precision = gauss_newton_covariance(
            analysis,
            model,
            batch,
            self.prior_cov_op,
            self.obs_cov_op,
            solver,
        )
        basis, coefficient_cov = coefficients or (None, None)
        return Posterior(
            mean=mean,
            cov=covariance,
            samples=None,
            basis=basis,
            coefficient_cov=coefficient_cov,
            precision=precision,
            provenance={
                "adapter": "GaussNewtonHessian",
                "n_krylov": self.n_krylov,
                "structure": structure,
            },
        )
