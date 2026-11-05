"""Laplace approximation at the analysis: Σ_x|y ≈ Λ_x|y⁻¹, the Gauss–Newton inverse.

The result is a lazy ``AbstractLinearOperator``, so samples and marginals compose
through ``lineax.CG`` without materialising Σ_x|y.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx

from ._precision import gauss_newton_covariance
from .container import Posterior

#: Step cap of the default CG solves; it bounds the Krylov space.
DEFAULT_MAX_STEPS = 200

#: Default solver for the lazy inverses.
DEFAULT_SOLVER = lx.CG(atol=1e-6, rtol=1e-6, max_steps=DEFAULT_MAX_STEPS)


class LaplaceCovariance(eqx.Module):
    """Laplace approximation of the posterior at the analysis.

        Σ_x|y ≈ Λ_x|y⁻¹,   Λ_x|y = Jᵀ Λ_ε J + Λ_x

    J is the observation Jacobian of the control. A single analysis time has
    J = J_h (``OptimalInterpolation``, ``ThreeDVar``). A 4D-Var window reaches x₀
    through f, so J stacks Jₜ = Mₜ J_h,t J_f,t with J_f,t = ∂xₜ/∂x₀ and Mₜ = diag(mₜ),
    and each time has its masked Λ̃_ε,t = Σ̃_ε,t⁻¹. For ``WeakFourDVar`` the control is
    (x₀, η₁, …, η_T) with Λ_η on each ηₜ, and the covariance is the x₀ block of the
    inverse.

    This is the default adapter for ``ThreeDVar``, ``StrongFourDVar``,
    ``WeakFourDVar`` and ``IncrementalFourDVar``, under a Gaussian likelihood, a
    Gaussian prior and a single posterior mode (verified by SBC). For multimodal
    posteriors or non-Gaussian likelihoods, use ``EnsembleCovariance``.

    Algorithm:
        1. Linearise h and, for 4D-Var, the rollout of f at the analysis.
        2. Assemble the Gauss–Newton precision Λ_x|y matrix-free.
        3. Return its inverse as a lazy operator applied by ``solver``.

    Attributes:
        prior_cov_op: Prior covariance Σ_x. Default ``None``: the model's
            ``prior_cov_op``. Unused for a linear control, whose prior is the
            basis's Σ_z.
        obs_cov_op: Observation-error covariance Σ_ε at one time. Default ``None``:
            the model's ``obs_cov_op``. Pass Σ_x and Σ_ε explicitly only to evaluate
            the posterior under other error statistics than the analysis used.
        solver: Linear solver for the lazy inverses; by default CG with
            ``atol = rtol = 1e-6`` and 200 steps.

    Examples:
        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> from types import SimpleNamespace
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float32))
        >>> laplace = vdx.LaplaceCovariance(prior_cov_op=eye, obs_cov_op=eye)
        >>> model = SimpleNamespace(obs_op=vdx.LinearObs(operator=eye))
        >>> post = laplace(jnp.zeros(4), model, batch=None)
        >>> post.mean.shape
        (4,)
        >>> # Σ_x = Σ_ε = H = I, so Σ_x|y = (I + I)⁻¹ = I / 2
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(4)), 0.5, atol=1e-2))
        True

        Weaker observations, Σ_ε = 3 I, leave a larger posterior,
        Σ_x|y = (1 + ⅓)⁻¹ I = ¾ I:

        >>> weak = jnp.float32(3.0) * eye
        >>> post = vdx.LaplaceCovariance(prior_cov_op=eye, obs_cov_op=weak)(
        ...     jnp.zeros(4), model, batch=None
        ... )
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(4)), 0.75, atol=1e-2))
        True

        A model with no ``obs_op`` is rejected:

        >>> laplace(
        ...     jnp.zeros(4), SimpleNamespace(), batch=None
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        AttributeError: ...
    """

    prior_cov_op: lx.AbstractLinearOperator | None = None
    obs_cov_op: lx.AbstractLinearOperator | None = None
    solver: lx.AbstractLinearSolver = eqx.field(static=True, default=DEFAULT_SOLVER)

    def __call__(
        self,
        analysis: Float[Array, ...] | tuple[Array, Array],
        model: Any,
        batch: Any,
    ) -> Posterior:
        """Build the Laplace posterior at ``analysis``.

        Args:
            analysis: Analysis of one window: x₀⋆, or (x₀⋆, η⋆) for ``WeakFourDVar``.
            model: The analysis model or its ``.as_analysis_step()`` wrapper. Its
                ``obs_op`` and, for 4D-Var, its rollout build the precision.
            batch: The single window the analysis was computed from; 4D-Var models
                read its length and mask. A batch with a leading sample axis must
                hold one sample.

        Returns:
            The [`Posterior`][vardax.Posterior] with ``mean`` x₀⋆ and ``cov`` a lazy
            ``AbstractLinearOperator`` representing Σ_x|y.

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
        # Σ_x|y = Λ_x|y⁻¹, or Φ Σ_z|y Φᵀ for a linear control
        mean, covariance, structure, coefficients, precision = gauss_newton_covariance(
            analysis,
            model,
            batch,
            self.prior_cov_op,
            self.obs_cov_op,
            self.solver,
        )
        basis, coefficient_cov = coefficients or (None, None)
        return Posterior(
            mean=mean,
            cov=covariance,
            samples=None,
            basis=basis,
            coefficient_cov=coefficient_cov,
            precision=precision,
            provenance={"adapter": "LaplaceCovariance", "structure": structure},
        )
