"""Optimal interpolation: the closed-form BLUE of a linear-Gaussian analysis.

Σ_x and Σ_ε are the prior and observation-error covariances, h the observation
operator with tangent H, and Σ_y = H Σ_x Hᵀ + Σ_ε the innovation covariance.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import jax
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D
from vardax._src.errors import ConfigurationError
from vardax._src.models._analysis_step import analyse_forecast
from vardax._src.models._observations import fill_gaps
from vardax._src.models._rollout import accepts_mask
from vardax._src.posterior.container import Posterior
from vardax._src.utils.covariance import (
    as_covariance,
    inverse_operator,
)


class OptimalInterpolation(eqx.Module):
    """Best linear unbiased estimate: the linear-Gaussian analysis in closed form.

        x⋆ = μ_x + G r,   r = m ⊙ y − m ⊙ h(μ_x)
        G = Σ_x Hᵀ Σ_y⁻¹,   Σ_y = H Σ_x Hᵀ + Σ̃_ε
        Σ̃_ε = M Σ_ε M + (I − M),   M = diag(m)

    H is the tangent of m ⊙ h(·) at μ_x, so the posterior is exact in one matrix
    expression, with no iteration and no convergence criterion. The solve runs against
    a ``lineax.AbstractLinearOperator``, so by default Σ_y is never materialised.

    Algorithm:
        1. For each sample, under ``jax.vmap``, form the masked operator m ⊙ h(·) and
           the residual r.
        2. Linearise the masked operator at μ_x to get H, and build Σ_y without
           materialising it.
        3. Solve Σ_y v = r with CG, or with ``solver`` when given.
        4. Return x⋆ = μ_x + Σ_x Hᵀ v.

    Attributes:
        obs_op: Observation operator h, expected to be linear. It is not checked: a
            nonlinear h is linearised once at μ_x and the result is that single
            linearised update, not the MAP. Use [`ThreeDVar`][vardax.ThreeDVar] for a
            nonlinear h.
        prior_mean: Prior mean μ_x.
        prior_cov_op: Prior covariance Σ_x.
        obs_cov_op: Observation-error covariance Σ_ε.
        cg_atol: CG absolute tolerance for the inner solve.
        cg_rtol: CG relative tolerance for the inner solve.
        cg_max_steps: CG iteration cap.
        solver: Optional linear solver for the innovation system, used instead of CG
            (the ``cg_*`` fields are then ignored), e.g. ``lineax.Cholesky()`` to
            factorise Σ_y exactly when it fits in memory, since CG at a fixed tolerance
            is inexact and can stall on an ill-conditioned system. It is also the
            default solver of the ``posterior`` adapter.
        form: ``"observation"`` (default) is the observation form, which solves the
            D_y × D_y system Σ_y v = r. ``"state"`` is the state form, which solves the
            D_x × D_x system (Σ_x⁻¹ + Hᵀ Σ̃_ε⁻¹ H) δx = Hᵀ Σ̃_ε⁻¹ r, the one to use when
            D_y ≫ D_x. Σ_x⁻¹ and Σ̃_ε⁻¹ are closed form for a diagonal operator and
            solved with ``solver`` otherwise, and the default posterior is then
            (Σ_x⁻¹ + Hᵀ Σ̃_ε⁻¹ H)⁻¹.

    Raises:
        ConfigurationError: If ``form`` is neither ``"observation"`` nor ``"state"``.

    Examples:
        With Σ_x = Σ_ε = I, identity h and everything observed, the analysis splits
        the residual in half, x⋆ = y / 2:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> structure = jax.ShapeDtypeStruct((1, 3), jnp.result_type(float))
        >>> eye = lx.IdentityLinearOperator(structure)
        >>> oi = vardax.OptimalInterpolation(
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros((1, 3)),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 1, 3)), mask=jnp.ones((1, 1, 3)))
        >>> xa = oi(batch)
        >>> xa.shape
        (1, 1, 3)
        >>> bool(jnp.allclose(xa, 0.5, atol=1e-4))
        True

        Unobserved entries (zero mask) keep the prior mean, and an exact Cholesky
        solver replaces CG:

        >>> exact = vardax.OptimalInterpolation(
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros((1, 3)),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ...     solver=lx.Cholesky(),
        ... )
        >>> gap = vardax.Batch1D(
        ...     input=jnp.ones((1, 1, 3)), mask=jnp.array([[[1.0, 0.0, 1.0]]])
        ... )
        >>> bool(jnp.allclose(exact(gap), jnp.array([0.5, 0.0, 0.5]), atol=1e-5))
        True

        Composed with the Laplace adapter, the posterior covariance of this problem is
        Σ_x|y = (I + I)⁻¹ = I / 2:

        >>> post = oi.posterior(batch)
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones((1, 3))), 0.5, atol=1e-2))
        True
    """

    # linear: a nonlinear h gets a single linearised update
    obs_op: Any
    prior_mean: Float[Array, "*state"]
    prior_cov_op: lx.AbstractLinearOperator = eqx.field(converter=as_covariance)
    obs_cov_op: lx.AbstractLinearOperator = eqx.field(converter=as_covariance)
    cg_atol: float = eqx.field(static=True, default=1e-6)
    cg_rtol: float = eqx.field(static=True, default=1e-6)
    cg_max_steps: int = eqx.field(static=True, default=200)
    solver: lx.AbstractLinearSolver | None = eqx.field(static=True, default=None)
    # "observation" or "state", checked in __check_init__
    form: str = eqx.field(static=True, default="observation")

    def __check_init__(self) -> None:
        if self.form not in ("observation", "state"):
            raise ConfigurationError(
                f"form must be 'observation' or 'state'; got {self.form!r}."
            )

    def __call__(self, batch: Batch1D) -> Float[Array, " batch *state"]:
        """BLUE analysis of each sample in the batch, x⋆ = μ_x + G r.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D].

        Returns:
            The analysis of every sample, with a leading batch axis.
        """

        def analyse_sample(
            sample_obs: Float[Array, "*obs"], sample_mask: Float[Array, "*obs"]
        ) -> Float[Array, "*state"]:
            return self._analyse(
                self.prior_mean, sample_obs, sample_mask, self.obs_op, self.obs_cov_op
            )

        return jax.vmap(analyse_sample)(fill_gaps(batch.input, batch.mask), batch.mask)

    def _analyse(
        self,
        prior_mean: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_covariance: lx.AbstractLinearOperator,
    ) -> Float[Array, "*state"]:
        """BLUE analysis of one sample around the prior mean μ_x."""
        # Σ̃_ε = M Σ_ε M + (I − M), so masked entries drop out of the likelihood
        masked_covariance = gaussx.masked_obs_covariance(obs_covariance, sample_mask)
        if self.form == "state":
            return self._analyse_state_space(
                prior_mean, sample_obs, sample_mask, obs_op, masked_covariance
            )
        masked_obs, obs_jacobian = _masked_system(prior_mean, sample_mask, obs_op)
        # m ⊙ y
        masked_observations = sample_obs * sample_mask
        # m ⊙ h(μ_x)
        predicted_obs = masked_obs(prior_mean)
        # r = m ⊙ y − m ⊙ h(μ_x)
        residual = masked_observations - predicted_obs
        # H μ_x
        linearised_obs = obs_jacobian.mv(prior_mean)
        # y′ = r + H μ_x, the observation a nonlinear h leaves for the linear update
        effective_obs = residual + linearised_obs
        solver = self._default_solver()
        # x⋆ = μ_x + Σ_x Hᵀ Σ_y⁻¹ r
        update = gaussx.linear_gaussian_update(
            prior_mean,
            self.prior_cov_op,
            obs_jacobian,
            masked_covariance,
            effective_obs,
            solver=solver,
        )
        return update.mean

    def _analyse_state_space(
        self,
        prior_mean: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        masked_covariance: lx.AbstractLinearOperator,
    ) -> Float[Array, "*state"]:
        """BLUE analysis in the state form, δx = (Σ_x⁻¹ + Hᵀ Σ̃_ε⁻¹ H)⁻¹ Hᵀ Σ̃_ε⁻¹ r."""
        masked_obs, obs_jacobian = _masked_system(prior_mean, sample_mask, obs_op)
        solver = self._default_solver()
        # Hᵀ
        adjoint = obs_jacobian.transpose()
        # Σ̃_ε⁻¹
        obs_precision = inverse_operator(masked_covariance, solver)
        # Hᵀ Σ̃_ε⁻¹ H; masked rows of H are zero, so gaps drop out
        information = adjoint @ obs_precision @ obs_jacobian
        # Λ_x|y = Hᵀ Σ̃_ε⁻¹ H + Σ_x⁻¹
        posterior_precision = information + inverse_operator(self.prior_cov_op, solver)
        tagged_precision = lx.TaggedLinearOperator(
            posterior_precision, lx.positive_semidefinite_tag
        )
        # m ⊙ y
        masked_observations = sample_obs * sample_mask
        # m ⊙ h(μ_x)
        predicted_obs = masked_obs(prior_mean)
        # r = m ⊙ y − m ⊙ h(μ_x)
        residual = masked_observations - predicted_obs
        # Σ̃_ε⁻¹ r
        weighted_residual = obs_precision.mv(residual)
        # Hᵀ Σ̃_ε⁻¹ r
        right_hand_side = adjoint.mv(weighted_residual)
        # δx = Λ_x|y⁻¹ Hᵀ Σ̃_ε⁻¹ r
        solution = lx.linear_solve(tagged_precision, right_hand_side, solver=solver)
        increment = solution.value
        # x⋆ = μ_x + δx
        return prior_mean + increment

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``. Without an adapter
        the covariance is the observation form Σ_x|y = Σ_x − Σ_x Hᵀ Σ_y⁻¹ H Σ_x: it
        solves the system the analysis solved, with the same solver, and never applies
        Σ_x⁻¹, which CG cannot invert on a smooth Σ_x. The state form instead defaults
        to [`LaplaceCovariance`][vardax.LaplaceCovariance] on Λ_x|y.

        Args:
            batch: A ``Batch1D`` holding one window.
            adapter: Optional [`PosteriorAdapter`][vardax.PosteriorAdapter],
                e.g. [`GaussNewtonHessian`][vardax.GaussNewtonHessian].

        Returns:
            The window's posterior: mean and lazy covariance.

        Raises:
            ShapeError: If ``batch`` holds more than one window.
            UnsupportedInputError: For a point batch.
        """
        # lazy: the posterior adapters import the models
        from vardax._src.posterior._precision import model_posterior

        if adapter is None and self.form == "state":
            # lazy: the posterior adapters import the models
            from vardax._src.posterior.laplace import LaplaceCovariance

            # the precision form is the system the analysis just solved
            adapter = LaplaceCovariance(solver=self._default_solver())
        elif adapter is None:
            adapter = _observation_space_posterior
        return model_posterior(self, batch, adapter)

    def _default_solver(self) -> lx.AbstractLinearSolver:
        """``solver``, else CG at ``cg_atol``, ``cg_rtol`` and ``cg_max_steps``."""
        return self.solver or lx.CG(
            atol=self.cg_atol,
            rtol=self.cg_rtol,
            max_steps=self.cg_max_steps,
        )

    def as_analysis_step(self) -> _OIAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        Returns:
            An analysis step that returns the BLUE analysis around a supplied
            forecast, treating ``NaN`` observations as missing.
        """
        return _OIAnalysisStep(self)


def _masked_system(
    prior_mean: Float[Array, "*state"],
    sample_mask: Float[Array, "*obs"],
    obs_op: Any,
) -> tuple[Any, lx.AbstractLinearOperator]:
    """m ⊙ h(·) and its tangent H at the prior mean μ_x.

    The effective observation operator is m ⊙ h(·), so the mask applies both to the
    predicted observations and to the tangent. Without it in the tangent, missing
    entries act as zero-residual observations and bias the analysis wherever Σ_x
    couples observed and missing components.

    Args:
        prior_mean: The linearisation point μ_x.
        sample_mask: The mask m.
        obs_op: The observation operator h.

    Returns:
        The masked observation map and H = ∂(m ⊙ h(x))/∂x at μ_x.
    """

    def masked_obs(state: Float[Array, "*state"]) -> Float[Array, "*obs"]:
        # h(x)
        raw = obs_op(state, mask=sample_mask) if accepts_mask(obs_op) else obs_op(state)
        # m ⊙ h(x)
        return sample_mask * raw

    # H = ∂(m ⊙ h(x))/∂x at μ_x
    obs_jacobian = lx.JacobianLinearOperator(
        lambda state, _args=None: masked_obs(state), prior_mean
    )
    return masked_obs, obs_jacobian


def _observation_space_posterior(
    analysis: Float[Array, "*state"],
    model: OptimalInterpolation,
    batch: Batch1D,
) -> Posterior:
    """The OI posterior Σ_x|y = Σ_x − Σ_x Hᵀ Σ_y⁻¹ H Σ_x, lazily.

    H is the masked tangent at μ_x, the one the analysis used, so for a linear
    ``obs_op`` this is the exact BLUE covariance.

    Args:
        analysis: The analysed state, kept as the posterior mean.
        model: The model.
        batch: The one window.

    Returns:
        The posterior, with gaussx's lazy Joseph-form covariance.
    """
    # the one window's mask
    mask = batch.mask[0]
    _, obs_jacobian = _masked_system(model.prior_mean, mask, model.obs_op)
    # Σ̃_ε = M Σ_ε M + (I − M)
    masked_covariance = gaussx.masked_obs_covariance(model.obs_cov_op, mask)
    # only the covariance is used; y = H μ_x makes the residual zero
    effective_obs = obs_jacobian.mv(model.prior_mean)
    update = gaussx.linear_gaussian_update(
        model.prior_mean,
        model.prior_cov_op,
        obs_jacobian,
        masked_covariance,
        effective_obs,
        solver=model._default_solver(),
    )
    return Posterior(
        mean=analysis,
        cov=update.cov,
        samples=None,
        provenance={
            "adapter": "OptimalInterpolation",
            "structure": "observation_space",
        },
    )


class _OIAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``OptimalInterpolation``.

    Returns the BLUE analysis x⋆ = μ_x + G r with μ_x = ``forecast``, so the model's
    ``prior_mean`` is not used. Σ_x is the model's ``prior_cov_op``; h is ``obs_op`` and
    Σ_ε is ``obs_err_cov`` when given, else the model's. H is the tangent at the
    forecast, so a nonlinear ``obs_op`` gives a single linearised (extended-Kalman)
    update.

    Attributes:
        model: The model to analyse with.
    """

    model: OptimalInterpolation

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        """Analyse around ``forecast``.

        Args:
            forecast: One state shaped like ``prior_mean``, or a batch of them.
            obs: Observations y, batched the same way; ``NaN`` entries are missing.
            obs_op: The cycle's observation operator h, or ``None``.
            obs_err_cov: The cycle's Σ_ε: ``None``, a
                ``lineax.AbstractLinearOperator`` on one observation sample, or a
                scalar or array of error variances (a diagonal Σ_ε).

        Returns:
            The analysis x⋆, with the shape of ``forecast``.
        """
        return analyse_forecast(
            self.model._analyse,
            state_shape=self.model.prior_mean.shape,
            default_obs_op=self.model.obs_op,
            default_obs_cov_op=self.model.obs_cov_op,
            forecast=forecast,
            obs=obs,
            obs_op=obs_op,
            obs_err_cov=obs_err_cov,
        )
