"""Dual (PSAS) 3D-Var: the 3D-Var increment solved in observation space.

The observation-space twin of [`ThreeDVar`][vardax.ThreeDVar]: CG runs on the
M × M system (H Σ_x Hᵀ + Σ̃_ε) λ = d instead of the D_x × D_x primal one, by
``argminax.solve.incremental_dual_solve`` (``argminax.solve.dual_solve`` at each
linearisation), which is the cheaper side when there are fewer observations than
state entries.
"""

from __future__ import annotations

from typing import Any

from argminax.solve import DualSolution, incremental_dual_solve
import equinox as eqx
import jax
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.errors import ConfigurationError, UnsupportedInputError
from vardax._src.models._analysis_step import analyse_forecast, centred_on
from vardax._src.models._observations import fill_gaps
from vardax._src.models._rollout import accepts_mask
from vardax._src.obs_operators.point import require_obs_cov
from vardax._src.utils.covariance import as_covariance, masked_covariance


class DualThreeDVar(eqx.Module):
    """3D variational analysis of a single time, solved in observation space (PSAS).

        (Hₖ Σ_x Hₖᵀ + Σ̃_ε) λₖ = dₖ,   xₖ₊₁ = μ_x + Σ_x Hₖᵀ λₖ,   x₀ = μ_x
        dₖ = m ⊙ (y − h(xₖ)) + Hₖ (xₖ − μ_x),   Hₖ = ∂(m ⊙ h)/∂x at xₖ
        Σ̃_ε = M Σ_ε M + (I − M),   M = diag(m)

    It minimises the cost of [`ThreeDVar`][vardax.ThreeDVar],
    𝒥(x) = ½ ‖x − μ_x‖²_Σ_x⁻¹ + ½ ‖m ⊙ (y − h(x))‖²_Σ̃_ε⁻¹, by Gauss–Newton steps whose
    linear systems are solved over the M observations instead of the state. For a
    linear h one outer iteration is exact, and the analysis is the BLUE of
    [`OptimalInterpolation`][vardax.OptimalInterpolation]; for a nonlinear h each outer
    iteration relinearises at the last analysis (incremental 3D-Var). A masked entry
    has zero innovation and an identity row in Σ̃_ε, so its multiplier is zero.

    Algorithm:
        1. For each sample, under ``jax.vmap``, start from x₀ = μ_x.
        2. Linearise m ⊙ h at xₖ (``jax.linearize``, and ``jax.linear_transpose`` for
           Hₖᵀ), form dₖ and solve (Hₖ Σ_x Hₖᵀ + Σ̃_ε) λₖ = dₖ by ``solver``
           (``argminax.solve.dual_solve``).
        3. Map back, xₖ₊₁ = μ_x + Σ_x Hₖᵀ λₖ, and repeat ``n_outer`` times.

    Attributes:
        obs_op: Observation operator h, linear or nonlinear.
        prior_mean: Prior mean μ_x.
        prior_cov_op: Prior covariance Σ_x; only its matvec is used.
        obs_cov_op: Observation-error covariance Σ_ε.
        solver: ``lineax`` solver of the dual system. Default ``None``: argminax's CG
            with rtol = atol = 1e-10 (about 1e-5 in float32) and 1000 steps,
            preconditioned by Σ̃_ε⁻¹. CG converges in the dual operator's norm, so its
            tolerance does not bound the error of the analysis directly.
        n_outer: Number of outer iterations (linearisations of h), at least 1; one
            is exact for a linear h.

    Raises:
        ConfigurationError: If ``n_outer`` is not an int ≥ 1.

    Examples:
        With Σ_x = Σ_ε = I and h = I the analysis is the BLUE, x⋆ = y / 2:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
        >>> dual = vardax.DualThreeDVar(
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(3),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 3)), mask=jnp.ones((1, 3)))
        >>> xa = dual(batch)
        >>> xa.shape
        (1, 3)
        >>> bool(jnp.allclose(xa, 0.5, atol=1e-4))
        True

        It agrees with the primal ``ThreeDVar`` on a nonlinear h after a few outer
        iterations, and a masked entry stays at μ_x:

        >>> square = vardax.CallableObs(fn=lambda x: x**2)
        >>> half = vardax.Batch1D(
        ...     input=jnp.full((1, 3), 4.0), mask=jnp.array([[1.0, 0.0, 0.0]])
        ... )
        >>> common = dict(
        ...     obs_op=square, prior_mean=jnp.ones(3), prior_cov_op=eye, obs_cov_op=eye
        ... )
        >>> dual_analysis = vardax.DualThreeDVar(**common, n_outer=10)(half)
        >>> primal_analysis = vardax.ThreeDVar(**common)(half)
        >>> bool(jnp.allclose(dual_analysis, primal_analysis, atol=1e-3))
        True
        >>> dual_analysis[0, 1:].tolist()
        [1.0, 1.0]

        ``analyse`` also returns argminax's ``DualSolution``, with the multiplier λ
        and the CG result of every sample:

        >>> xa, solution = dual.analyse(batch)
        >>> solution.multiplier.shape
        (1, 3)
        >>> bool(jnp.all(solution.result == lx.RESULTS.successful))
        True

    References:
        Courtier, P. (1997). Dual formulation of four-dimensional variational
        assimilation. Q. J. R. Meteorol. Soc. 123, 2449.
        [doi:10.1002/qj.49712354414](https://doi.org/10.1002/qj.49712354414).
    """

    obs_op: Any
    prior_mean: Float[Array, "*state"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator | None
    solver: lx.AbstractLinearSolver | None = eqx.field(static=True)
    n_outer: int = eqx.field(static=True)

    def __init__(
        self,
        obs_op: Any,
        prior_mean: Float[Array, "*state"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator | None,
        solver: lx.AbstractLinearSolver | None = None,
        n_outer: int = 1,
    ) -> None:
        if isinstance(n_outer, bool) or not isinstance(n_outer, int) or n_outer < 1:
            raise ConfigurationError(f"n_outer must be an int >= 1; got {n_outer!r}.")
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = as_covariance(prior_cov_op)
        self.obs_cov_op = None if obs_cov_op is None else as_covariance(obs_cov_op)
        self.solver = solver
        self.n_outer = n_outer

    def __call__(self, batch: Batch1D | PointObsBatch) -> Float[Array, " batch *state"]:
        """Analyse each sample of the batch in observation space.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D].

        Returns:
            The analysis of every sample, with a leading batch axis.

        Raises:
            UnsupportedInputError: For a ``PointObsBatch``.
            MissingInputError: If ``obs_cov_op`` is ``None``.
        """
        return self.analyse(batch)[0]

    def analyse(
        self, batch: Batch1D | PointObsBatch
    ) -> tuple[Float[Array, " batch *state"], DualSolution]:
        """The analysis, as ``__call__``, with the dual solve of every sample.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D].

        Returns:
            ``(analysis, solution)``: the analysis of every sample and argminax's
            ``DualSolution`` of the last outer iteration, whose leaves carry a leading
            sample axis: ``value`` (the analysis), ``multiplier`` λ, the CG ``result``
            (check it before trusting the analysis) and ``num_steps``.

        Raises:
            UnsupportedInputError: For a ``PointObsBatch``.
            MissingInputError: If ``obs_cov_op`` is ``None``.
        """
        if isinstance(batch, PointObsBatch):
            raise UnsupportedInputError(
                "DualThreeDVar takes a gridded Batch1D; for a PointObsBatch use "
                "ThreeDVar."
            )
        obs_covariance = require_obs_cov(self.obs_cov_op)

        def analyse_sample(
            sample_obs: Float[Array, "*obs"], sample_mask: Float[Array, "*obs"]
        ) -> tuple[Float[Array, "*state"], DualSolution]:
            solution = self._solve(
                self.prior_mean, sample_obs, sample_mask, self.obs_op, obs_covariance
            )
            return solution.value, solution

        return jax.vmap(analyse_sample)(fill_gaps(batch.input, batch.mask), batch.mask)

    def _analyse(
        self,
        prior_mean: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_covariance: lx.AbstractLinearOperator,
    ) -> Float[Array, "*state"]:
        """The analysis of one sample around the prior mean μ_x."""
        return self._solve(
            prior_mean, sample_obs, sample_mask, obs_op, obs_covariance
        ).value

    def _solve(
        self,
        prior_mean: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_covariance: lx.AbstractLinearOperator,
    ) -> DualSolution:
        """Incremental dual solve of one sample, from x₀ = μ_x.

        Args:
            prior_mean: The prior mean μ_x, also the first linearisation point.
            sample_obs: Observations y, zero where missing.
            sample_mask: The mask m.
            obs_op: The observation operator h.
            obs_covariance: Σ_ε of one sample.

        Returns:
            argminax's ``DualSolution``, with the analysis in ``value``.
        """
        # Σ̃_ε = M Σ_ε M + (I − M), so masked entries drop out of the likelihood
        masked = masked_covariance(obs_covariance, sample_mask)

        def observe(state: Float[Array, "*state"]) -> Float[Array, "*obs"]:
            # h(x)
            predicted_obs = (
                obs_op(state, mask=sample_mask)
                if accepts_mask(obs_op)
                else obs_op(state)
            )
            # m ⊙ h(x), so a masked entry has a zero row in Hₖ
            return sample_mask * predicted_obs

        # xₖ₊₁ = μ_x + Σ_x Hₖᵀ (Hₖ Σ_x Hₖᵀ + Σ̃_ε)⁻¹ dₖ, n_outer times
        return incremental_dual_solve(
            observe,
            prior_mean,
            self.prior_cov_op,
            masked,
            # m ⊙ y
            sample_mask * sample_obs,
            num_outer=self.n_outer,
            solver=self.solver,
        )

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the default adapter
        is [`LaplaceCovariance`][vardax.LaplaceCovariance] with this model's Σ_x and
        Σ_ε, as for [`ThreeDVar`][vardax.ThreeDVar].

        Args:
            batch: A ``Batch1D`` holding one window.
            adapter: Optional [`PosteriorAdapter`][vardax.PosteriorAdapter].

        Returns:
            The window's posterior: mean and lazy covariance.

        Raises:
            ShapeError: If ``batch`` holds more than one window.
            UnsupportedInputError: For a point batch.
        """
        # lazy: the posterior adapters import the models
        from vardax._src.posterior._precision import model_posterior

        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _DualThreeDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

        Returns:
            An analysis step that solves around a supplied forecast, treating ``NaN``
            observations as missing.
        """
        return _DualThreeDVarAnalysisStep(self)


class _DualThreeDVarAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``DualThreeDVar``.

    Solves with μ_x = ``forecast``, which is also the first linearisation point, so the
    model's ``prior_mean`` is not used. Σ_x is the model's ``prior_cov_op``; h is
    ``obs_op`` and Σ_ε is ``obs_err_cov`` when given, else the model's.

    Attributes:
        model: The model to analyse with.
    """

    model: DualThreeDVar

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

        def analyse_one(prior_mean: Array, *args: Any) -> Array:
            return centred_on(self.model, prior_mean)._analyse(prior_mean, *args)

        return analyse_forecast(
            analyse_one,
            state_shape=self.model.prior_mean.shape,
            default_obs_op=self.model.obs_op,
            default_obs_cov_op=self.model.obs_cov_op,
            forecast=forecast,
            obs=obs,
            obs_op=obs_op,
            obs_err_cov=obs_err_cov,
        )
