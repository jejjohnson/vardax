r"""Optimal Interpolation / BLUE (Decision D16).

Closed-form linear-Gaussian analysis: the Best Linear Unbiased
Estimator. When $H$ is linear and $B$, $R$ are Gaussian, the
posterior is exact in one matrix expression — no iteration, no
convergence criterion.

$$
x^* = x_b + K (y - H x_b), \quad K = B H^\top (H B H^\top + R)^{-1}.
$$

Implementation notes:

- By default the inversion is solved by ``lineax.CG`` against an
  ``AbstractLinearOperator``, so $HBH^\top + R$ is never materialised.
  Pass ``solver=`` (e.g. ``lineax.Cholesky()``) for an exact direct
  factorisation when that matrix fits in memory: CG at a fixed tolerance
  is inexact and can stall on an ill-conditioned innovation system.
- ``B_op`` and ``R_op`` are supplied as ``lineax.AbstractLinearOperator``
  (typically ``gaussx`` structured operators for Matérn priors and
  diagonal $R$).
- ``obs_op`` is not checked for linearity. A nonlinear $H$ is linearised
  once at $x_b$ and the class returns that single update (one outer loop
  of incremental 3D-Var), not the MAP; use ``ThreeDVar`` for the MAP.
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
from vardax._src.posterior.container import Posterior
from vardax._src.utils.covariance import as_covariance


class OptimalInterpolation(eqx.Module):
    r"""BLUE / OI — closed-form linear-Gaussian analysis.

    $$x^* = x_b + B H^\top (H B H^\top + R)^{-1} (y - H x_b)$$

    Algorithm:
        1. For each sample (``jax.vmap``), form the masked operator
           $m \odot H(\cdot)$ and the innovation
           $d = m \odot y - m \odot H(x_b)$.
        2. Linearise the masked operator at $x_b$ to get $H$ and build the
           innovation operator $H B H^\top + R$ without materialising it.
        3. Solve $(H B H^\top + R) v = d$ with CG, or with ``solver`` when
           given.
        4. Return $x_b + B H^\top v$.

    Attributes:
        obs_op: Observation operator, expected to be linear. It is not
            checked: a nonlinear ``H`` is linearised once at $x_b$ and
            the result is that single linearised update, not the MAP.
            Use [`ThreeDVar`][vardax.ThreeDVar] for non-linear ``H``.
        prior_mean: Background $x_b$.
        prior_cov_op: Background-error covariance $B$ as a
            ``lineax.AbstractLinearOperator``.
        obs_cov_op: Observation-error covariance $R$ as a
            ``lineax.AbstractLinearOperator``.
        cg_atol: CG absolute tolerance for the inner solve.
        cg_rtol: CG relative tolerance for the inner solve.
        cg_max_steps: CG iteration cap.
        solver: Optional linear solver for the innovation system, used
            instead of CG (the ``cg_*`` fields are then ignored), e.g.
            ``lineax.Cholesky()`` to factorise $HBH^\top + R$ exactly. It
            is also the default solver of the ``posterior`` adapter.
        form: ``"observation"`` (default) solves the $D_y \times D_y$
            innovation system $(H B H^\top + R) v = d$. ``"state"`` solves
            the $D_x \times D_x$ information system
            $(B^{-1} + H^\top R^{-1} H) \delta x = H^\top R^{-1} d$, the
            one to use when $D_y \gg D_x$; $B^{-1}$ and $R^{-1}$ are closed
            form for a diagonal operator and solved with ``solver``
            otherwise, and the default posterior is then
            $(B^{-1} + H^\top R^{-1} H)^{-1}$.

    Examples:
        With $B = R = I$, identity $H$ and everything observed, the
        analysis splits the innovation in half: $x^* = y / 2$.

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float32))
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

        Unobserved entries (zero mask) keep the background, and an exact
        Cholesky solver replaces CG:

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

        Composed with the Laplace adapter, the posterior covariance of this
        problem is $(I + I)^{-1} = I / 2$:

        >>> post = oi.posterior(batch)
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones((1, 3))), 0.5, atol=1e-2))
        True
    """

    obs_op: Any  # ObservationOperator (linear)
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
        """BLUE analysis for each sample in the batch.

        Args:
            batch: A gridded [`Batch1D`][vardax.Batch1D].

        Returns:
            The analysis of every sample, with a leading batch axis.
        """

        def _one(
            sample_obs: Float[Array, "*obs"], sample_mask: Float[Array, "*obs"]
        ) -> Float[Array, "*state"]:
            return self._analyse(
                self.prior_mean, sample_obs, sample_mask, self.obs_op, self.obs_cov_op
            )

        return jax.vmap(_one)(fill_gaps(batch.input, batch.mask), batch.mask)

    def _analyse(
        self,
        background: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
    ) -> Float[Array, "*state"]:
        """BLUE analysis of one sample around the ``background``."""
        if self.form == "state":
            return self._analyse_state_space(
                background, sample_obs, sample_mask, obs_op, obs_cov_op
            )
        masked_obs, obs_jacobian, inner_op, solver = self._innovation_system(
            background, sample_mask, obs_op, obs_cov_op
        )
        # d = m ⊙ y − m ⊙ H(x_b)
        innovation = (sample_obs * sample_mask) - masked_obs(background)
        # v = (H B Hᵀ + R)⁻¹ d
        dual_weights = lx.linear_solve(inner_op, innovation, solver=solver).value

        # xₐ = x_b + B Hᵀ v
        return background + self.prior_cov_op.mv(
            obs_jacobian.transpose().mv(dual_weights)
        )

    def _analyse_state_space(
        self,
        background: Float[Array, "*state"],
        sample_obs: Float[Array, "*obs"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
    ) -> Float[Array, "*state"]:
        """BLUE analysis in the state-space (information) form."""
        masked_obs, obs_jacobian, precision, solver = self._information_system(
            background, sample_mask, obs_op, obs_cov_op
        )
        # d = m ⊙ y − m ⊙ H(x_b)
        innovation = (sample_obs * sample_mask) - masked_obs(background)
        # Hᵀ R⁻¹ d
        right_hand_side = obs_jacobian.transpose().mv(
            _inverse(obs_cov_op, solver).mv(innovation)
        )
        # δx = (B⁻¹ + Hᵀ R⁻¹ H)⁻¹ Hᵀ R⁻¹ d
        increment = lx.linear_solve(precision, right_hand_side, solver=solver).value
        # xₐ = x_b + δx
        return background + increment

    def _information_system(
        self,
        background: Float[Array, "*state"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
    ) -> tuple[Any, lx.AbstractLinearOperator, lx.AbstractLinearOperator, Any]:
        """m ⊙ H(·), its tangent at ``background``, B⁻¹ + Hᵀ R⁻¹ H and its solver."""
        masked_obs, obs_jacobian, _, solver = self._innovation_system(
            background, sample_mask, obs_op, obs_cov_op
        )
        # B⁻¹ + Hᵀ R⁻¹ H; masked rows of H are zero, so gaps drop out
        precision_raw = obs_jacobian.transpose() @ _inverse(
            obs_cov_op, solver
        ) @ obs_jacobian + _inverse(self.prior_cov_op, solver)
        precision = lx.TaggedLinearOperator(precision_raw, lx.positive_semidefinite_tag)
        return masked_obs, obs_jacobian, precision, solver

    def _innovation_system(
        self,
        background: Float[Array, "*state"],
        sample_mask: Float[Array, "*obs"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
    ) -> tuple[Any, lx.AbstractLinearOperator, lx.AbstractLinearOperator, Any]:
        """m ⊙ H(·), its tangent at ``background``, H B Hᵀ + R and its solver."""

        # The effective observation operator is mask ⊙ H(·) — apply the
        # mask both to the predicted obs (innovation) AND to the
        # tangent-linear used to assemble (H B H^T + R). Without the
        # mask in `linearize`, missing entries are treated as
        # zero-innovation observations and bias the analysis whenever
        # B couples observed and missing components.
        def masked_obs(state: Float[Array, "*state"]) -> Float[Array, "*obs"]:
            raw = (
                obs_op(state, mask=sample_mask)
                if _accepts_mask(obs_op)
                else obs_op(state)
            )
            return sample_mask * raw

        # Build (H B H^T + R) as a composed lineax operator. We
        # tag it positive-semidefinite so lineax.CG accepts it
        # (the composition itself doesn't carry the tag).
        # H = ∂(m ⊙ H(x))/∂x at x_b
        obs_jacobian = lx.JacobianLinearOperator(
            lambda state, _args=None: masked_obs(state), background
        )
        # H B Hᵀ + R
        inner_raw = (
            obs_jacobian @ self.prior_cov_op @ obs_jacobian.transpose() + obs_cov_op
        )
        inner_op = lx.TaggedLinearOperator(inner_raw, lx.positive_semidefinite_tag)

        return masked_obs, obs_jacobian, inner_op, self._default_solver()

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        r"""Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``. Without
        an adapter the covariance is the observation-space form
        $P^* = B - B H^\top (H B H^\top + R)^{-1} H B$: it solves the
        innovation system the analysis solved, with the same solver, and
        never applies $B^{-1}$, which CG cannot invert on a smooth $B$.

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
        from vardax._src.posterior._precision import model_posterior

        if adapter is None and self.form == "state":
            from vardax._src.posterior.laplace import LaplaceCovariance

            # the precision form is the system the analysis just solved
            adapter = LaplaceCovariance(solver=self._default_solver())
        elif adapter is None:
            adapter = _ObservationSpaceCovariance()
        return model_posterior(self, batch, adapter)

    def _default_solver(self) -> lx.AbstractLinearSolver:
        """``solver``, else CG at ``cg_atol``, ``cg_rtol`` and ``cg_max_steps``."""
        return self.solver or lx.CG(
            atol=self.cg_atol,
            rtol=self.cg_rtol,
            max_steps=self.cg_max_steps,
        )

    def as_analysis_step(self) -> _OIAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8).

        Returns:
            An analysis step that returns the BLUE analysis around a
            supplied forecast, treating ``NaN`` observations as missing.
        """
        return _OIAnalysisStep(self)


def _inverse(
    operator: lx.AbstractLinearOperator, solver: lx.AbstractLinearSolver
) -> lx.AbstractLinearOperator:
    """A⁻¹ of a PSD operator, lazily: closed form when A is diagonal."""
    structure = operator.in_structure()
    if lx.is_diagonal(operator):
        diagonal = lx.diagonal(operator)

        def divide(vector: Array) -> Array:
            # A⁻¹ v = v / diag(A)
            return vector / diagonal.reshape(vector.shape)

        return lx.FunctionLinearOperator(
            divide, structure, lx.positive_semidefinite_tag
        )
    # wrapped so lineax can linearise it inside the composed precision
    inverse = gaussx.inv(operator, solver=solver)
    return lx.FunctionLinearOperator(
        inverse.mv, structure, lx.positive_semidefinite_tag
    )


def _accepts_mask(obs_op: Any) -> bool:
    """Whether ``obs_op.__call__`` takes a ``mask`` kwarg.

    ``MaskedIdentity`` does; ``LinearObs`` doesn't. The dispatch keeps
    the same OI implementation usable across operator types.
    """
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _ObservationSpaceCovariance(eqx.Module):
    r"""OI posterior $P^* = B - B H^\top (H B H^\top + R)^{-1} H B$, lazily.

    $H$ is the masked tangent at the background, the one the analysis
    used, so for a linear ``obs_op`` this is the exact BLUE covariance.
    """

    def __call__(
        self,
        analysis: Float[Array, "*state"],
        model: OptimalInterpolation,
        batch: Batch1D,
    ) -> Posterior:
        # the one window's mask, (T, N) or (N,)
        mask = batch.mask[0]
        _, obs_jacobian, inner_op, solver = model._innovation_system(
            model.prior_mean, mask, model.obs_op, model.obs_cov_op
        )
        prior_cov = model.prior_cov_op
        obs_adjoint = obs_jacobian.transpose()

        def covariance_mv(direction: Float[Array, "*state"]) -> Float[Array, "*state"]:
            # B v
            prior_direction = prior_cov.mv(direction)
            # (H B Hᵀ + R)⁻¹ H B v
            weights = lx.linear_solve(
                inner_op, obs_jacobian.mv(prior_direction), solver=solver
            ).value
            # B v − B Hᵀ (H B Hᵀ + R)⁻¹ H B v
            return prior_direction - prior_cov.mv(obs_adjoint.mv(weights))

        covariance = lx.FunctionLinearOperator(
            covariance_mv, prior_cov.in_structure(), lx.positive_semidefinite_tag
        )
        return Posterior(
            mean=analysis,
            cov=covariance,
            samples=None,
            provenance={
                "adapter": "OptimalInterpolation",
                "structure": "observation_space",
            },
        )


class _OIAnalysisStep(eqx.Module):
    r"""``pipekit_cycle.AnalysisStep`` adapter for ``OptimalInterpolation``.

    Returns the BLUE analysis around the forecast,
    $x_a = x_b + B H^\top (H B H^\top + R)^{-1} (y - H x_b)$ with
    $x_b$ = ``forecast``. ``B`` is the model's ``prior_cov_op``; ``H`` is
    ``obs_op`` and ``R`` is ``obs_err_cov`` when given, else the model's
    ``obs_op`` / ``obs_cov_op``. The model's ``prior_mean`` is not used.

    ``forecast`` is one state shaped like ``prior_mean`` or a batch of
    them, ``(B, T, N)``, and ``obs`` is batched the same way; ``NaN``
    observations are missing. ``obs_err_cov`` may be ``None``, a
    ``lineax.AbstractLinearOperator`` on one observation sample, or a
    scalar / array of error variances (a diagonal $R$). ``H`` is
    linearised at the forecast, so a nonlinear ``obs_op`` gives a single
    linearised (extended-Kalman) update.
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
