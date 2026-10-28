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
- The class refuses non-linear observation operators on construction.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D
from vardax._src.models._analysis_step import analyse_forecast


class OptimalInterpolation(eqx.Module):
    r"""BLUE / OI — closed-form linear-Gaussian analysis.

    Attributes:
        obs_op: Observation operator. Must be linear (its
            ``linearize(x)`` must return an operator independent of
            ``x``). Use [`ThreeDVar`][vardax.ThreeDVar] for non-linear
            ``H``.
        prior_mean: Background $x_b$ of shape ``(T, N)``.
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
    """

    obs_op: Any  # ObservationOperator (linear)
    prior_mean: Float[Array, "*state"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    cg_atol: float = eqx.field(static=True, default=1e-6)
    cg_rtol: float = eqx.field(static=True, default=1e-6)
    cg_max_steps: int = eqx.field(static=True, default=200)
    solver: lx.AbstractLinearSolver | None = eqx.field(static=True, default=None)

    def __call__(self, batch: Batch1D) -> Float[Array, " batch *state"]:
        """BLUE analysis for each sample in the batch."""

        def _one(
            input_i: Float[Array, "*obs"], mask_i: Float[Array, "*obs"]
        ) -> Float[Array, "*state"]:
            return self._analyse(
                self.prior_mean, input_i, mask_i, self.obs_op, self.obs_cov_op
            )

        return jax.vmap(_one)(batch.input, batch.mask)

    def _analyse(
        self,
        x_b: Float[Array, "*state"],
        input_i: Float[Array, "*obs"],
        mask_i: Float[Array, "*obs"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
    ) -> Float[Array, "*state"]:
        """BLUE analysis of one sample around the background ``x_b``."""

        # The effective observation operator is mask ⊙ H(·) — apply the
        # mask both to the predicted obs (innovation) AND to the
        # tangent-linear used to assemble (H B H^T + R). Without the
        # mask in `linearize`, missing entries are treated as
        # zero-innovation observations and bias the analysis whenever
        # B couples observed and missing components.
        def masked_obs(x: Float[Array, "*state"]) -> Float[Array, "*obs"]:
            raw = obs_op(x, mask=mask_i) if _accepts_mask(obs_op) else obs_op(x)
            return mask_i * raw

        y_pred = masked_obs(x_b)
        innovation = (input_i * mask_i) - y_pred

        # Build (H B H^T + R) as a composed lineax operator. We
        # tag it positive-semidefinite so lineax.CG accepts it
        # (the composition itself doesn't carry the tag).
        H = lx.JacobianLinearOperator(lambda x, _args=None: masked_obs(x), x_b)
        inner_raw = H @ self.prior_cov_op @ H.transpose() + obs_cov_op
        inner_op = lx.TaggedLinearOperator(inner_raw, lx.positive_semidefinite_tag)

        solver = self.solver or lx.CG(
            atol=self.cg_atol,
            rtol=self.cg_rtol,
            max_steps=self.cg_max_steps,
        )
        v = lx.linear_solve(inner_op, innovation, solver=solver).value

        # x_star = x_b + B H^T v
        return x_b + self.prior_cov_op.mv(H.transpose().mv(v))

    def posterior(self, batch: Batch1D, adapter: Any = None) -> Any:
        """Analyse one window and return its [`Posterior`][vardax.Posterior].

        Runs the analysis, then ``adapter(analysis, self, batch)``; the
        default adapter is [`LaplaceCovariance`][vardax.LaplaceCovariance]
        with this model's $B$ and $R$, and with this model's ``solver``
        when one is set.

        Args:
            batch: A ``Batch1D`` holding one window.
            adapter: Optional [`PosteriorAdapter`][vardax.PosteriorAdapter],
                e.g. [`GaussNewtonHessian`][vardax.GaussNewtonHessian].

        Returns:
            The window's posterior: mean and lazy covariance.

        Raises:
            ValueError: If ``batch`` holds more than one window.
        """
        from vardax._src.posterior._precision import model_posterior

        if adapter is None and self.solver is not None:
            from vardax._src.posterior.laplace import LaplaceCovariance

            adapter = LaplaceCovariance(solver=self.solver)
        return model_posterior(self, batch, adapter)

    def as_analysis_step(self) -> _OIAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
        return _OIAnalysisStep(self)


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
