r"""3D Variational analysis.

Minimises

$$
J(x) = \tfrac{1}{2} \|x - x_b\|^2_{B^{-1}}
     + \tfrac{1}{2} \|y - H(x)\|^2_{R^{-1}}
$$

over a single timestep. When $H$ is linear and $B$, $R$ Gaussian,
``ThreeDVar`` reduces to [`OptimalInterpolation`][vardax.OptimalInterpolation]
— the canonical correctness gate enforced by the linear-Gaussian
agreement test.

The inner minimisation is delegated to an
``optimistix.AbstractMinimiser`` (``GaussNewton``, ``BFGS``,
``NonlinearCG``, ``LevenbergMarquardt``, …). The choice of minimiser
is the user's; vardax provides a sensible default.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx
import optimistix as optx

from vardax._src._types import Batch1D
from vardax._src.models._analysis_step import analyse_forecast


class ThreeDVar(eqx.Module):
    r"""3D variational analysis.

    Attributes:
        obs_op: Observation operator (linear or nonlinear).
        prior_mean: Background $x_b$ of shape ``(T, N)``.
        prior_cov_op: $B$. ``lineax.AbstractLinearOperator``.
        obs_cov_op: $R$. ``lineax.AbstractLinearOperator``.
        minimiser: ``optimistix.AbstractMinimiser`` for the inner
            iteration. Default ``optimistix.BFGS(rtol=1e-6, atol=1e-6)``.
        minimiser_adjoint: ``optimistix.AbstractAdjoint`` for
            differentiating *through* the minimum (used only if the
            ``ThreeDVar`` analysis is itself nested inside a larger
            training loop). Default ``ImplicitAdjoint`` — exact at the
            optimum.
        max_steps: Iteration cap on the inner solver.

    Examples:
        With $B = R = I$, identity $H$ and everything observed, the
        analysis agrees with the closed-form BLUE, $x^* = y / 2$.

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 3), jnp.float32))
        >>> three = vardax.ThreeDVar(
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros((1, 3)),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> batch = vardax.Batch1D(input=jnp.ones((1, 1, 3)), mask=jnp.ones((1, 1, 3)))
        >>> xa = three(batch)
        >>> xa.shape
        (1, 1, 3)
        >>> bool(jnp.allclose(xa, 0.5, atol=1e-3))
        True
    """

    obs_op: Any
    prior_mean: Float[Array, "T N"]
    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    minimiser: optx.AbstractMinimiser = eqx.field(static=True)
    minimiser_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    max_steps: int = eqx.field(static=True, default=200)

    def __init__(
        self,
        obs_op: Any,
        prior_mean: Float[Array, "T N"],
        prior_cov_op: lx.AbstractLinearOperator,
        obs_cov_op: lx.AbstractLinearOperator,
        minimiser: optx.AbstractMinimiser | None = None,
        minimiser_adjoint: optx.AbstractAdjoint | None = None,
        max_steps: int = 200,
    ) -> None:
        self.obs_op = obs_op
        self.prior_mean = prior_mean
        self.prior_cov_op = prior_cov_op
        self.obs_cov_op = obs_cov_op
        self.minimiser = minimiser or optx.BFGS(rtol=1e-6, atol=1e-6)
        self.minimiser_adjoint = minimiser_adjoint or optx.ImplicitAdjoint()
        self.max_steps = max_steps

    def __call__(self, batch: Batch1D) -> Float[Array, "B T N"]:
        """3DVar analysis for each sample in the batch."""

        def _one(
            input_i: Float[Array, "T N"], mask_i: Float[Array, "T N"]
        ) -> Float[Array, "T N"]:
            return self._analyse(
                self.prior_mean, input_i, mask_i, self.obs_op, self.obs_cov_op
            )

        return jax.vmap(_one)(batch.input, batch.mask)

    def _analyse(
        self,
        x_b: Float[Array, "T N"],
        input_i: Float[Array, "T N"],
        mask_i: Float[Array, "T N"],
        obs_op: Any,
        obs_cov_op: lx.AbstractLinearOperator,
    ) -> Float[Array, "T N"]:
        """Minimise the 3D-Var cost of one sample around the background ``x_b``."""

        def cost(x: Float[Array, "T N"], _args: Any) -> Float[Array, ""]:
            # Background term — apply B^{-1} via lineax solve.
            dx = x - x_b
            B_inv_dx = lx.linear_solve(
                self.prior_cov_op,
                dx,
                solver=lx.CG(atol=1e-6, rtol=1e-6),
            ).value
            j_bg = 0.5 * jnp.sum(dx * B_inv_dx)
            # Observation term — apply R^{-1} via lineax solve.
            # Mask predictions AND observations symmetrically: for
            # obs_ops that don't take a `mask` kwarg (e.g.
            # LinearObs), unmasked predictions at missing entries
            # would otherwise penalise the analysis there.
            y_pred = obs_op(x, mask=mask_i) if _accepts_mask(obs_op) else obs_op(x)
            residual = mask_i * (input_i - y_pred)
            R_inv_r = lx.linear_solve(
                obs_cov_op,
                residual,
                solver=lx.CG(atol=1e-6, rtol=1e-6),
            ).value
            j_obs = 0.5 * jnp.sum(residual * R_inv_r)
            return j_bg + j_obs

        result = optx.minimise(
            fn=cost,
            solver=self.minimiser,
            y0=x_b,
            args=None,
            max_steps=self.max_steps,
            adjoint=self.minimiser_adjoint,
            throw=False,
        )
        return result.value

    def as_analysis_step(self) -> _ThreeDVarAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
        return _ThreeDVarAnalysisStep(self)


def _accepts_mask(obs_op: Any) -> bool:
    import inspect

    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


class _ThreeDVarAnalysisStep(eqx.Module):
    r"""``pipekit_cycle.AnalysisStep`` adapter for ``ThreeDVar``.

    Minimises the 3D-Var cost around the forecast,
    $J(x) = \tfrac12 \|x - x_b\|^2_{B^{-1}} + \tfrac12 \|y - H(x)\|^2_{R^{-1}}$
    with $x_b$ = ``forecast`` (also the starting point of the
    minimiser). ``B`` is the model's ``prior_cov_op``; ``H`` is ``obs_op``
    and ``R`` is ``obs_err_cov`` when given, else the model's ``obs_op`` /
    ``obs_cov_op``. The model's ``prior_mean`` is not used.

    ``forecast`` is one state shaped like ``prior_mean`` or a batch of
    them, ``(B, T, N)``, and ``obs`` is batched the same way; ``NaN``
    observations are missing. ``obs_err_cov`` may be ``None``, a
    ``lineax.AbstractLinearOperator`` on one observation sample, or a
    scalar / array of error variances (a diagonal $R$).
    """

    model: ThreeDVar

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
