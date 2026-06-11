r"""Gauss-Newton Hessian posterior (Decision D10).

For ``IncrementalFourDVar`` the Gauss-Newton Hessian is assembled
during the last outer iteration as part of the algorithm. Reuse it
for posterior covariance:

$$
P^* = \big(B^{-1} + \sum_t (H'_t M'_t)^\top R_t^{-1} (H'_t M'_t)\big)^{-1}.
$$

For methods that don't assemble the GN Hessian as a side effect
(``StrongFourDVar``, ``WeakFourDVar``), this adapter computes it
on demand via Krylov / Lanczos over the same composed operator.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
from jaxtyping import Array, Float
import lineax as lx

from .container import Posterior
from .laplace import _CG_SOLVER, _inverse_op


class GaussNewtonHessian(eqx.Module):
    r"""Gauss-Newton Hessian inversion at MAP.

    Functionally similar to
    [`LaplaceCovariance`][vardax.LaplaceCovariance] — both compute
    $(B^{-1} + H^\top R^{-1} H)^{-1}$ (or its 4D extension)
    — but the GN-Hessian adapter is the recommended path for
    ``IncrementalFourDVar`` where the Hessian is already
    materialised by the inner CG solver.

    Attributes:
        prior_cov_op: $B$.
        obs_cov_op: $R$.
        n_krylov: Maximum Krylov iterations for mat-vec evaluations.
    """

    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    n_krylov: int = eqx.field(static=True, default=50)

    def __call__(
        self,
        analysis: Float[Array, ...],
        model: Any,
        batch: Any,
    ) -> Posterior:
        """Build the GN-Hessian posterior at ``analysis``."""
        underlying = getattr(model, "model", model)
        obs_op = getattr(underlying, "obs_op", None)
        if obs_op is None:
            raise AttributeError(
                "GaussNewtonHessian requires the model to expose `obs_op`."
            )

        H = obs_op.linearize(analysis)
        precision = H.transpose() @ _inverse_op(self.obs_cov_op) @ H + _inverse_op(
            self.prior_cov_op
        )
        precision_tagged = lx.TaggedLinearOperator(
            precision, lx.positive_semidefinite_tag
        )

        return Posterior(
            mean=analysis,
            cov=gaussx.inv(precision_tagged, solver=_CG_SOLVER),
            samples=None,
            provenance={
                "adapter": "GaussNewtonHessian",
                "n_krylov": self.n_krylov,
            },
        )
