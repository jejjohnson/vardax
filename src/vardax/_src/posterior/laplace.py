r"""Laplace approximation at MAP.

For Gaussian-likelihood / Gaussian-prior problems at the MAP, the
posterior covariance is the inverse Gauss–Newton Hessian of the cost the
analysis minimised. For a single analysis time (``OptimalInterpolation``,
``ThreeDVar``):

$$
P^* \approx \big(B^{-1} + H'^\top R^{-1} H'\big)^{-1}.
$$

For a 4DVar analysis the window's observations depend on $x_0$ through
the model, so the tangent-linear model $M'_{t \leftarrow 0}$ enters:

$$
P^* \approx \Big(B^{-1} + \sum_t (H'_t M'_{t\leftarrow 0})^\top R_t^{-1}
              (H'_t M'_{t\leftarrow 0})\Big)^{-1},
$$

applied matrix-free through the model's own rollout (``jax.linearize`` /
``jax.vjp``). For ``WeakFourDVar`` the control is $(x_0, \eta)$ and the
returned covariance is the $x_0$ block of the inverse augmented
precision, with $Q^{-1}$ on each $\eta_t$.

The result is returned as an ``AbstractLinearOperator`` so mat-vecs
(samples, marginals) compose lazily via ``lineax.CG`` without
materialising $P^*$.

Use when:

- Gaussian likelihood + Gaussian prior + single posterior mode
  (verified by SBC).
- Default for ``ThreeDVar``, ``StrongFourDVar``, ``WeakFourDVar``,
  ``IncrementalFourDVar``.

For multimodal posteriors / non-Gaussian likelihoods, use
``EnsembleCovariance``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx

from ._precision import gauss_newton_covariance
from .container import Posterior

#: Default solver for the lazy inverses (max_steps bounds the Krylov space).
DEFAULT_SOLVER = lx.CG(atol=1e-6, rtol=1e-6, max_steps=200)


class LaplaceCovariance(eqx.Module):
    r"""Laplace approximation at MAP.

    Attributes:
        prior_cov_op: Background-error covariance $B$.
        obs_cov_op: Observation-error covariance $R$ at one time.
        solver: Linear solver for the lazy inverses (default CG,
            ``atol = rtol = 1e-6``, 200 steps).

    Both required so the adapter can build $P^*$ lazily. They
    should match the operators used by the analysis method that
    produced ``analysis``.

    Examples:
        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> from types import SimpleNamespace
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float32))
        >>> laplace = vdx.LaplaceCovariance(prior_cov_op=eye, obs_cov_op=eye)
        >>> model = SimpleNamespace(obs_op=vdx.LinearObs(H_mat=eye))
        >>> post = laplace(jnp.zeros(4), model, batch=None)
        >>> post.mean.shape
        (4,)
        >>> # B = R = H = I, so P* = (I + I)^{-1} = I / 2
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(4)), 0.5, atol=1e-2))
        True
    """

    prior_cov_op: lx.AbstractLinearOperator
    obs_cov_op: lx.AbstractLinearOperator
    solver: lx.AbstractLinearSolver = eqx.field(static=True, default=DEFAULT_SOLVER)

    def __call__(
        self,
        analysis: Float[Array, ...],
        model: Any,  # AnalysisStep-compliant
        batch: Any,
    ) -> Posterior:
        r"""Build the Laplace posterior at ``analysis``.

        Args:
            analysis: MAP of one window: $x_0^*$, or $(x_0^*, \eta^*)$ for
                ``WeakFourDVar``.
            model: The analysis model or its ``.as_analysis_step()``
                wrapper. Its ``obs_op`` (and, for 4DVar, its rollout) build
                the precision.
            batch: The single window the analysis was computed from; 4DVar
                models read its length and mask. A batch with a leading
                sample axis must hold one sample.

        Returns:
            [`Posterior`][vardax.Posterior] with ``mean`` $= x_0^*$ and
            ``cov`` a lazy ``AbstractLinearOperator`` representing $P^*$.

        Raises:
            AttributeError: If the model exposes no ``obs_op``.
            ValueError: For a point batch, a multi-sample batch, or a
                ``basis=`` / ``parameterization=`` control.
        """
        mean, covariance, structure = gauss_newton_covariance(
            analysis,
            model,
            batch,
            self.prior_cov_op,
            self.obs_cov_op,
            self.solver,
        )
        return Posterior(
            mean=mean,
            cov=covariance,
            samples=None,
            provenance={"adapter": "LaplaceCovariance", "structure": structure},
        )
