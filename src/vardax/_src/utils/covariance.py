"""Covariance operators as the models store them."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx


def as_covariance(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Tag a covariance operator positive semidefinite.

    A covariance is PSD by definition, but lineax drops the tag when an
    operator is scaled (``σ² I``) and never sets it on a
    ``DiagonalLinearOperator``, and ``lx.CG`` refuses an untagged operator.

    Args:
        operator: A covariance operator.

    Returns:
        ``operator`` unchanged if it is already PSD, else ``operator``
        wrapped in a PSD ``TaggedLinearOperator``.
    """
    if lx.is_positive_semidefinite(operator):
        return operator
    return lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)


def masked_obs_cov(
    obs_cov_op: lx.AbstractLinearOperator, mask: Float[Array, "*obs"]
) -> lx.AbstractLinearOperator:
    r"""The observation covariance with masked entries marginalised out.

    $\tilde R = M R M + (I - M)$ with $M = \mathrm{diag}(m)$ for a 0/1
    mask $m$. It holds $R_{oo}$ on the observed block, the identity on
    the unobserved block and no cross terms, so for a masked residual
    $r = m \odot (y - H(x))$ the cost $\tfrac12 r^\top \tilde R^{-1} r$
    is $\tfrac12 r_o^\top R_{oo}^{-1} r_o$, the likelihood of the
    observed entries alone. Solving the full $R$ against $r$ would
    instead weight them by $(R^{-1})_{oo}$, which treats each masked
    entry as a perfect observation when $R$ is correlated.

    Args:
        obs_cov_op: $R$, a PSD operator over the observation structure.
        mask: $m$, 1 where observed and 0 where not; its size matches
            ``obs_cov_op``'s structure.

    Returns:
        $\tilde R$, PSD-tagged: a ``DiagonalLinearOperator`` when $R$ is
        diagonal, so diagonal solves stay closed form, else a lazy
        ``FunctionLinearOperator``.
    """
    structure = obs_cov_op.in_structure()
    mask = jnp.reshape(mask, structure.shape).astype(structure.dtype)
    if lx.is_diagonal(obs_cov_op):
        diagonal = jnp.reshape(lx.diagonal(obs_cov_op), structure.shape)
        # diag(R̃) = m ⊙ diag(R) + (1 − m)
        return as_covariance(lx.DiagonalLinearOperator(mask * diagonal + (1.0 - mask)))

    def masked_mv(vector: Array) -> Array:
        # R̃ v = m ⊙ R (m ⊙ v) + (1 − m) ⊙ v
        return mask * obs_cov_op.mv(mask * vector) + (1.0 - mask) * vector

    return lx.FunctionLinearOperator(masked_mv, structure, lx.positive_semidefinite_tag)
