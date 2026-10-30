"""gaussx structured operators as ``prior_cov_op`` (#141).

Before gaussx registered ``lx.linearise``, ``lx.materialise`` and
``lx.diagonal`` for its operators, a ``BlockDiag`` or ``Kronecker`` B made
every model raise ``NotImplementedError`` on its first call. Each case
checks the OI analysis against the dense closed form.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from vardax import Batch1D, MaskedIdentity, OptimalInterpolation

_FACTOR = jnp.array([[2.0, 0.5], [0.5, 1.0]])


def _psd(matrix):
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


@pytest.mark.parametrize(
    "prior_cov",
    [
        gx.BlockDiag(_psd(_FACTOR), _psd(2.0 * _FACTOR)),
        gx.Kronecker(_psd(_FACTOR), _psd(_FACTOR)),
    ],
    ids=["block_diag", "kronecker"],
)
def test_structured_prior_cov_matches_dense(prior_cov):
    obs = jnp.array([[1.0, -0.5, 0.25, 2.0]])
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), obs.dtype))
    model = OptimalInterpolation(
        obs_op=MaskedIdentity(),
        prior_mean=jnp.zeros(4),
        prior_cov_op=prior_cov,
        obs_cov_op=eye,
    )
    analysis = model(Batch1D(input=obs, mask=jnp.ones_like(obs)))
    B = prior_cov.as_matrix()
    # x* = B (B + I)⁻¹ y, with x_b = 0 and H = R = I
    expected = B @ jnp.linalg.solve(B + jnp.eye(4), obs[0])
    assert jnp.allclose(analysis[0], expected, atol=1e-5)
