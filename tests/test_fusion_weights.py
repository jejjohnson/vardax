"""``MultiInstrumentFusion.weights`` (#145).

The weights used to be a field nothing read. ``cost`` now sums
αᵢ ½ ‖mᵢ ⊙ (yᵢ − Hᵢ(x))‖²_{Rᵢ⁻¹}, and ``obs_cov_op`` is the flattened
block-diagonal R with blocks Rᵢ / αᵢ, so the flattened adapter and that R
give a model the same weighted likelihood.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

import vardax as vdx


def _fusion(weights=None):
    # a: 2 cells of the state, R_a = diag(0.5, 2); b: their sum, R_b = 0.25
    cov_a = lx.DiagonalLinearOperator(jnp.array([0.5, 2.0]))
    cov_b = lx.DiagonalLinearOperator(jnp.array([0.25]))
    summed = lx.MatrixLinearOperator(jnp.array([[1.0, 1.0]]))
    first = vdx.InstrumentSpec(vdx.MaskedIdentity(), jnp.array([1.0, 0.0]), cov_a, "a")
    second = vdx.InstrumentSpec(vdx.LinearObs(operator=summed), jnp.ones(1), cov_b, "b")
    registry = vdx.InstrumentRegistry(entries={"b": second, "a": first})
    return vdx.MultiInstrumentFusion(registry, weights=weights)


STATE = jnp.array([0.3, -0.7])
OBS = {"a": jnp.array([1.0, jnp.nan]), "b": jnp.array([2.0])}


def test_cost_matches_the_weighted_sum():
    fusion = _fusion(weights={"a": 3.0})
    # a: m = (1, 0), so only cell 0 counts: ½ (1 − 0.3)² / 0.5, weighted by 3
    term_a = 3.0 * 0.5 * (1.0 - 0.3) ** 2 / 0.5
    # b: ½ (2 − (0.3 − 0.7))² / 0.25, weight 1 by default
    term_b = 0.5 * (2.0 + 0.4) ** 2 / 0.25
    np.testing.assert_allclose(fusion.cost(STATE, OBS), term_a + term_b, rtol=1e-6)


def test_weights_change_the_cost():
    plain = _fusion().cost(STATE, OBS)
    weighted = _fusion(weights={"b": 10.0}).cost(STATE, OBS)
    assert not np.isclose(plain, weighted)


def test_flattened_likelihood_matches_the_cost():
    fusion = _fusion(weights={"a": 3.0, "b": 0.5})
    flat_op = fusion.to_observation_operator()
    masks = jnp.concatenate([jnp.array([1.0, 0.0]), jnp.ones(1)])
    observed = jnp.where(masks != 0, jnp.concatenate([OBS["a"], OBS["b"]]), 0.0)
    residual = masks * (observed - flat_op(STATE))
    flat_cov = fusion.obs_cov_op().as_matrix()
    # ½ rᵀ R⁻¹ r with blocks Rᵢ / αᵢ
    flat_cost = 0.5 * residual @ jnp.linalg.solve(flat_cov, residual)
    np.testing.assert_allclose(flat_cost, fusion.cost(STATE, OBS), rtol=1e-6)


def test_obs_cov_op_blocks_follow_sorted_ids():
    fusion = _fusion(weights={"b": 0.5})
    diagonal = jax.vmap(fusion.obs_cov_op().mv)(jnp.eye(3)).diagonal()
    # a first (R_a), then b (R_b / 0.5)
    np.testing.assert_allclose(diagonal, [0.5, 2.0, 0.5], rtol=1e-6)
