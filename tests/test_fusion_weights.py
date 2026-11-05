"""``MultiInstrumentFusion.weights`` (#145).

The weights used to be a field nothing read. ``cost`` now sums
αᵢ ½ ‖mᵢ ⊙ (yᵢ − Hᵢ(x))‖²_{Rᵢ⁻¹}, and ``obs_cov_op`` is the flattened
block-diagonal R with blocks Rᵢ / αᵢ, so the flattened adapter and that R
give a model the same weighted likelihood.
"""

from __future__ import annotations

import einx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

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


def _registry_of_three():
    """One 2-D and two 1-D observation shapes, with diagonal and dense R."""
    dtype = jnp.result_type(float)
    identity = vdx.MaskedIdentity()
    dense = lx.MatrixLinearOperator(
        jnp.array([[2.0, 0.5], [0.5, 1.0]], dtype), lx.positive_semidefinite_tag
    )
    entries = {
        "a": vdx.InstrumentSpec(
            identity,
            jnp.ones((2, 2), dtype),
            lx.DiagonalLinearOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype)),
            "a",
        ),
        "b": vdx.InstrumentSpec(identity, jnp.ones(2, dtype), dense, "b"),
        "c": vdx.InstrumentSpec(
            identity,
            jnp.ones(3, dtype),
            lx.DiagonalLinearOperator(jnp.array([1.0, 0.5, 0.25], dtype)),
            "c",
        ),
    }
    return vdx.MultiInstrumentFusion(
        vdx.InstrumentRegistry(entries=entries), weights={"a": 2.0, "b": 0.5, "c": 1.0}
    )


def _dense_fused_r():
    dtype = jnp.result_type(float)
    return jax.scipy.linalg.block_diag(
        jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0], dtype)) / 2.0,
        jnp.array([[2.0, 0.5], [0.5, 1.0]], dtype) / 0.5,
        jnp.diag(jnp.array([1.0, 0.5, 0.25], dtype)),
    )


def test_obs_cov_op_is_block_diag_and_solves_like_the_dense_r():
    """R = blockdiag(Rᵢ / αᵢ), with a flattened 2-D block (#191)."""
    covariance = _registry_of_three().obs_cov_op()
    assert isinstance(covariance, gaussx.BlockDiag)
    assert lx.is_positive_semidefinite(covariance)
    dense = _dense_fused_r()
    vector = jnp.arange(1.0, 10.0, dtype=dense.dtype)
    assert jnp.allclose(covariance.mv(vector), dense @ vector, rtol=1e-5)
    solved = gaussx.solve(covariance, vector)
    assert jnp.allclose(solved, jnp.linalg.solve(dense, vector), rtol=1e-4)
    assert jnp.allclose(
        gaussx.logdet(covariance), jnp.linalg.slogdet(dense)[1], rtol=1e-4
    )


def test_a_shaped_dense_block_is_reshaped():
    """A 2-D block that is not diagonal goes through the lazy flatten."""
    dtype = jnp.result_type(float)
    shaped = lx.FunctionLinearOperator(
        lambda v: 3.0 * v,
        jax.ShapeDtypeStruct((2, 2), dtype),
        lx.positive_semidefinite_tag,
    )
    fusion = vdx.MultiInstrumentFusion(
        vdx.InstrumentRegistry(
            entries={
                "a": vdx.InstrumentSpec(
                    vdx.MaskedIdentity(), jnp.ones((2, 2), dtype), shaped, "a"
                )
            }
        )
    )
    vector = jnp.arange(4.0, dtype=dtype)
    assert jnp.allclose(fusion.obs_cov_op().mv(vector), 3.0 * vector)


def test_an_empty_registry_has_an_empty_covariance():
    fusion = vdx.MultiInstrumentFusion(vdx.InstrumentRegistry(entries={}))
    covariance = fusion.obs_cov_op()
    assert covariance.in_size() == 0
    assert covariance.mv(jnp.zeros(0)).shape == (0,)
    assert fusion.cost(jnp.zeros(2), {}) == 0.0


@pytest.mark.parametrize(
    "mask_values",
    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]],
    ids=["full", "gap"],
)
def test_default_solver_handles_a_non_diagonal_covariance_on_a_2d_observation(
    x64, mask_values
):
    mask = jnp.array(mask_values)
    covariance_matrix = jnp.eye(4) + 0.2

    def covariance_mv(vector):
        flat = einx.id("a b -> (a b)", vector)
        return einx.id("(a b) -> a b", covariance_matrix @ flat, a=2, b=2)

    covariance = lx.FunctionLinearOperator(
        covariance_mv,
        jax.eval_shape(lambda: jnp.zeros((2, 2))),
        lx.positive_semidefinite_tag,
    )
    spec = vdx.InstrumentSpec(vdx.MaskedIdentity(), mask, covariance, "a")
    fusion = vdx.MultiInstrumentFusion(vdx.InstrumentRegistry(entries={"a": spec}))

    cost = fusion.cost(jnp.zeros((2, 2)), {"a": jnp.ones((2, 2))})

    # ½ rᵀ Σ̃_ε⁻¹ r, with Σ̃_ε = M Σ_ε M + (I − M)
    flat_mask = einx.id("a b -> (a b)", mask)
    masked_matrix = jnp.diag(flat_mask) @ covariance_matrix @ jnp.diag(
        flat_mask
    ) + jnp.diag(1.0 - flat_mask)
    expected = 0.5 * flat_mask @ jnp.linalg.solve(masked_matrix, flat_mask)
    assert jnp.allclose(cost, expected)
