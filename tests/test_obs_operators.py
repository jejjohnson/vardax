"""Tests for the Layer 1 observation operator family."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from vardax import (
    AveragingKernel,
    InstrumentRegistry,
    InstrumentSpec,
    LinearObs,
    MaskedIdentity,
    MultiInstrumentFusion,
    ObservationOperator,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestObservationOperatorProtocol:
    def test_masked_identity(self):
        assert isinstance(MaskedIdentity(), ObservationOperator)

    def test_linear_obs(self, rng):
        H = lx.MatrixLinearOperator(jax.random.normal(rng, (3, 4)))
        assert isinstance(LinearObs(H_mat=H), ObservationOperator)

    def test_averaging_kernel(self, rng):
        n = 4
        A = lx.MatrixLinearOperator(jax.random.normal(rng, (n, n)))
        ak = AveragingKernel(A=A, x_a=jnp.zeros(n), h=jnp.ones(n))
        assert isinstance(ak, ObservationOperator)

    def test_multi_instrument_fusion_via_flatten_adapter(self, rng):
        n = 4
        A = lx.MatrixLinearOperator(jnp.eye(n))
        spec = InstrumentSpec(
            obs_op=AveragingKernel(A=A, x_a=jnp.zeros(n), h=jnp.ones(n)),
            mask=jnp.ones(n),
            R_op=lx.DiagonalLinearOperator(jnp.ones(n)),
            instrument_id="tropomi",
        )
        fusion = MultiInstrumentFusion(
            registry=InstrumentRegistry(entries={"tropomi": spec}),
        )
        # The fusion itself returns a dict, not Array — for strict
        # ObservationOperator conformance, use .to_observation_operator()
        flattened = fusion.to_observation_operator()
        assert isinstance(flattened, ObservationOperator)


# ---------------------------------------------------------------------------
# MaskedIdentity
# ---------------------------------------------------------------------------


class TestMaskedIdentity:
    def test_passthrough_without_mask(self, rng):
        op = MaskedIdentity()
        x = jax.random.normal(rng, (3, 4))
        assert jnp.allclose(op(x), x)

    def test_applies_mask(self, rng):
        op = MaskedIdentity()
        x = jnp.ones((3, 4))
        mask = jnp.array([[1.0, 0.0, 1.0, 0.0]] * 3)
        out = op(x, mask=mask)
        assert jnp.allclose(out, mask)

    def test_linearize_returns_operator(self, rng):
        op = MaskedIdentity()
        x = jax.random.normal(rng, (4,))
        H = op.linearize(x)
        assert isinstance(H, lx.AbstractLinearOperator)


# ---------------------------------------------------------------------------
# LinearObs
# ---------------------------------------------------------------------------


class TestLinearObs:
    def test_applies_matrix(self, rng):
        H_mat = jax.random.normal(rng, (3, 5))
        op = LinearObs(H_mat=lx.MatrixLinearOperator(H_mat))
        x = jax.random.normal(rng, (5,))
        out = op(x)
        assert jnp.allclose(out, H_mat @ x, atol=1e-5)

    def test_linearize_is_self(self, rng):
        H_mat = lx.MatrixLinearOperator(jax.random.normal(rng, (3, 5)))
        op = LinearObs(H_mat=H_mat)
        H = op.linearize(jnp.zeros(5))
        # For a linear operator, the tangent linear is the operator itself.
        assert H is H_mat


# ---------------------------------------------------------------------------
# AveragingKernel (Decision D9)
# ---------------------------------------------------------------------------


class TestAveragingKernel:
    def test_with_full_weighting_recovers_A_x(self, rng):
        n = 6
        A_mat = jax.random.normal(rng, (n, n))
        op = AveragingKernel(
            A=lx.MatrixLinearOperator(A_mat),
            x_a=jnp.zeros(n),
            h=jnp.ones(n),  # full weight on the model state
        )
        x = jax.random.normal(rng, (n,))
        # h = 1 ⇒ inner = x, output = A @ x
        assert jnp.allclose(op(x), A_mat @ x, atol=1e-5)

    def test_with_zero_weighting_returns_A_x_a(self, rng):
        n = 6
        A_mat = jax.random.normal(rng, (n, n))
        x_a = jax.random.normal(rng, (n,))
        op = AveragingKernel(
            A=lx.MatrixLinearOperator(A_mat),
            x_a=x_a,
            h=jnp.zeros(n),  # all weight on the retrieval prior
        )
        x = jax.random.normal(rng, (n,))
        # h = 0 ⇒ inner = x_a, output = A @ x_a, independent of x
        assert jnp.allclose(op(x), A_mat @ x_a, atol=1e-5)

    def test_linearize_adjoint_test(self, rng):
        """Adjoint test: <H'u, v> == <u, H'^T v> for random u, v."""
        n = 5
        A_mat = jax.random.normal(rng, (n, n))
        h = jax.random.uniform(rng, (n,))
        op = AveragingKernel(
            A=lx.MatrixLinearOperator(A_mat),
            x_a=jnp.zeros(n),
            h=h,
        )
        H = op.linearize(jnp.zeros(n))
        k1, k2 = jax.random.split(rng)
        u = jax.random.normal(k1, (n,))
        v = jax.random.normal(k2, (n,))
        forward_inner = jnp.dot(H.mv(u), v)
        adjoint_inner = jnp.dot(u, H.transpose().mv(v))
        assert jnp.isclose(forward_inner, adjoint_inner, atol=1e-4)


# ---------------------------------------------------------------------------
# MultiInstrumentFusion
# ---------------------------------------------------------------------------


class TestMultiInstrumentFusion:
    def test_per_instrument_dict_output(self, rng):
        n = 4
        spec_a = InstrumentSpec(
            obs_op=MaskedIdentity(),
            mask=jnp.ones(n),
            R_op=lx.DiagonalLinearOperator(jnp.ones(n)),
            instrument_id="A",
        )
        spec_b = InstrumentSpec(
            obs_op=LinearObs(H_mat=lx.MatrixLinearOperator(jnp.eye(n) * 2.0)),
            mask=jnp.ones(n),
            R_op=lx.DiagonalLinearOperator(jnp.ones(n)),
            instrument_id="B",
        )
        fusion = MultiInstrumentFusion(
            registry=InstrumentRegistry(entries={"A": spec_a, "B": spec_b}),
        )
        x = jax.random.normal(rng, (n,))
        out = fusion(x)
        assert set(out.keys()) == {"A", "B"}
        assert jnp.allclose(out["A"], x, atol=1e-5)
        assert jnp.allclose(out["B"], 2.0 * x, atol=1e-5)
