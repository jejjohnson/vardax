"""KStepAdjoint, the pipekit spec mapping, and diffrax-backed dynamics.

The inner-solve layer of the shared adjoint-strategies design:
``KStepAdjoint(k)`` generalises ``OneStepAdjoint`` (the ``k=1`` alias),
``to_optimistix_adjoint`` interprets the ``pipekit_cycle.adjoints``
vocabulary, and ``pipekit_jax.DiffraxForwardModel`` slots into
``StrongFourDVar`` as a plain ``ForwardModel``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import pytest

import vardax as vdx
from vardax._src._types import Batch1D
from vardax.adjoints import KStepAdjoint, OneStepAdjoint, to_optimistix_adjoint


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def batch_1d(rng):
    B, T, N = 2, 4, 8
    return Batch1D(
        input=jax.random.normal(rng, (B, T, N)),
        mask=jnp.ones((B, T, N)),
        target=None,
    )


def _model(rng, batch, adjoint, n_steps=3):
    _, T, N = batch.input.shape
    return vdx.FourDVarNet1D(
        state_dim=N,
        n_time=T,
        latent_dim=8,
        hidden_dim=16,
        n_solver_steps=n_steps,
        solver_adjoint=adjoint,
        key=rng,
    )


def _grad_norm(model, batch):
    def loss(m):
        return jnp.sum(m(batch) ** 2)

    grads = eqx.filter_grad(loss)(model)
    leaves = [g for g in jax.tree.leaves(grads) if g is not None]
    return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))


class TestKStepAdjoint:
    def test_k1_matches_one_step_exactly(self, rng, batch_1d):
        m_one = _model(rng, batch_1d, OneStepAdjoint())
        m_k1 = _model(rng, batch_1d, KStepAdjoint(k=1))
        assert jnp.allclose(m_one(batch_1d), m_k1(batch_1d))
        assert jnp.allclose(_grad_norm(m_one, batch_1d), _grad_norm(m_k1, batch_1d))

    def test_k_equals_n_steps_matches_unrolled_gradient(self, rng, batch_1d):
        """No warmup left to detach: identical to the unrolled default."""
        m_full = _model(rng, batch_1d, optx.RecursiveCheckpointAdjoint(), n_steps=3)
        m_k3 = _model(rng, batch_1d, KStepAdjoint(k=3), n_steps=3)
        assert jnp.allclose(m_full(batch_1d), m_k3(batch_1d))
        assert jnp.allclose(
            _grad_norm(m_full, batch_1d), _grad_norm(m_k3, batch_1d), rtol=1e-5
        )

    def test_intermediate_k_differs_from_both_ends(self, rng, batch_1d):
        g1 = _grad_norm(_model(rng, batch_1d, KStepAdjoint(k=1)), batch_1d)
        g2 = _grad_norm(_model(rng, batch_1d, KStepAdjoint(k=2)), batch_1d)
        g3 = _grad_norm(_model(rng, batch_1d, KStepAdjoint(k=3)), batch_1d)
        assert not jnp.allclose(g1, g2)
        assert not jnp.allclose(g2, g3)

    def test_forward_values_identical_for_all_k(self, rng, batch_1d):
        outs = [
            _model(rng, batch_1d, KStepAdjoint(k=k))(batch_1d) for k in (1, 2, 3, 99)
        ]
        for out in outs[1:]:
            assert jnp.allclose(outs[0], out)

    def test_one_step_is_kstep_subclass(self):
        assert isinstance(OneStepAdjoint(), KStepAdjoint)
        assert OneStepAdjoint().k == 1

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="k >= 1"):
            KStepAdjoint(k=0)


class TestToOptimistixAdjoint:
    def test_pipekit_spec_mapping(self):
        adjoints = pytest.importorskip("pipekit_cycle.adjoints")
        mapped = to_optimistix_adjoint(adjoints.TruncatedAdjoint(k=4))
        assert isinstance(mapped, KStepAdjoint)
        assert mapped.k == 4
        assert isinstance(
            to_optimistix_adjoint(adjoints.ImplicitAdjoint()), optx.ImplicitAdjoint
        )
        assert isinstance(
            to_optimistix_adjoint(adjoints.RecursiveCheckpointAdjoint(checkpoints=8)),
            optx.RecursiveCheckpointAdjoint,
        )

    def test_raw_optimistix_passthrough(self):
        raw = optx.ImplicitAdjoint()
        assert to_optimistix_adjoint(raw) is raw

    def test_layer_inappropriate_specs_rejected(self):
        adjoints = pytest.importorskip("pipekit_cycle.adjoints")
        with pytest.raises(ValueError, match="inner-solve layer"):
            to_optimistix_adjoint(adjoints.BacksolveAdjoint())
        with pytest.raises(ValueError, match="inner-solve layer"):
            to_optimistix_adjoint(adjoints.DirectAdjoint())

    def test_unknown_spec_rejected(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            to_optimistix_adjoint(object())

    def test_model_accepts_mapped_spec(self, rng, batch_1d):
        adjoints = pytest.importorskip("pipekit_cycle.adjoints")
        model = _model(
            rng, batch_1d, to_optimistix_adjoint(adjoints.TruncatedAdjoint(k=2))
        )
        out = model(batch_1d)
        assert out.shape == batch_1d.input.shape


class TestDiffraxDynamics:
    def test_strong_fourdvar_with_diffrax_forward(self, rng):
        """DiffraxForwardModel is a pipekit ForwardModel — it slots straight in."""
        pipekit_jax = pytest.importorskip("pipekit_jax")
        pytest.importorskip("diffrax")

        N = 4
        fwd = pipekit_jax.DiffraxForwardModel(
            vector_field=lambda t, y, args: -0.1 * y,
            dt0=0.1,
            dt=1.0,
        )
        model = vdx.StrongFourDVar(
            forward=fwd,
            obs_op=vdx.MaskedIdentity(),
            prior_mean=jnp.zeros(N),
            prior_cov_op=lx.IdentityLinearOperator(
                jax.ShapeDtypeStruct((N,), jnp.float32)
            ),
            obs_cov_op=lx.IdentityLinearOperator(
                jax.ShapeDtypeStruct((N,), jnp.float32)
            ),
        )
        batch = Batch1D(
            input=jax.random.normal(rng, (2, 1, N)),
            mask=jnp.ones((2, 1, N)),
            target=None,
        )
        out = model(batch)
        assert out.shape == (2, N)
        assert jnp.all(jnp.isfinite(out))

    def test_adjoint_swap_is_configuration(self):
        pipekit_jax = pytest.importorskip("pipekit_jax")
        adjoints = pytest.importorskip("pipekit_cycle.adjoints")
        pytest.importorskip("diffrax")

        fwd = pipekit_jax.DiffraxForwardModel(
            vector_field=lambda t, y, args: -y, dt0=0.1
        )
        swapped = fwd.with_adjoint(adjoints.DirectAdjoint())
        assert swapped.adjoint == adjoints.DirectAdjoint()
        y = jnp.ones(3)
        assert jnp.allclose(fwd.step(y, 1.0), swapped.step(y, 1.0), atol=1e-5)
