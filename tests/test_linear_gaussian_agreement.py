"""The Decision D14 invariant — all DA methods agree in the linear-Gaussian limit.

When :math:`H` is linear and :math:`B, R` are Gaussian, every analysis
method must converge to the same posterior mean as
``OptimalInterpolation`` (the closed-form BLUE). This is the canonical
correctness gate: if a method disagrees with OI in this limit, the
implementation is wrong.

This module exercises the gate across:

- ``OptimalInterpolation`` (the oracle)
- ``ThreeDVar`` (iterative, single time)
- ``StrongFourDVar`` (with an identity forward model so T_plus_1=1
  reduces to the 3D case)
- ``WeakFourDVar`` (same, with model error trivially zero at T=0)

``FourDVarNet`` is not included here — its learned components must
be initialised + trained to match, which isn't a unit test concern.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from vardax import (
    Batch1D,
    IncrementalConfig,
    IncrementalFourDVar,
    MaskedIdentity,
    OptimalInterpolation,
    StrongFourDVar,
    ThreeDVar,
    WeakFourDVar,
)


class IdentityForward(eqx.Module):
    """Trivial dynamics: ``M_t(x) = x``. Makes 4DVar reduce to 3DVar."""

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def state_signature(self):
        return None

    def step(self, state, dt):
        return state


@pytest.fixture
def linear_gaussian_setup():
    """Identity B, R, MaskedIdentity H, batch with all observations valid."""
    rng = jax.random.PRNGKey(42)
    N, B = 4, 2

    prior_mean_state = jnp.zeros(N)
    prior_mean_with_time = prior_mean_state[None, :]  # shape (1, N)

    B_op_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    R_op_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    B_op_with_time = lx.IdentityLinearOperator(
        jax.ShapeDtypeStruct((1, N), jnp.float32)
    )
    R_op_with_time = lx.IdentityLinearOperator(
        jax.ShapeDtypeStruct((1, N), jnp.float32)
    )

    batch = Batch1D(
        input=jax.random.normal(rng, (B, 1, N)),
        mask=jnp.ones((B, 1, N)),
        target=None,
    )

    return {
        "rng": rng,
        "N": N,
        "B": B,
        "prior_mean_state": prior_mean_state,
        "prior_mean_with_time": prior_mean_with_time,
        "B_op_state": B_op_state,
        "R_op_state": R_op_state,
        "B_op_with_time": B_op_with_time,
        "R_op_with_time": R_op_with_time,
        "batch": batch,
    }


def _analytical_blue(batch: Batch1D) -> jax.Array:
    """For ``B = R = I``, ``H = identity``, fully observed: ``x* = y / 2``."""
    return batch.input[:, 0, :] / 2.0


class TestLinearGaussianAgreement:
    """All methods must agree in the linear-Gaussian limit (D14)."""

    def test_oi_matches_analytical(self, linear_gaussian_setup):
        s = linear_gaussian_setup
        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        out = oi(s["batch"])
        # OI returns shape (B, T, N); collapse the T=1 axis
        assert jnp.allclose(out[:, 0, :], _analytical_blue(s["batch"]), atol=1e-4)

    def test_3dvar_agrees_with_oi(self, linear_gaussian_setup):
        s = linear_gaussian_setup
        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        three = ThreeDVar(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        assert jnp.allclose(oi(s["batch"]), three(s["batch"]), atol=1e-3)

    def test_strong_4dvar_agrees_with_oi_in_3d_limit(self, linear_gaussian_setup):
        """T_plus_1 = 1 ⇒ StrongFourDVar with IdentityForward = 3DVar."""
        s = linear_gaussian_setup
        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        strong = StrongFourDVar(
            forward=IdentityForward(),
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
        )
        # OI shape (B, T=1, N) → squeeze; strong shape (B, N)
        oi_out = oi(s["batch"])[:, 0, :]
        strong_out = strong(s["batch"])
        assert jnp.allclose(oi_out, strong_out, atol=1e-3)

    def test_incremental_4dvar_agrees_with_oi_in_3d_limit(self, linear_gaussian_setup):
        """T_plus_1 = 1 ⇒ IncrementalFourDVar with IdentityForward = 3DVar."""
        s = linear_gaussian_setup
        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        inc = IncrementalFourDVar(
            forward=IdentityForward(),
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
            config=IncrementalConfig(n_outer=3, n_inner=30),
        )
        oi_out = oi(s["batch"])[:, 0, :]
        inc_out = inc(s["batch"])
        assert jnp.allclose(oi_out, inc_out, atol=1e-2)

    def test_weak_4dvar_x0_agrees_with_oi_in_3d_limit(self, linear_gaussian_setup):
        """T_plus_1 = 1 ⇒ WeakFourDVar has no η; reduces to 3DVar on x_0."""
        s = linear_gaussian_setup
        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        weak = WeakFourDVar(
            forward=IdentityForward(),
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
            model_err_cov_op=s["B_op_state"],  # ignored when T_plus_1=1
        )
        oi_out = oi(s["batch"])[:, 0, :]
        x0_star, _ = weak(s["batch"])
        assert jnp.allclose(oi_out, x0_star, atol=1e-3)


class TestAnalysisStepConformance:
    """Every classical method exposes ``.as_analysis_step()`` returning an AnalysisStep."""

    def test_oi(self, linear_gaussian_setup):
        s = linear_gaussian_setup
        from vardax import AnalysisStep

        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        assert isinstance(oi.as_analysis_step(), AnalysisStep)

    def test_three_dvar(self, linear_gaussian_setup):
        s = linear_gaussian_setup
        from vardax import AnalysisStep

        three = ThreeDVar(
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        assert isinstance(three.as_analysis_step(), AnalysisStep)

    def test_strong_4dvar(self, linear_gaussian_setup):
        s = linear_gaussian_setup
        from vardax import AnalysisStep

        strong = StrongFourDVar(
            forward=IdentityForward(),
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
        )
        assert isinstance(strong.as_analysis_step(), AnalysisStep)

    def test_weak_4dvar(self, linear_gaussian_setup):
        s = linear_gaussian_setup
        from vardax import AnalysisStep

        weak = WeakFourDVar(
            forward=IdentityForward(),
            obs_op=MaskedIdentity(),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
            model_err_cov_op=s["B_op_state"],
        )
        assert isinstance(weak.as_analysis_step(), AnalysisStep)
