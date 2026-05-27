"""Regression tests for the symmetric-masking bug fix (PR #41 review).

For obs_ops that don't accept a ``mask`` kwarg (e.g. ``LinearObs``), the
v0.2-dev classical methods originally formed residuals as
``mask * y - obs_op(x)``, which left ``obs_op(x)`` unmasked at missing
entries. That pulled the analysis toward zero in gaps. The fix is to
mask both sides:

    residual = mask * (y - obs_op(x))

This module exercises that fix across ``ThreeDVar``, ``StrongFourDVar``,
``WeakFourDVar``, and ``IncrementalFourDVar`` by checking that
**partially-masked** problems with a **non-mask-aware** obs operator
recover the analytical BLUE answer (which itself is mask-aware).

The same masking is required inside ``OptimalInterpolation``'s
``linearize()`` call — verified by exercising the OI path with a
gap-mask and comparing against a hand-written BLUE.
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
    LinearObs,
    OptimalInterpolation,
    StrongFourDVar,
    ThreeDVar,
    WeakFourDVar,
)


class IdentityForward(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    @property
    def state_signature(self):
        return None

    def step(self, state, dt):
        return state


@pytest.fixture
def masked_linear_gaussian():
    """Linear-Gaussian setup with a gap mask.

    obs_op is ``LinearObs(H_mat=I)`` (not mask-aware). Mask is
    ``[1, 0, 1, 0]`` along the spatial axis so the analysis should
    only update the *observed* entries.
    """
    rng = jax.random.PRNGKey(7)
    N = 4
    B = 1
    # mask: 1, 0, 1, 0 (alternating gap pattern)
    mask = jnp.array([[1.0, 0.0, 1.0, 0.0]])  # shape (T=1, N=4)
    mask_batched = mask[None]  # (B=1, T=1, N=4)

    y_truth = jax.random.normal(rng, (B, 1, N))
    input_batched = y_truth * mask_batched

    prior_mean_with_time = jnp.zeros((1, N))
    prior_mean_state = jnp.zeros(N)

    B_op_with_time = lx.IdentityLinearOperator(
        jax.ShapeDtypeStruct((1, N), jnp.float32)
    )
    R_op_with_time = lx.IdentityLinearOperator(
        jax.ShapeDtypeStruct((1, N), jnp.float32)
    )
    B_op_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    R_op_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))

    return {
        "rng": rng,
        "N": N,
        "B": B,
        "mask": mask_batched,
        "input": input_batched,
        "prior_mean_with_time": prior_mean_with_time,
        "prior_mean_state": prior_mean_state,
        "B_op_with_time": B_op_with_time,
        "R_op_with_time": R_op_with_time,
        "B_op_state": B_op_state,
        "R_op_state": R_op_state,
        # The analytical answer: observed entries → y/2, gaps → 0.
        "expected": (input_batched / 2.0)[:, 0, :],
    }


def _identity_linear_obs_state(N: int) -> LinearObs:
    """``LinearObs`` on a flat state ``(N,)`` — used by 4DVar variants."""
    return LinearObs(
        H_mat=lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    )


def _identity_linear_obs_with_time(N: int) -> LinearObs:
    """``LinearObs`` on a ``(T=1, N)`` shape — used by OI / ThreeDVar."""
    return LinearObs(
        H_mat=lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, N), jnp.float32))
    )


class TestMaskedResidualRegression:
    """Each classical method must recover ``y/2`` at observed entries
    and stay at ``0`` at gaps (the analytical BLUE with B=R=I)."""

    def test_oi_with_non_mask_aware_obs(self, masked_linear_gaussian):
        s = masked_linear_gaussian
        batch = Batch1D(input=s["input"], mask=s["mask"], target=None)
        oi = OptimalInterpolation(
            obs_op=_identity_linear_obs_with_time(s["N"]),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        out = oi(batch)
        assert jnp.allclose(out[:, 0, :], s["expected"], atol=1e-3)

    def test_threedvar_with_non_mask_aware_obs(self, masked_linear_gaussian):
        s = masked_linear_gaussian
        batch = Batch1D(input=s["input"], mask=s["mask"], target=None)
        three = ThreeDVar(
            obs_op=_identity_linear_obs_with_time(s["N"]),
            prior_mean=s["prior_mean_with_time"],
            prior_cov_op=s["B_op_with_time"],
            obs_cov_op=s["R_op_with_time"],
        )
        out = three(batch)
        assert jnp.allclose(out[:, 0, :], s["expected"], atol=1e-2)

    def test_strong_4dvar_with_non_mask_aware_obs(self, masked_linear_gaussian):
        s = masked_linear_gaussian
        batch = Batch1D(input=s["input"], mask=s["mask"], target=None)
        strong = StrongFourDVar(
            forward=IdentityForward(),
            obs_op=_identity_linear_obs_state(s["N"]),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
        )
        out = strong(batch)
        assert jnp.allclose(out, s["expected"], atol=1e-2)

    def test_incremental_4dvar_with_non_mask_aware_obs(self, masked_linear_gaussian):
        s = masked_linear_gaussian
        batch = Batch1D(input=s["input"], mask=s["mask"], target=None)
        inc = IncrementalFourDVar(
            forward=IdentityForward(),
            obs_op=_identity_linear_obs_state(s["N"]),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
            config=IncrementalConfig(n_outer=4, n_inner=30),
        )
        out = inc(batch)
        assert jnp.allclose(out, s["expected"], atol=1e-2)

    def test_weak_4dvar_with_non_mask_aware_obs(self, masked_linear_gaussian):
        s = masked_linear_gaussian
        batch = Batch1D(input=s["input"], mask=s["mask"], target=None)
        weak = WeakFourDVar(
            forward=IdentityForward(),
            obs_op=_identity_linear_obs_state(s["N"]),
            prior_mean=s["prior_mean_state"],
            prior_cov_op=s["B_op_state"],
            obs_cov_op=s["R_op_state"],
            model_err_cov_op=s["B_op_state"],
        )
        x0_star, _ = weak(batch)
        assert jnp.allclose(x0_star, s["expected"], atol=1e-2)
