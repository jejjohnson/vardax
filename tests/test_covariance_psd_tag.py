"""Untagged covariances under the default CG solve (#146).

lineax drops the PSD tag when an operator is scaled, so ``σ² I`` written as
``0.01 * eye`` used to make every model's default ``lx.CG`` raise. The
models now tag their covariances at construction. Each case checks the
analysis with an untagged ``σ² I`` against the same model given the
tagged operator.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import vardax as vdx

N = 3
STATE = jax.ShapeDtypeStruct((N,), jnp.float32)


class _Decay(eqx.Module):
    """``x_{t+1} = 0.9 x_t``."""

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return 0.9 * state


def _scaled_eye(variance):
    # σ² I, which lineax leaves untagged
    return jnp.float32(variance) * lx.IdentityLinearOperator(STATE)


def _tagged(operator):
    return lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)


def _covs(tag):
    wrap = _tagged if tag else (lambda op: op)
    return dict(
        prior_cov_op=wrap(_scaled_eye(1.0)),
        obs_cov_op=wrap(_scaled_eye(0.01)),
    )


def _common():
    return dict(obs_op=vdx.MaskedIdentity(), prior_mean=jnp.zeros(N))


def _with_forward():
    return dict(_common(), forward=_Decay())


BUILDERS = {
    "threedvar": lambda tag: vdx.ThreeDVar(**_common(), **_covs(tag)),
    "optimal_interpolation": lambda tag: vdx.OptimalInterpolation(
        **_common(), **_covs(tag)
    ),
    "strong_fourdvar": lambda tag: vdx.StrongFourDVar(**_with_forward(), **_covs(tag)),
    "weak_fourdvar": lambda tag: vdx.WeakFourDVar(
        **_with_forward(),
        **_covs(tag),
        model_err_cov_op=(_tagged if tag else (lambda op: op))(_scaled_eye(0.1)),
    ),
    "incremental_fourdvar": lambda tag: vdx.IncrementalFourDVar(
        **_with_forward(), **_covs(tag)
    ),
    "reduced_basis_fourdvar": lambda tag: vdx.ReducedBasisFourDVar(
        **_with_forward(),
        **_covs(tag),
        basis=vdx.linear_basis(jnp.eye(N)[:, :2], variance=jnp.ones(2)),
    ),
}

ONE_TIME = {"threedvar", "optimal_interpolation"}


def _batch(name):
    n_times = 1 if name in ONE_TIME else 3
    obs = jnp.linspace(0.5, 1.5, n_times * N).reshape(1, n_times, N)
    if name in ONE_TIME:
        obs = obs[:, 0]
    return vdx.Batch1D(input=obs, mask=jnp.ones_like(obs))


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_untagged_scaled_identity_matches_tagged(name):
    untagged = BUILDERS[name](tag=False)
    assert lx.is_positive_semidefinite(untagged.prior_cov_op)
    assert lx.is_positive_semidefinite(untagged.obs_cov_op)
    batch = _batch(name)
    analysis = untagged(batch)
    expected = BUILDERS[name](tag=True)(batch)
    assert jax.tree.all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=1e-5), analysis, expected)
    )


def test_threedvar_issue_reproduction():
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, N), jnp.float32))
    model = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros((1, N)),
        prior_cov_op=eye,
        obs_cov_op=jnp.float32(0.01) * eye,
    )
    batch = vdx.Batch1D(input=jnp.ones((1, 1, N)), mask=jnp.ones((1, 1, N)))
    # x* = B (B + R)⁻¹ y = y / 1.01
    assert jnp.allclose(model(batch), 1.0 / 1.01, atol=1e-4)


def test_laplace_adapter_accepts_untagged_covariances():
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, N), jnp.float32))
    model = vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros((1, N)),
        prior_cov_op=_tagged(eye),
        obs_cov_op=_tagged(jnp.float32(0.01) * eye),
    )
    batch = vdx.Batch1D(input=jnp.ones((1, 1, N)), mask=jnp.ones((1, 1, N)))
    analysis = model(batch)[0]
    adapter = vdx.LaplaceCovariance(
        prior_cov_op=eye, obs_cov_op=jnp.float32(0.01) * eye
    )
    posterior = adapter(analysis, model, batch)
    # P* = (B⁻¹ + R⁻¹)⁻¹ = I / 101
    unit = jnp.zeros((1, N)).at[0, 0].set(1.0)
    assert jnp.allclose(posterior.cov.mv(unit), unit / 101.0, atol=1e-5)
