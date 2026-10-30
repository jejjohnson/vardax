"""Reduced weak-constraint 4D-Var: ``WeakFourDVar(model_err_basis=...)`` (#156).

η = χ W Φᵀ with W = Λ_t^{1/2} ω Λ_s^{1/2}, control (x₀, ω), cost
½ ‖x₀ − x_b‖²_{B⁻¹} + ½ ‖ω‖² + ½ Σₖ ‖yₖ − Hₖ(zₖ)‖²_{R⁻¹}. With χ = I_T,
Φ = I_N and Λ_t ⊗ Λ_s = Q it is the full weak-constraint cost.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest

import vardax as vdx

N, T = 3, 2  # T model-error steps, so T + 1 = 3 times
TRANSITION = np.array([[0.9, 0.2, 0.0], [-0.1, 0.8, 0.1], [0.0, 0.3, 0.7]])
Q_VAR = 0.2


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


class _Linear(eqx.Module):
    """x_{t+1} = M x_t."""

    mat: jax.Array

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return self.mat @ state


class _Biased(eqx.Module):
    """x_{t+1} = M x_t + b: the truth the free model lacks."""

    mat: jax.Array
    bias: float

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return self.mat @ state + self.bias


def _psd(matrix):
    return lx.MatrixLinearOperator(jnp.asarray(matrix), lx.positive_semidefinite_tag)


def _common(n_state=N):
    return dict(
        forward=_Linear(jnp.asarray(TRANSITION[:n_state, :n_state])),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(n_state),
        prior_cov_op=_psd(np.eye(n_state)),
        obs_cov_op=_psd(0.5 * np.eye(n_state)),
        minimiser=optx.BFGS(rtol=1e-13, atol=1e-13),
        cov_solver=lx.Cholesky(),
        max_steps=1000,
    )


def _window(seed=0, n_times=T + 1, n_state=N):
    obs = np.random.default_rng(seed).normal(size=(1, n_times, n_state))
    return vdx.Batch1D(input=jnp.asarray(obs), mask=jnp.ones((1, n_times, n_state)))


@pytest.mark.slow
def test_identity_basis_is_the_full_weak_analysis(x64):
    identity = vdx.ModelErrorBasis(
        temporal=jnp.eye(T),
        spatial=jnp.eye(N),
        temporal_variance=jnp.ones(T),
        spatial_variance=jnp.full(N, Q_VAR),
    )
    window = _window()
    full_x0, full_eta = vdx.WeakFourDVar(
        **_common(), model_err_cov_op=_psd(Q_VAR * np.eye(N))
    )(window)
    reduced_x0, coefficients = vdx.WeakFourDVar(**_common(), model_err_basis=identity)(
        window
    )
    assert coefficients.shape == (1, T, N)
    # both sides are BFGS minima of the same quadratic, which stop ~1e-9 apart
    np.testing.assert_allclose(reduced_x0, full_x0, rtol=1e-8, atol=1e-10)
    # with χ = Φ = I the coefficients are the model errors themselves
    np.testing.assert_allclose(coefficients, full_eta, rtol=1e-8, atol=1e-10)


def test_one_coefficient_recovers_a_constant_bias(x64):
    """A model missing +0.5 per step: C = P = 1 recovers it."""
    n_state, n_steps, bias = 6, 20, 0.5
    truth = _Biased(jnp.asarray(0.8 * np.eye(n_state)), bias)
    state = jnp.linspace(-1.0, 1.0, n_state)
    trajectory = [state]
    for _ in range(n_steps):
        state = truth.step(state, 1.0)
        trajectory.append(state)
    noise = 0.01 * np.random.default_rng(1).normal(size=(n_steps + 1, n_state))
    obs = jnp.stack(trajectory) + noise
    window = vdx.Batch1D(input=obs[None], mask=jnp.ones_like(obs)[None])
    constant = vdx.ModelErrorBasis(
        temporal=jnp.ones((n_steps, 1)),
        spatial=jnp.ones((n_state, 1)),
        temporal_variance=jnp.ones(1),
        spatial_variance=jnp.ones(1),
    )
    common = _common(n_state) | dict(
        forward=_Linear(jnp.asarray(0.8 * np.eye(n_state))),
        obs_cov_op=_psd(1e-4 * np.eye(n_state)),
    )
    _, coefficients = vdx.WeakFourDVar(**common, model_err_basis=constant)(window)
    # recovered within 2 % of the true bias
    assert coefficients.shape == (1, 1, 1)
    np.testing.assert_allclose(coefficients[0, 0, 0], bias, rtol=2e-2)


@pytest.mark.slow
def test_gradient_through_the_reduced_analysis(x64):
    """ImplicitAdjoint gradient of the coefficients in y matches central differences."""
    basis = vdx.ModelErrorBasis(
        temporal=jnp.asarray([[1.0, -1.0], [1.0, 1.0]]),
        spatial=jnp.asarray([[1.0], [0.5], [-0.5]]),
        temporal_variance=jnp.asarray([1.0, 0.5]),
        spatial_variance=jnp.asarray([0.3]),
    )
    model = vdx.WeakFourDVar(**_common(), model_err_basis=basis)
    window = _window(seed=2)

    def loss(obs):
        batch = vdx.Batch1D(input=obs, mask=window.mask)
        x0, coefficients = model(batch)
        return jnp.sum(x0**2) + jnp.sum(coefficients**2)

    grad = jax.grad(loss)(window.input)
    direction = jnp.asarray(np.random.default_rng(3).normal(size=window.input.shape))
    # the loss is quadratic in y for a linear model, so central differences
    # are exact and a wide step keeps the minimiser's stopping error small
    step = 1e-2
    central = (
        loss(window.input + step * direction) - loss(window.input - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(jnp.sum(grad * direction), central, rtol=1e-6)


def test_composes_with_a_reduced_initial_state(x64):
    basis = vdx.ModelErrorBasis(
        temporal=jnp.ones((T, 1)),
        spatial=jnp.ones((N, 1)),
        temporal_variance=jnp.ones(1),
        spatial_variance=jnp.ones(1),
    )
    param = vdx.LinearParameterization(
        vdx.linear_basis(jnp.eye(N)[:, :2], variance=1.0), background=jnp.zeros(N)
    )
    x0, coefficients = vdx.WeakFourDVar(
        **_common(), model_err_basis=basis, parameterization=param
    )(_window())
    assert x0.shape == (1, N) and coefficients.shape == (1, 1, 1)
    # x₀ lies in span(e₁, e₂)
    assert abs(float(x0[0, 2])) < 1e-12


def test_needs_exactly_one_model_error_prior():
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    basis = vdx.ModelErrorBasis(
        temporal=jnp.ones((T, 1)),
        spatial=jnp.ones((N, 1)),
        temporal_variance=jnp.ones(1),
        spatial_variance=jnp.ones(1),
    )
    common = dict(
        forward=_Linear(jnp.eye(N)),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(N),
        prior_cov_op=eye,
        obs_cov_op=eye,
    )
    with pytest.raises(vdx.ConfigurationError, match="exactly one"):
        vdx.WeakFourDVar(**common)
    with pytest.raises(vdx.ConfigurationError, match="exactly one"):
        vdx.WeakFourDVar(**common, model_err_cov_op=eye, model_err_basis=basis)
    wrong_length = vdx.Batch1D(input=jnp.ones((1, 5, N)), mask=jnp.ones((1, 5, N)))
    with pytest.raises(vdx.ShapeError, match="model-error steps"):
        vdx.WeakFourDVar(**common, model_err_basis=basis)(wrong_length)
