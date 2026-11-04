"""A trial step that blows up must not end the solve (#209).

A line search tries points whose rollout can overflow, and it can only reject a
trial that returns. The costs used to raise from ``lx.linear_solve`` instead.
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import vardax as vdx
from vardax._src.utils.covariance import guarded_solve, half_weighted_norm

N = 4
PSD = lx.positive_semidefinite_tag
CG = lx.CG(atol=1e-6, rtol=1e-6)


def _cov(variance=0.25, n=N):
    return lx.MatrixLinearOperator(variance * jnp.eye(n), PSD)


class _Explosive(eqx.Module):
    """x_{t+1} = x_t + x_t⁵: gentle near the truth, overflows on a long step."""

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return state + 0.1 * state**5


class _Lorenz96(eqx.Module):
    """Lorenz-96 at F = 8, as in the issue."""

    @property
    def dt(self) -> float:
        return 0.05

    @staticmethod
    def rhs(t, x, args):
        return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + 8.0

    def step(self, x, dt):
        return dfx.diffeqsolve(dfx.ODETerm(self.rhs), dfx.Tsit5(), 0.0, dt, dt, x).ys[
            -1
        ]


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param(jnp.array([jnp.nan, 1.0, 1.0, 1.0]), id="nan"),
        pytest.param(jnp.array([jnp.inf, 1.0, 1.0, 1.0]), id="inf"),
        pytest.param(jnp.full(N, 1e25), id="huge"),
    ],
)
def test_an_unusable_vector_is_reported_and_not_solved(vector):
    solved, usable = guarded_solve(_cov(), vector, CG)
    assert not bool(usable)
    assert bool(jnp.all(jnp.isnan(solved)))
    assert jnp.isposinf(half_weighted_norm(_cov(), vector, CG))
    # the gradient path is where throw=False alone still raises
    gradient = jax.grad(lambda v: half_weighted_norm(_cov(), v, CG))(vector)
    assert bool(jnp.all(gradient == 0.0))


def test_a_usable_vector_is_the_weighted_norm():
    vector = jnp.array([1.0, -2.0, 3.0, -4.0])
    norm = half_weighted_norm(_cov(), vector, CG)
    assert jnp.allclose(norm, 0.5 * jnp.sum(vector**2) / 0.25, rtol=1e-5)
    gradient = jax.grad(lambda v: half_weighted_norm(_cov(), v, CG))(vector)
    assert jnp.allclose(gradient, vector / 0.25, rtol=1e-5)


def _window():
    obs = jnp.array([[1.0, 1.1, 0.9, 1.0], [1.1, 1.2, 1.0, 1.1], [1.3, 1.4, 1.1, 1.3]])
    return vdx.Batch1D(input=obs[None], mask=jnp.ones_like(obs)[None])


def _common():
    return dict(
        forward=_Explosive(),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.full(N, 1.5),
        prior_cov_op=_cov(),
        obs_cov_op=_cov(),
    )


MODELS = [
    pytest.param(lambda: vdx.StrongFourDVar(**_common()), id="strong"),
    pytest.param(
        lambda: vdx.WeakFourDVar(**_common(), model_err_cov_op=_cov(0.01)), id="weak"
    ),
    pytest.param(lambda: vdx.ReducedBasisFourDVar(**_common()), id="reduced-basis"),
]


@pytest.mark.parametrize("make_model", MODELS)
def test_an_exploding_trial_step_is_rejected(make_model):
    analysis = make_model()(_window())
    first_state = analysis[0] if isinstance(analysis, tuple) else analysis
    assert bool(jnp.all(jnp.isfinite(first_state)))


def test_threedvar_rejects_an_exploding_trial_step():
    model = vdx.ThreeDVar(
        obs_op=vdx.CallableObs(fn=lambda state: state**41),
        prior_mean=jnp.full(N, 1.5),
        prior_cov_op=_cov(),
        obs_cov_op=_cov(),
    )
    batch = vdx.Batch1D(input=jnp.ones((1, N)), mask=jnp.ones((1, N)))
    assert bool(jnp.all(jnp.isfinite(model(batch))))


@pytest.mark.slow
def test_incremental_reports_a_failed_window_as_nan():
    """Gauss-Newton has no line search, so its failure comes back as NaN."""
    model = vdx.IncrementalFourDVar(
        **_common() | {"prior_mean": jnp.full(N, 5.0)},
        config=vdx.IncrementalConfig(n_outer=8),
    )
    analysis = model(_window())
    assert analysis.shape == (1, N)
    assert bool(jnp.all(jnp.isnan(analysis))) or bool(jnp.all(jnp.isfinite(analysis)))


@pytest.mark.slow
@pytest.mark.parametrize("truth_index", [1, 3])
def test_chaotic_lorenz96_window_finishes(truth_index):
    """The two truths of the issue's script that raised, with its exact setup."""
    forward = _Lorenz96()
    n = 12
    x = 8.0 * jnp.ones(n).at[0].add(0.01)
    for _ in range(200):
        x = forward.step(x, forward.dt)
    for index in range(truth_index + 1):
        window = [x]
        for _ in range(3):
            window.append(forward.step(window[-1], forward.dt))
        window = jnp.stack(window)
        if index < truth_index:
            for _ in range(40):
                x = forward.step(x, forward.dt)
    noise = 0.2 * jax.random.normal(jax.random.key(truth_index), window.shape)
    batch = vdx.Batch1D(input=(window + noise)[None], mask=jnp.ones_like(window)[None])
    prior_mean = x + 0.3 * jax.random.normal(jax.random.key(100 + truth_index), (n,))
    common = dict(
        forward=forward,
        obs_op=vdx.MaskedIdentity(),
        prior_mean=prior_mean,
        prior_cov_op=_cov(0.5, n),
        obs_cov_op=_cov(0.04, n),
    )
    analysis = vdx.StrongFourDVar(**common)(batch)[0]
    background_rmse = jnp.sqrt(jnp.mean((prior_mean - x) ** 2))
    assert jnp.sqrt(jnp.mean((analysis - x) ** 2)) < background_rmse
    # both methods minimise the same cost, so they agree where both finish
    gauss_newton = vdx.IncrementalFourDVar(
        **common, config=vdx.IncrementalConfig(jac="bwd")
    )(batch)[0]
    assert jnp.allclose(analysis, gauss_newton, atol=1e-2)
