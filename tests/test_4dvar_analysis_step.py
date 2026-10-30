"""The 4D-Var ``AnalysisStep`` adapters analyse a window around the forecast (#79).

``SmootherCycle`` calls ``step(forecasts, observations, obs_op=H,
obs_err_cov=R)`` with length-W lists and steps the next window from the
return value. The adapters used to ignore the forecast, H and R, so a cycle
analysed every window around the model's construction-time background.
Now the forecast at the window start is x_b, the cycle's H and R apply at
every time, and the return is the analysed state at the end of the window.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from pipekit import Lambda
import pipekit_cycle as pc
import pytest

import vardax as vdx

N, M, W = 4, 3, 3  # state size, observations per time, times per window
TRANSITION = jnp.array(
    [
        [0.9, 0.2, 0.0, 0.0],
        [-0.1, 0.8, 0.1, 0.0],
        [0.0, 0.3, 0.7, 0.1],
        [0.0, 0.0, 0.2, 0.9],
    ]
)


class _Linear(eqx.Module):
    """x_{t+1} = A x_t."""

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def state_signature(self):
        return None

    def step(self, state, dt):
        return TRANSITION @ state


def _psd(matrix):
    return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)


def _problem():
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    root = jax.random.normal(keys[0], (N, N))
    return {
        "B": root @ root.T / N + 0.5 * jnp.eye(N),
        "H": jax.random.normal(keys[1], (M, N)),  # the cycle's H
        "R": jnp.diag(jnp.array([0.3, 0.5, 0.8])),  # the cycle's R
        "Q": 0.1 * jnp.eye(N),
        "forecast": jax.random.normal(keys[2], (N,)) + 1.0,
        "ys": [jax.random.normal(k, (M,)) for k in jax.random.split(keys[3], W)],
        # the model's own H and R: different from the cycle's on purpose
        "model_H": jax.random.normal(keys[4], (M, N)),
        "model_R": 2.0 * jnp.eye(M),
    }


def _common(p):
    return dict(
        forward=_Linear(),
        obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(p["model_H"])),
        prior_mean=jnp.zeros(N),
        prior_cov_op=_psd(p["B"]),
        obs_cov_op=_psd(p["model_R"]),
    )


def _strong(p):
    return vdx.StrongFourDVar(
        **_common(p),
        minimiser=optx.BFGS(rtol=1e-12, atol=1e-12),
        cov_solver=lx.Cholesky(),
        max_steps=1000,
    )


def _incremental(p):
    return vdx.IncrementalFourDVar(
        **_common(p),
        config=vdx.IncrementalConfig(
            n_outer=1, n_inner=100, cg_atol=1e-13, cg_rtol=1e-13, covariance_tol=1e-13
        ),
    )


def _weak(p):
    return vdx.WeakFourDVar(
        **_common(p),
        model_err_cov_op=_psd(p["Q"]),
        minimiser=optx.BFGS(rtol=1e-12, atol=1e-12),
        cov_solver=lx.Cholesky(),
        max_steps=2000,
    )


def _powers():
    return [jnp.linalg.matrix_power(TRANSITION, t) for t in range(W)]


def _strong_oracle(p, x_b, ys):
    """End state Aᵂ⁻¹ x₀* of the strong-constraint BLUE over the window."""
    G = jnp.concatenate([p["H"] @ power for power in _powers()])
    R = jnp.kron(jnp.eye(W), p["R"])
    y = jnp.concatenate(ys)
    gain = p["B"] @ G.T @ jnp.linalg.inv(G @ p["B"] @ G.T + R)
    x0 = x_b + gain @ (y - G @ x_b)
    return _powers()[-1] @ x0


def _weak_oracle(p, x_b, ys):
    """End state of the weak-constraint BLUE over z = (x₀, η₁, η₂)."""
    powers = _powers()
    size = N * W
    # x_t = Aᵗ x₀ + Σ_{s ≤ t} Aᵗ⁻ˢ η_s; S maps z to the stacked trajectory
    S = jnp.zeros((size, size))
    for t in range(W):
        S = S.at[t * N : (t + 1) * N, :N].set(powers[t])
        for s in range(1, t + 1):
            S = S.at[t * N : (t + 1) * N, s * N : (s + 1) * N].set(powers[t - s])
    G = jnp.kron(jnp.eye(W), p["H"]) @ S
    P = jax.scipy.linalg.block_diag(p["B"], *([p["Q"]] * (W - 1)))
    R = jnp.kron(jnp.eye(W), p["R"])
    z_b = jnp.concatenate([x_b, jnp.zeros(N * (W - 1))])
    gain = P @ G.T @ jnp.linalg.inv(G @ P @ G.T + R)
    z = z_b + gain @ (jnp.concatenate(ys) - G @ z_b)
    return (S @ z)[-N:]


def _forecasts(x_b):
    return [power @ x_b for power in _powers()]


def _call(step, p, x_b, ys):
    return eqx.filter_jit(step)(
        _forecasts(x_b),
        ys,
        obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(p["H"])),
        obs_err_cov=_psd(p["R"]),
    )


MODELS = [
    pytest.param(_strong, _strong_oracle, id="strong"),
    pytest.param(_incremental, _strong_oracle, id="incremental"),
    pytest.param(_weak, _weak_oracle, id="weak"),
]


@pytest.mark.slow
@pytest.mark.parametrize(("make_model", "oracle"), MODELS)
def test_window_end_state_matches_the_closed_form(make_model, oracle):
    with jax.enable_x64(True):
        p = _problem()
        step = make_model(p).as_analysis_step()
        end_state = _call(step, p, p["forecast"], p["ys"])
        expected = oracle(p, p["forecast"], p["ys"])
        assert jnp.allclose(end_state, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(("make_model", "oracle"), MODELS)
def test_changing_the_forecast_changes_the_analysis(make_model, oracle):
    """Regression: the adapters used to ignore the forecast entirely."""
    p = _problem()
    step = make_model(p).as_analysis_step()
    first = _call(step, p, p["forecast"], p["ys"])
    second = _call(step, p, p["forecast"] + 1.0, p["ys"])
    assert not jnp.allclose(first, second, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize(("make_model", "oracle"), MODELS)
def test_two_window_smoother_threads_the_analysis(make_model, oracle):
    """The second window steps from the first window's analysed end state."""
    with jax.enable_x64(True):
        p = _problem()
        ys = p["ys"] + [-y + 0.5 for y in p["ys"]]
        cycle = vdx.VarSmootherCycle(
            forward=_Linear(),
            obs_op=vdx.LinearObs(operator=lx.MatrixLinearOperator(p["H"])),
            model=make_model(p),
            window=W,
            stride=2,
            obs_source=Lambda(lambda i: ys[i]),
        )
        x0 = p["forecast"]
        final, _ = cycle(x0, pc.DAState(obs_err_cov=_psd(p["R"])))
        # window 1 starts one step after x0; window 2 one step after its end
        end_1 = oracle(p, TRANSITION @ x0, ys[:W])
        end_2 = oracle(p, TRANSITION @ end_1, ys[W:])
        assert jnp.allclose(final, end_2, rtol=0.0, atol=1e-6)
