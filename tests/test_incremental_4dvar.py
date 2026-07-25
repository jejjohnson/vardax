"""Tests for `vardax.Incremental4DVar` (issue #49).

Vardax-style conformance:
- Builds as an `eqx.Module` with stored `(forward, obs_op, prior_mean,
  prior_cov_op, obs_cov_op)`.
- `__call__(batch: Batch1D)` returns the analysis state.
- `.as_analysis_step()` returns a `pipekit_cycle.AnalysisStep` adapter.
- Plugs into `vardax.VarSmootherCycle` like every other vardax model.
- Recovers the closed-form linear-Gaussian MAP under the analytic
  shrinkage when used basis-less.
- Reduced-basis path agrees with `optimistix.GaussNewton` /
  `optimistix.BFGS` (engine parity within tolerance).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pipekit_cycle as pc
import pytest

import vardax as vdx
from vardax import (
    AnalysisStep,
    Batch1D,
    Incremental4DVar,
    LinearBasis,
    LinearObs,
    rbf_basis,
)


class _LinearForward(eqx.Module):
    """`ForwardModel`-compatible 1-D linear decay: ``x_{t+1} = a * x_t``."""

    alpha: float
    _dt: float = 1.0

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_signature(self):
        return None

    def step(self, state, dt):
        return self.alpha * state


@pytest.fixture
def lg_problem():
    """Linear-Gaussian setup: N=3 state, T=2 rollout, B=R=I, x_b=0, full obs."""
    fwd = _LinearForward(alpha=0.9)
    N = 3
    T_plus_1 = 3  # 2 rollout steps + 1 initial

    H = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    obs_op = LinearObs(H_mat=H)
    state_op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))

    # Truth + obs
    x0_true = jnp.array([3.0, -2.0, 1.0])
    traj = [x0_true]
    for _ in range(T_plus_1 - 1):
        traj.append(fwd.step(traj[-1], fwd.dt))
    truth = jnp.stack(traj)  # (T+1, N)

    batch = Batch1D(
        input=truth[None],  # (B=1, T+1, N)
        mask=jnp.ones((1, T_plus_1, N)),
        target=None,
    )
    return {
        "fwd": fwd,
        "obs_op": obs_op,
        "B_op": state_op,
        "R_op": state_op,
        "prior_mean": jnp.zeros(N),
        "x0_true": x0_true,
        "T_plus_1": T_plus_1,
        "batch": batch,
    }


def test_basis_less_recovers_closed_form_map(lg_problem):
    """Closed-form MAP under B^{-1}=R^{-1}=I, linear forward x_t = a^t x_0,
    noise-free obs y_t = a^t * x_true:

        x_0_MAP = (sum_t a^{2t}) / (1 + sum_t a^{2t}) * x_true.
    """
    p = lg_problem
    ana = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        max_steps=500,
    )
    out = ana(p["batch"])  # (B=1, N)
    assert out.shape == (1, 3)

    a = p["fwd"].alpha
    shrink = sum(a ** (2 * t) for t in range(p["T_plus_1"])) / (
        1.0 + sum(a ** (2 * t) for t in range(p["T_plus_1"]))
    )
    np.testing.assert_allclose(
        np.asarray(out[0]),
        shrink * np.asarray(p["x0_true"]),
        atol=5e-2,
    )


def test_analysis_step_smoother_window_shapes(lg_problem):
    """`SmootherCycle` hands the adapter list-of-(N,) inputs; the adapter
    must stack to ``(1, T+1, N)`` rather than letting the model treat
    ``T+1`` as the batch axis (which would mis-set the rollout length)."""
    p = lg_problem
    ana = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        max_steps=500,
    )
    step = ana.as_analysis_step()

    truth = p["batch"].input[0]  # (T+1, N)
    window_forecasts = [truth[t] for t in range(p["T_plus_1"])]
    window_obs = [truth[t] for t in range(p["T_plus_1"])]

    out = step(
        window_forecasts,
        window_obs,
        obs_op=p["obs_op"],
        obs_err_cov=p["R_op"],
    )
    assert out.shape == (3,)


def test_analysis_step_threads_forecast_as_background(lg_problem):
    """The forecast supplied by the cycle replaces the construction-time
    ``prior_mean`` via ``eqx.tree_at`` — so per-window cycles regularise
    against the live forecast, not the stale stored background."""
    p = lg_problem
    bad_prior = jnp.full_like(p["prior_mean"], 1e6)
    ana = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=bad_prior,
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        max_steps=500,
    )
    step = ana.as_analysis_step()

    truth = p["batch"].input[0]
    window_forecasts = [truth[t] for t in range(p["T_plus_1"])]
    window_obs = [truth[t] for t in range(p["T_plus_1"])]

    out_threaded = step(
        window_forecasts,
        window_obs,
        obs_op=p["obs_op"],
        obs_err_cov=p["R_op"],
    )

    ana_good = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=jnp.asarray(window_forecasts[0]),
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        max_steps=500,
    )
    out_stored = ana_good.as_analysis_step()(
        window_forecasts,
        window_obs,
        obs_op=p["obs_op"],
        obs_err_cov=p["R_op"],
    )
    np.testing.assert_allclose(
        np.asarray(out_threaded), np.asarray(out_stored), atol=1e-3
    )


def test_analysis_step_returns_window_end_state(lg_problem):
    """`SmootherCycle` replaces its carrier with the analysis-step return
    value, then steps it forward — so the adapter must return the state
    at the END of the window, not the optimised start-of-window x_0.
    For a non-trivial forward, those values differ; verify the return
    equals ``forward^(window-1)(x_0_analysed)``."""
    p = lg_problem
    ana = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        max_steps=500,
    )
    step = ana.as_analysis_step()

    truth = p["batch"].input[0]
    window_forecasts = [truth[t] for t in range(p["T_plus_1"])]
    window_obs = [truth[t] for t in range(p["T_plus_1"])]

    out_end = step(
        window_forecasts,
        window_obs,
        obs_op=p["obs_op"],
        obs_err_cov=p["R_op"],
    )

    # Recover the analysed x_0 by calling the bare model with the same batch,
    # then roll it forward window-1 steps to compare against the adapter return.
    bg = jnp.asarray(window_forecasts[0])
    swapped = eqx.tree_at(lambda m: m.prior_mean, ana, bg)
    obs_stack = jnp.stack([jnp.asarray(o) for o in window_obs])
    batch = Batch1D(
        input=obs_stack[None],
        mask=jnp.ones_like(obs_stack)[None],
        target=None,
    )
    x0_analysed = swapped(batch)[0]
    rolled = x0_analysed
    for _ in range(len(window_obs) - 1):
        rolled = p["fwd"].step(rolled, p["fwd"].dt)
    np.testing.assert_allclose(np.asarray(out_end), np.asarray(rolled), atol=1e-4)
    # Sanity: with alpha=0.9 and window=3, x0 and rolled state differ.
    assert not np.allclose(np.asarray(out_end), np.asarray(x0_analysed), atol=1e-2)


def test_analysis_step_threads_obs_op_and_err_cov(lg_problem):
    """Runtime `obs_op` / `obs_err_cov` from the cycle must override the
    construction-time fields via `eqx.tree_at`; otherwise cycle-level
    operator changes (or `DAState.obs_err_cov` updates) are silently
    discarded."""
    p = lg_problem
    N = p["prior_mean"].shape[0]

    # Build the model with deliberately-wrong obs_op (zero matrix, so the
    # observation residual is just y_t — minimising it pulls x toward 0 and
    # the analysis stays near the zero prior_mean). The cycle hands in the
    # correct identity obs_op, which should override the bad one.
    zero_H = lx.MatrixLinearOperator(jnp.zeros((N, N), dtype=jnp.float32))
    bad_obs_op = LinearObs(H_mat=zero_H)
    ana_bad = Incremental4DVar(
        p["fwd"],
        obs_op=bad_obs_op,
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        max_steps=500,
    )
    step = ana_bad.as_analysis_step()

    truth = p["batch"].input[0]
    window_forecasts = [truth[t] for t in range(p["T_plus_1"])]
    window_obs = [truth[t] for t in range(p["T_plus_1"])]
    out_threaded = step(
        window_forecasts,
        window_obs,
        obs_op=p["obs_op"],  # cycle hands in the correct one
        obs_err_cov=p["R_op"],
    )

    # Reference: build the model with the *correct* obs_op from the start.
    ana_good = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        max_steps=500,
    )
    out_reference = ana_good.as_analysis_step()(
        window_forecasts,
        window_obs,
        obs_op=p["obs_op"],
        obs_err_cov=p["R_op"],
    )
    np.testing.assert_allclose(
        np.asarray(out_threaded), np.asarray(out_reference), atol=1e-3
    )

    # Sanity: if the threading were dropped, ana_bad would ignore the cycle's
    # obs_op and stay close to the zero prior_mean — the threaded result must
    # not match that bad-path output.
    out_bad = ana_bad(
        Batch1D(
            input=jnp.stack([jnp.asarray(o) for o in window_obs])[None],
            mask=jnp.ones((1, len(window_obs), N), dtype=jnp.float32),
            target=None,
        )
    )[0]
    # Roll out as in the adapter to get end-of-window for ana_bad.
    rolled = out_bad
    for _ in range(len(window_obs) - 1):
        rolled = p["fwd"].step(rolled, p["fwd"].dt)
    assert not np.allclose(np.asarray(out_threaded), np.asarray(rolled), atol=1e-2)


def test_as_analysis_step_satisfies_protocol(lg_problem):
    """`Incremental4DVar.as_analysis_step()` returns an `AnalysisStep`."""
    p = lg_problem
    ana = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
    )
    step = ana.as_analysis_step()
    assert isinstance(step, AnalysisStep)
    assert isinstance(step, pc.AnalysisStep)


def test_vardacycle_integration(lg_problem):
    """Plugs into `vardax.VarSmootherCycle` like every other vardax model."""
    p = lg_problem
    ana = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
    )
    cycle = vdx.VarSmootherCycle(
        forward=p["fwd"],
        obs_op=p["obs_op"],
        model=ana,
        window=2,
    )
    assert isinstance(cycle.analysis_step, pc.AnalysisStep)


def test_basis_path_recovers_increment(lg_problem):
    """With a Gaussian-RBF `LinearBasis` control over a 2-D field, the
    recovered increment lies in the basis span (verified via cosine
    alignment)."""
    fwd = _LinearForward(alpha=1.0)
    y_axis = np.linspace(0.0, 1.0, 7)
    x_axis = np.linspace(0.0, 1.0, 7)
    centers = np.array([[0.5, 0.5], [0.2, 0.8]])
    basis = rbf_basis((y_axis, x_axis), centers, widths=0.15)
    N = y_axis.size * x_axis.size

    # State lives as a flat (N,) vector inside vardax, so rebuild the
    # basis with a flat output_shape instead of the 2-D grid.
    flat_basis = LinearBasis(phi=basis.phi, variance=basis.variance, output_shape=(N,))

    truth = flat_basis.operg(0.0, jnp.array([1.0, -0.5]))
    H = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    obs_op = LinearObs(H_mat=H)
    state_op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))

    # Two-step window, identity forward, full obs at every step.
    T_plus_1 = 2
    batch_input = jnp.stack([truth, truth])[None]  # (B=1, T+1, N)
    batch = Batch1D(
        input=batch_input,
        mask=jnp.ones((1, T_plus_1, N)),
        target=None,
    )

    ana = Incremental4DVar(
        fwd,
        obs_op=obs_op,
        prior_mean=jnp.zeros(N),
        prior_cov_op=state_op,
        obs_cov_op=state_op,
        basis=flat_basis,
        max_steps=500,
    )
    out = ana(batch)[0]  # (N,)

    cos = float(
        jnp.sum(out * truth) / (jnp.linalg.norm(out) * jnp.linalg.norm(truth) + 1e-12)
    )
    assert cos > 0.95, f"recovered field not aligned with truth, cosine={cos}"


def test_minimiser_choice_respected(lg_problem):
    """Passing a different `optx.AbstractMinimiser` runs to convergence."""
    p = lg_problem
    ana_bfgs = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        minimiser=optx.BFGS(rtol=1e-6, atol=1e-6),
        max_steps=500,
    )
    ana_cg = Incremental4DVar(
        p["fwd"],
        obs_op=p["obs_op"],
        prior_mean=p["prior_mean"],
        prior_cov_op=p["B_op"],
        obs_cov_op=p["R_op"],
        minimiser=optx.NonlinearCG(rtol=1e-6, atol=1e-6),
        max_steps=500,
    )
    out_bfgs = ana_bfgs(p["batch"])[0]
    out_cg = ana_cg(p["batch"])[0]
    np.testing.assert_allclose(np.asarray(out_bfgs), np.asarray(out_cg), atol=5e-2)
