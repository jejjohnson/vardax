"""The OI and 3D-Var ``AnalysisStep`` adapters analyse around the forecast.

``pipekit_cycle.AnalysisStep`` is called as
``step(forecast, obs, obs_op=H, obs_err_cov=R)``. The adapters used to
ignore ``forecast``, ``obs_op`` and ``obs_err_cov`` and analyse around
the model's construction-time ``prior_mean`` with its own ``H`` and
``R``, so a ``DACycle`` never fed one cycle's forecast into the next
analysis. These tests pin the contract:

- in a linear-Gaussian problem the OI adapter returns the closed-form
  BLUE $x_b + B H^T (H B H^T + R)^{-1} (y - H x_b)$ with $x_b$ the
  forecast, $H$ the cycle's ``obs_op`` and $R$ the cycle's
  ``obs_err_cov`` (none of them the model's own);
- 3D-Var converges to the same analysis;
- a two-cycle ``VarDACycle`` threads the first cycle's analysis, through
  the forecast, into the second analysis.

The closed-form checks run in float64 (``jax.enable_x64``, scoped to the
test so the rest of the suite stays in float32).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from pipekit import Lambda
import pipekit_cycle as pc
import pytest

from vardax import (
    LinearObs,
    MaskedIdentity,
    OptimalInterpolation,
    ThreeDVar,
    VarDACycle,
)

N, M = 5, 3  # state and observation sizes


def _psd(mat: jax.Array) -> lx.AbstractLinearOperator:
    return lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)


def _problem() -> dict:
    """A linear-Gaussian problem whose model fields differ from the cycle's.

    Call inside ``jax.enable_x64(True)`` for float64 arrays.
    """
    k = jax.random.split(jax.random.PRNGKey(0), 6)
    a = jax.random.normal(k[0], (N, N))
    B = a @ a.T / N + 0.5 * jnp.eye(N)  # SPD background covariance
    H = jax.random.normal(k[1], (M, N))  # the cycle's H
    R = jnp.diag(jnp.array([0.3, 0.5, 0.8]))  # the cycle's R
    return {
        "B": B,
        "H": H,
        "R": R,
        # The model's own H, R and background: deliberately different
        # from the cycle's, so using them would give a different answer.
        "model_H": jax.random.normal(k[2], (M, N)),
        "model_R": 2.0 * jnp.eye(M),
        "prior_mean": jnp.zeros(N),
        "forecast": jax.random.normal(k[3], (N,)) + 1.0,
        "y": jax.random.normal(k[4], (M,)),
    }


def _blue(B, H, R, x_b, y):
    """Closed-form BLUE: x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)."""
    return x_b + B @ H.T @ jnp.linalg.solve(H @ B @ H.T + R, y - H @ x_b)


def _oi(p: dict) -> OptimalInterpolation:
    return OptimalInterpolation(
        obs_op=LinearObs(H_mat=lx.MatrixLinearOperator(p["model_H"])),
        prior_mean=p["prior_mean"],
        prior_cov_op=_psd(p["B"]),
        obs_cov_op=_psd(p["model_R"]),
        # Tight CG tolerances: the comparison is with a dense solve.
        cg_atol=1e-12,
        cg_rtol=1e-12,
    )


def _threedvar(p: dict) -> ThreeDVar:
    import optimistix as optx

    return ThreeDVar(
        obs_op=LinearObs(H_mat=lx.MatrixLinearOperator(p["model_H"])),
        prior_mean=p["prior_mean"],
        prior_cov_op=_psd(p["B"]),
        obs_cov_op=_psd(p["model_R"]),
        minimiser=optx.BFGS(rtol=1e-10, atol=1e-10),
        max_steps=500,
    )


def _call(step, p: dict, forecast, obs, *, obs_err_cov="R"):
    # filter_jit compiles the analysis once instead of op by op.
    return eqx.filter_jit(step)(
        forecast,
        obs,
        obs_op=LinearObs(H_mat=lx.MatrixLinearOperator(p["H"])),
        obs_err_cov=_psd(p["R"]) if obs_err_cov == "R" else obs_err_cov,
    )


class TestOIAnalysisStep:
    @pytest.mark.slow
    def test_matches_closed_form_blue_around_forecast(self):
        with jax.enable_x64(True):
            p = _problem()
            step = _oi(p).as_analysis_step()
            xa = _call(step, p, p["forecast"], p["y"])
            expected = _blue(p["B"], p["H"], p["R"], p["forecast"], p["y"])
            assert xa.dtype == jnp.float64
            assert xa.shape == (N,)
            assert jnp.allclose(xa, expected, rtol=0.0, atol=1e-6)
            # Not the analysis around the model's prior_mean.
            wrong = _blue(p["B"], p["H"], p["R"], p["prior_mean"], p["y"])
            assert not jnp.allclose(xa, wrong, atol=1e-3)

    def test_variances_as_obs_err_cov(self):
        """An array of error variances is a diagonal R."""
        with jax.enable_x64(True):
            p = _problem()
            step = _oi(p).as_analysis_step()
            xa = _call(step, p, p["forecast"], p["y"], obs_err_cov=jnp.diag(p["R"]))
            expected = _blue(p["B"], p["H"], p["R"], p["forecast"], p["y"])
            assert jnp.allclose(xa, expected, rtol=0.0, atol=1e-6)

    def test_none_falls_back_to_model_obs_op_and_cov(self):
        with jax.enable_x64(True):
            p = _problem()
            step = _oi(p).as_analysis_step()
            xa = step(p["forecast"], p["y"], obs_op=None, obs_err_cov=None)
            expected = _blue(p["B"], p["model_H"], p["model_R"], p["forecast"], p["y"])
            assert jnp.allclose(xa, expected, rtol=0.0, atol=1e-6)

    def test_batched_forecasts_each_use_their_own_background(self):
        with jax.enable_x64(True):
            p = _problem()
            step = _oi(p).as_analysis_step()
            forecasts = jnp.stack([p["forecast"], -p["forecast"]])
            ys = jnp.stack([p["y"], 2.0 * p["y"]])
            xa = _call(step, p, forecasts, ys)
            assert xa.shape == (2, N)
            for i in range(2):
                expected = _blue(p["B"], p["H"], p["R"], forecasts[i], ys[i])
                assert jnp.allclose(xa[i], expected, rtol=0.0, atol=1e-6)

    def test_changing_the_forecast_changes_the_analysis(self):
        """Regression: the adapter used to ignore ``forecast`` entirely."""
        p = _problem()
        step = _oi(p).as_analysis_step()
        xa_1 = _call(step, p, p["forecast"], p["y"])
        xa_2 = _call(step, p, p["forecast"] + 1.0, p["y"])
        assert not jnp.allclose(xa_1, xa_2, atol=1e-3)

    def test_missing_observations_are_masked(self):
        """B = R = I, H = I: observed x_a = (x_b + y) / 2, missing x_a = x_b."""
        eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1, 4), jnp.float32))
        oi = OptimalInterpolation(
            obs_op=MaskedIdentity(),
            prior_mean=jnp.zeros((1, 4)),
            prior_cov_op=eye,
            obs_cov_op=eye,
        )
        x_b = jnp.arange(4.0).reshape(1, 4) + 1.0
        y = jnp.array([[3.0, jnp.nan, -1.0, jnp.nan]])
        xa = oi.as_analysis_step()(x_b, y, obs_op=None, obs_err_cov=None)
        expected = jnp.array([[2.0, 2.0, 1.0, 4.0]])
        assert jnp.allclose(xa, expected, atol=1e-5)

    def test_standalone_call_unchanged(self):
        """``model(batch)`` still analyses around ``prior_mean``."""
        from vardax import Batch1D

        with jax.enable_x64(True):
            p = _problem()
            oi = _oi(p)
            batch = Batch1D(input=p["y"][None], mask=jnp.ones((1, M)))
            xa = oi(batch)[0]
            expected = _blue(
                p["B"], p["model_H"], p["model_R"], p["prior_mean"], p["y"]
            )
            assert jnp.allclose(xa, expected, rtol=0.0, atol=1e-6)

    def test_rejects_forecast_of_the_wrong_shape(self):
        p = _problem()
        step = _oi(p).as_analysis_step()
        with pytest.raises(ValueError, match="forecast must be one state"):
            _call(step, p, jnp.zeros(N + 1), p["y"])

    def test_rejects_obs_err_cov_of_the_wrong_shape(self):
        p = _problem()
        step = _oi(p).as_analysis_step()
        with pytest.raises(ValueError, match="obs_err_cov acts on shape"):
            _call(step, p, p["forecast"], p["y"], obs_err_cov=_psd(jnp.eye(M + 1)))
        with pytest.raises(ValueError, match="do not broadcast"):
            _call(step, p, p["forecast"], p["y"], obs_err_cov=jnp.ones(M + 1))


class TestThreeDVarAnalysisStep:
    def test_converges_to_closed_form_blue_around_forecast(self):
        # BFGS at rtol = atol = 1e-10 on a 5-dimensional quadratic; the
        # B⁻¹ / R⁻¹ CG solves inside the cost are exact for 5-by-5 / 3-by-3
        # systems up to rounding. The converged error is ~4e-10 (OI: ~2e-16);
        # 1e-6 leaves margin for other BLAS / platforms.
        with jax.enable_x64(True):
            p = _problem()
            step = _threedvar(p).as_analysis_step()
            xa = _call(step, p, p["forecast"], p["y"])
            expected = _blue(p["B"], p["H"], p["R"], p["forecast"], p["y"])
            assert xa.dtype == jnp.float64
            assert jnp.allclose(xa, expected, rtol=0.0, atol=1e-6)
            wrong = _blue(p["B"], p["H"], p["R"], p["prior_mean"], p["y"])
            assert not jnp.allclose(xa, wrong, atol=1e-3)

    def test_changing_the_forecast_changes_the_analysis(self):
        """Regression: the adapter used to ignore ``forecast`` entirely."""
        p = _problem()
        step = _threedvar(p).as_analysis_step()
        xa_1 = _call(step, p, p["forecast"], p["y"])
        xa_2 = _call(step, p, p["forecast"] + 1.0, p["y"])
        assert not jnp.allclose(xa_1, xa_2, atol=1e-3)

    def test_agrees_with_oi_adapter(self):
        with jax.enable_x64(True):
            p = _problem()
            xa_3dvar = _call(_threedvar(p).as_analysis_step(), p, p["forecast"], p["y"])
            xa_oi = _call(_oi(p).as_analysis_step(), p, p["forecast"], p["y"])
            assert jnp.allclose(xa_3dvar, xa_oi, rtol=0.0, atol=1e-6)


class AffineForward(eqx.Module):
    """``M(x) = 0.5 x + 1``: the forecast differs from the analysis."""

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def state_signature(self):
        return None

    def step(self, state, dt):
        return 0.5 * state + 1.0


class TestTwoCycleVarDACycle:
    @pytest.mark.parametrize("make_model", [_oi, _threedvar], ids=["oi", "3dvar"])
    def test_second_analysis_uses_first_cycles_forecast(self, make_model):
        with jax.enable_x64(True):
            p = _problem()
            ys = [p["y"], -p["y"] + 0.5]
            forward = AffineForward()
            cycle = VarDACycle(
                forward=forward,
                obs_op=LinearObs(H_mat=lx.MatrixLinearOperator(p["H"])),
                model=make_model(p),
                obs_source=Lambda(lambda i: ys[i]),
                n_steps=2,
            )
            x0 = p["forecast"]
            state = pc.DAState(obs_err_cov=_psd(p["R"]))
            final, _ = cycle(x0, state)

            # Cycle 1: forecast from x0, analyse around it.
            f1 = forward.step(x0, 1.0)
            a1 = _blue(p["B"], p["H"], p["R"], f1, ys[0])
            # Cycle 2: forecast from the first analysis, analyse around it.
            f2 = forward.step(a1, 1.0)
            a2 = _blue(p["B"], p["H"], p["R"], f2, ys[1])

            (hf1, ha1), (hf2, ha2) = cycle.history
            assert jnp.allclose(hf1, f1, atol=1e-6)
            assert jnp.allclose(ha1, a1, atol=1e-6)
            assert jnp.allclose(hf2, f2, atol=1e-6)
            assert jnp.allclose(ha2, a2, atol=1e-6)
            assert jnp.allclose(final, a2, atol=1e-6)

            # The old behaviour — analyse around prior_mean every cycle —
            # gives a different second analysis.
            stale = _blue(p["B"], p["H"], p["R"], p["prior_mean"], ys[1])
            assert not jnp.allclose(final, stale, atol=1e-3)
