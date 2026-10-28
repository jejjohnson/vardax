"""Four-test suite (oracle, property, gradient, transforms) for the 3D/4D models."""

from __future__ import annotations

import functools

import equinox as eqx
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx

import vardax as vdx

N = 3
# Dense SPD background covariance, diagonal observation covariance, background.
B_MAT = np.array([[1.0, 0.3, 0.1], [0.3, 2.0, -0.4], [0.1, -0.4, 1.5]])
R_DIAG = np.array([0.5, 1.0, 2.0])
XB = np.array([0.2, -0.5, 1.0])
DECAY = 0.7  # linear dynamics M(x) = DECAY * x for the 4D-Var window

_floats = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
_vec = hnp.arrays(np.float64, (N,), elements=_floats)
_mask = st.lists(st.sampled_from([0.0, 1.0]), min_size=N, max_size=N)


def _psd(matvec, shape):
    return lx.FunctionLinearOperator(
        matvec, jax.ShapeDtypeStruct(shape, jnp.float64), lx.positive_semidefinite_tag
    )


@functools.cache
def _grid_ops(shape):
    """B (dense) and R (diagonal) as operators on states of ``shape``.

    Cached so the operators' closures keep one identity: they are static
    pytree data, and a fresh closure per model would recompile every call.
    """
    b = jnp.asarray(B_MAT)
    r = jnp.asarray(R_DIAG)
    return _psd(lambda z: z @ b, shape), _psd(lambda z: z * r, shape)


def _blue(gain_obs, x_b, y, b_mat, r_mat):
    """Dense BLUE x_b + B Gᵀ (G B Gᵀ + R)⁻¹ (y − G x_b), the independent oracle."""
    s = gain_obs @ b_mat @ gain_obs.T + r_mat
    return x_b + b_mat @ gain_obs.T @ np.linalg.solve(s, y - gain_obs @ x_b)


def _central_jacobian(fn, y, eps=1e-6):
    """Central-difference Jacobian of a vector function, columns per input."""
    y = np.asarray(y, dtype=np.float64)
    cols = []
    for i in range(y.size):
        step = np.zeros_like(y)
        step[i] = eps
        cols.append((np.asarray(fn(y + step)) - np.asarray(fn(y - step))) / (2 * eps))
    return np.stack(cols, axis=-1)


@eqx.filter_jit
def _run(model, batch):
    """Call ``model`` under one cached jit: array leaves do not recompile."""
    return model(batch)


def _batch(y, mask):
    """A ``Batch1D`` of ``(B, T, N)`` observations."""
    return vdx.Batch1D(input=jnp.asarray(y), mask=jnp.asarray(mask))


# --------------------------------------------------------------------------- #
# OptimalInterpolation
# --------------------------------------------------------------------------- #


def _oi(x_b, solver=None):
    b_op, r_op = _grid_ops((1, N))
    return vdx.OptimalInterpolation(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.asarray(x_b)[None],
        prior_cov_op=b_op,
        obs_cov_op=r_op,
        solver=solver if solver is not None else lx.Cholesky(),
    )


class TestOptimalInterpolation:
    def test_closed_form_oracle(self, x64):
        # Cholesky is a direct solve, so the BLUE is reproduced to rounding.
        y = np.array([[0.9, -1.2, 0.4], [2.0, 0.3, -0.7]])
        m = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        out = _run(_oi(XB), _batch(y[:, None], m[:, None]))
        for k in range(2):
            h = np.diag(m[k])
            expected = _blue(h, XB, m[k] * y[k], B_MAT, np.diag(R_DIAG))
            np.testing.assert_allclose(out[k, 0], expected, rtol=1e-12)

    @settings(max_examples=10, deadline=None)
    @given(x_b=_vec, y=_vec, mask=_mask, scale=st.floats(0.5, 3.0))
    def test_property_no_obs_background_and_linearity(self, x_b, y, mask, scale):
        with jax.enable_x64(True):
            m = np.asarray(mask)
            # (a) with every observation masked the analysis is the background
            none = _run(_oi(x_b), _batch(y[None, None], np.zeros((1, 1, N))))
            np.testing.assert_allclose(none[0, 0], x_b, rtol=1e-12, atol=1e-12)
            # (b) the analysis is linear in (x_b, y) for a fixed mask
            full = _run(_oi(x_b), _batch(y[None, None], m[None, None]))[0, 0]
            scaled = _run(
                _oi(scale * x_b), _batch(scale * y[None, None], m[None, None])
            )
            np.testing.assert_allclose(scaled[0, 0], scale * full, rtol=1e-9, atol=1e-9)

    def test_gradient_matches_central_differences(self, x64):
        m = np.array([1.0, 0.0, 1.0])
        y = np.array([0.9, -1.2, 0.4])
        model = _oi(XB)

        def analysis(obs):
            return _run(model, _batch(obs[None, None], m[None, None]))[0, 0]

        jac = jax.jacrev(analysis)(jnp.asarray(y))
        np.testing.assert_allclose(
            jac, _central_jacobian(analysis, y), rtol=1e-6, atol=1e-9
        )
        # d xa / dy = K H = B H (H B H + R)⁻¹ H with H = diag(m)
        h = np.diag(m)
        gain = B_MAT @ h @ np.linalg.inv(h @ B_MAT @ h + np.diag(R_DIAG)) @ h
        np.testing.assert_allclose(jac, gain, rtol=1e-9, atol=1e-12)

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(3, 1, N))
        m = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=float)[:, None]
        model = _oi(XB)
        batched = _run(model, _batch(y, m))
        loop = jnp.concatenate(
            [_run(model, _batch(y[k : k + 1], m[k : k + 1])) for k in range(3)]
        )
        jitted = eqx.filter_jit(model)(_batch(y, m))
        np.testing.assert_allclose(batched, loop, rtol=1e-12)
        np.testing.assert_allclose(jitted, loop, rtol=1e-12)


# --------------------------------------------------------------------------- #
# ThreeDVar
# --------------------------------------------------------------------------- #


def _3dvar(x_b):
    b_op, r_op = _grid_ops((1, N))
    # Tight BFGS and a direct covariance solve: the cost is exactly quadratic,
    # so the analysis reaches the BLUE to the optimiser's convergence.
    return vdx.ThreeDVar(
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.asarray(x_b)[None],
        prior_cov_op=b_op,
        obs_cov_op=r_op,
        minimiser=optx.BFGS(rtol=1e-13, atol=1e-13),
        max_steps=500,
        cov_solver=lx.Cholesky(),
    )


class TestThreeDVar:
    def test_closed_form_oracle(self, x64):
        y = np.array([[0.9, -1.2, 0.4], [2.0, 0.3, -0.7]])
        m = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        out = _run(_3dvar(XB), _batch(y[:, None], m[:, None]))
        for k in range(2):
            expected = _blue(np.diag(m[k]), XB, m[k] * y[k], B_MAT, np.diag(R_DIAG))
            # Iterative minimiser: BFGS at rtol=atol=1e-13 reliably gets ~1e-9.
            np.testing.assert_allclose(out[k, 0], expected, rtol=1e-8, atol=1e-10)

    @settings(max_examples=8, deadline=None)
    @given(x_b=_vec, y=_vec, mask=_mask, scale=st.floats(0.5, 3.0))
    def test_property_no_obs_background_and_linearity(self, x_b, y, mask, scale):
        with jax.enable_x64(True):
            m = np.asarray(mask)
            none = _run(_3dvar(x_b), _batch(y[None, None], np.zeros((1, 1, N))))
            np.testing.assert_allclose(none[0, 0], x_b, atol=1e-8)
            full = _run(_3dvar(x_b), _batch(y[None, None], m[None, None]))[0, 0]
            scaled = _run(
                _3dvar(scale * x_b), _batch(scale * y[None, None], m[None, None])
            )
            np.testing.assert_allclose(scaled[0, 0], scale * full, rtol=1e-6, atol=1e-7)

    def test_gradient_matches_central_differences(self, x64):
        m = np.array([1.0, 0.0, 1.0])
        y = np.array([0.9, -1.2, 0.4])
        model = _3dvar(XB)

        def analysis(obs):
            return _run(model, _batch(obs[None, None], m[None, None]))[0, 0]

        # The analysis is exactly linear in y, so a wide step is exact and keeps
        # the BFGS convergence noise (~1e-9) out of the difference quotient.
        jac = jax.jacrev(analysis)(jnp.asarray(y))
        np.testing.assert_allclose(
            jac, _central_jacobian(analysis, y, eps=1e-3), rtol=1e-6, atol=1e-7
        )
        h = np.diag(m)
        gain = B_MAT @ h @ np.linalg.inv(h @ B_MAT @ h + np.diag(R_DIAG)) @ h
        # implicit adjoint at the optimiser's tolerance, not machine precision
        np.testing.assert_allclose(jac, gain, rtol=1e-6, atol=1e-8)

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(3, 1, N))
        m = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=float)[:, None]
        model = _3dvar(XB)
        batched = _run(model, _batch(y, m))
        loop = jnp.concatenate(
            [_run(model, _batch(y[k : k + 1], m[k : k + 1])) for k in range(3)]
        )
        jitted = eqx.filter_jit(model)(_batch(y, m))
        np.testing.assert_allclose(batched, loop, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(jitted, loop, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# IncrementalFourDVar
# --------------------------------------------------------------------------- #


class _Decay(eqx.Module):
    """Linear dynamics ``M(x) = a x``, so the 4D-Var problem is linear-Gaussian."""

    a: float = DECAY

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return self.a * state


def _inc(x_b, *, cvt):
    b_op, r_op = _grid_ops((N,))
    # Linear problem: one Gauss-Newton step is exact if the inner CG converges,
    # and the covariance CG solves are tightened from their 1e-6 default.
    config = vdx.IncrementalConfig(
        n_outer=1,
        n_inner=100,
        cg_atol=1e-14,
        cg_rtol=1e-14,
        cvt=cvt,
        covariance_tol=1e-14,
    )
    return vdx.IncrementalFourDVar(
        forward=_Decay(),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.asarray(x_b),
        prior_cov_op=b_op,
        obs_cov_op=r_op,
        config=config,
    )


def _stack_obs(m):
    """G = [diag(m_0); a diag(m_1)] for observations at t = 0 and t = 1."""
    return np.vstack([np.diag(m[0]), DECAY * np.diag(m[1])])


class TestIncrementalFourDVar:
    def test_closed_form_oracle(self, x64):
        y = np.array([[0.9, -1.2, 0.4], [0.5, 0.2, -0.6]])
        m = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        r_stack = np.diag(np.tile(R_DIAG, 2))
        expected = _blue(_stack_obs(m), XB, (m * y).reshape(-1), B_MAT, r_stack)
        for cvt in (True, False):
            out = _run(_inc(XB, cvt=cvt), _batch(y[None], m[None]))
            # Truncated-CG GN on a 3-dim quadratic: CG hits ~1e-13 residuals.
            np.testing.assert_allclose(out[0], expected, rtol=1e-8, atol=1e-10)

    @settings(max_examples=6, deadline=None)
    @given(x_b=_vec, y=_vec, mask=_mask, scale=st.floats(0.5, 3.0))
    def test_property_no_obs_background_and_linearity(self, x_b, y, mask, scale):
        with jax.enable_x64(True):
            yy = np.stack([y, 0.5 * y])[None]
            m = np.stack([mask, mask]).astype(float)[None]
            none = _run(_inc(x_b, cvt=True), _batch(yy, np.zeros_like(m)))
            np.testing.assert_allclose(none[0], x_b, atol=1e-10)
            full = _run(_inc(x_b, cvt=True), _batch(yy, m))[0]
            scaled = _run(_inc(scale * x_b, cvt=True), _batch(scale * yy, m))
            np.testing.assert_allclose(scaled[0], scale * full, rtol=1e-7, atol=1e-8)

    def test_gradient_matches_central_differences(self, x64):
        m = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        y0 = np.array([[0.9, -1.2, 0.4], [0.5, 0.2, -0.6]])
        model = _inc(XB, cvt=True)

        def analysis(flat):
            return _run(model, _batch(flat.reshape(1, 2, N), m[None]))[0]

        jac = jax.jacrev(analysis)(jnp.asarray(y0.reshape(-1)))
        np.testing.assert_allclose(
            jac, _central_jacobian(analysis, y0.reshape(-1)), rtol=1e-6, atol=1e-8
        )
        # d xa / dy = B Gᵀ (G B Gᵀ + R)⁻¹ diag(m), G = stacked H M^t
        g = _stack_obs(m)
        s = g @ B_MAT @ g.T + np.diag(np.tile(R_DIAG, 2))
        gain = B_MAT @ g.T @ np.linalg.inv(s) @ np.diag(m.reshape(-1))
        np.testing.assert_allclose(jac, gain, rtol=1e-7, atol=1e-9)

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(2)
        y = rng.normal(size=(3, 2, N))
        m = np.ones((3, 2, N))
        m[1, :, 1] = 0.0
        m[2, 1] = 0.0
        model = _inc(XB, cvt=True)
        batched = _run(model, _batch(y, m))
        loop = jnp.concatenate(
            [_run(model, _batch(y[k : k + 1], m[k : k + 1])) for k in range(3)]
        )
        jitted = eqx.filter_jit(model)(_batch(y, m))
        np.testing.assert_allclose(batched, loop, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(jitted, loop, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# SpaceTimeFieldVar
# --------------------------------------------------------------------------- #

COORDS = np.array([[0.0, -1.0], [0.2, -0.3], [0.5, 0.4], [0.7, 1.0], [1.0, 2.0]])
SIGMA = np.array([0.5, 1.0, 0.8, 1.2, 0.6])
VALUES = np.array([-0.8, 0.1, 0.9, 1.3, 2.4])
ALPHA = 0.3


class _Line(eqx.Module):
    """f_w(t, x) = w0 + w1 x: linear in its parameters, so the fit is ridge."""

    w: jax.Array

    def __call__(self, s):
        return (self.w[0] + self.w[1] * s[1])[None]


def _obs(values, valid=None):
    return vdx.PointObsBatch(
        coords=jnp.asarray(COORDS),
        values=jnp.asarray(values),
        sigma=jnp.asarray(SIGMA),
        valid=jnp.ones(len(COORDS), bool) if valid is None else jnp.asarray(valid),
    )


def _field_var(alpha=ALPHA):
    # Tight BFGS: the cost is exactly quadratic in w.
    return vdx.SpaceTimeFieldVar(
        field=_Line(jnp.zeros(2)),
        weight_decay=alpha,
        minimiser=vdx.OptimistixMinimiser(
            optx.BFGS(rtol=1e-13, atol=1e-13), max_steps=500
        ),
    )


@eqx.filter_jit
def _fit_w(model, obs):
    return model(obs).field.w


def _ridge(values, alpha, valid=None):
    """w = (Aᵀ W A + α I)⁻¹ Aᵀ W y with W = diag(valid / σ²)."""
    a = np.stack([np.ones(len(COORDS)), COORDS[:, 1]], axis=1)
    keep = np.ones(len(COORDS)) if valid is None else np.asarray(valid, float)
    w = np.diag(keep / SIGMA**2)
    return np.linalg.solve(a.T @ w @ a + alpha * np.eye(2), a.T @ w @ values)


class TestSpaceTimeFieldVar:
    def test_closed_form_oracle(self, x64):
        # The family contains the ridge solution, so the fit must equal it;
        # BFGS at 1e-13 reaches it to about 1e-9 relative.
        result = _field_var()(_obs(VALUES))
        np.testing.assert_allclose(
            result.field.w, _ridge(VALUES, ALPHA), rtol=1e-8, atol=1e-10
        )

    @settings(max_examples=8, deadline=None)
    @given(
        alpha=st.floats(0.05, 5.0),
        factor=st.floats(1.5, 10.0),
        values=hnp.arrays(np.float64, (5,), elements=_floats),
    )
    def test_property_weight_decay_shrinks_parameters(self, alpha, factor, values):
        with jax.enable_x64(True):
            low = _field_var(alpha)(_obs(values)).field.w
            high = _field_var(alpha * factor)(_obs(values)).field.w
            # ridge parameter norm is non-increasing in the penalty weight
            assert float(jnp.linalg.norm(high)) <= float(jnp.linalg.norm(low)) + 1e-9
            # and padding rows (any value, NaN included) change nothing
            valid = np.array([True, True, False, True, True])
            junk = values.copy()
            junk[2] = np.nan
            masked = _field_var(alpha)(_obs(junk, valid)).field.w
            np.testing.assert_allclose(
                masked, _ridge(np.where(valid, values, 0.0), alpha, valid), atol=1e-7
            )

    def test_gradient_matches_central_differences(self, x64):
        model = _field_var()

        def fitted(values):
            return _fit_w(model, _obs(values))

        # w is exactly linear in y: a wide step is exact and hides BFGS noise
        jac = jax.jacrev(fitted)(jnp.asarray(VALUES))
        np.testing.assert_allclose(
            jac, _central_jacobian(fitted, VALUES, eps=1e-3), rtol=1e-6, atol=1e-8
        )
        # d w / d y = (Aᵀ W A + α I)⁻¹ Aᵀ W
        a = np.stack([np.ones(5), COORDS[:, 1]], axis=1)
        w = np.diag(1.0 / SIGMA**2)
        expected = np.linalg.solve(a.T @ w @ a + ALPHA * np.eye(2), a.T @ w)
        np.testing.assert_allclose(jac, expected, rtol=1e-6, atol=1e-8)

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(3)
        values = VALUES + 0.3 * rng.normal(size=(3, 5))
        model = _field_var()
        loop = jnp.stack([model(_obs(v)).field.w for v in values])

        via_filter_vmap = eqx.filter_vmap(lambda v: _fit_w(model, _obs(v)))(
            jnp.asarray(values)
        )
        via_jit = eqx.filter_jit(lambda v: model(_obs(v)).field.w)
        via_jax_vmap = jax.vmap(via_jit)(jnp.asarray(values))
        np.testing.assert_allclose(via_filter_vmap, loop, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(via_jax_vmap, loop, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(via_jit(jnp.asarray(values[0])), loop[0], rtol=1e-10)
