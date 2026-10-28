"""Four-test suite for the 4D-Var forwards on a small linear-Gaussian window."""

from __future__ import annotations

import equinox as eqx
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest

import vardax as vdx

N = 3
T = 2  # rollout steps, so the window holds T + 1 = 3 times

M_NP = np.array([[0.9, 0.2, 0.0], [-0.1, 0.8, 0.1], [0.0, 0.3, 0.7]])
B_NP = np.array([[1.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 0.8]])
R_NP = np.array([[0.5, 0.1, 0.0], [0.1, 0.7, 0.0], [0.0, 0.0, 0.4]])
Q_NP = np.array([[0.2, 0.05, 0.0], [0.05, 0.3, 0.0], [0.0, 0.0, 0.25]])
PHI_NP = np.array([[1.0, 0.0], [0.5, 1.0], [-0.3, 0.7]])
PHI_VAR_NP = np.array([1.5, 0.6])
XB_NP = np.array([0.4, -0.7, 1.1])
Y_NP = np.array([[0.5, -0.4, 0.9], [0.1, -0.2, 1.3], [0.3, 0.6, 0.2]])

# A tight solver pair: with a direct covariance solve the only error left
# is BFGS stopping on a quadratic, which reaches about 1e-9 in float64.
TOL = 1e-9


class _Linear(eqx.Module):
    """``x_{t+1} = M x_t`` as a pipekit-style forward."""

    mat: jax.Array

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def state_signature(self):
        return None

    def step(self, state, dt):
        return self.mat @ state


def _spd(mat):
    return lx.MatrixLinearOperator(jnp.asarray(mat), lx.positive_semidefinite_tag)


def _kwargs():
    return dict(
        forward=_Linear(jnp.asarray(M_NP)),
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.asarray(XB_NP),
        prior_cov_op=_spd(B_NP),
        obs_cov_op=_spd(R_NP),
        minimiser=optx.BFGS(rtol=1e-12, atol=1e-12),
        cov_solver=lx.Cholesky(),
        max_steps=500,
    )


def _strong():
    return vdx.StrongFourDVar(**_kwargs())


def _weak():
    return vdx.WeakFourDVar(**_kwargs(), model_err_cov_op=_spd(Q_NP))


def _reduced():
    basis = vdx.linear_basis(jnp.asarray(PHI_NP), variance=jnp.asarray(PHI_VAR_NP))
    return vdx.ReducedBasisFourDVar(**_kwargs(), basis=basis)


def _batch(y, mask=None):
    """Batch1D of ``(B, T + 1, N)`` observations, fully observed by default."""
    y = jnp.asarray(y)
    mask = jnp.ones_like(y) if mask is None else jnp.asarray(mask)
    return vdx.Batch1D(input=y, mask=mask, target=None)


def _blue(G, P, Rb, zb, y):
    """Dense BLUE ``z_a = z_b + K (y - G z_b)`` with ``K = P Gᵀ (G P Gᵀ + R)⁻¹``."""
    S = G @ P @ G.T + Rb
    K = P @ G.T @ np.linalg.inv(S)
    return zb + K @ (y - G @ zb), K


def _strong_G():
    # Block t maps x_0 to M^t x_0, the state observed at time t.
    return np.concatenate([np.linalg.matrix_power(M_NP, t) for t in range(T + 1)])


def _weak_G():
    # z = (x_0, eta_1..eta_T); x_t = M^t x_0 + sum_{s<=t} M^(t-s) eta_s.
    G = np.zeros(((T + 1) * N, (T + 1) * N))
    for t in range(T + 1):
        G[t * N : (t + 1) * N, :N] = np.linalg.matrix_power(M_NP, t)
        for s in range(1, t + 1):
            G[t * N : (t + 1) * N, s * N : (s + 1) * N] = np.linalg.matrix_power(
                M_NP, t - s
            )
    return G


def _weak_P():
    P = np.zeros(((T + 1) * N, (T + 1) * N))
    P[:N, :N] = B_NP
    for s in range(1, T + 1):
        P[s * N : (s + 1) * N, s * N : (s + 1) * N] = Q_NP
    return P


def _Rb():
    return np.kron(np.eye(T + 1), R_NP)


def _strong_oracle(xb, y):
    za, K = _blue(_strong_G(), B_NP, _Rb(), xb, y.reshape(-1))
    return za, K.reshape(N, T + 1, N)


def _weak_oracle(xb, y):
    zb = np.concatenate([xb, np.zeros(T * N)])
    za, K = _blue(_weak_G(), _weak_P(), _Rb(), zb, y.reshape(-1))
    return za[:N], za[N:].reshape(T, N), K.reshape(T + 1, N, T + 1, N)


def _reduced_oracle(xb, y):
    # Coefficient prior diag(variance); the state increment is Phi X.
    G = _strong_G() @ PHI_NP
    Xa, K = _blue(
        G, np.diag(PHI_VAR_NP), _Rb(), np.zeros(2), y.reshape(-1) - _strong_G() @ xb
    )
    return xb + PHI_NP @ Xa, (PHI_NP @ K).reshape(N, T + 1, N)


def _central_diff(f, y, h=1e-2):
    """Central differences of ``f`` (flattened output) w.r.t. every entry of y.

    The analysis is linear in y, so a large step costs no truncation error;
    h=1e-6 would amplify the minimiser's ~1e-10 stopping noise to ~1e-4.
    """
    cols = []
    for i in range(y.size):
        e = np.zeros(y.size)
        e[i] = h
        e = e.reshape(y.shape)
        cols.append((np.ravel(f(y + e)) - np.ravel(f(y - e))) / (2 * h))
    return np.stack(cols, axis=-1)


def _windows(shift=0.0):
    """Three distinct windows so batching mistakes show up."""
    return np.stack([Y_NP, 0.5 * Y_NP + shift, -Y_NP + 0.1])


vecs = hnp.arrays(np.float64, (N,), elements=st.floats(-3, 3))
wins = hnp.arrays(np.float64, (1, T + 1, N), elements=st.floats(-3, 3))
coef = st.floats(-2, 2)


def _prop_settings():
    return settings(max_examples=6, deadline=None)


class TestStrongFourDVar:
    def test_closed_form_oracle(self, x64):
        # BLUE over the stacked operator H M^t; BFGS(1e-12) + Cholesky reach TOL.
        want, _ = _strong_oracle(XB_NP, Y_NP)
        got = _strong()(_batch(Y_NP[None]))
        assert got.shape == (1, N)
        np.testing.assert_allclose(np.asarray(got[0]), want, rtol=TOL, atol=TOL)

    @pytest.mark.slow
    @_prop_settings()
    @given(xb1=vecs, xb2=vecs, y1=wins, y2=wins, a=coef, b=coef)
    def test_property_affine_homogeneous_linearity(self, xb1, xb2, y1, y2, a, b):
        # A linear-Gaussian analysis is linear in (x_b, y) jointly.
        with jax.enable_x64(True):
            model = _strong()

            @eqx.filter_jit
            def run(xb, y):
                m = eqx.tree_at(lambda s: s.prior_mean, model, xb)
                return m(_batch(y))[0]

            lhs = run(jnp.asarray(a * xb1 + b * xb2), jnp.asarray(a * y1 + b * y2))
            rhs = a * run(jnp.asarray(xb1), jnp.asarray(y1)) + b * run(
                jnp.asarray(xb2), jnp.asarray(y2)
            )
            np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-7)

    @pytest.mark.slow
    @_prop_settings()
    @given(xb=vecs, y=wins)
    def test_property_no_observations_returns_background(self, xb, y):
        with jax.enable_x64(True):
            m = eqx.tree_at(lambda s: s.prior_mean, _strong(), jnp.asarray(xb))
            out = m(_batch(y, mask=np.zeros_like(y)))
            np.testing.assert_allclose(np.asarray(out[0]), xb, atol=1e-9)

    @pytest.mark.slow
    def test_gradient_matches_central_differences(self, x64):
        model = _strong()

        @eqx.filter_jit
        def f(y):
            return model(_batch(y[None]))[0]

        jac = jax.jacrev(f)(jnp.asarray(Y_NP))  # (N, T+1, N)
        fd = _central_diff(lambda y: np.asarray(f(jnp.asarray(y))), Y_NP)
        np.testing.assert_allclose(
            np.asarray(jac).reshape(N, -1), fd, rtol=1e-6, atol=1e-6
        )
        # For a linear-Gaussian problem the Jacobian is the gain K.
        _, K = _strong_oracle(XB_NP, Y_NP)
        np.testing.assert_allclose(np.asarray(jac), K, rtol=1e-6, atol=1e-6)

    @pytest.mark.slow
    def test_jit_vmap_match_loop(self, x64):
        model = _strong()
        ys = jnp.asarray(_windows())
        loop = jnp.stack([model(_batch(w[None]))[0] for w in ys])
        eager = model(_batch(ys))
        jitted = eqx.filter_jit(model)(_batch(ys))
        vmapped = jax.vmap(lambda w: model(_batch(w[None]))[0])(ys)
        for out in (eager, jitted, vmapped):
            np.testing.assert_allclose(np.asarray(out), np.asarray(loop), atol=1e-9)


class TestWeakFourDVar:
    def test_closed_form_oracle(self, x64):
        # Augmented BLUE over (x_0, eta_1..eta_T) with prior blockdiag(B, Q, Q).
        x0, eta, _ = _weak_oracle(XB_NP, Y_NP)
        got_x0, got_eta = _weak()(_batch(Y_NP[None]))
        assert got_x0.shape == (1, N) and got_eta.shape == (1, T, N)
        # The augmented problem is worse conditioned, so BFGS stops near 1e-8.
        np.testing.assert_allclose(np.asarray(got_x0[0]), x0, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(np.asarray(got_eta[0]), eta, rtol=1e-7, atol=1e-7)

    @pytest.mark.slow
    @_prop_settings()
    @given(xb1=vecs, xb2=vecs, y1=wins, y2=wins, a=coef, b=coef)
    def test_property_affine_homogeneous_linearity(self, xb1, xb2, y1, y2, a, b):
        # Both returned blocks (x_0, eta) are linear in (x_b, y).
        with jax.enable_x64(True):
            model = _weak()

            @eqx.filter_jit
            def run(xb, y):
                m = eqx.tree_at(lambda s: s.prior_mean, model, xb)
                x0, eta = m(_batch(y))
                return x0[0], eta[0]

            lhs = run(jnp.asarray(a * xb1 + b * xb2), jnp.asarray(a * y1 + b * y2))
            r1 = run(jnp.asarray(xb1), jnp.asarray(y1))
            r2 = run(jnp.asarray(xb2), jnp.asarray(y2))
            for left, p, q in zip(lhs, r1, r2, strict=True):
                np.testing.assert_allclose(
                    np.asarray(left), np.asarray(a * p + b * q), atol=1e-7
                )

    @pytest.mark.slow
    @_prop_settings()
    @given(xb=vecs, y=wins)
    def test_property_no_observations_returns_background(self, xb, y):
        # Without data the minimum is x_0 = x_b and zero model error.
        with jax.enable_x64(True):
            m = eqx.tree_at(lambda s: s.prior_mean, _weak(), jnp.asarray(xb))
            x0, eta = m(_batch(y, mask=np.zeros_like(y)))
            np.testing.assert_allclose(np.asarray(x0[0]), xb, atol=1e-9)
            np.testing.assert_allclose(np.asarray(eta), 0.0, atol=1e-9)

    @pytest.mark.slow
    def test_gradient_matches_central_differences(self, x64):
        model = _weak()

        @eqx.filter_jit
        def f(y):
            x0, eta = model(_batch(y[None]))
            return x0[0], eta[0]

        jac_x0, jac_eta = jax.jacrev(f)(jnp.asarray(Y_NP))
        _, _, K = _weak_oracle(XB_NP, Y_NP)

        def flat(y):
            x0, eta = f(jnp.asarray(y))
            return np.concatenate([np.ravel(x0), np.ravel(eta)])

        fd = _central_diff(flat, Y_NP)
        jac = np.concatenate(
            [np.asarray(jac_x0).reshape(N, -1), np.asarray(jac_eta).reshape(T * N, -1)]
        )
        np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=1e-6)
        # Gain of the augmented system, stacked as (x_0 rows; eta rows).
        np.testing.assert_allclose(
            jac, K.reshape((T + 1) * N, -1), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.slow
    def test_jit_vmap_match_loop(self, x64):
        model = _weak()
        ys = jnp.asarray(_windows())

        def one(w):
            x0, eta = model(_batch(w[None]))
            return x0[0], eta[0]

        loop = [one(w) for w in ys]
        loop = tuple(jnp.stack(parts) for parts in zip(*loop, strict=True))
        eager = model(_batch(ys))
        jitted = eqx.filter_jit(model)(_batch(ys))
        vmapped = jax.vmap(one)(ys)
        for out in (eager, jitted, vmapped):
            for got, want in zip(out, loop, strict=True):
                np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-9)


class TestReducedBasisFourDVar:
    def test_closed_form_oracle(self, x64):
        # BLUE in coefficient space with prior diag(variance) and G = H M^t Phi.
        want, _ = _reduced_oracle(XB_NP, Y_NP)
        got = _reduced()(_batch(Y_NP[None]))
        assert got.shape == (1, N)
        np.testing.assert_allclose(np.asarray(got[0]), want, rtol=TOL, atol=TOL)

    def test_closed_form_square_basis_equals_state_space_blue(self, x64):
        # An invertible square Phi with coefficient prior Q means B = Phi Q Phi^T.
        phi = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, -0.3], [0.1, 0.0, 1.0]])
        var = np.array([1.0, 2.0, 0.5])
        kwargs = _kwargs()
        model = vdx.ReducedBasisFourDVar(
            **kwargs,
            basis=vdx.linear_basis(jnp.asarray(phi), variance=jnp.asarray(var)),
        )
        B = phi @ np.diag(var) @ phi.T
        want, _ = _blue(_strong_G(), B, _Rb(), XB_NP, Y_NP.reshape(-1))
        got = model(_batch(Y_NP[None]))
        np.testing.assert_allclose(np.asarray(got[0]), want, rtol=TOL, atol=TOL)

    @pytest.mark.slow
    @_prop_settings()
    @given(xb1=vecs, xb2=vecs, y1=wins, y2=wins, a=coef, b=coef)
    def test_property_affine_homogeneous_linearity(self, xb1, xb2, y1, y2, a, b):
        with jax.enable_x64(True):
            model = _reduced()

            @eqx.filter_jit
            def run(xb, y):
                m = eqx.tree_at(lambda s: s.prior_mean, model, xb)
                return m(_batch(y))[0]

            lhs = run(jnp.asarray(a * xb1 + b * xb2), jnp.asarray(a * y1 + b * y2))
            rhs = a * run(jnp.asarray(xb1), jnp.asarray(y1)) + b * run(
                jnp.asarray(xb2), jnp.asarray(y2)
            )
            np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-7)

    @pytest.mark.slow
    @_prop_settings()
    @given(xb=vecs, y=wins)
    def test_property_increment_in_basis_span(self, xb, y):
        # x_a - x_b = Phi X, so it has no component outside span(Phi).
        with jax.enable_x64(True):
            m = eqx.tree_at(lambda s: s.prior_mean, _reduced(), jnp.asarray(xb))
            inc = np.asarray(m(_batch(y))[0]) - xb
            resid = inc - PHI_NP @ np.linalg.lstsq(PHI_NP, inc, rcond=None)[0]
            np.testing.assert_allclose(resid, 0.0, atol=1e-8)

    @pytest.mark.slow
    @_prop_settings()
    @given(xb=vecs, y=wins)
    def test_property_no_observations_returns_background(self, xb, y):
        with jax.enable_x64(True):
            m = eqx.tree_at(lambda s: s.prior_mean, _reduced(), jnp.asarray(xb))
            out = m(_batch(y, mask=np.zeros_like(y)))
            np.testing.assert_allclose(np.asarray(out[0]), xb, atol=1e-9)

    @pytest.mark.slow
    def test_gradient_matches_central_differences(self, x64):
        model = _reduced()

        @eqx.filter_jit
        def f(y):
            return model(_batch(y[None]))[0]

        jac = jax.jacrev(f)(jnp.asarray(Y_NP))
        fd = _central_diff(lambda y: np.asarray(f(jnp.asarray(y))), Y_NP)
        np.testing.assert_allclose(
            np.asarray(jac).reshape(N, -1), fd, rtol=1e-6, atol=1e-6
        )
        _, K = _reduced_oracle(XB_NP, Y_NP)
        np.testing.assert_allclose(np.asarray(jac), K, rtol=1e-6, atol=1e-6)

    @pytest.mark.slow
    def test_jit_vmap_match_loop(self, x64):
        model = _reduced()
        ys = jnp.asarray(_windows())
        loop = jnp.stack([model(_batch(w[None]))[0] for w in ys])
        eager = model(_batch(ys))
        jitted = eqx.filter_jit(model)(_batch(ys))
        vmapped = jax.vmap(lambda w: model(_batch(w[None]))[0])(ys)
        for out in (eager, jitted, vmapped):
            np.testing.assert_allclose(np.asarray(out), np.asarray(loop), atol=1e-9)
