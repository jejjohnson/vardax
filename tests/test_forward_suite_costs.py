"""Four-test suite (oracle, property, gradient, transforms) for the cost forwards."""

from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp
import jax
import jax.numpy as jnp
import numpy as np

import vardax as vdx

SETTINGS = settings(max_examples=20, deadline=None)
_FLOATS = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)


def _arrays(shape):
    return hnp.arrays(np.float64, shape, elements=_FLOATS)


def _masks(shape):
    return hnp.arrays(np.bool_, shape).map(lambda m: m.astype(np.float64))


def _central_diff(f, x, step=1e-6):
    """Central-difference gradient of a scalar function of one array."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        up, down = x.copy(), x.copy()
        up[idx] += step
        down[idx] -= step
        grad[idx] = (float(f(jnp.asarray(up))) - float(f(jnp.asarray(down)))) / (
            2 * step
        )
    return grad


def _close_grad(actual, expected):
    # atol covers gradient entries that are zero up to cancellation error
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


class TestObsCost1d:
    shape = (2, 3, 4)

    def test_closed_form_oracle(self, x64):
        state = np.array([[[1.0, 2.0], [3.0, -1.0]]])
        obs = np.array([[[0.5, np.nan], [1.0, 2.0]]])
        mask = np.array([[[1.0, 0.0], [1.0, 1.0]]])
        # mean over all 4 elements, not over the 3 observed ones
        expected = np.mean((mask * (state - np.nan_to_num(obs))) ** 2)
        got = vdx.obs_cost_1d(
            jnp.asarray(state), jnp.asarray(obs), jnp.asarray(mask), nan_to_num=True
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    @SETTINGS
    @given(
        state=_arrays((2, 3, 4)),
        obs=_arrays((2, 3, 4)),
        other=_arrays((2, 3, 4)),
        mask=_masks((2, 3, 4)),
    )
    def test_property_nonneg_mask_invariant_and_quadratic(
        self, state, obs, other, mask
    ):
        with jax.enable_x64(True):
            # jax arrays are built here so that they are float64
            state, obs, other, mask = map(jnp.asarray, (state, obs, other, mask))
            cost = vdx.obs_cost_1d(state, obs, mask)
            assert cost >= 0
            # unobserved entries must not influence the cost
            swapped = jnp.where(mask > 0, obs, other)
            np.testing.assert_allclose(
                vdx.obs_cost_1d(state, swapped, mask), cost, rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                vdx.obs_cost_1d(2 * state, 2 * obs, mask),
                4 * cost,
                rtol=1e-12,
                atol=1e-14,
            )

    def test_gradient_matches_central_differences(self, x64):
        rng = np.random.default_rng(0)
        state = rng.normal(size=self.shape)
        obs = jnp.asarray(rng.normal(size=self.shape))
        mask = jnp.asarray((rng.uniform(size=self.shape) > 0.3).astype(float))
        f = lambda s: vdx.obs_cost_1d(s, obs, mask)
        _close_grad(jax.grad(f)(jnp.asarray(state)), _central_diff(f, state))

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(1)
        states = jnp.asarray(rng.normal(size=(3, *self.shape)))
        obs = jnp.asarray(rng.normal(size=self.shape))
        mask = jnp.asarray((rng.uniform(size=self.shape) > 0.3).astype(float))
        f = lambda s: vdx.obs_cost_1d(s, obs, mask)
        loop = np.stack([f(s) for s in states])
        np.testing.assert_allclose(jax.jit(f)(states[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(states), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(states), loop, rtol=1e-12)


class TestObsCost2d:
    shape = (1, 2, 3, 3)

    def test_closed_form_oracle(self, x64):
        state = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
        obs = np.full((1, 1, 3, 3), 2.0)
        mask = np.zeros((1, 1, 3, 3))
        mask[0, 0, 0, :] = 1.0
        mask[0, 0, 2, 1] = 1.0
        # observed departures are 1-2, 2-2, 3-2 and 8-2: squares 1, 0, 1, 36
        expected = 38.0 / 9.0
        got = vdx.obs_cost_2d(jnp.asarray(state), jnp.asarray(obs), jnp.asarray(mask))
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    @SETTINGS
    @given(state=_arrays((1, 2, 3, 3)), mask=_masks((1, 2, 3, 3)), c=_FLOATS)
    def test_property_zero_at_obs_and_homogeneous(self, state, mask, c):
        with jax.enable_x64(True):
            state, mask = jnp.asarray(state), jnp.asarray(mask)
            np.testing.assert_allclose(
                vdx.obs_cost_2d(state, state, mask), 0.0, atol=1e-14
            )
            obs = state + 1.0
            cost = vdx.obs_cost_2d(state, obs, mask)
            # a uniform departure of 1 costs the observed fraction
            np.testing.assert_allclose(cost, mask.mean(), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(
                vdx.obs_cost_2d(state, state + c, mask),
                c**2 * mask.mean(),
                rtol=1e-12,
                atol=1e-12,
            )
            # no observations means no cost, whatever the state
            assert vdx.obs_cost_2d(state, obs, jnp.zeros_like(mask)) == 0

    def test_gradient_matches_central_differences(self, x64):
        rng = np.random.default_rng(2)
        state = rng.normal(size=self.shape)
        obs = jnp.asarray(rng.normal(size=self.shape))
        mask = jnp.asarray((rng.uniform(size=self.shape) > 0.3).astype(float))
        f = lambda s: vdx.obs_cost_2d(s, obs, mask)
        _close_grad(jax.grad(f)(jnp.asarray(state)), _central_diff(f, state))

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(3)
        states = jnp.asarray(rng.normal(size=(3, *self.shape)))
        obs = jnp.asarray(rng.normal(size=self.shape))
        mask = jnp.asarray((rng.uniform(size=self.shape) > 0.3).astype(float))
        f = lambda s: vdx.obs_cost_2d(s, obs, mask)
        loop = np.stack([f(s) for s in states])
        np.testing.assert_allclose(jax.jit(f)(states[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(states), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(states), loop, rtol=1e-12)


class TestPriorCost:
    shape = (2, 3)

    def test_closed_form_oracle(self, x64):
        state = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        recon = np.array([[1.5, 2.0, 1.0], [4.0, 7.0, 6.0]])
        expected = (0.25 + 0.0 + 4.0 + 0.0 + 4.0 + 0.0) / 6
        got = vdx.prior_cost(jnp.asarray(state), jnp.asarray(recon))
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    @SETTINGS
    @given(x=_arrays((2, 3)), r=_arrays((2, 3)), shift=_FLOATS)
    def test_property_symmetric_nonneg_shift_invariant(self, x, r, shift):
        with jax.enable_x64(True):
            cost = vdx.prior_cost(x, r)
            assert cost >= 0
            np.testing.assert_allclose(vdx.prior_cost(r, x), cost, rtol=1e-12)
            # only the departure matters, not the common offset
            np.testing.assert_allclose(
                vdx.prior_cost(x + shift, r + shift), cost, rtol=1e-9, atol=1e-12
            )
            np.testing.assert_allclose(vdx.prior_cost(x, x), 0.0, atol=1e-30)

    def test_gradient_matches_central_differences(self, x64):
        rng = np.random.default_rng(4)
        x = rng.normal(size=self.shape)
        r = jnp.asarray(rng.normal(size=self.shape))
        f = lambda s: vdx.prior_cost(s, r)
        _close_grad(jax.grad(f)(jnp.asarray(x)), _central_diff(f, x))

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(5)
        xs = jnp.asarray(rng.normal(size=(3, *self.shape)))
        r = jnp.asarray(rng.normal(size=self.shape))
        f = lambda s: vdx.prior_cost(s, r)
        loop = np.stack([f(s) for s in xs])
        np.testing.assert_allclose(jax.jit(f)(xs[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(xs), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(xs), loop, rtol=1e-12)


class TestBackgroundCost:
    shape = (5,)

    def test_closed_form_oracle(self, x64):
        x0 = np.array([1.0, 2.0, 3.0, 4.0])
        xb = np.array([0.0, 2.5, 3.0, 1.0])
        expected = (1.0 + 0.25 + 0.0 + 9.0) / 4
        got = vdx.background_cost(jnp.asarray(x0), jnp.asarray(xb))
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    @SETTINGS
    @given(x0=_arrays((5,)), xb=_arrays((5,)), c=_FLOATS)
    def test_property_zero_at_background_and_quadratic(self, x0, xb, c):
        with jax.enable_x64(True):
            assert vdx.background_cost(x0, x0) == 0
            cost = vdx.background_cost(x0, xb)
            assert cost >= 0
            # scaling both states by c scales the departure, so the cost by c^2
            np.testing.assert_allclose(
                vdx.background_cost(c * x0, c * xb),
                c**2 * cost,
                rtol=1e-12,
                atol=1e-14,
            )

    def test_gradient_matches_central_differences(self, x64):
        rng = np.random.default_rng(6)
        x0 = rng.normal(size=self.shape)
        xb = jnp.asarray(rng.normal(size=self.shape))
        f = lambda s: vdx.background_cost(s, xb)
        _close_grad(jax.grad(f)(jnp.asarray(x0)), _central_diff(f, x0))

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(7)
        xs = jnp.asarray(rng.normal(size=(3, *self.shape)))
        xb = jnp.asarray(rng.normal(size=self.shape))
        f = lambda s: vdx.background_cost(s, xb)
        loop = np.stack([f(s) for s in xs])
        np.testing.assert_allclose(jax.jit(f)(xs[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(xs), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(xs), loop, rtol=1e-12)


def _prior(x):
    # smooth nonlinear stand-in for an autoencoder
    return 0.9 * jnp.tanh(x)


class TestVariationalCost:
    shape = (1, 2, 3)

    def _batch(self, y, m):
        return vdx.Batch1D(input=jnp.asarray(y), mask=jnp.asarray(m))

    def test_closed_form_oracle(self, x64):
        x = np.array([[[0.5, -1.0, 2.0], [0.0, 1.0, -0.5]]])
        y = np.array([[[0.0, 0.5, 1.0], [1.0, 1.0, 1.0]]])
        m = np.array([[[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]])
        a_o, a_p = 0.3, 0.7
        j_o = np.mean((m * (x - y)) ** 2)
        j_p = np.mean((x - 0.9 * np.tanh(x)) ** 2)
        got = vdx.variational_cost(
            jnp.asarray(x), self._batch(y, m), _prior, alpha_obs=a_o, alpha_prior=a_p
        )
        np.testing.assert_allclose(got, a_o * j_o + a_p * j_p, rtol=1e-12)

    @SETTINGS
    @given(
        x=_arrays((1, 2, 3)),
        y=_arrays((1, 2, 3)),
        m=_masks((1, 2, 3)),
        a=st.floats(0.0, 2.0),
        b=st.floats(0.0, 2.0),
    )
    def test_property_nonneg_and_linear_in_weights(self, x, y, m, a, b):
        with jax.enable_x64(True):
            batch = self._batch(y, m)
            cost = vdx.variational_cost(x, batch, _prior, a, b)
            assert cost >= 0
            j_o = vdx.variational_cost(x, batch, _prior, 1.0, 0.0)
            j_p = vdx.variational_cost(x, batch, _prior, 0.0, 1.0)
            np.testing.assert_allclose(cost, a * j_o + b * j_p, rtol=1e-12, atol=1e-14)

    def test_gradient_matches_central_differences(self, x64):
        rng = np.random.default_rng(8)
        x = rng.normal(size=self.shape)
        batch = self._batch(
            rng.normal(size=self.shape),
            (rng.uniform(size=self.shape) > 0.3).astype(float),
        )
        f = lambda s: vdx.variational_cost(s, batch, _prior, 0.4, 0.6)
        expected = _central_diff(f, x)
        _close_grad(jax.grad(f)(jnp.asarray(x)), expected)
        # the packaged gradient is the same derivative
        got = vdx.variational_cost_grad(jnp.asarray(x), batch, _prior, 0.4, 0.6)
        _close_grad(got, expected)

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(9)
        xs = jnp.asarray(rng.normal(size=(3, *self.shape)))
        y = jnp.asarray(rng.normal(size=self.shape))
        m = jnp.asarray((rng.uniform(size=self.shape) > 0.3).astype(float))
        f = lambda s: vdx.variational_cost(s, vdx.Batch1D(y, m), _prior)
        loop = np.stack([f(s) for s in xs])
        np.testing.assert_allclose(jax.jit(f)(xs[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(xs), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(xs), loop, rtol=1e-12)


def _forward(x0, ts):
    # (B, N) control and (T,) times to a (B, T, N) trajectory
    return jnp.tanh(x0)[:, None, :] * ts[None, :, None]


class TestStrongVariationalCost:
    x_shape = (1, 2)
    y_shape = (1, 2, 2)
    ts = np.array([1.0, 2.0])

    def _batch(self, y, m):
        return vdx.Batch1D(input=jnp.asarray(y), mask=jnp.asarray(m))

    def test_closed_form_oracle(self, x64):
        x0 = np.array([[0.5, -1.0]])
        xb = np.array([[0.0, -0.5]])
        y = np.array([[[0.1, np.nan], [0.2, -1.0]]])
        m = np.array([[[1.0, 0.0], [1.0, 1.0]]])
        a_o, a_b = 0.8, 0.2
        traj = np.tanh(x0)[:, None, :] * self.ts[None, :, None]
        j_o = np.mean((m * (traj - np.nan_to_num(y))) ** 2)
        j_b = np.mean((x0 - xb) ** 2)
        got = vdx.strong_variational_cost(
            jnp.asarray(x0),
            jnp.asarray(self.ts),
            self._batch(y, m),
            _forward,
            xb=jnp.asarray(xb),
            alpha_obs=a_o,
            alpha_bg=a_b,
            nan_to_num=True,
        )
        np.testing.assert_allclose(got, a_o * j_o + a_b * j_b, rtol=1e-12)

    @SETTINGS
    @given(
        x0=_arrays((1, 2)),
        xb=_arrays((1, 2)),
        y=_arrays((1, 2, 2)),
        m=_masks((1, 2, 2)),
    )
    def test_property_nonneg_and_background_default(self, x0, xb, y, m):
        with jax.enable_x64(True):
            ts, batch = jnp.asarray(self.ts), self._batch(y, m)
            cost = vdx.strong_variational_cost(x0, ts, batch, _forward, xb=xb)
            assert cost >= 0
            # xb defaults to x0, which switches the background term off
            default = vdx.strong_variational_cost(x0, ts, batch, _forward)
            explicit = vdx.strong_variational_cost(x0, ts, batch, _forward, xb=x0)
            np.testing.assert_allclose(default, explicit, rtol=1e-12, atol=1e-15)
            # without a background weight xb cannot matter
            kw = {"alpha_bg": 0.0}
            np.testing.assert_allclose(
                vdx.strong_variational_cost(x0, ts, batch, _forward, xb=xb, **kw),
                default,
                rtol=1e-12,
                atol=1e-14,
            )

    def test_gradient_matches_central_differences(self, x64):
        rng = np.random.default_rng(10)
        x0 = rng.normal(size=self.x_shape)
        xb = jnp.asarray(rng.normal(size=self.x_shape))
        batch = self._batch(
            rng.normal(size=self.y_shape),
            (rng.uniform(size=self.y_shape) > 0.3).astype(float),
        )
        ts = jnp.asarray(self.ts)
        f = lambda s: vdx.strong_variational_cost(s, ts, batch, _forward, xb=xb)
        _close_grad(jax.grad(f)(jnp.asarray(x0)), _central_diff(f, x0))

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(11)
        x0s = jnp.asarray(rng.normal(size=(3, *self.x_shape)))
        y = jnp.asarray(rng.normal(size=self.y_shape))
        m = jnp.ones(self.y_shape)
        ts = jnp.asarray(self.ts)
        f = lambda s: vdx.strong_variational_cost(s, ts, vdx.Batch1D(y, m), _forward)
        loop = np.stack([f(s) for s in x0s])
        np.testing.assert_allclose(jax.jit(f)(x0s[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(x0s), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(x0s), loop, rtol=1e-12)


def _residual(field, s):
    # N[f](s) = f(s) - 1, one residual component
    return field(s) - 1.0


def _linear_field(theta):
    return lambda s: jnp.array([theta * s[0] + s[1]])


class TestResidualPenalty:
    points = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])

    def _penalty(self, points, weight):
        return vdx.ResidualPenalty(_residual, jnp.asarray(points), weight)

    def test_closed_form_oracle(self, x64):
        theta, weight = 2.0, 0.5
        # residuals 2*c0 + c1 - 1 are 0, 2 and 2
        expected = weight / (2 * 3) * (0.0 + 4.0 + 4.0)
        got = self._penalty(self.points, weight)(_linear_field(theta))
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        # a sampler supplies the same points through the key
        sampled = vdx.ResidualPenalty(
            _residual, None, weight, sampler=lambda key: jnp.asarray(self.points)
        )
        got = sampled(_linear_field(theta), key=jax.random.PRNGKey(0))
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    @SETTINGS
    @given(
        pts=_arrays((4, 2)),
        theta=_FLOATS,
        w=st.floats(0.0, 3.0),
        c=st.floats(0.1, 3.0),
        perm=st.permutations(range(4)),
    )
    def test_property_nonneg_linear_in_weight_permutation_invariant(
        self, pts, theta, w, c, perm
    ):
        with jax.enable_x64(True):
            field = _linear_field(theta)
            cost = self._penalty(pts, w)(field)
            assert cost >= 0
            np.testing.assert_allclose(
                self._penalty(pts, c * w)(field), c * cost, rtol=1e-12, atol=1e-14
            )
            # the cost is a sum over points, so their order is irrelevant
            shuffled = self._penalty(pts[list(perm)], w)(field)
            np.testing.assert_allclose(shuffled, cost, rtol=1e-12, atol=1e-14)
            # a field that solves N[f] = 0 everywhere costs nothing
            assert self._penalty(pts, w)(lambda s: jnp.ones(1)) == 0

    def test_gradient_matches_central_differences(self, x64):
        pen = self._penalty(self.points, 0.7)

        def f(theta):
            return pen(_linear_field(theta))

        theta = 1.3
        expected = (f(theta + 1e-6) - f(theta - 1e-6)) / 2e-6
        np.testing.assert_allclose(jax.grad(f)(theta), expected, rtol=1e-6)

    def test_jit_vmap_match_loop(self, x64):
        pen = self._penalty(self.points, 0.7)

        def f(theta):
            return pen(_linear_field(theta))

        thetas = jnp.array([-1.0, 0.3, 2.0])
        loop = np.stack([f(t) for t in thetas])
        np.testing.assert_allclose(jax.jit(f)(thetas[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(thetas), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(thetas), loop, rtol=1e-12)


def _point_batch(values, sigma, valid):
    return vdx.PointObsBatch(
        coords=jnp.zeros((len(values), 1)),
        values=jnp.asarray(values),
        sigma=jnp.asarray(sigma),
        valid=jnp.asarray(valid),
    )


class TestPointObsCost:
    def test_closed_form_oracle(self, x64):
        values = np.array([1.0, 2.0, np.nan, -1.0])
        sigma = np.array([1.0, 2.0, 0.0, 0.5])
        valid = np.array([True, True, False, True])
        pred = np.array([0.0, 0.0, 5.0, 0.0])
        # the padded NaN row with zero sigma contributes nothing
        expected = 0.5 * (1.0 / 1.0 + 4.0 / 4.0 + 1.0 / 0.25)
        got = vdx.point_obs_cost(jnp.asarray(pred), _point_batch(values, sigma, valid))
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    @SETTINGS
    @given(
        y=_arrays((5,)),
        pred=_arrays((5,)),
        junk=_arrays((5,)),
        sigma=hnp.arrays(np.float64, (5,), elements=st.floats(0.2, 3.0)),
        valid=hnp.arrays(np.bool_, (5,)),
        c=st.floats(0.1, 3.0),
    )
    def test_property_nonneg_padding_blind_and_quadratic(
        self, y, pred, junk, sigma, valid, c
    ):
        with jax.enable_x64(True):
            y, pred, junk = map(jnp.asarray, (y, pred, junk))
            cost = vdx.point_obs_cost(pred, _point_batch(y, sigma, valid))
            assert cost >= 0
            # padding rows must not see their values, prediction or sigma
            bad = jnp.where(valid, y, jnp.nan)
            blind = vdx.point_obs_cost(
                jnp.where(valid, pred, junk), _point_batch(bad, sigma, valid)
            )
            np.testing.assert_allclose(blind, cost, rtol=1e-12, atol=1e-14)
            scaled = vdx.point_obs_cost(c * pred, _point_batch(c * y, sigma, valid))
            np.testing.assert_allclose(scaled, c**2 * cost, rtol=1e-10, atol=1e-12)
            assert vdx.point_obs_cost(y, _point_batch(y, sigma, valid)) == 0

    def test_gradient_matches_central_differences(self, x64):
        rng = np.random.default_rng(12)
        values = rng.normal(size=6)
        values[4] = np.nan
        sigma = rng.uniform(0.5, 2.0, size=6)
        valid = np.array([True, True, False, True, False, True])
        batch = _point_batch(values, sigma, valid)
        pred = rng.normal(size=6)
        f = lambda p: vdx.point_obs_cost(p, batch)
        got = jax.grad(f)(jnp.asarray(pred))
        assert np.all(np.isfinite(got))
        _close_grad(got, _central_diff(f, pred))

    def test_jit_vmap_match_loop(self, x64):
        rng = np.random.default_rng(13)
        batch = _point_batch(
            [1.0, np.nan, -0.5, 2.0],
            [1.0, 1.0, 0.5, 2.0],
            [True, False, True, True],
        )
        preds = jnp.asarray(rng.normal(size=(3, 4)))
        f = lambda p: vdx.point_obs_cost(p, batch)
        loop = np.stack([f(p) for p in preds])
        np.testing.assert_allclose(jax.jit(f)(preds[0]), loop[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(f)(preds), loop, rtol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(preds), loop, rtol=1e-12)
