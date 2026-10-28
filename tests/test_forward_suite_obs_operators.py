"""Four-test suite (oracle, property, gradient, transforms) for the observation operators."""

import einx
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import vardax as vdx

RTOL = 1e-12
SETTINGS = settings(max_examples=25, deadline=None)


def _floats(lo, hi):
    return st.floats(
        lo, hi, allow_nan=False, allow_infinity=False, allow_subnormal=False
    )


def _arrays(shape, lo=-5.0, hi=5.0):
    # no subnormals: XLA on CPU flushes them to zero
    elements = _floats(lo, hi)
    return hnp.arrays(np.float64, shape, elements=elements)


def _central_jacobian(fn, x, step=1e-6):
    """Central-difference Jacobian of shape ``(out.size, x.size)``."""
    base = np.asarray(x, dtype=np.float64)
    columns = []
    for i in range(base.size):
        shift = np.zeros(base.size)
        shift[i] = step
        up = fn(jnp.asarray(base + shift.reshape(base.shape)))
        down = fn(jnp.asarray(base - shift.reshape(base.shape)))
        columns.append(np.ravel(up) - np.ravel(down))
    return einx.rearrange("c o -> o c", np.stack(columns)) / (2.0 * step)


def _flat_jacobian(fn, x):
    """``jax.jacrev`` reshaped to ``(out.size, x.size)``."""
    jac = jax.jacrev(fn)(x)
    return np.asarray(jac).reshape(-1, np.size(x))


def _assert_matches_central(fn, x):
    np.testing.assert_allclose(
        _flat_jacobian(fn, x), _central_jacobian(fn, x), rtol=1e-6, atol=1e-8
    )


def _assert_transforms_match_loop(fn, batch):
    """Eager vmap, jitted vmap and per-sample jit agree with a Python loop."""

    def stack(outputs):
        return jax.tree.map(lambda *leaves: np.stack(leaves), *outputs)

    expected = stack([fn(sample) for sample in batch])
    candidates = [
        jax.vmap(fn)(batch),
        jax.jit(jax.vmap(fn))(batch),
        stack([jax.jit(fn)(sample) for sample in batch]),
    ]
    for got in candidates:
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=RTOL, atol=1e-14),
            got,
            expected,
        )


def _as_fn(op):
    # a plain function: jax.jit cannot hash a module that holds arrays
    def fn(x):
        return op(x)

    return fn


def _point_batch(coords, valid=None, time_index=None, n_times=None):
    coords = jnp.asarray(coords)
    n_obs = coords.shape[0]
    return vdx.PointObsBatch(
        coords=coords,
        values=jnp.zeros(n_obs),
        sigma=jnp.ones(n_obs),
        valid=jnp.ones(n_obs, bool) if valid is None else jnp.asarray(valid),
        time_index=None if time_index is None else jnp.asarray(time_index),
        n_times=n_times,
    )


_H_LIN = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]])


class TestLinearObs:
    def test_closed_form_oracle(self, x64):
        obs = vdx.LinearObs(operator=lx.MatrixLinearOperator(jnp.asarray(_H_LIN)))
        # rows: 1*1 + 2*2 + 3*(-1) = 2 and 0*1 - 1*2 + 4*(-1) = -6
        got = obs(jnp.array([1.0, 2.0, -1.0]))
        np.testing.assert_allclose(got, [2.0, -6.0], rtol=RTOL)

    @given(x=_arrays((3,)), y=_arrays((3,)), a=_floats(-3, 3), b=_floats(-3, 3))
    @SETTINGS
    def test_property_linearity(self, x, y, a, b):
        with jax.enable_x64(True):
            obs = vdx.LinearObs(operator=lx.MatrixLinearOperator(jnp.asarray(_H_LIN)))
            lhs = obs(a * jnp.asarray(x) + b * jnp.asarray(y))
            rhs = a * obs(jnp.asarray(x)) + b * obs(jnp.asarray(y))
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(obs(jnp.zeros(3)), 0.0, atol=1e-15)

    def test_gradient_matches_central_differences(self, x64):
        obs = vdx.LinearObs(operator=lx.MatrixLinearOperator(jnp.asarray(_H_LIN)))
        x = jnp.array([0.3, -0.7, 1.1])
        _assert_matches_central(obs, x)
        # the linearisation is the Jacobian, at any point
        for point in (x, jnp.zeros(3)):
            np.testing.assert_allclose(
                obs.linearize(point).as_matrix(), jax.jacfwd(obs)(x), rtol=RTOL
            )

    def test_jit_vmap_match_loop(self, x64):
        obs = vdx.LinearObs(operator=lx.MatrixLinearOperator(jnp.asarray(_H_LIN)))
        batch = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 0.0, 1.0]])
        _assert_transforms_match_loop(_as_fn(obs), batch)


class TestMaskedIdentity:
    def test_closed_form_oracle(self, x64):
        op = vdx.MaskedIdentity()
        x = jnp.array([1.5, -2.0, 3.0])
        got = op(x, mask=jnp.array([1.0, 0.0, 0.5]))
        np.testing.assert_allclose(got, [1.5, 0.0, 1.5], rtol=RTOL)
        np.testing.assert_allclose(op(x), x, rtol=RTOL)

    @given(x=_arrays((6,)), bits=hnp.arrays(np.bool_, (6,)))
    @SETTINGS
    def test_property_binary_mask_is_idempotent(self, x, bits):
        with jax.enable_x64(True):
            op = vdx.MaskedIdentity()
            mask = jnp.asarray(bits, dtype=jnp.float64)
            once = op(jnp.asarray(x), mask=mask)
            # masking twice changes nothing; dropped entries are exactly zero
            np.testing.assert_array_equal(op(once, mask=mask), once)
            np.testing.assert_array_equal(np.asarray(once)[~bits], 0.0)
            np.testing.assert_array_equal(np.asarray(once)[bits], x[bits])

    def test_gradient_matches_central_differences(self, x64):
        op = vdx.MaskedIdentity()
        mask = jnp.array([1.0, 0.0, 0.5, 1.0])
        x = jnp.array([0.3, -0.7, 1.1, 2.0])
        _assert_matches_central(lambda z: op(z, mask=mask), x)
        np.testing.assert_allclose(
            _flat_jacobian(lambda z: op(z, mask=mask), x), np.diag(mask), rtol=RTOL
        )
        # linearize ignores the mask, so it matches the unmasked Jacobian
        np.testing.assert_allclose(
            op.linearize(x).as_matrix(), jax.jacfwd(op)(x), rtol=RTOL
        )

    def test_jit_vmap_match_loop(self, x64):
        op = vdx.MaskedIdentity()
        mask = jnp.array([1.0, 0.0, 0.5])
        batch = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 4.0, 1.0]])
        _assert_transforms_match_loop(lambda x: op(x, mask=mask), batch)


def _nonlinear_map(x):
    return jnp.stack([x[0] * x[1], jnp.sin(x[2]), x[0] ** 2 + x[2]])


class TestCallableObs:
    def test_closed_form_oracle(self, x64):
        obs = vdx.CallableObs(fn=_nonlinear_map)
        got = obs(jnp.array([2.0, 3.0, 0.5]))
        expected = [2.0 * 3.0, np.sin(0.5), 2.0**2 + 0.5]
        np.testing.assert_allclose(got, expected, rtol=RTOL)

    @given(x=_arrays((3,)), shift=_arrays((3,)))
    @SETTINGS
    def test_property_linear_map_has_constant_linearisation(self, x, shift):
        with jax.enable_x64(True):
            matrix = jnp.asarray(_H_LIN)
            obs = vdx.CallableObs(fn=lambda z: matrix @ z, materialise=True)
            # for an affine map the tangent-linear is the matrix at every point
            np.testing.assert_allclose(
                obs.linearize(jnp.asarray(x)).as_matrix(), _H_LIN, rtol=RTOL
            )
            np.testing.assert_allclose(
                obs(jnp.asarray(x + shift)) - obs(jnp.asarray(x)),
                _H_LIN @ shift,
                rtol=1e-9,
                atol=1e-9,
            )

    def test_gradient_matches_central_differences(self, x64):
        x = jnp.array([2.0, 3.0, 0.5])
        for obs in (
            vdx.CallableObs(fn=_nonlinear_map),
            vdx.CallableObs(fn=_nonlinear_map, materialise=True),
            vdx.CallableObs(fn=_nonlinear_map, jac="bwd", materialise=True),
        ):
            _assert_matches_central(obs, x)
            np.testing.assert_allclose(
                obs.linearize(x).as_matrix(), jax.jacfwd(obs)(x), rtol=1e-10
            )

    def test_jit_vmap_match_loop(self, x64):
        obs = vdx.CallableObs(fn=_nonlinear_map)
        batch = jnp.array([[2.0, 3.0, 0.5], [-1.0, 0.5, 2.0], [0.0, 4.0, 1.0]])
        _assert_transforms_match_loop(_as_fn(obs), batch)


_AXIS_Y = np.array([0.0, 1.0, 3.0])
_AXIS_X = np.array([0.0, 2.0, 3.0])


def _bilinear_field(y, x):
    # bilinear in (y, x), so interpolation on any grid reproduces it exactly
    return 1.0 + 2.0 * y + 3.0 * x + 0.5 * y * x


def _interp_op():
    obs_points = jnp.array([[2.0, 1.0], [0.5, 2.5], [5.0, -1.0]])
    return vdx.interp_obs_from_coords((_AXIS_Y, _AXIS_X), obs_points)


class TestInterpObs:
    def test_closed_form_oracle(self, x64):
        op = _interp_op()
        yy, xx = np.meshgrid(_AXIS_Y, _AXIS_X, indexing="ij")
        field = jnp.asarray(_bilinear_field(yy, xx))
        # (2, 1) -> 1 + 4 + 3 + 1; (0.5, 2.5) -> 1 + 1 + 7.5 + 0.625;
        # (5, -1) clamps to (3, 0) -> 1 + 6
        np.testing.assert_allclose(op(field), [9.0, 10.125, 7.0], rtol=RTOL)
        mask = jnp.array([1.0, 0.0, 1.0])
        np.testing.assert_allclose(op(field, mask=mask), [9.0, 0.0, 7.0], rtol=RTOL)

    @given(field=_arrays((3, 3), -10, 10), bits=hnp.arrays(np.bool_, (3,)))
    @SETTINGS
    def test_property_convex_combination_and_mask(self, field, bits):
        with jax.enable_x64(True):
            op = _interp_op()
            state = jnp.asarray(field)
            got = np.asarray(op(state))
            # weights are non-negative and sum to one, so values stay in range
            assert np.all(got >= field.min() - 1e-12)
            assert np.all(got <= field.max() + 1e-12)
            mask = jnp.asarray(bits, dtype=jnp.float64)
            np.testing.assert_allclose(op(state, mask=mask), got * bits, rtol=RTOL)
            np.testing.assert_allclose(
                op(state + 2.5), got + 2.5, rtol=1e-12, atol=1e-12
            )

    def test_gradient_matches_central_differences(self, x64):
        op = _interp_op()
        field = jnp.asarray(np.linspace(-1.0, 2.0, 9).reshape(3, 3) ** 2)
        _assert_matches_central(op, field)
        np.testing.assert_allclose(
            op.linearize(field).as_matrix(), _flat_jacobian(op, field), rtol=1e-10
        )

    def test_jit_vmap_match_loop(self, x64):
        op = _interp_op()
        mask = jnp.array([1.0, 0.0, 1.0])
        batch = jnp.asarray(np.random.default_rng(0).normal(size=(3, 3, 3)))
        _assert_transforms_match_loop(lambda s: op(s, mask=mask), batch)


_GRID_Y = np.array([0.0, 1.0, 3.0])
_GRID_X = np.array([0.0, 2.0])


def _grid_trajectory():
    yy, xx = np.meshgrid(_GRID_Y, _GRID_X, indexing="ij")
    return jnp.asarray(np.stack([10.0 * t + _bilinear_field(yy, xx) for t in (0, 1)]))


class TestGridPointObs:
    @pytest.mark.slow
    def test_closed_form_oracle(self, x64):
        op = vdx.GridPointObs((jnp.asarray(_GRID_Y), jnp.asarray(_GRID_X)))
        coords = [[2.0, 1.0], [0.5, 1.5], [3.0, 2.0], [-1.0, 5.0]]
        batch = _point_batch(coords, time_index=[0, 1, 1, 0], n_times=2)
        # f_t = 10 t + 1 + 2y + 3x + y x / 2 evaluated by hand; the last point
        # clamps to the corner (0, 2)
        expected = [9.0, 16.875, 26.0, 7.0]
        np.testing.assert_allclose(op(_grid_trajectory(), batch), expected, rtol=RTOL)

        # 1-D grid: f(x) = 2 + 3x is reproduced exactly by linear interpolation
        op1 = vdx.GridPointObs((jnp.array([0.0, 2.0, 3.0]),))
        state = jnp.array([[2.0, 8.0, 11.0]])
        batch1 = _point_batch([[2.5], [0.5], [9.0]])
        np.testing.assert_allclose(op1(state, batch1), [9.5, 3.5, 11.0], rtol=RTOL)

    @pytest.mark.slow
    @given(
        coords=_arrays((5, 2), -1.0, 4.0),
        level=_floats(-5, 5),
    )
    @SETTINGS
    def test_property_constant_field_and_permutation(self, coords, level):
        with jax.enable_x64(True):
            op = vdx.GridPointObs((jnp.asarray(_GRID_Y), jnp.asarray(_GRID_X)))
            flat = jnp.full((1, 3, 2), level)
            got = op(flat, _point_batch(coords))
            # linear weights sum to one everywhere, clamping included
            np.testing.assert_allclose(got, level, rtol=1e-12, atol=1e-12)

            state = _grid_trajectory()[:1]
            out = np.asarray(op(state, _point_batch(coords)))
            perm = np.array([3, 0, 4, 1, 2])
            shuffled = op(state, _point_batch(coords[perm]))
            np.testing.assert_allclose(shuffled, out[perm], rtol=RTOL, atol=1e-14)

    def test_gradient_matches_central_differences(self, x64):
        op = vdx.GridPointObs((jnp.asarray(_GRID_Y), jnp.asarray(_GRID_X)))
        coords = [[2.0, 1.0], [0.5, 1.5], [3.0, 2.0], [1.7, 0.3]]
        batch = _point_batch(coords, time_index=[0, 1, 1, 0], n_times=2)
        state = _grid_trajectory() ** 2
        _assert_matches_central(lambda s: op(s, batch), state)

    def test_jit_vmap_match_loop(self, x64):
        op = vdx.GridPointObs((jnp.asarray(_GRID_Y), jnp.asarray(_GRID_X)))
        coords = [[2.0, 1.0], [0.5, 1.5], [9.0, 9.0], [1.7, 0.3]]
        valid = [True, True, False, True]
        batch = _point_batch(coords, valid, time_index=[0, 1, 1, 0], n_times=2)
        states = jnp.asarray(np.random.default_rng(1).normal(size=(3, 2, 3, 2)))
        _assert_transforms_match_loop(lambda s: op(s, batch), states)


def _scalar_field(s):
    return s[0] * s[1] + jnp.sin(s[1])


class TestFieldPointObs:
    def test_closed_form_oracle(self, x64):
        coords = np.array([[1.0, 2.0], [-0.5, 0.25], [2.0, -1.0]])
        got = vdx.FieldPointObs()(_scalar_field, _point_batch(coords))
        expected = coords[:, 0] * coords[:, 1] + np.sin(coords[:, 1])
        np.testing.assert_allclose(got, expected, rtol=RTOL)
        # a field returning shape (1,) gives the same values
        got1 = vdx.FieldPointObs()(
            lambda s: _scalar_field(s)[None], _point_batch(coords)
        )
        np.testing.assert_allclose(got1, expected, rtol=RTOL)

    @given(
        coords=_arrays((5, 2), -3, 3),
        junk=_arrays((5, 2), -3, 3),
        bits=hnp.arrays(np.bool_, (5,)),
    )
    @SETTINGS
    def test_property_padding_is_ignored(self, coords, junk, bits):
        with jax.enable_x64(True):
            op = vdx.FieldPointObs()
            if not bits.any():
                bits = bits.copy()
                bits[0] = True
            clean = op(_scalar_field, _point_batch(coords, bits))
            scrambled_coords = np.where(bits[:, None], coords, junk)
            scrambled = op(_scalar_field, _point_batch(scrambled_coords, bits))
            # padding rows are evaluated at the first valid row, whatever they hold
            np.testing.assert_array_equal(scrambled, clean)
            expected = coords[:, 0] * coords[:, 1] + np.sin(coords[:, 1])
            np.testing.assert_allclose(
                np.asarray(clean)[bits], expected[bits], rtol=1e-12, atol=1e-14
            )

    def test_gradient_matches_central_differences(self, x64):
        op = vdx.FieldPointObs()
        coords = jnp.array([[1.0, 2.0], [-0.5, 0.25], [2.0, -1.0]])
        _assert_matches_central(lambda c: op(_scalar_field, _point_batch(c)), coords)

    def test_jit_vmap_match_loop(self, x64):
        op = vdx.FieldPointObs()
        valid = [True, True, False]
        batch = jnp.asarray(np.random.default_rng(2).normal(size=(3, 3, 2)))
        _assert_transforms_match_loop(
            lambda c: op(_scalar_field, _point_batch(c, valid)), batch
        )


def _kernel(x64_dtype=jnp.float64):
    A = jnp.array([[0.5, 0.5], [0.2, 0.8]], dtype=x64_dtype)
    return A, vdx.AveragingKernel(
        kernel_matrix=lx.MatrixLinearOperator(A),
        prior_profile=jnp.array([2.0, 4.0], dtype=x64_dtype),
        pressure_weights=jnp.array([0.5, 1.0], dtype=x64_dtype),
    )


class TestAveragingKernel:
    def test_closed_form_oracle(self, x64):
        _, op = _kernel()
        # blend = [0.5 * 1 + 0.5 * 2, 3] = [1.5, 3]; A @ blend = [2.25, 2.7]
        got = op(jnp.array([1.0, 3.0]))
        np.testing.assert_allclose(got, [2.25, 2.7], rtol=RTOL)

    @given(x=_arrays((2,)), y=_arrays((2,)))
    @SETTINGS
    def test_property_affine_with_kernel_slope(self, x, y):
        with jax.enable_x64(True):
            A, op = _kernel()
            diag_h = np.diag([0.5, 1.0])
            # H(x) - H(y) = A diag(h) (x - y): the prior cancels
            np.testing.assert_allclose(
                op(jnp.asarray(x)) - op(jnp.asarray(y)),
                np.asarray(A) @ diag_h @ (x - y),
                rtol=1e-10,
                atol=1e-10,
            )
            # with h = 0 the state is ignored and only the prior is seen
            blind = vdx.AveragingKernel(
                kernel_matrix=op.kernel_matrix,
                prior_profile=op.prior_profile,
                pressure_weights=jnp.zeros(2),
            )
            np.testing.assert_allclose(
                blind(jnp.asarray(x)),
                np.asarray(A) @ np.asarray(op.prior_profile),
                rtol=RTOL,
            )

    def test_gradient_matches_central_differences(self, x64):
        _, op = _kernel()
        x = jnp.array([0.4, -1.3])
        _assert_matches_central(op, x)
        np.testing.assert_allclose(
            op.linearize(x).as_matrix(), jax.jacfwd(op)(x), rtol=1e-10
        )

    def test_jit_vmap_match_loop(self, x64):
        _, op = _kernel()
        batch = jnp.array([[1.0, 3.0], [-1.0, 0.5], [0.0, 4.0]])
        _assert_transforms_match_loop(_as_fn(op), batch)


def _fusion():
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float64))
    H = lx.MatrixLinearOperator(jnp.array([[1.0, 1.0]]))
    first = vdx.InstrumentSpec(vdx.MaskedIdentity(), jnp.ones(2), eye, "a")
    second = vdx.InstrumentSpec(vdx.LinearObs(operator=H), jnp.ones(1), eye, "b")
    registry = vdx.InstrumentRegistry(entries={"b": second, "a": first})
    return vdx.MultiInstrumentFusion(registry)


def _fusion_flat(fusion, x):
    preds = fusion(x)
    # sorted id order, matching the flattened adapter
    return jnp.concatenate([jnp.ravel(preds[k]) for k in sorted(preds)])


class TestMultiInstrumentFusion:
    def test_closed_form_oracle(self, x64):
        fusion = _fusion()
        x = jnp.array([1.5, -2.0])
        preds = fusion(x)
        assert set(preds) == {"a", "b"}
        np.testing.assert_allclose(preds["a"], [1.5, -2.0], rtol=RTOL)
        np.testing.assert_allclose(preds["b"], [-0.5], rtol=RTOL)
        # the adapter concatenates in sorted id order: a then b
        flat = fusion.to_observation_operator()(x)
        np.testing.assert_allclose(flat, [1.5, -2.0, -0.5], rtol=RTOL)

    @given(x=_arrays((2,)), y=_arrays((2,)), a=_floats(-3, 3))
    @SETTINGS
    def test_property_linear_per_instrument(self, x, y, a):
        with jax.enable_x64(True):
            fusion = _fusion()
            lhs = fusion(a * jnp.asarray(x) + jnp.asarray(y))
            base_x, base_y = fusion(jnp.asarray(x)), fusion(jnp.asarray(y))
            for key in lhs:
                np.testing.assert_allclose(
                    lhs[key],
                    a * base_x[key] + base_y[key],
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_gradient_matches_central_differences(self, x64):
        fusion = _fusion()
        x = jnp.array([0.4, -1.3])
        _assert_matches_central(lambda z: _fusion_flat(fusion, z), x)
        # per-instrument linearisations equal the per-instrument Jacobians
        jacobians = jax.jacfwd(fusion)(x)
        for key, tangent in fusion.linearize(x).items():
            np.testing.assert_allclose(
                tangent.as_matrix(), jacobians[key], rtol=RTOL, atol=1e-14
            )

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            "_FlattenedMultiInstrument.linearize passes the one-argument "
            "self.__call__ to lx.JacobianLinearOperator, which calls fn(x, args)."
        ),
    )
    def test_flattened_linearize_matches_jacobian(self, x64):
        fusion = _fusion()
        x = jnp.array([0.4, -1.3])
        flat = fusion.to_observation_operator()
        np.testing.assert_allclose(
            flat.linearize(x).as_matrix(), jax.jacfwd(flat)(x), rtol=RTOL
        )

    def test_jit_vmap_match_loop(self, x64):
        fusion = _fusion()
        batch = jnp.array([[1.0, 2.0], [-1.0, 0.5], [0.0, 4.0]])
        _assert_transforms_match_loop(_as_fn(fusion), batch)
