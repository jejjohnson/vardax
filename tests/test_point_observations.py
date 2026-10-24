"""PointObsBatch, GridPointObs, FieldPointObs and point batches in the models (#83)."""

from __future__ import annotations

import equinox as eqx
import geonnax
import jax
import jax.numpy as jnp
import lineax as lx
import optax
import pytest

import vardax as vdx
from vardax import (
    Batch1D,
    FieldPointObs,
    GridPointObs,
    IncrementalFourDVar,
    MaskedIdentity,
    NeuralFieldParameterization,
    OptaxMinimiser,
    OptimalInterpolation,
    PointObsBatch,
    ReducedBasisFourDVar,
    StrongFourDVar,
    ThreeDVar,
    WeakFourDVar,
    interp_obs_from_coords,
    point_obs_cost,
)

N = 6
AXIS = jnp.arange(float(N))


class IdentityForward(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return state


class Lorenz96RK4(eqx.Module):
    """Lorenz-96 (F = 8) stepped with classical RK4 (a ``ForwardModel``)."""

    dt: float = 0.05

    def step(self, x, dt):
        f = vdx.Lorenz96(F=8.0)
        k1 = f(0.0, x, None)
        k2 = f(0.0, x + 0.5 * dt * k1, None)
        k3 = f(0.0, x + 0.5 * dt * k2, None)
        k4 = f(0.0, x + dt * k3, None)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _batch(coords, values, sigma=None, valid=None, **kw):
    values = jnp.asarray(values)
    n = values.shape[0]
    return PointObsBatch(
        coords=jnp.asarray(coords).reshape(n, -1),
        values=values,
        sigma=jnp.ones(n) if sigma is None else jnp.asarray(sigma),
        valid=jnp.ones(n, bool) if valid is None else jnp.asarray(valid),
        **kw,
    )


def _pad(batch: PointObsBatch, n_pad: int) -> PointObsBatch:
    """Append ``n_pad`` padding rows full of garbage (NaN, zero sigma)."""
    d = batch.coords.shape[1]
    nan = jnp.full(n_pad, jnp.nan)
    return PointObsBatch(
        coords=jnp.concatenate([batch.coords, jnp.full((n_pad, d), jnp.nan)]),
        values=jnp.concatenate([batch.values, nan]),
        sigma=jnp.concatenate([batch.sigma, jnp.zeros(n_pad)]),
        valid=jnp.concatenate([batch.valid, jnp.zeros(n_pad, bool)]),
        time_index=None
        if batch.time_index is None
        else jnp.concatenate(
            [batch.time_index, jnp.full(n_pad, 10**6, batch.time_index.dtype)]
        ),
        n_times=batch.n_times,
    )


class TestPointObsCost:
    def test_closed_form(self):
        batch = _batch(jnp.zeros(3), [1.0, 2.0, 3.0], sigma=[1.0, 2.0, 0.5])
        pred = jnp.array([0.0, 1.0, 3.5])
        expected = 0.5 * (1.0 + (1.0 / 2.0) ** 2 + (0.5 / 0.5) ** 2)
        assert jnp.allclose(point_obs_cost(pred, batch), expected)

    def test_padding_invariance_of_value_and_gradient(self):
        key = jax.random.PRNGKey(0)
        coords = jax.random.uniform(key, (8,), maxval=N - 1.0)
        times = jnp.array([0, 1, 2, 2, 1, 0, 2, 1])
        batch = _batch(
            coords,
            jnp.sin(coords),
            sigma=0.3 + coords / 10,
            time_index=times,
            n_times=3,
        )
        padded = _pad(batch, 5)
        H = GridPointObs((AXIS,))
        traj = jax.random.normal(jax.random.PRNGKey(1), (3, N))

        def cost(x, b):
            return point_obs_cost(H(x, b), b)

        v, g = jax.value_and_grad(cost)(traj, batch)
        v_pad, g_pad = jax.value_and_grad(cost)(traj, padded)
        # The padding rows are exactly zero: identical up to the order of a
        # float32 sum over 8 vs 13 terms (one ulp of the total).
        assert jnp.isfinite(v_pad)
        assert jnp.allclose(v, v_pad, rtol=1e-6)
        assert jnp.all(jnp.isfinite(g_pad))
        assert jnp.allclose(g, g_pad, rtol=1e-6, atol=1e-7)

    def test_all_padding_is_zero(self):
        empty = PointObsBatch(
            coords=jnp.zeros((0, 1)),
            values=jnp.zeros(0),
            sigma=jnp.ones(0),
            valid=jnp.zeros(0, bool),
        )
        batch = _pad(empty, 4)
        g = jax.grad(lambda x: point_obs_cost(GridPointObs((AXIS,))(x, batch), batch))(
            jnp.ones((1, N))
        )
        assert jnp.array_equal(g, jnp.zeros((1, N)))


class TestGridPointObs:
    def test_1d_linear_interpolation(self):
        H = GridPointObs((AXIS,))
        state = (3.0 * AXIS + 1.0)[None]
        coords = jnp.array([0.0, 0.25, 2.5, 5.0])
        # A linear function is reproduced exactly by linear interpolation.
        assert jnp.allclose(H(state, _batch(coords, jnp.zeros(4))), 3.0 * coords + 1.0)

    def test_clamps_outside_grid(self):
        H = GridPointObs((AXIS,))
        state = AXIS[None]
        out = H(state, _batch(jnp.array([-3.0, 9.0]), jnp.zeros(2)))
        assert jnp.allclose(out, jnp.array([0.0, N - 1.0]))

    def test_time_index_selects_trajectory_state(self):
        H = GridPointObs((AXIS,))
        traj = jnp.stack([AXIS, 10 * AXIS, 100 * AXIS])
        batch = _batch(
            jnp.array([1.0, 1.0, 1.0]),
            jnp.zeros(3),
            time_index=jnp.array([0, 1, 2]),
            n_times=3,
        )
        assert jnp.allclose(H(traj, batch), jnp.array([1.0, 10.0, 100.0]))

    def test_2d_matches_interp_obs(self):
        ya, xa = jnp.arange(4.0), jnp.linspace(0.0, 2.0, 5)
        pts = jax.random.uniform(jax.random.PRNGKey(2), (7, 2)) * jnp.array([3.0, 2.0])
        field = jax.random.normal(jax.random.PRNGKey(3), (4, 5))
        ref = interp_obs_from_coords((ya, xa), pts)(field)
        out = GridPointObs((ya, xa))(field[None], _batch(pts, jnp.zeros(7)))
        # Same bilinear stencil, computed in float32 vs float64-then-cast.
        assert jnp.allclose(out, ref, atol=1e-5)

    def test_flat_state_is_reshaped_to_grid(self):
        ya, xa = jnp.arange(2.0), jnp.arange(3.0)
        H = GridPointObs((ya, xa))
        state = jnp.arange(6.0)[None]  # (1, 6) flat over a (2, 3) grid
        out = H(state, _batch(jnp.array([[1.0, 2.0]]), jnp.zeros(1)))
        assert jnp.allclose(out, 5.0)

    @pytest.mark.parametrize(
        "axes",
        [(), (AXIS, AXIS, AXIS), (jnp.array([0.0]),), (jnp.array([0.0, 2.0, 1.0]),)],
    )
    def test_validation(self, axes):
        with pytest.raises(ValueError):
            GridPointObs(axes)


class TestFieldPointObs:
    def test_evaluates_field(self):
        batch = _batch(jnp.array([[1.0, 2.0], [3.0, -1.0]]), jnp.zeros(2))
        out = FieldPointObs()(lambda s: (s[0] * s[1])[None], batch)
        assert jnp.allclose(out, jnp.array([2.0, -3.0]))

    def test_padding_rows_are_finite(self):
        batch = _pad(_batch(jnp.array([[1.0, 2.0]]), jnp.zeros(1)), 2)
        out = FieldPointObs()(lambda s: jnp.log1p(s[:1] ** 2), batch)
        assert jnp.all(jnp.isfinite(out))


# --- the models -------------------------------------------------------------


def _ops():
    """Correlated B (exponential, length 2), diagonal R with varied sigma."""
    dist = jnp.abs(AXIS[:, None] - AXIS[None, :])
    B = jnp.exp(-dist / 2.0)
    sigma = jnp.array([0.5, 1.0, 0.8, 1.5, 0.7, 1.2])
    return B, sigma


def _grid_and_points(observed):
    y = jax.random.normal(jax.random.PRNGKey(4), (N,))
    _, sigma = _ops()
    mask = observed.astype(jnp.float32)
    grid = Batch1D(input=(y * mask)[None, None], mask=mask[None, None])
    points = _batch(AXIS, y, sigma=sigma, valid=observed)
    return grid, points, y


def _psd(m):
    return lx.MatrixLinearOperator(m, lx.positive_semidefinite_tag)


def _diag(v):
    return lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(v), lx.positive_semidefinite_tag
    )


@pytest.mark.parametrize("all_observed", [True, False])
def test_point_batch_reproduces_grid_3dvar(all_observed):
    B, sigma = _ops()
    observed = (
        jnp.ones(N, bool) if all_observed else jnp.array([1, 0, 1, 1, 0, 1], bool)
    )
    grid, points, _ = _grid_and_points(observed)
    x_b = jnp.linspace(-1.0, 1.0, N)[None]
    common = {"prior_mean": x_b, "prior_cov_op": _psd_time(B)}
    grid_model = ThreeDVar(
        obs_op=MaskedIdentity(),
        obs_cov_op=_diag((sigma**2)[None]),
        **common,
    )
    point_model = ThreeDVar(obs_op=GridPointObs((AXIS,)), obs_cov_op=None, **common)
    xa_grid = grid_model(grid)[0]
    xa_point = point_model(points)
    assert xa_point.shape == x_b.shape
    # The two costs are the same quadratic (the point cost at the grid
    # nodes is the masked diagonal-R grid cost), so each analysis is within
    # its gradient bound of the common minimiser, and of each other.
    tol = _quad_bound(xa_grid[0], x_b[0], B, points) + _quad_bound(
        xa_point[0], x_b[0], B, points
    )
    assert jnp.linalg.norm(xa_grid - xa_point) <= tol


EPS = float(jnp.finfo(jnp.float32).eps)


def _quad_bound(x, x_b, B, points):
    """‖x - x*‖ for J(x) = ½‖x - x_b‖²_{B⁻¹} + point cost (one state).

    J is quadratic with Hessian ⪰ B⁻¹ ⪰ I / λ_max(B), so
    ‖x - x*‖ ≤ λ_max(B) ‖∇J(x)‖ exactly. ∇J is evaluated in float32, which
    resolves x only to ~eps·cond(B)·‖x‖ (8 for the op count): added as the
    round-off floor.
    """
    B_inv = jnp.linalg.inv(B)
    H = GridPointObs((AXIS,))

    def cost(x):
        d = x - x_b
        return 0.5 * d @ B_inv @ d + point_obs_cost(H(x[None], points), points)

    eig = jnp.linalg.eigvalsh(B)
    lam_max, cond = float(eig[-1]), float(eig[-1] / eig[0])
    g = float(jnp.linalg.norm(jax.grad(cost)(x)))
    return lam_max * g + 8 * EPS * cond * float(jnp.linalg.norm(x))


def _psd_time(B):
    """B on a (1, N) state as a lineax operator."""
    return lx.TaggedLinearOperator(
        lx.FunctionLinearOperator(
            lambda x: (B @ x[0])[None], jax.ShapeDtypeStruct((1, N), jnp.float32)
        ),
        lx.positive_semidefinite_tag,
    )


@pytest.mark.parametrize("which", ["strong", "weak", "incremental", "incremental_gn"])
def test_point_batch_linear_gaussian_agreement(which):
    """Every 4D-Var model on a point batch agrees with OI on the grid batch."""
    B, sigma = _ops()
    observed = jnp.array([1, 0, 1, 1, 0, 1], bool)
    grid, points, _ = _grid_and_points(observed)
    points = _pad(points, 3)
    x_b = jnp.linspace(-1.0, 1.0, N)
    oi = OptimalInterpolation(
        obs_op=MaskedIdentity(),
        prior_mean=x_b[None],
        prior_cov_op=_psd_time(B),
        obs_cov_op=_diag((sigma**2)[None]),
    )
    x_oi = oi(grid)[0, 0]
    common = {
        "forward": IdentityForward(),
        "obs_op": GridPointObs((AXIS,)),
        "prior_mean": x_b,
        "prior_cov_op": _psd(B),
        "obs_cov_op": None,
    }
    if which == "strong":
        xa = StrongFourDVar(**common)(points)
    elif which == "weak":
        xa, eta = WeakFourDVar(**common, model_err_cov_op=_psd(jnp.eye(N)))(points)
        assert eta.shape == (0, N)
    elif which == "incremental":
        xa = ReducedBasisFourDVar(**common)(points)
    else:
        xa = IncrementalFourDVar(**common)(points)
    # Linear-Gaussian: every method minimises the same quadratic as OI
    # (exact up to float32), so ‖x̂ - x_OI‖ is within the gradient bound.
    assert jnp.linalg.norm(xa - x_oi) <= _quad_bound(xa, x_b, B, points)


def test_padding_invariance_of_strong_4dvar_analysis():
    B, _ = _ops()
    _, points, _ = _grid_and_points(jnp.array([1, 0, 1, 1, 0, 1], bool))
    model = StrongFourDVar(
        forward=IdentityForward(),
        obs_op=GridPointObs((AXIS,)),
        prior_mean=jnp.zeros(N),
        prior_cov_op=_psd(B),
        obs_cov_op=None,
    )
    # Padding changes the cost only by float32 summation order, so BFGS
    # follows the same path to within its stopping tolerance.
    assert jnp.allclose(model(points), model(_pad(points, 7)), atol=1e-5)


def test_batched_point_batch():
    B, _ = _ops()
    _, points, _ = _grid_and_points(jnp.ones(N, bool))
    stacked = jax.tree.map(lambda a: jnp.stack([a, a]), points)
    model = StrongFourDVar(
        forward=IdentityForward(),
        obs_op=GridPointObs((AXIS,)),
        prior_mean=jnp.zeros(N),
        prior_cov_op=_psd(B),
        obs_cov_op=None,
    )
    out = model(stacked)
    assert out.shape == (2, N)
    assert jnp.allclose(out[0], model(points), atol=1e-6)


def test_point_batch_needs_point_operator():
    _, points, _ = _grid_and_points(jnp.ones(N, bool))
    model = StrongFourDVar(
        forward=IdentityForward(),
        obs_op=MaskedIdentity(),
        prior_mean=jnp.zeros(N),
        prior_cov_op=_psd(jnp.eye(N)),
        obs_cov_op=None,
    )
    with pytest.raises(TypeError, match="GridPointObs"):
        model(points)


def test_time_index_needs_n_times():
    batch = _batch(AXIS, jnp.zeros(N), time_index=jnp.zeros(N, jnp.int32))
    model = StrongFourDVar(
        forward=IdentityForward(),
        obs_op=GridPointObs((AXIS,)),
        prior_mean=jnp.zeros(N),
        prior_cov_op=_psd(jnp.eye(N)),
        obs_cov_op=None,
    )
    with pytest.raises(ValueError, match="n_times"):
        model(batch)


def test_threedvar_field_point_obs_with_neural_field():
    angle = 2 * jnp.pi * jnp.arange(16) / 16
    coords = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    siren = geonnax.SIREN.init(
        2, 8, 1, depth=2, key=jax.random.PRNGKey(0), first_omega=3.0
    )
    param = NeuralFieldParameterization(siren, coords, 1e-4)
    obs_coords = coords[::2] * 0.9  # off-grid points inside the ring
    batch = _batch(obs_coords, jnp.cos(2 * angle[::2]), sigma=0.1 * jnp.ones(8))

    def cost(theta):
        pred = FieldPointObs()(param.as_field(theta), batch)
        return param.penalty(theta) + point_obs_cost(pred, batch)

    model = ThreeDVar(
        obs_op=FieldPointObs(),
        prior_mean=jnp.zeros((1, 16)),
        prior_cov_op=_psd(jnp.eye(16)),
        obs_cov_op=None,
        parameterization=param,
        minimiser=OptaxMinimiser(optax.adam(1e-2), max_steps=200),
    )
    xa = model(batch)
    assert xa.shape == (1, 16)
    theta_star = model.minimiser(lambda th, _: cost(th), param.init_control(), None)
    assert cost(theta_star) < 0.5 * cost(param.init_control())
    with pytest.raises(TypeError, match="NeuralFieldParameterization"):
        ThreeDVar(
            obs_op=FieldPointObs(),
            prior_mean=jnp.zeros((1, 16)),
            prior_cov_op=_psd(jnp.eye(16)),
            obs_cov_op=None,
        )(batch)


def _l96_problem(n_obs, key):
    k_truth, k_b, k_pos, k_t, k_o = jax.random.split(key, 5)
    forward = Lorenz96RK4()
    _, states = vdx.simulate_lorenz96(k_truth, N=40, n_steps=1, n_burn_in=500)
    x_true = states[0]
    n_times = 6
    traj = [x_true]
    for _ in range(n_times - 1):
        traj.append(forward.step(traj[-1], forward.dt))
    truth = jnp.stack(traj)
    axis = jnp.arange(40.0)
    pos = jax.random.uniform(k_pos, (n_obs,), maxval=39.0)
    t_idx = jax.random.randint(k_t, (n_obs,), 0, n_times)
    clean = GridPointObs((axis,))(
        truth, _batch(pos, jnp.zeros(n_obs), time_index=t_idx, n_times=n_times)
    )
    sigma = 0.5 * jnp.ones(n_obs)
    batch = _batch(
        pos,
        clean + sigma * jax.random.normal(k_o, (n_obs,)),
        sigma=sigma,
        time_index=t_idx,
        n_times=n_times,
    )
    x_b = x_true + jax.random.normal(k_b, (40,))
    model = StrongFourDVar(
        forward=forward,
        obs_op=GridPointObs((axis,)),
        prior_mean=x_b,
        prior_cov_op=_psd(jnp.eye(40)),
        obs_cov_op=None,
        max_steps=500,
    )

    def cost(x0):
        xs = [x0]
        for _ in range(n_times - 1):
            xs.append(forward.step(xs[-1], forward.dt))
        return 0.5 * jnp.sum((x0 - x_b) ** 2) + point_obs_cost(
            GridPointObs((axis,))(jnp.stack(xs), batch), batch
        )

    return model, _pad(batch, 10), cost, x_true, x_b


def _check_l96(n_obs):
    model, batch, cost, x_true, x_b = _l96_problem(n_obs, jax.random.PRNGKey(5))
    xa = model(batch)
    g_b = jnp.linalg.norm(jax.grad(cost)(x_b))
    g_a = jnp.linalg.norm(jax.grad(cost)(xa))
    # Converged: BFGS(rtol = atol = 1e-6) leaves a gradient orders of
    # magnitude below the background's (float32 gradient round-off of a
    # cost ~ n_obs is ~1e-4 relative).
    assert g_a < 1e-3 * g_b
    assert cost(xa) < cost(x_b)
    assert jnp.linalg.norm(xa - x_true) < jnp.linalg.norm(x_b - x_true)


def test_strong_4dvar_lorenz96_scattered_obs_smoke():
    _check_l96(n_obs=60)


@pytest.mark.slow
def test_strong_4dvar_lorenz96_scattered_obs_converges():
    _check_l96(n_obs=200)


# --- review follow-up -------------------------------------------------------


def test_field_point_obs_padding_is_finite_for_field_singular_at_origin():
    # Padding used to be evaluated at the origin; log|s| is singular there.
    batch = PointObsBatch(
        coords=jnp.array([[1.0], [jnp.nan]]),
        values=jnp.zeros(2),
        sigma=jnp.ones(2),
        valid=jnp.array([True, False]),
    )

    def cost(scale):
        pred = FieldPointObs()(lambda s: scale * jnp.log(jnp.abs(s)), batch)
        return point_obs_cost(pred, batch)

    v, g = jax.value_and_grad(cost)(1.0)
    # The only valid row is at s = 1, where log|s| = 0: value and gradient 0.
    assert v == 0.0
    assert g == 0.0


def test_out_of_range_valid_time_index_is_an_error():
    batch = _batch(
        jnp.array([1.0, 2.0]),
        jnp.zeros(2),
        time_index=jnp.array([0, 7]),
        n_times=3,
    )
    model = StrongFourDVar(
        forward=IdentityForward(),
        obs_op=GridPointObs((AXIS,)),
        prior_mean=jnp.zeros(N),
        prior_cov_op=_psd(jnp.eye(N)),
        obs_cov_op=None,
    )
    with pytest.raises(ValueError, match="time_index"):
        model(batch)
    # Padding rows may hold any index.
    padded = _pad(
        _batch(jnp.array([1.0]), jnp.zeros(1), time_index=jnp.array([2]), n_times=3), 2
    )
    assert model(padded).shape == (N,)


def test_traced_time_index_is_clamped():
    H = GridPointObs((AXIS,))
    traj = jnp.stack([AXIS, 10 * AXIS, 100 * AXIS])

    @jax.jit
    def apply(t_idx):
        b = _batch(jnp.array([1.0]), jnp.zeros(1), time_index=t_idx, n_times=3)
        return H(traj, b)

    assert jnp.allclose(apply(jnp.array([7])), 100.0)


@pytest.mark.parametrize(
    "cov", [_diag(jnp.ones(N)), _psd(jnp.eye(N)), _psd(2.0 * jnp.eye(N))]
)
def test_point_batch_rejects_obs_cov_op(cov):
    _, points, _ = _grid_and_points(jnp.ones(N, bool))
    model = StrongFourDVar(
        forward=IdentityForward(),
        obs_op=GridPointObs((AXIS,)),
        prior_mean=jnp.zeros(N),
        prior_cov_op=_psd(jnp.eye(N)),
        obs_cov_op=cov,
    )
    with pytest.raises(ValueError, match="obs_cov_op"):
        model(points)
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    assert eqx.tree_at(lambda m: m.obs_cov_op, model, identity)(points).shape == (N,)


class MixingForward(eqx.Module):
    """Non-trivial linear dynamics: x ← 0.8 x + 0.1 roll(x, 1)."""

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return 0.8 * state + 0.1 * jnp.roll(state, 1)


@pytest.mark.parametrize("which", ["strong", "weak", "incremental", "incremental_gn"])
def test_scattered_time_points_match_grid_batch(which):
    """Scattered (t, i) point observations reproduce the grid-batch analysis.

    Three states of mixing dynamics, five observations at scattered
    (time, site) pairs, R = I. The point cost equals the masked grid cost
    term by term, so both runs minimise the same function from the same
    start and follow the same iterates up to float32 summation order (the
    reviewer's probe saw 6e-8). 1e-5 is 5x the solvers' step tolerance
    (BFGS rtol = atol = 1e-6 on an O(1) state), covering a stop one
    iteration apart.
    """
    T1 = 3
    y = jax.random.normal(jax.random.PRNGKey(3), (T1, N))
    mask = (
        jnp.zeros((T1, N))
        .at[0, 1]
        .set(1)
        .at[1, 3]
        .set(1)
        .at[2, 0]
        .set(1)
        .at[2, 5]
        .set(1)
        .at[1, 2]
        .set(1)
    )
    grid = Batch1D(input=(y * mask)[None], mask=mask[None])
    t, i = jnp.nonzero(mask)
    points = _pad(
        _batch(AXIS[i], y[t, i], time_index=t.astype(jnp.int32), n_times=T1), 2
    )
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float32))
    common = {
        "forward": MixingForward(),
        "prior_mean": jnp.linspace(-1.0, 1.0, N),
        "prior_cov_op": eye,
    }
    cls, extra = {
        "strong": (StrongFourDVar, {}),
        "weak": (WeakFourDVar, {"model_err_cov_op": eye}),
        "incremental": (ReducedBasisFourDVar, {}),
        "incremental_gn": (IncrementalFourDVar, {}),
    }[which]
    xg = cls(obs_op=MaskedIdentity(), obs_cov_op=eye, **common, **extra)(grid)
    xp = cls(obs_op=GridPointObs((AXIS,)), obs_cov_op=None, **common, **extra)(points)
    if which == "weak":
        (xg, eta_g), (xp, eta_p) = xg, xp
        assert jnp.allclose(eta_g[0], eta_p, atol=1e-5)
    assert jnp.allclose(xg[0], xp, atol=1e-5)
