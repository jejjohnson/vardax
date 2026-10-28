"""StateParameterization, its two implementations and ``parameterization=`` (#84)."""

from __future__ import annotations

import equinox as eqx
import geonnax
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import pytest

import vardax as vdx
from vardax import (
    Batch1D,
    LinearParameterization,
    MaskedIdentity,
    NeuralFieldParameterization,
    OptaxMinimiser,
    ReducedBasisFourDVar,
    StateParameterization,
    StrongFourDVar,
    ThreeDVar,
    WeakFourDVar,
    linear_basis,
    rbf_basis,
)


class IdentityForward(eqx.Module):
    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return state


class DecayForward(eqx.Module):
    """``x_{t+1} = 0.9 x_t``: linear dynamics with a non-trivial rollout."""

    @property
    def dt(self) -> float:
        return 1.0

    def step(self, state, dt):
        return 0.9 * state


class Lorenz96RK4(eqx.Module):
    """Lorenz-96 (F = 8) stepped with classical RK4 (a ``ForwardModel``)."""

    dt: float = 0.01

    def step(self, x, dt):
        f = vdx.Lorenz96(forcing=8.0)
        k1 = f(0.0, x, None)
        k2 = f(0.0, x + 0.5 * dt * k1, None)
        k3 = f(0.0, x + 0.5 * dt * k2, None)
        k4 = f(0.0, x + dt * k3, None)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _eye(shape):
    return lx.IdentityLinearOperator(jax.ShapeDtypeStruct(shape, jnp.float32))


def _ring(n=40):
    angle = 2.0 * jnp.pi * jnp.arange(n) / n
    return jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)


EPS = float(jnp.finfo(jnp.float32).eps)


def _rbf():
    return rbf_basis(
        (jnp.arange(6.0),), centers=jnp.array([[0.0], [2.5], [5.0]]), widths=2.0
    )


def _phi(basis):
    return jnp.stack([basis.operg(0.0, e) for e in jnp.eye(basis.nbasis)], axis=1)


def _closed_form(batch, x_b, basis):
    """Normal equations of the reduced-basis cost, in float64 (NumPy).

    J(X) = ½ XᵀQ⁻¹X + ½ Σ_t ‖m_t ⊙ (y_t - 0.9^t (x_b + ΦX))‖², so
    (Q⁻¹ + Σ_t A_tᵀA_t) X = Σ_t A_tᵀ(y_t - m_t 0.9^t x_b), A_t = 0.9^t m_t Φ.
    """
    phi = np.asarray(_phi(basis), dtype=np.float64)
    m = phi.shape[1]
    q_inv = np.stack(
        [np.asarray(basis.prior_inv(jnp.asarray(e))) for e in np.eye(m)], axis=1
    )
    y = np.asarray(batch.input[0], dtype=np.float64)
    mask = np.asarray(batch.mask[0], dtype=np.float64)
    xb = np.asarray(x_b, dtype=np.float64)
    h, g = q_inv.astype(np.float64), np.zeros(m)
    for t in range(y.shape[0]):
        a = (0.9**t) * mask[t][:, None] * phi
        h = h + a.T @ a
        g = g + a.T @ (y[t] - mask[t] * 0.9**t * xb)
    X = np.linalg.solve(h, g)
    j_star = 0.5 * X @ q_inv @ X
    for t in range(y.shape[0]):
        r = mask[t] * (y[t] - 0.9**t * (xb + phi @ X))
        j_star = j_star + 0.5 * r @ r
    x_cf = jnp.asarray(xb + phi @ X, dtype=jnp.float32)
    return (
        x_cf,
        float(j_star),
        float(np.linalg.eigvalsh(h)[0]),
        float(np.linalg.norm(phi, 2)),
    )


@pytest.fixture
def window():
    """N = 6 state, 3 rollout steps of decay dynamics, partial observations."""
    key_x, key_m = jax.random.split(jax.random.PRNGKey(0))
    N, T1 = 6, 4
    x0 = jax.random.normal(key_x, (N,))
    traj = jnp.stack([0.9**t * x0 for t in range(T1)])
    mask = (jax.random.uniform(key_m, (T1, N)) > 0.3).astype(jnp.float32)
    batch = Batch1D(input=(traj * mask)[None], mask=mask[None])
    x_b = 0.5 * jnp.ones(N)
    return batch, x_b


class TestLinearParameterization:
    def test_protocol(self):
        basis = linear_basis(jnp.eye(4)[:, :2])
        param = LinearParameterization(basis, jnp.zeros(4))
        assert isinstance(param, StateParameterization)

    def test_decode_and_penalty_closed_form(self):
        phi = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
        variance = jnp.array([1.0, 2.0, 4.0])
        basis = linear_basis(phi, variance=variance)
        x_b = jnp.arange(5.0)
        param = LinearParameterization(basis, x_b)
        X = jnp.array([1.0, -1.0, 2.0])
        assert jnp.allclose(param.decode(X), x_b + phi @ X)
        assert jnp.allclose(param.penalty(X), 0.5 * jnp.sum(X**2 / variance))
        assert jnp.array_equal(param.init_control(), jnp.zeros(3))

    @pytest.mark.slow
    def test_matches_incremental_4dvar_basis_and_closed_form(self, window):
        batch, x_b = window
        basis = _rbf()
        common = {
            "forward": DecayForward(),
            "obs_op": MaskedIdentity(),
            "prior_mean": x_b,
            "prior_cov_op": _eye((6,)),
            "obs_cov_op": _eye((6,)),
        }
        x_basis = ReducedBasisFourDVar(**common, basis=basis)(batch)[0]
        x_param = ReducedBasisFourDVar(
            **common, parameterization=LinearParameterization(basis, x_b)
        )(batch)[0]
        x_cf, j_star, lam_min, phi_norm = _closed_form(batch, x_b, basis)
        # Tied to BFGS(rtol = atol = 1e-6): it stops once the cost decrease
        # is below atol + rtol·|J|. Allowing the remaining gap to be 10x that
        # last decrease, J(X̂) - J* ≤ δ := 10 (1e-6 + 1e-6 |J*|), and for this
        # quadratic ½ λ_min ‖X̂ - X*‖² ≤ J(X̂) - J*, so
        # ‖x̂ - x*‖ ≤ ‖Φ‖₂ sqrt(2δ / λ_min), plus float32 round-off of x.
        delta = 10 * (1e-6 + 1e-6 * abs(j_star))
        bound = phi_norm * np.sqrt(2 * delta / lam_min)
        bound = bound + 8 * EPS * float(jnp.max(jnp.abs(x_cf)))
        assert jnp.linalg.norm(x_basis - x_cf) <= bound
        assert jnp.linalg.norm(x_param - x_cf) <= bound
        # Both minimise the same cost; they agree within twice that bound.
        assert jnp.linalg.norm(x_basis - x_param) <= 2 * bound

    def test_basis_path_matches_pre_pr_reference(self, window):
        # ReducedBasisFourDVar(basis=...) output on main before this PR (the
        # basis path is the same code). 1e-5 allows cross-platform float32
        # differences in the BFGS iterates; routing it through
        # LinearParameterization moved it by 6.6e-5.
        batch, x_b = window
        x = ReducedBasisFourDVar(
            forward=DecayForward(),
            obs_op=MaskedIdentity(),
            prior_mean=x_b,
            prior_cov_op=_eye((6,)),
            obs_cov_op=_eye((6,)),
            basis=_rbf(),
        )(batch)[0]
        reference = jnp.array(
            [-0.4702341, -0.7426063, -0.8532791, -0.7294558, -0.41439342, -0.05025512]
        )
        assert jnp.allclose(x, reference, rtol=0.0, atol=1e-5)

    def test_strong_4dvar_matches_incremental_basis(self, window):
        batch, x_b = window
        basis = rbf_basis(
            (jnp.arange(6.0),), centers=jnp.array([[0.0], [2.5], [5.0]]), widths=2.0
        )
        x_basis = ReducedBasisFourDVar(
            forward=DecayForward(),
            obs_op=MaskedIdentity(),
            prior_mean=x_b,
            prior_cov_op=_eye((6,)),
            obs_cov_op=_eye((6,)),
            basis=basis,
        )(batch)
        x_strong = StrongFourDVar(
            forward=DecayForward(),
            obs_op=MaskedIdentity(),
            prior_mean=x_b,
            prior_cov_op=_eye((6,)),
            obs_cov_op=_eye((6,)),
            parameterization=LinearParameterization(basis, x_b),
        )(batch)
        # Same cost (R = I, 0/1 mask: the masked-R solve is the identity),
        # different code path: BFGS(rtol=atol=1e-6) agreement.
        assert jnp.allclose(x_basis, x_strong, atol=1e-4)

    @pytest.mark.parametrize("which", ["three", "strong", "weak", "incremental"])
    def test_identity_basis_reproduces_raw_control(self, which):
        # Φ = I, Q = B = I: pen(X) = ½‖x - x_b‖², the default cost.
        y = jax.random.normal(jax.random.PRNGKey(2), (2, 1, 4))
        batch = Batch1D(input=y, mask=jnp.ones_like(y))
        basis = linear_basis(jnp.eye(4))
        x_b = jnp.full(4, 0.3)
        if which == "three":
            param = LinearParameterization(
                linear_basis(jnp.eye(4), output_shape=(1, 4)), x_b[None]
            )

            def build(**kw):
                return ThreeDVar(
                    obs_op=MaskedIdentity(),
                    prior_mean=x_b[None],
                    prior_cov_op=_eye((1, 4)),
                    obs_cov_op=_eye((1, 4)),
                    **kw,
                )

            ref, out = build()(batch), build(parameterization=param)(batch)
        else:
            cls = {
                "strong": StrongFourDVar,
                "weak": WeakFourDVar,
                "incremental": ReducedBasisFourDVar,
            }[which]
            kw = {"model_err_cov_op": _eye((4,))} if which == "weak" else {}

            def build(**extra):
                return cls(
                    forward=IdentityForward(),
                    obs_op=MaskedIdentity(),
                    prior_mean=x_b,
                    prior_cov_op=_eye((4,)),
                    obs_cov_op=_eye((4,)),
                    **kw,
                    **extra,
                )

            param = LinearParameterization(basis, x_b)
            ref, out = build()(batch), build(parameterization=param)(batch)
            if which == "weak":
                ref, out = ref[0], out[0]
        # Both BFGS(rtol=atol=1e-6) on the same quadratic.
        assert jnp.allclose(ref, out, atol=1e-4)
        assert jnp.allclose(out.reshape(2, 4), (x_b + y[:, 0]) / 2, atol=1e-4)

    def test_basis_and_parameterization_are_exclusive(self):
        basis = linear_basis(jnp.eye(4))
        with pytest.raises(ValueError, match="not both"):
            ReducedBasisFourDVar(
                forward=IdentityForward(),
                obs_op=MaskedIdentity(),
                prior_mean=jnp.zeros(4),
                prior_cov_op=_eye((4,)),
                obs_cov_op=_eye((4,)),
                basis=basis,
                parameterization=LinearParameterization(basis, jnp.zeros(4)),
            )


class TestNeuralFieldParameterization:
    def _param(self, weight_decay=1e-4):
        siren = geonnax.SIREN.init(
            2, 16, 1, depth=3, key=jax.random.PRNGKey(0), first_omega=3.0
        )
        return NeuralFieldParameterization(siren, _ring(), weight_decay)

    @pytest.mark.slow
    def test_protocol_and_shapes(self):
        param = self._param()
        assert isinstance(param, StateParameterization)
        theta = param.init_control()
        assert param.decode(theta).shape == (40,)
        # A time column for a space-time field: D + 1 = 3 inputs.
        siren_t = geonnax.SIREN.init(3, 8, 1, depth=2, key=jax.random.PRNGKey(1))
        param_t = NeuralFieldParameterization(siren_t, _ring(), 0.0)
        assert param_t.decode(param_t.init_control(), t=0.5).shape == (40,)

    def test_decode_matches_field(self):
        param = self._param()
        expected = jax.vmap(param.field)(_ring())[:, 0]
        assert jnp.allclose(param.decode(param.init_control()), expected)

    def test_penalty_is_weight_decay(self):
        param = self._param(weight_decay=0.1)
        theta = param.init_control()
        sq = sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(theta))
        assert jnp.allclose(param.penalty(theta), 0.05 * sq)

    @pytest.mark.unchecked_types
    def test_validation(self):
        siren = geonnax.SIREN.init(2, 8, 1, depth=2, key=jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="coords"):
            NeuralFieldParameterization(siren, jnp.zeros(40), 0.0)
        with pytest.raises(ValueError, match="weight_decay"):
            NeuralFieldParameterization(siren, _ring(), -1.0)

    @pytest.mark.slow
    def test_frozen_leaves_stay_fixed_with_filter_spec(self):
        # An RFF field: frozen frequencies B behind stop_gradient (the geonnax
        # convention) and a trainable linear head.
        class RFFField(eqx.Module):
            B: jax.Array
            lin: eqx.nn.Linear

            def __call__(self, s):
                z = s @ jax.lax.stop_gradient(self.B)
                return self.lin(jnp.concatenate([jnp.cos(z), jnp.sin(z)]))

        field = RFFField(
            3.0 * jax.random.normal(jax.random.PRNGKey(0), (2, 8)),
            eqx.nn.Linear(16, 1, key=jax.random.PRNGKey(1)),
        )
        spec = eqx.tree_at(
            lambda f: f.B, jax.tree.map(eqx.is_inexact_array, field), False
        )
        coords = _ring(20)
        param = NeuralFieldParameterization(field, coords, 1.0, filter_spec=spec)
        theta = param.init_control()
        assert theta.B is None  # not a control
        assert jnp.allclose(param.decode(theta), jax.vmap(field)(coords)[:, 0])
        # The penalty only sees the head; minimising it leaves B untouched.
        head_sq = sum(jnp.sum(a**2) for a in jax.tree.leaves(theta))
        assert jnp.allclose(param.penalty(theta), 0.5 * head_sq)
        y = jnp.sin(3 * 2 * jnp.pi * jnp.arange(20) / 20)
        theta_star = OptaxMinimiser(optax.adam(1e-2), max_steps=300)(
            lambda th, _: (
                param.penalty(th) + 0.5 * jnp.sum((param.decode(th) - y) ** 2)
            ),
            theta,
            None,
        )
        assert theta_star.B is None
        rebuilt = eqx.combine(theta_star, eqx.filter(field, spec, inverse=True))
        assert jnp.array_equal(rebuilt.B, field.B)

    @pytest.mark.slow
    def test_grad_flows_to_parameters(self):
        param = self._param()
        g = jax.grad(lambda th: jnp.sum(param.decode(th) ** 2))(param.init_control())
        assert any(jnp.any(leaf != 0) for leaf in jax.tree.leaves(g))

    @pytest.mark.slow
    def test_siren_strong_4dvar_lorenz96_reduces_cost(self):
        key_x, key_o = jax.random.split(jax.random.PRNGKey(3))
        forward = Lorenz96RK4()
        _, states = vdx.simulate_lorenz96(key_x, n_sites=40, n_steps=1, n_burn_in=500)
        x_true = states[0]
        T = 10
        traj = [x_true]
        for _ in range(T):
            traj.append(forward.step(traj[-1], forward.dt))
        truth = jnp.stack(traj)
        y = truth + jax.random.normal(key_o, truth.shape)
        batch = Batch1D(input=y[None], mask=jnp.ones_like(y)[None])

        # A small SIREN over the 40-site ring, (cos, sin) inputs so the
        # field is periodic; first_omega=10 resolves the L96 wavenumbers.
        siren = geonnax.SIREN.init(
            2, 32, 1, depth=3, key=jax.random.PRNGKey(0), first_omega=10.0
        )
        param = NeuralFieldParameterization(siren, _ring(), 1e-4)
        minimiser = OptaxMinimiser(optax.adam(1e-2), max_steps=400, patience=50)
        model = StrongFourDVar(
            forward=forward,
            obs_op=MaskedIdentity(),
            prior_mean=jnp.zeros(40),
            prior_cov_op=_eye((40,)),
            obs_cov_op=_eye((40,)),
            parameterization=param,
            minimiser=minimiser,
        )

        # The cost the model minimises (R = I, everything observed).
        def cost(theta, _):
            xs = [param.decode(theta)]
            for _ in range(T):
                xs.append(forward.step(xs[-1], forward.dt))
            return param.penalty(theta) + 0.5 * jnp.sum((y - jnp.stack(xs)) ** 2)

        theta0 = param.init_control()
        result = minimiser.run(cost, theta0, None)
        # The 4D-Var cost falls by an order of magnitude (from ~4e3 to the
        # noise level ~ n_obs / 2 = 220).
        assert result.cost < 0.1 * cost(theta0, None)
        # And the model's analysis is far closer to the truth than the
        # untrained field.
        x0_star = model(batch)[0]
        err0 = jnp.linalg.norm(param.decode(theta0) - x_true)
        assert jnp.linalg.norm(x0_star - x_true) < 0.25 * err0
