"""Checkpointed rollouts and the ``forward_adjoint`` slot (issue #95).

Before #95, ``StrongFourDVar(forward_adjoint=...)`` stored the adjoint and
ignored it, and no 4DVar model could bound rollout memory. These tests pin:

- ``checkpoint=`` gives the same gradient as plain reverse mode (1e-10,
  float64) and lowers the compiled gradient's temp memory on a long window;
- every 4DVar model gives the same analysis with and without it;
- ``forward_adjoint=`` reaches a diffrax-backed forward through
  ``with_adjoint`` and raises for a per-step forward.
"""

from __future__ import annotations

import dataclasses

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import vardax as vdx
from vardax._src.models._rollout import checkpoint_every, rollout

N_SITES, DT, FORCING = 40, 0.01, 8.0


class L96RK4(eqx.Module):
    """Lorenz-96, one RK4 step per call."""

    dt: float = DT

    def step(self, x, dt):
        def tendency(x):
            # dx_k/dt = (x_{k+1} - x_{k-2}) x_{k-1} - x_k + F
            return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + FORCING

        k1 = tendency(x)
        k2 = tendency(x + 0.5 * dt * k1)
        k3 = tendency(x + 0.5 * dt * k2)
        k4 = tendency(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclasses.dataclass(frozen=True)
class DiffraxDecay:
    """dx/dt = −λx integrated by diffrax, honouring ``with_adjoint``."""

    dt: float = 0.1
    rate: float = 0.7
    adjoint: dfx.AbstractAdjoint = dataclasses.field(
        default_factory=dfx.RecursiveCheckpointAdjoint
    )

    def step(self, x, dt):
        solution = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, args: -self.rate * y),
            dfx.Tsit5(),
            t0=0.0,
            t1=dt,
            dt0=dt / 4,
            y0=x,
            adjoint=self.adjoint,
        )
        return solution.ys[-1]

    def with_adjoint(self, adjoint):
        return dataclasses.replace(self, adjoint=adjoint)


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _l96_cost(checkpoint, n_steps):
    target = jnp.cos(jnp.arange((n_steps + 1) * N_SITES)).reshape(n_steps + 1, -1)

    def cost(x0):
        trajectory = rollout(L96RK4(), x0, n_steps, checkpoint=checkpoint)
        # J = ½ ∑ₜ ‖xₜ - yₜ‖²
        return 0.5 * jnp.sum((trajectory - target) ** 2)

    return cost


class TestRolloutHelper:
    @pytest.mark.parametrize("checkpoint", [True, 1, 3, 7, 50])
    def test_trajectory_identical(self, checkpoint):
        x0 = FORCING + jnp.sin(jnp.arange(N_SITES, dtype=jnp.float32))
        plain = rollout(L96RK4(), x0, 23)
        checkpointed = rollout(L96RK4(), x0, 23, checkpoint=checkpoint)
        assert plain.shape == (24, N_SITES)
        assert jnp.array_equal(plain, checkpointed)

    def test_forcing_with_tail_block(self):
        forcing = jnp.arange(10.0).reshape(5, 2)
        plain = rollout(L96RK4(dt=0.0), jnp.zeros(2), 5, forcing=forcing)
        blocked = rollout(
            L96RK4(dt=0.0), jnp.zeros(2), 5, forcing=forcing, checkpoint=2
        )
        # dt = 0 makes the step the identity: xₜ = ∑_{s<t} ηₛ
        expected = jnp.concatenate([jnp.zeros((1, 2)), jnp.cumsum(forcing, axis=0)])
        assert jnp.allclose(plain, expected)
        assert jnp.array_equal(plain, blocked)

    def test_zero_steps(self):
        out = rollout(L96RK4(), jnp.ones(3), 0, checkpoint=True)
        assert out.shape == (1, 3)

    @pytest.mark.unchecked_types
    @pytest.mark.parametrize("bad", [0, -2, 1.5, "yes"])
    def test_rejects_bad_checkpoint(self, bad):
        with pytest.raises(ValueError, match="checkpoint"):
            checkpoint_every(bad)

    @pytest.mark.parametrize("checkpoint", [True, 10])
    def test_gradient_matches_reverse_mode(self, x64, checkpoint):
        # 2 time units of L96: long enough to matter, short enough that
        # chaos does not amplify round-off of the recomputed steps
        n_steps = 200
        x0 = FORCING + jnp.sin(jnp.arange(N_SITES, dtype=jnp.float64))
        reference = jax.grad(_l96_cost(False, n_steps))(x0)
        gradient = jax.grad(_l96_cost(checkpoint, n_steps))(x0)
        scale = jnp.max(jnp.abs(reference))
        assert jnp.max(jnp.abs(gradient - reference)) / scale < 1e-10

    @pytest.mark.parametrize("checkpoint", [True, 10])
    def test_gradient_temp_memory_drops(self, x64, checkpoint):
        n_steps = 500
        x0 = FORCING + jnp.sin(jnp.arange(N_SITES, dtype=jnp.float64))

        def temp_bytes(ck):
            gradient = jax.jit(jax.grad(_l96_cost(ck, n_steps)))
            analysis = gradient.lower(x0).compile().memory_analysis()
            return analysis.temp_size_in_bytes

        plain, checkpointed = temp_bytes(False), temp_bytes(checkpoint)
        # measured on CPU: 1.61 MB → 0.49 MB (True), 0.37 MB (10)
        assert checkpointed < 0.5 * plain


def _models(forward, n_sites, **slots):
    eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n_sites,), jnp.float32))
    common = dict(
        forward=forward,
        obs_op=vdx.MaskedIdentity(),
        prior_mean=jnp.zeros(n_sites),
        prior_cov_op=eye,
        obs_cov_op=eye,
    )
    return {
        "strong": lambda: vdx.StrongFourDVar(**common, **slots),
        "weak": lambda: vdx.WeakFourDVar(**common, model_err_cov_op=eye, **slots),
        "incremental": lambda: vdx.IncrementalFourDVar(**common, **slots),
        "reduced_basis": lambda: vdx.ReducedBasisFourDVar(**common, **slots),
    }


class _Decay(eqx.Module):
    dt: float = 1.0

    def step(self, x, dt):
        return 0.9 * x


def _decay_batch(n_sites=3, n_times=4):
    truth = 0.9 ** jnp.arange(n_times)[:, None] * jnp.linspace(1.0, 2.0, n_sites)
    return vdx.Batch1D(input=truth[None], mask=jnp.ones((1, n_times, n_sites)))


class TestModelSlots:
    @pytest.mark.parametrize("name", ["strong", "weak", "incremental", "reduced_basis"])
    def test_checkpoint_keeps_the_analysis(self, name):
        batch = _decay_batch()
        plain = _models(_Decay(), 3)[name]()(batch)
        checkpointed = _models(_Decay(), 3, checkpoint=True)[name]()(batch)
        if name == "weak":
            plain, checkpointed = plain[0], checkpointed[0]
        assert jnp.allclose(plain, checkpointed, atol=1e-6)

    @pytest.mark.parametrize("name", ["strong", "weak", "incremental", "reduced_basis"])
    def test_forward_adjoint_on_per_step_forward_raises(self, name):
        with pytest.raises(ValueError, match="with_adjoint"):
            _models(_Decay(), 3, forward_adjoint=dfx.BacksolveAdjoint())[name]()

    @pytest.mark.parametrize("name", ["strong", "weak", "incremental", "reduced_basis"])
    def test_bad_checkpoint_raises_at_construction(self, name):
        with pytest.raises(ValueError, match="checkpoint"):
            _models(_Decay(), 3, checkpoint=0)[name]()

    def test_forward_adjoint_reaches_the_integrator(self, monkeypatch):
        seen = []
        original = DiffraxDecay.with_adjoint

        def spy(self, adjoint):
            seen.append(adjoint)
            return original(self, adjoint)

        monkeypatch.setattr(DiffraxDecay, "with_adjoint", spy)
        adjoint = dfx.BacksolveAdjoint()
        model = _models(DiffraxDecay(), 3, forward_adjoint=adjoint)["strong"]()
        seen.clear()
        trajectory = model._rollout(jnp.ones(3), 4)
        assert trajectory.shape == (5, 3)
        assert seen == [adjoint]

    def test_backsolve_gradient_matches_default(self):
        """Same gradient through a diffrax forward with either adjoint."""
        batch = _decay_batch()

        def cost_at(model):
            def cost(x0):
                trajectory = model._rollout(x0, 3)
                return 0.5 * jnp.sum((trajectory - batch.input[0]) ** 2)

            return jax.grad(cost)(jnp.ones(3))

        default = _models(DiffraxDecay(), 3)["strong"]()
        backsolve = _models(DiffraxDecay(), 3, forward_adjoint=dfx.BacksolveAdjoint())[
            "strong"
        ]()
        assert jnp.allclose(cost_at(default), cost_at(backsolve), rtol=1e-3)

    def test_protocol_conformance(self):
        assert isinstance(DiffraxDecay(), vdx.SupportsForwardAdjoint)
        assert not isinstance(_Decay(), vdx.SupportsForwardAdjoint)
