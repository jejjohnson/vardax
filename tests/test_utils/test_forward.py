"""Tests for the ``SoftBoundedForward`` adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pipekit_cycle as pc

from vardax import DynTrajectory, Lorenz63, SoftBoundedForward


class Identity:
    dt = 0.5
    state_signature = None

    def step(self, x, dt):
        return x


def test_identity_inside_box():
    fwd = SoftBoundedForward(Identity(), bound=10.0)
    x = jnp.array([-9.9, 0.0, 3.0, 10.0])
    assert jnp.allclose(fwd.step(x, 0.5), x)


def test_confined_outside_box():
    fwd = SoftBoundedForward(Identity(), bound=10.0)
    x = jnp.array([-1e9, 15.0, 1e3])
    out = fwd.step(x, 0.5)
    assert jnp.all(jnp.abs(out) < 20.0)
    assert jnp.all(jnp.sign(out) == jnp.sign(x))
    # Monotone just outside the box: further out stays further out.
    assert float(out[1]) < float(fwd.step(jnp.array([25.0]), 0.5)[0]) < 20.0


def test_identity_gradient_inside_box_including_zero():
    fwd = SoftBoundedForward(Identity(), bound=10.0)
    x = jnp.array([0.0, -0.0, 2.5, -9.0])
    jac = jax.jacfwd(lambda v: fwd.step(v, 0.5))(x)
    assert jnp.allclose(jac, jnp.eye(4))


def test_gradient_nonzero_outside_box():
    fwd = SoftBoundedForward(Identity(), bound=10.0)
    g = jax.grad(lambda x: jnp.sum(fwd.step(x, 0.5)))(jnp.array([25.0, -40.0]))
    assert jnp.all(jnp.isfinite(g))
    assert jnp.all(g > 0.0)


def test_satisfies_forward_model_protocol_and_delegates_dt():
    ode = DynTrajectory(model=Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0))
    fwd = SoftBoundedForward(ode.as_forward_model(dt=0.05), bound=60.0)
    assert isinstance(fwd, pc.ForwardModel)
    assert fwd.dt == 0.05
    x = jnp.array([1.0, 2.0, 30.0])
    # On the attractor the adapter is exactly the wrapped model.
    assert jnp.allclose(fwd.step(x, 0.05), ode.as_forward_model(dt=0.05).step(x, 0.05))
