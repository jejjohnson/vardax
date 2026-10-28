"""Lorenz-63 and Lorenz-96 dynamical system simulations using Diffrax."""

from __future__ import annotations

from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Lorenz63(eqx.Module):
    """Lorenz-63 vector field as an Equinox module.

    Implements the system:
        dx/dt = σ(y − x)
        dy/dt = x(ρ − z) − y
        dz/dt = xy − βz

    Attributes:
        sigma: Prandtl number (default 10.0).
        rho: Rayleigh number (default 28.0).
        beta: Geometric factor (default 8/3).
    """

    sigma: float
    rho: float
    beta: float

    def __call__(
        self,
        t: float | Float[Array, ""],
        y: Float[Array, " 3"],
        args: None,
    ) -> Float[Array, " 3"]:
        """Evaluate the vector field at state ``y`` and time ``t``."""
        state_x, state_y, state_z = y[0], y[1], y[2]
        # dx/dt = σ(y − x)
        dx_dt = self.sigma * (state_y - state_x)
        # dy/dt = x(ρ − z) − y
        dy_dt = state_x * (self.rho - state_z) - state_y
        # dz/dt = xy − βz
        dz_dt = state_x * state_y - self.beta * state_z
        return jnp.array([dx_dt, dy_dt, dz_dt])


def simulate_lorenz63(
    key: Array,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    n_steps: int = 5000,
    n_burn_in: int = 1000,
    x0: Float[Array, " 3"] | None = None,
) -> tuple[Float[Array, " T"], Float[Array, "T 3"]]:
    """Simulate the Lorenz-63 system and return state trajectory.

    Args:
        key: JAX PRNG key used to perturb the initial condition when
            ``x0`` is ``None``.
        sigma: Prandtl number.
        rho: Rayleigh number.
        beta: Geometric factor.
        dt: Integration time step.
        n_steps: Total number of integration steps *after* burn-in.
        n_burn_in: Number of initial steps to discard (burn-in).
        x0: Optional explicit initial condition of shape ``(3,)``.  When
            ``None``, the classic off-attractor point ``(1, 1, 1)`` is used,
            perturbed by ``key``. From there the transient onto the
            attractor takes a couple of time units, so the default burn-in
            lands on it. (Starting at the unstable fixed point instead, as
            older versions did, leaves the trajectory within a fraction of
            a unit of it for ~40 time units — far longer than any burn-in
            the tutorials use.)

    Returns:
        ``(time_coords, states)`` — time coordinates of shape ``(T,)``
        (starting at 0) and the Lorenz-63 state trajectory of shape
        ``(T, 3)``.
    """
    model = Lorenz63(sigma=sigma, rho=rho, beta=beta)

    if x0 is None:
        noise = jax.random.normal(key, shape=(3,)) * 0.01
        x0 = jnp.array([1.0, 1.0, 1.0]) + noise

    total_steps = n_burn_in + n_steps
    start_time = 0.0
    end_time = total_steps * dt

    save_times = jnp.linspace(start_time, end_time, total_steps + 1)

    solution = diffeqsolve(
        ODETerm(model),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        Tsit5(),
        t0=start_time,
        t1=end_time,
        dt0=dt,
        y0=x0,
        saveat=SaveAt(ts=save_times),
        # Fixed-step integration takes exactly ``total_steps`` steps; diffrax's
        # default cap (4096) would abort long simulations (e.g. 5000 + 1000).
        max_steps=total_steps + 1,
    )

    # Discard burn-in steps; keep indices n_burn_in .. total_steps (inclusive)
    states = solution.ys[n_burn_in:]
    time_coords = save_times[n_burn_in:] - save_times[n_burn_in]
    return time_coords, states


class Lorenz96(eqx.Module):
    """Lorenz-96 vector field as an Equinox module.

    Implements the system (with periodic boundary conditions):
        dx_k/dt = (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + F

    Attributes:
        F: Forcing constant (default 8.0).
    """

    F: float

    def __call__(
        self,
        t: float | Float[Array, ""],
        y: Float[Array, " N"],
        args: None,
    ) -> Float[Array, " N"]:
        """Evaluate the vector field at state ``y`` and time ``t``."""
        # Periodic shifts using jnp.roll
        y_km2 = jnp.roll(y, 2)  # x_{k-2}
        y_km1 = jnp.roll(y, 1)  # x_{k-1}
        y_kp1 = jnp.roll(y, -1)  # x_{k+1}
        return (y_kp1 - y_km2) * y_km1 - y + self.F


def simulate_lorenz96(
    key: Array,
    *,
    N: int = 40,
    F: float = 8.0,
    dt: float = 0.01,
    n_steps: int = 5000,
    n_burn_in: int = 1000,
) -> tuple[Float[Array, " T"], Float[Array, "T N"]]:
    """Simulate the Lorenz-96 system and return state trajectory.

    Args:
        key: JAX PRNG key used to perturb the initial condition.
        N: Number of variables (spatial dimension).
        F: Forcing constant.
        dt: Integration time step.
        n_steps: Total number of integration steps *after* burn-in.
        n_burn_in: Number of initial steps to discard (burn-in).

    Returns:
        ``(time_coords, states)`` — time coordinates of shape ``(T,)``
        (starting at 0) and the state trajectory of shape
        ``(n_steps + 1, N)``.
    """
    model = Lorenz96(F=F)

    # Standard L96 initialization: uniform forcing with a small random perturbation
    initial_state = jnp.full((N,), F)
    noise = jax.random.normal(key, shape=(N,)) * 0.01
    initial_state = initial_state + noise

    total_steps = n_burn_in + n_steps
    start_time = 0.0
    end_time = total_steps * dt

    save_times = jnp.linspace(start_time, end_time, total_steps + 1)

    solution = diffeqsolve(
        ODETerm(model),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        Tsit5(),
        t0=start_time,
        t1=end_time,
        dt0=dt,
        y0=initial_state,
        saveat=SaveAt(ts=save_times),
        # Fixed-step integration takes exactly ``total_steps`` steps; diffrax's
        # default cap (4096) would abort long simulations (e.g. 5000 + 1000).
        max_steps=total_steps + 1,
    )

    states = solution.ys[n_burn_in:]
    time_coords = save_times[n_burn_in:] - save_times[n_burn_in]
    return time_coords, states
