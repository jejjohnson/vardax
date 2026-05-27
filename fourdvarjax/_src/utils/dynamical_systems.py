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
        t: float,
        y: Float[Array, 3],
        args: None,
    ) -> Float[Array, 3]:
        """Evaluate the vector field at state ``y`` and time ``t``."""
        x_, y_, z_ = y[0], y[1], y[2]
        dx = self.sigma * (y_ - x_)
        dy = x_ * (self.rho - z_) - y_
        dz = x_ * y_ - self.beta * z_
        return jnp.array([dx, dy, dz])


def simulate_lorenz63(
    key: Array,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    n_steps: int = 5000,
    n_burn_in: int = 1000,
    x0: Float[Array, 3] | None = None,
) -> tuple[Float[Array, T], Float[Array, "T 3"]]:  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
    """Simulate the Lorenz-63 system and return state trajectory.

    Parameters
    ----------
    key:
        JAX PRNG key used to perturb the initial condition when ``x0`` is
        ``None``.
    sigma:
        Prandtl number.
    rho:
        Rayleigh number.
    beta:
        Geometric factor.
    dt:
        Integration time step.
    n_steps:
        Total number of integration steps *after* burn-in.
    n_burn_in:
        Number of initial steps to discard (burn-in).
    x0:
        Optional explicit initial condition of shape ``(3,)``.  When
        ``None``, the classic fixed-point perturbation is used.

    Returns
    -------
    time_coords : Float[Array, "T"]
        Time coordinates for the returned trajectory (starting at 0).
    states : Float[Array, "T 3"]
        State trajectory of the Lorenz-63 system.
    """
    model = Lorenz63(sigma=sigma, rho=rho, beta=beta)

    if x0 is None:
        fp = jnp.array(
            [
                jnp.sqrt(beta * (rho - 1)),
                jnp.sqrt(beta * (rho - 1)),
                rho - 1.0,
            ]
        )
        noise = jax.random.normal(key, shape=(3,)) * 0.01
        x0 = fp + noise

    total_steps = n_burn_in + n_steps
    t0 = 0.0
    t1 = total_steps * dt

    save_times = jnp.linspace(t0, t1, total_steps + 1)

    sol = diffeqsolve(
        ODETerm(model),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        Tsit5(),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=x0,
        saveat=SaveAt(ts=save_times),
    )

    # Discard burn-in steps; keep indices n_burn_in .. total_steps (inclusive)
    states = sol.ys[n_burn_in:]
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
        t: float,
        y: Float[Array, N],  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
        args: None,
    ) -> Float[Array, N]:  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
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
) -> tuple[Float[Array, T], Float[Array, "T N"]]:  # type: ignore[unresolved-reference]  # ty:ignore[unresolved-reference]
    """Simulate the Lorenz-96 system and return state trajectory.

    Parameters
    ----------
    key:
        JAX PRNG key used to perturb the initial condition.
    N:
        Number of variables (spatial dimension).
    F:
        Forcing constant.
    dt:
        Integration time step.
    n_steps:
        Total number of integration steps *after* burn-in.
    n_burn_in:
        Number of initial steps to discard (burn-in).

    Returns
    -------
    time_coords : Float[Array, "T"]
        Time coordinates for the returned trajectory (starting at 0).
    states : Float[Array, "T N"]
        State trajectory of shape ``(n_steps + 1, N)``.
    """
    model = Lorenz96(F=F)

    # Standard L96 initialization: uniform forcing with a small random perturbation
    x0 = jnp.full((N,), F)
    noise = jax.random.normal(key, shape=(N,)) * 0.01
    x0 = x0 + noise

    total_steps = n_burn_in + n_steps
    t0 = 0.0
    t1 = total_steps * dt

    save_times = jnp.linspace(t0, t1, total_steps + 1)

    sol = diffeqsolve(
        ODETerm(model),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        Tsit5(),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=x0,
        saveat=SaveAt(ts=save_times),
    )

    states = sol.ys[n_burn_in:]
    time_coords = save_times[n_burn_in:] - save_times[n_burn_in]
    return time_coords, states
