"""Lorenz-63 and Lorenz-96 dynamical system simulations using Diffrax."""

from __future__ import annotations

from typing import Any

from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Lorenz63(eqx.Module):
    """Lorenz-63 vector field as an Equinox module.

        ẋ = σ (y − x),   ẏ = x (ρ − z) − y,   ż = x y − β z

    The module is a diffrax-compatible right-hand side ``f(t, y, args)``, whose
    ``y`` is the state (x, y, z).

    Attributes:
        sigma: Prandtl number σ (classically 10).
        rho: Rayleigh number ρ (classically 28).
        beta: Geometric factor β (classically 8/3).

    Examples:
        The vector field at (1, 1, 1) for the classical parameters:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> f = vdx.Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        >>> dydt = f(0.0, jnp.ones(3), None)
        >>> (
        ...     dydt.shape,
        ...     bool(jnp.allclose(dydt, jnp.array([0.0, 26.0, 1.0 - 8.0 / 3.0]))),
        ... )
        ((3,), True)

        The origin is a fixed point:

        >>> bool(jnp.all(f(0.0, jnp.zeros(3), None) == 0))
        True

        It drives a [`DynTrajectory`][vardax.DynTrajectory] prior:

        >>> ts = jnp.linspace(0.0, 0.05, 6)
        >>> vdx.DynTrajectory(model=f)(jnp.ones(3), ts).shape
        (6, 3)
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
        """Evaluate the vector field at state ``y`` and time ``t``.

        Args:
            t: Time (unused; the system is autonomous).
            y: State (x, y, z).
            args: Unused, present for the diffrax signature.

        Returns:
            The time derivative (ẋ, ẏ, ż).
        """
        state_x, state_y, state_z = y[0], y[1], y[2]
        # ẋ = σ (y − x)
        dx_dt = self.sigma * (state_y - state_x)
        # ẏ = x (ρ − z) − y
        dy_dt = state_x * (self.rho - state_z) - state_y
        # ż = x y − β z
        dz_dt = state_x * state_y - self.beta * state_z
        return jnp.array([dx_dt, dy_dt, dz_dt])

    def get_config(self) -> dict[str, Any]:
        """Return the parameters as a plain dictionary."""
        return {
            # float(): the parameters are leaves and may be 0-d arrays
            "sigma": float(self.sigma),
            "rho": float(self.rho),
            "beta": float(self.beta),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Lorenz63:
        """Rebuild the module from ``get_config`` output."""
        return cls(**config)


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
    """Simulate the Lorenz-63 system and return the state trajectory.

        ẋ = σ (y − x),   ẏ = x (ρ − z) − y,   ż = x y − β z

    Integrates the system with diffrax and discards a burn-in so the trajectory lies
    on the attractor.

    Algorithm:
        1. Build the initial condition, (1, 1, 1) perturbed by ``key``, or ``x0``.
        2. Integrate with ``Tsit5`` over ``n_burn_in + n_steps`` steps of size
           ``dt``, saving every step.
        3. Drop the first ``n_burn_in`` states and shift time to start at 0.

    Args:
        key: JAX PRNG key used to perturb the initial condition when ``x0`` is
            ``None``.
        sigma: Prandtl number σ.
        rho: Rayleigh number ρ.
        beta: Geometric factor β.
        dt: Integration time step.
        n_steps: Total number of integration steps after burn-in.
        n_burn_in: Number of initial steps to discard (burn-in).
        x0: Optional explicit initial condition x₀. When ``None``, the classic
            off-attractor point (1, 1, 1) is used, perturbed by ``key``. From there
            the transient onto the attractor takes a couple of time units, so the
            default burn-in lands on it. (Starting at the unstable fixed point
            instead, as older versions did, leaves the trajectory within a fraction
            of a unit of it for ~40 time units, far longer than any burn-in the
            tutorials use.)

    Returns:
        ``(time_coords, states)``: the time coordinates (starting at 0) and the
        Lorenz-63 state trajectory, with ``n_steps + 1`` rows.

    Examples:
        A short run without burn-in:

        >>> import jax
        >>> import vardax as vdx
        >>> t, states = vdx.simulate_lorenz63(
        ...     jax.random.PRNGKey(0), n_steps=10, n_burn_in=0
        ... )
        >>> t.shape, states.shape
        ((11,), (11, 3))
        >>> float(t[0])
        0.0

        An explicit initial condition is used as given:

        >>> import jax.numpy as jnp
        >>> x0 = jnp.array([1.0, 2.0, 3.0])
        >>> _, states = vdx.simulate_lorenz63(
        ...     jax.random.PRNGKey(0), n_steps=5, n_burn_in=0, x0=x0
        ... )
        >>> bool(jnp.allclose(states[0], x0))
        True

        A burn-in is dropped from the output but the time axis restarts at 0:

        >>> t, states = vdx.simulate_lorenz63(
        ...     jax.random.PRNGKey(0), n_steps=5, n_burn_in=5
        ... )
        >>> states.shape, float(t[0])
        ((6, 3), 0.0)
    """
    model = Lorenz63(sigma=sigma, rho=rho, beta=beta)

    if x0 is None:
        # x₀ = (1, 1, 1) + 0.01 ξ, ξ ~ 𝒩(0, I)
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
    # tₖ − t_burn, so the time axis restarts at 0
    time_coords = save_times[n_burn_in:] - save_times[n_burn_in]
    return time_coords, states


class Lorenz96(eqx.Module):
    """Lorenz-96 vector field as an Equinox module.

        ẋₙ = (xₙ₊₁ − xₙ₋₂) xₙ₋₁ − xₙ + F

    with n indexing the sites of a periodic ring. The module is a diffrax-compatible
    right-hand side ``f(t, y, args)``, whose ``y`` is the state x.

    Attributes:
        forcing: Forcing constant F (classically 8).

    Examples:
        The uniform state xₙ = F is a fixed point, and at the origin the field equals
        the forcing:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> f = vdx.Lorenz96(forcing=8.0)
        >>> bool(jnp.allclose(f(0.0, jnp.full(6, 8.0), None), 0.0))
        True
        >>> bool(jnp.allclose(f(0.0, jnp.zeros(6), None), 8.0))
        True

        It maps an N-vector to an N-vector, for any N:

        >>> f(0.0, jnp.arange(5.0), None).shape
        (5,)

        It drives a [`DynTrajectory`][vardax.DynTrajectory] prior:

        >>> ts = jnp.linspace(0.0, 0.05, 6)
        >>> vdx.DynTrajectory(model=f)(jnp.arange(5.0), ts).shape
        (6, 5)
    """

    forcing: float

    def __call__(
        self,
        t: float | Float[Array, ""],
        y: Float[Array, " N"],
        args: None,
    ) -> Float[Array, " N"]:
        """Evaluate the vector field at state ``y`` and time ``t``.

        Args:
            t: Time (unused; the system is autonomous).
            y: State x on the periodic ring.
            args: Unused, present for the diffrax signature.

        Returns:
            The time derivative ẋ.
        """
        # xₙ₋₂
        back_two = jnp.roll(y, 2)
        # xₙ₋₁
        back_one = jnp.roll(y, 1)
        # xₙ₊₁
        ahead_one = jnp.roll(y, -1)
        # (xₙ₊₁ − xₙ₋₂) xₙ₋₁
        advection = (ahead_one - back_two) * back_one
        # ẋₙ = (xₙ₊₁ − xₙ₋₂) xₙ₋₁ − xₙ + F
        return advection - y + self.forcing

    def get_config(self) -> dict[str, Any]:
        """Return the forcing as a plain dictionary."""
        return {
            # float(): the forcing is a leaf and may be a 0-d array
            "forcing": float(self.forcing),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Lorenz96:
        """Rebuild the module from ``get_config`` output."""
        return cls(**config)


def simulate_lorenz96(
    key: Array,
    *,
    n_sites: int = 40,
    forcing: float = 8.0,
    dt: float = 0.01,
    n_steps: int = 5000,
    n_burn_in: int = 1000,
) -> tuple[Float[Array, " T"], Float[Array, "T N"]]:
    """Simulate the Lorenz-96 system and return the state trajectory.

        ẋₙ = (xₙ₊₁ − xₙ₋₂) xₙ₋₁ − xₙ + F

    Integrates the system on a periodic ring with diffrax and discards a burn-in.

    Algorithm:
        1. Start from the uniform state xₙ = F plus a small perturbation drawn from
           ``key``.
        2. Integrate with ``Tsit5`` over ``n_burn_in + n_steps`` steps of size
           ``dt``, saving every step.
        3. Drop the first ``n_burn_in`` states and shift time to start at 0.

    Args:
        key: JAX PRNG key used to perturb the initial condition.
        n_sites: Number of variables (spatial dimension).
        forcing: Forcing constant F.
        dt: Integration time step.
        n_steps: Total number of integration steps after burn-in.
        n_burn_in: Number of initial steps to discard (burn-in).

    Returns:
        ``(time_coords, states)``: the time coordinates (starting at 0) and the
        state trajectory, with ``n_steps + 1`` rows.

    Examples:
        >>> import jax
        >>> import vardax as vdx
        >>> t, states = vdx.simulate_lorenz96(
        ...     jax.random.PRNGKey(0), n_sites=8, n_steps=10, n_burn_in=0
        ... )
        >>> t.shape, states.shape
        ((11,), (11, 8))

        The perturbation of the initial state depends on the key:

        >>> import jax.numpy as jnp
        >>> _, other = vdx.simulate_lorenz96(
        ...     jax.random.PRNGKey(1), n_sites=8, n_steps=10, n_burn_in=0
        ... )
        >>> bool(jnp.allclose(states[0], other[0]))
        False

        The result feeds a plotting helper directly:

        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> fig, ax = vdx.plot_l96_grid(states, t)
        >>> type(ax).__name__
        'Axes'
    """
    model = Lorenz96(forcing=forcing)

    # the standard L96 start: x₀,ₙ = F + 0.01 ξₙ, ξₙ ~ 𝒩(0, 1)
    initial_state = jnp.full((n_sites,), forcing)
    noise = jax.random.normal(key, shape=(n_sites,)) * 0.01
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
    # tₖ − t_burn, so the time axis restarts at 0
    time_coords = save_times[n_burn_in:] - save_times[n_burn_in]
    return time_coords, states
