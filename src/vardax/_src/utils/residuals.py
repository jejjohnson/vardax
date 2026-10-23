r"""Reference PDE residuals for [`ResidualPenalty`][vardax.ResidualPenalty].

Each constructor returns a residual ``residual_fn(field, s) -> (R,)`` that
evaluates a PDE operator $\mathcal N[f](s)$ of a continuous field $f$ at one
space-time point $s = (t, x, y)$. Derivatives come from automatic
differentiation through `geonnax.diffops`, so any differentiable callable
``field: (3,) -> (K,)`` works: a neural field, a closed-form solution, or an
interpolant of a gridded state.

The residuals are callables because vardax may not import a dynamical-model
library (somax sits above it in the stack); bring your own PDE the same way.
"""

from __future__ import annotations

from collections.abc import Callable

from geonnax import diffops
import jax.numpy as jnp
from jaxtyping import Array, Float

Field = Callable[[Float[Array, " D"]], Float[Array, " K"]]
Residual = Callable[[Field, Float[Array, " D"]], Float[Array, " R"]]


def _at_time(
    f: Callable[[Float[Array, " 3"]], Float[Array, ""]], t: Float[Array, ""]
) -> Callable[[Float[Array, " 2"]], Float[Array, ""]]:
    """Restrict a field of $(t, x, y)$ to a field of $(x, y)$ at time ``t``."""
    return lambda xy: f(jnp.concatenate([t[None], xy]))


def advection_diffusion_2d(
    velocity: tuple[float, float] | Float[Array, " 2"],
    diffusivity: float,
) -> Residual:
    r"""Residual of 2-D advection–diffusion with constant coefficients.

    For a scalar tracer $u(t, x, y)$ advected by a constant velocity
    $c = (c_x, c_y)$ and diffused with diffusivity $\kappa \ge 0$,

    $$
    \mathcal N[u] = \partial_t u + c \cdot \nabla u - \kappa\, \Delta u ,
    $$

    with $\nabla$ and $\Delta$ acting on $(x, y)$ only.

    Args:
        velocity: Constant advecting velocity $(c_x, c_y)$.
        diffusivity: Diffusivity $\kappa \ge 0$.

    The field must be twice differentiable in $(x, y)$ and once in $t$ at
    every collocation point; a raw `jnp.sqrt` or `jnp.linalg.norm` that
    vanishes there gives NaN derivatives, so guard it with the double-`where`
    idiom.

    Returns:
        ``residual_fn(field, s) -> (1,)``, where ``s = (t, x, y)`` and the
        tracer is ``field(s)[0]``.

    Raises:
        ValueError: If ``velocity`` does not have two components or
            ``diffusivity`` is negative.

    Examples:
        A Gaussian bump translated by $c$ and spread by diffusion solves the
        equation exactly:

        >>> import jax.numpy as jnp
        >>> from vardax.examples import advection_diffusion_2d
        >>> c, kappa = jnp.array([1.0, 0.5]), 0.1
        >>> def bump(s):
        ...     var = 0.04 + 2 * kappa * s[0]
        ...     r2 = jnp.sum((s[1:] - c * s[0]) ** 2)
        ...     return jnp.array([0.04 / var * jnp.exp(-r2 / (2 * var))])
        >>> res = advection_diffusion_2d(c, kappa)
        >>> r = res(bump, jnp.array([0.3, 0.25, 0.2]))
        >>> r.shape, bool(jnp.abs(r[0]) < 1e-4)
        ((1,), True)
    """
    c = jnp.asarray(velocity)
    if c.shape != (2,):
        raise ValueError(f"velocity must have two components; got shape {c.shape}.")
    if diffusivity < 0:
        raise ValueError(f"diffusivity must be non-negative; got {diffusivity}.")

    def residual(field: Field, s: Float[Array, " 3"]) -> Float[Array, " 1"]:
        def u(z: Float[Array, " 3"]) -> Float[Array, ""]:
            return field(z)[0]

        # ∇ₛu = (∂ₜu, ∂ₓu, ∂ᵧu): (3,)
        du = diffops.grad(u)(s)
        # Δu = ∂ₓₓu + ∂ᵧᵧu at fixed t: ()
        lap = diffops.laplacian(_at_time(u, s[0]))(s[1:])
        # N[u] = ∂ₜu + c·∇u - κΔu: ()
        r = du[0] + jnp.dot(c.astype(du.dtype), du[1:]) - diffusivity * lap
        return r[None]

    return residual


def navier_stokes_vorticity_2d(
    viscosity: float,
    forcing: Callable[[Float[Array, " 3"]], Float[Array, ""]] | None = None,
) -> Residual:
    r"""Residual of 2-D incompressible Navier–Stokes in vorticity form.

    The field provides the **streamfunction** $\psi(t, x, y)$ (its first
    output); velocity and vorticity are derived from it,

    $$
    u = \nabla^\perp \psi = (-\partial_y \psi,\ \partial_x \psi),
    \qquad \omega = \partial_x u_y - \partial_y u_x = \Delta \psi ,
    $$

    and the residual is the vorticity equation

    $$
    \mathcal N[\psi] = \partial_t \omega + u \cdot \nabla \omega
        - \nu\, \Delta \omega - F(t, x, y) .
    $$

    Deriving $u$ from $\psi$ makes the flow divergence-free and the Poisson
    relation $\Delta\psi = \omega$ hold exactly, so one residual component
    is enough. The price is fourth derivatives of the field ($\Delta\omega =
    \Delta^2\psi$), computed by nested automatic differentiation; for a field
    that returns $(\omega, \psi)$ separately, add $\Delta\psi - \omega$ as a
    second residual instead.

    The streamfunction must be four times differentiable in $(x, y)$ (and
    $\Delta\psi$ once in $t$) at every collocation point; a raw `jnp.sqrt`
    or `jnp.linalg.norm` that vanishes there gives NaN derivatives, so guard
    it with the double-`where` idiom.

    Args:
        viscosity: Kinematic viscosity $\nu \ge 0$.
        forcing: Optional vorticity forcing $F(s)$, a scalar function of
            ``s = (t, x, y)`` (for Kolmogorov flow,
            $F = -n f_0 \cos(n y)$ for the body force $f_0 \sin(n y)\,\hat x$).
            ``None`` means unforced.

    Returns:
        ``residual_fn(field, s) -> (1,)``, where ``s = (t, x, y)`` and the
        streamfunction is ``field(s)[0]``.

    Raises:
        ValueError: If ``viscosity`` is negative.

    Examples:
        The decaying Taylor–Green vortex $\psi = \sin x \sin y\,
        e^{-2\nu t}$ solves the unforced equation exactly:

        >>> import jax.numpy as jnp
        >>> from vardax.examples import navier_stokes_vorticity_2d
        >>> nu = 0.1
        >>> def tg(s):
        ...     return jnp.array(
        ...         [jnp.sin(s[1]) * jnp.sin(s[2]) * jnp.exp(-2 * nu * s[0])]
        ...     )
        >>> res = navier_stokes_vorticity_2d(nu)
        >>> r = res(tg, jnp.array([0.5, 0.3, 1.1]))
        >>> r.shape, bool(jnp.abs(r[0]) < 1e-5)
        ((1,), True)
    """
    if viscosity < 0:
        raise ValueError(f"viscosity must be non-negative; got {viscosity}.")

    def residual(field: Field, s: Float[Array, " 3"]) -> Float[Array, " 1"]:
        def psi(z: Float[Array, " 3"]) -> Float[Array, ""]:
            return field(z)[0]

        def omega(z: Float[Array, " 3"]) -> Float[Array, ""]:
            # ω = Δψ at fixed t: ()
            return diffops.laplacian(_at_time(psi, z[0]))(z[1:])

        # ∇ₛψ = (∂ₜψ, ∂ₓψ, ∂ᵧψ): (3,)
        dpsi = diffops.grad(psi)(s)
        # u = ∇^⊥ψ = (-∂ᵧψ, ∂ₓψ): (2,)
        u = jnp.stack([-dpsi[2], dpsi[1]])
        # ∇ₛω = (∂ₜω, ∂ₓω, ∂ᵧω): (3,)
        domega = diffops.grad(omega)(s)
        # Δω at fixed t: ()
        lap_omega = diffops.laplacian(_at_time(omega, s[0]))(s[1:])
        # N[ψ] = ∂ₜω + u·∇ω - nu Δω - F: ()
        r = domega[0] + jnp.dot(u, domega[1:]) - viscosity * lap_omega
        if forcing is not None:
            r = r - forcing(s)
        return r[None]

    return residual
