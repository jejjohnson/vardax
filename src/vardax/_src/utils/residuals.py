"""Reference PDE residuals for [`ResidualPenalty`][vardax.ResidualPenalty].

Each constructor returns ``residual_fn(field, s)``, a PDE operator 𝒩 of a continuous
field at one space-time point s = (t, x, y), with x, y Cartesian coordinates, q a
tracer, v = (u, v) velocity, ψ streamfunction and ζ relative vorticity.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
from geonnax import diffops
import jax.numpy as jnp
from jaxtyping import Array, Float

from vardax._src.errors import ConfigurationError, ShapeError

# Callables, because vardax may not import a dynamical-model library (somax sits
# above it in the stack); bring your own PDE the same way. Any differentiable field
# works: a neural field, a closed-form solution or an interpolant of a gridded state.
Field = Callable[[Float[Array, " D"]], Float[Array, " K"]]
Residual = Callable[[Field, Float[Array, " D"]], Float[Array, " R"]]


def _at_time(
    field: Callable[[Float[Array, " 3"]], Float[Array, ""]], time: Float[Array, ""]
) -> Callable[[Float[Array, " 2"]], Float[Array, ""]]:
    """Restrict a field of (t, x, y) to a field of (x, y) at time ``time``."""

    def field_at_time(space_point: Float[Array, " 2"]) -> Float[Array, ""]:
        # s = (t, x, y)
        space_time_point = einx.id(", d -> (1 + d)", time, space_point)
        return field(space_time_point)

    return field_at_time


def advection_diffusion_2d(
    velocity: tuple[float, float] | Float[Array, " 2"],
    diffusivity: float,
) -> Residual:
    """Residual of 2-D advection–diffusion with constant coefficients.

        𝒩[q] = ∂ₜq + v · ∇q − κ Δq

    For a scalar tracer q(t, x, y) advected by a constant velocity v = (u, v) and
    diffused with diffusivity κ ≥ 0, with ∇ and Δ acting on (x, y) only.

    The field must be twice differentiable in (x, y) and once in t at every
    collocation point; a raw `jnp.sqrt` or `jnp.linalg.norm` that vanishes there
    gives NaN derivatives, so guard it with the double-`where` idiom.

    Args:
        velocity: Constant advecting velocity v = (u, v).
        diffusivity: Diffusivity κ ≥ 0.

    Returns:
        ``residual_fn(field, s)`` returning one component, where ``s = (t, x, y)``
        and the tracer is ``field(s)[0]``.

    Raises:
        ShapeError: If ``velocity`` does not have two components.
        ConfigurationError: If ``diffusivity`` is negative.

    Examples:
        A Gaussian bump translated by v and spread by diffusion solves the equation
        exactly:

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

        A steady uniform tracer has zero residual, whatever the flow:

        >>> uniform = lambda s: jnp.ones(1)
        >>> r = advection_diffusion_2d((2.0, -1.0), 0.5)(uniform, jnp.zeros(3))
        >>> bool(jnp.allclose(r, 0.0))
        True

        A negative diffusivity is rejected:

        >>> advection_diffusion_2d(
        ...     (1.0, 0.0), -0.1
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: diffusivity must be non-negative; got -0.1.
    """
    advection_velocity = jnp.asarray(velocity)
    if advection_velocity.shape != (2,):
        raise ShapeError(
            f"velocity must have two components; got shape {advection_velocity.shape}."
        )
    if diffusivity < 0:
        raise ConfigurationError(
            f"diffusivity must be non-negative; got {diffusivity}."
        )

    def residual(field: Field, point: Float[Array, " 3"]) -> Float[Array, " 1"]:
        def tracer(space_time_point: Float[Array, " 3"]) -> Float[Array, ""]:
            return field(space_time_point)[0]

        # (∂ₜq, ∂ₓq, ∂_y q)
        tracer_gradient = diffops.grad(tracer)(point)
        # Δq = ∂ₓₓq + ∂_yy q at fixed t
        tracer_laplacian = diffops.laplacian(_at_time(tracer, point[0]))(point[1:])
        # v · ∇q
        advection = einx.dot(
            "d, d ->",
            advection_velocity.astype(tracer_gradient.dtype),
            tracer_gradient[1:],
        )
        # 𝒩[q] = ∂ₜq + v · ∇q − κ Δq
        pde_residual = tracer_gradient[0] + advection - diffusivity * tracer_laplacian
        return einx.id(" -> 1", pde_residual)

    return residual


def navier_stokes_vorticity_2d(
    viscosity: float,
    forcing: Callable[[Float[Array, " 3"]], Float[Array, ""]] | None = None,
) -> Residual:
    """Residual of 2-D incompressible Navier–Stokes in vorticity form.

        𝒩[ψ] = ∂ₜζ + v · ∇ζ − ν Δζ − F(t, x, y)
        v = (u, v) = (−∂_y ψ, ∂ₓψ),   ζ = ∂ₓv − ∂_y u = Δψ

    The field provides the streamfunction ψ(t, x, y) as its first output; velocity
    and vorticity are derived from it. Deriving v from ψ makes the flow
    divergence-free and the Poisson relation Δψ = ζ hold exactly, so one residual
    component is enough. The price is fourth derivatives of the field
    (Δζ = Δ²ψ), computed by nested automatic differentiation; for a field that
    returns (ζ, ψ) separately, add Δψ − ζ as a second residual instead.

    The streamfunction must be four times differentiable in (x, y) (and Δψ once in
    t) at every collocation point; a raw `jnp.sqrt` or `jnp.linalg.norm` that
    vanishes there gives NaN derivatives, so guard it with the double-`where` idiom.

    Args:
        viscosity: Kinematic viscosity ν ≥ 0.
        forcing: Optional vorticity forcing F(s), a scalar function of
            ``s = (t, x, y)`` (for Kolmogorov flow, F = −n f₀ cos(n y) for the body
            force f₀ sin(n y) x̂). ``None`` means unforced.

    Returns:
        ``residual_fn(field, s)`` returning one component, where ``s = (t, x, y)``
        and the streamfunction is ``field(s)[0]``.

    Raises:
        ConfigurationError: If ``viscosity`` is negative.

    Examples:
        The decaying Taylor–Green vortex ψ = sin x sin y e^(−2νt) solves the unforced
        equation exactly:

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

        A forcing is subtracted from the residual:

        >>> still = lambda s: jnp.zeros(1)
        >>> forced = navier_stokes_vorticity_2d(nu, forcing=lambda s: jnp.ones(()))
        >>> bool(jnp.allclose(forced(still, jnp.zeros(3)), -1.0))
        True

        A negative viscosity is rejected:

        >>> navier_stokes_vorticity_2d(-1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: viscosity must be non-negative; got -1.0.
    """
    if viscosity < 0:
        raise ConfigurationError(f"viscosity must be non-negative; got {viscosity}.")

    def residual(field: Field, point: Float[Array, " 3"]) -> Float[Array, " 1"]:
        def streamfunction(space_time_point: Float[Array, " 3"]) -> Float[Array, ""]:
            return field(space_time_point)[0]

        def vorticity(space_time_point: Float[Array, " 3"]) -> Float[Array, ""]:
            # ζ = Δψ at fixed t
            return diffops.laplacian(_at_time(streamfunction, space_time_point[0]))(
                space_time_point[1:]
            )

        # (∂ₜψ, ∂ₓψ, ∂_y ψ)
        streamfunction_gradient = diffops.grad(streamfunction)(point)
        # v = (−∂_y ψ, ∂ₓψ)
        velocity = jnp.stack([-streamfunction_gradient[2], streamfunction_gradient[1]])
        # (∂ₜζ, ∂ₓζ, ∂_y ζ)
        vorticity_gradient = diffops.grad(vorticity)(point)
        # Δζ at fixed t
        vorticity_laplacian = diffops.laplacian(_at_time(vorticity, point[0]))(
            point[1:]
        )
        # v · ∇ζ
        advection = einx.dot("d, d ->", velocity, vorticity_gradient[1:])
        # ∂ₜζ + v · ∇ζ − ν Δζ
        pde_residual = (
            vorticity_gradient[0] + advection - viscosity * vorticity_laplacian
        )
        if forcing is not None:
            # 𝒩[ψ] = ∂ₜζ + v · ∇ζ − ν Δζ − F
            pde_residual = pde_residual - forcing(point)
        return einx.id(" -> 1", pde_residual)

    return residual
