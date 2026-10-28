"""Tests for the reference PDE residuals in vardax._src.utils.residuals.

Exact solutions must give a residual at float32 round-off. Each residual is
a sum of terms (∂ₜ, advection, diffusion, forcing), each evaluated by
autodiff of a closed form to within a few ulps of its own magnitude, so the
cancellation leaves an absolute error of a few ulps of the summed term
magnitudes. The tests allow ``ROUNDOFF_ULPS`` of them: wide enough for
platform differences, and still four to six orders of magnitude below the
residual of a wrong coefficient (the negative controls).
"""

from __future__ import annotations

from geonnax import diffops
import jax
import jax.numpy as jnp
import pytest

from vardax.examples import advection_diffusion_2d, navier_stokes_vorticity_2d

EPS32 = float(jnp.finfo(jnp.float32).eps)
ROUNDOFF_ULPS = 32


def _at_time(f, t):
    return lambda xy: f(jnp.concatenate([t[None], xy]))


def _points(n: int, scale) -> jax.Array:
    # (t, x, y) uniform on [0, 1] x [0, Lx] x [0, Ly]
    return jax.random.uniform(jax.random.PRNGKey(0), (n, 3)) * jnp.asarray(scale)


# ---------------------------------------------------------------------------
# Advection-diffusion
# ---------------------------------------------------------------------------

C = jnp.array([0.8, -0.4])
KAPPA = 0.05
SIGMA0 = 0.1
X0 = jnp.array([0.5, 0.5])


def gaussian_bump(s):
    """Exact solution: a Gaussian translated by C and spread by diffusion.

    u = σ₀²/σ²(t) exp(−|x − x₀ − c t|² / (2σ²(t))), σ²(t) = σ₀² + 2κt.
    """
    var = SIGMA0**2 + 2 * KAPPA * s[0]
    r2 = jnp.sum((s[1:] - X0 - C * s[0]) ** 2)
    return jnp.array([SIGMA0**2 / var * jnp.exp(-r2 / (2 * var))])


def _adv_diff_terms(field, s):
    u = lambda z: field(z)[0]
    du = diffops.grad(u)(s)
    lap = diffops.laplacian(_at_time(u, s[0]))(s[1:])
    return jnp.stack([du[0], jnp.dot(C, du[1:]), -KAPPA * lap])


@pytest.mark.slow
def test_advection_diffusion_shape():
    r = advection_diffusion_2d(C, KAPPA)(gaussian_bump, jnp.array([0.1, 0.4, 0.6]))
    assert r.shape == (1,)
    assert r.dtype == jnp.float32


def test_advection_diffusion_gaussian_bump_is_exact():
    pts = _points(256, [1.0, 1.0, 1.0])
    r = jax.jit(jax.vmap(lambda s: advection_diffusion_2d(C, KAPPA)(gaussian_bump, s)))(
        pts
    )
    terms = jax.jit(jax.vmap(lambda s: _adv_diff_terms(gaussian_bump, s)))(pts)
    # The test's term split sums to the residual (same operator).
    assert jnp.allclose(r[:, 0], terms.sum(-1), atol=1e-6)
    tol = ROUNDOFF_ULPS * EPS32 * jnp.abs(terms).sum(-1)
    assert jnp.all(jnp.abs(r[:, 0]) <= tol)
    # Negative control: a wrong diffusivity leaves an O(1) residual.
    bad = advection_diffusion_2d(C, 2 * KAPPA)
    r_bad = jax.jit(jax.vmap(lambda s: bad(gaussian_bump, s)))(pts)
    assert jnp.max(jnp.abs(r_bad)) > 1e4 * jnp.max(tol)


def test_advection_diffusion_tuple_velocity():
    s = jnp.array([0.2, 0.3, 0.7])
    a = advection_diffusion_2d((0.8, -0.4), KAPPA)(gaussian_bump, s)
    b = advection_diffusion_2d(C, KAPPA)(gaussian_bump, s)
    assert jnp.allclose(a, b)


@pytest.mark.unchecked_types
@pytest.mark.parametrize(
    ("velocity", "kappa"), [(jnp.ones(3), 0.1), ((1.0, 0.0), -0.1)]
)
def test_advection_diffusion_validation(velocity, kappa):
    with pytest.raises(ValueError):
        advection_diffusion_2d(velocity, kappa)


# ---------------------------------------------------------------------------
# Navier-Stokes, vorticity form
# ---------------------------------------------------------------------------

NU = 0.05
K = 2.0
DOMAIN = [1.0, 2 * jnp.pi, 2 * jnp.pi]


def taylor_green(s):
    """Decaying Taylor–Green vortex ψ = sin kx sin ky e^{−2νk²t}."""
    return jnp.array(
        [jnp.sin(K * s[1]) * jnp.sin(K * s[2]) * jnp.exp(-2 * NU * K**2 * s[0])]
    )


def kolmogorov_decaying(s):
    """Decaying Kolmogorov shear mode ψ = sin(ky) e^{−νk²t}."""
    return jnp.array([jnp.sin(K * s[2]) * jnp.exp(-NU * K**2 * s[0])])


def kolmogorov_steady(s):
    """Steady forced Kolmogorov flow ψ = sin(ky); balanced by F below."""
    return jnp.array([jnp.sin(K * s[2])])


def kolmogorov_forcing(s):
    # ω = -k² sin(ky), Δω = k⁴ sin(ky); -nuΔω - F = 0 ⇒ F = -nuk⁴ sin(ky)
    return -NU * K**4 * jnp.sin(K * s[2])


def _ns_terms(field, s, forcing=None):
    psi = lambda z: field(z)[0]
    omega = lambda z: diffops.laplacian(_at_time(psi, z[0]))(z[1:])
    dpsi = diffops.grad(psi)(s)
    domega = diffops.grad(omega)(s)
    lap_omega = diffops.laplacian(_at_time(omega, s[0]))(s[1:])
    f = 0.0 if forcing is None else forcing(s)
    # u·∇ω split into its two products, which cancel for these flows.
    return jnp.stack(
        [
            domega[0],
            -dpsi[2] * domega[1],
            dpsi[1] * domega[2],
            -NU * lap_omega,
            -jnp.asarray(f, dtype=domega.dtype),
        ]
    )


@pytest.mark.parametrize(
    ("field", "forcing"),
    [
        (taylor_green, None),
        (kolmogorov_decaying, None),
        (kolmogorov_steady, kolmogorov_forcing),
    ],
    ids=["taylor-green", "kolmogorov-decaying", "kolmogorov-forced"],
)
def test_navier_stokes_exact_solutions(field, forcing):
    pts = _points(128, DOMAIN)
    res = navier_stokes_vorticity_2d(NU, forcing)
    r = jax.jit(jax.vmap(lambda s: res(field, s)))(pts)
    assert r.shape == (128, 1)
    terms = jax.jit(jax.vmap(lambda s: _ns_terms(field, s, forcing)))(pts)
    assert jnp.allclose(r[:, 0], terms.sum(-1), atol=1e-5)
    tol = ROUNDOFF_ULPS * EPS32 * jnp.abs(terms).sum(-1)
    assert jnp.all(jnp.abs(r[:, 0]) <= tol)


@pytest.mark.slow
def test_navier_stokes_negative_controls():
    pts = _points(64, DOMAIN)
    # Wrong viscosity for Taylor-Green, and the forced flow without forcing.
    ns_bad, ns = navier_stokes_vorticity_2d(2 * NU), navier_stokes_vorticity_2d(NU)
    r1 = jax.jit(jax.vmap(lambda s: ns_bad(taylor_green, s)))(pts)
    r2 = jax.jit(jax.vmap(lambda s: ns(kolmogorov_steady, s)))(pts)
    assert jnp.max(jnp.abs(r1)) > 0.1
    assert jnp.max(jnp.abs(r2)) > 0.1


@pytest.mark.slow
def test_navier_stokes_nonlinear_term_is_active():
    # The exact solutions above all have ω a function of ψ, so u·∇ω = 0 there.
    # This ψ does not, so the advection term is non-zero and must enter the
    # residual as ∂ₜω + u·∇ω - nuΔω.
    field = lambda s: jnp.array([jnp.sin(s[1]) * jnp.sin(2 * s[2]) + 0.5 * s[1] ** 2])
    s = jnp.array([0.0, 0.7, 0.3])
    terms = _ns_terms(field, s)
    assert jnp.abs(terms[1] + terms[2]) > 0.1
    r = navier_stokes_vorticity_2d(NU)(field, s)
    assert jnp.allclose(r[0], terms.sum(), atol=1e-5)


def test_vorticity_convention_matches_curl():
    """ω = Δψ is the curl of u = ∇^⊥ψ = (−∂ᵧψ, ∂ₓψ)."""
    psi = lambda xy: jnp.sin(1.3 * xy[0]) * jnp.cos(0.7 * xy[1]) + xy[0] * xy[1] ** 2
    u = lambda xy: jnp.array([-1.0, 1.0]) * diffops.grad(psi)(xy)[::-1]
    xy = jnp.array([0.4, -0.9])
    assert jnp.allclose(diffops.curl2d(u)(xy), diffops.laplacian(psi)(xy), atol=1e-5)


def test_navier_stokes_validation():
    with pytest.raises(ValueError):
        navier_stokes_vorticity_2d(-1.0)
