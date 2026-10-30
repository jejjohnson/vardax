r"""Rodgers optimal estimation: MAP retrieval with its diagnostics.

Rodgers (2000), *Inverse Methods for Atmospheric Sounding*, chapters 2–5.
With $K = \partial F / \partial x$ at the retrieval,

$$
J(x) = (y - F(x))^\top S_\varepsilon^{-1} (y - F(x)) + (x - x_a)^\top S_a^{-1} (x - x_a),
$$

$$
\hat S = (K^\top S_\varepsilon^{-1} K + S_a^{-1})^{-1}, \quad
A = \hat S K^\top S_\varepsilon^{-1} K, \quad
d_s = \operatorname{tr} A, \quad
H = \tfrac12 \ln\det(S_a \hat S^{-1}).
$$

The retrieval observes the true state through $\hat x \approx x_a + A (x - x_a)$,
which is what ``OEResult.as_observation_operator`` returns for assimilating
it downstream.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import lineax as lx
import optimistix as optx

from vardax._src.obs_operators.callable import CallableObs


class OEResult(eqx.Module):
    r"""The output of [`optimal_estimation`][vardax.optimal_estimation].

    Attributes:
        x_hat: The retrieval $\hat x$, shape ``(n,)``.
        S_hat: Its covariance $\hat S$, a PSD ``(n, n)`` operator.
        averaging_kernel: $A = \hat S K^\top S_\varepsilon^{-1} K$,
            shape ``(n, n)``.
        dofs: Degrees of freedom for signal, $d_s = \operatorname{tr} A$.
        information: Shannon information content,
            $H = \tfrac12 \ln\det(S_a \hat S^{-1})$, in nats.
        chi2: The cost $J(\hat x)$ (both terms, without the ½).
        d2: Rodgers' convergence measure of one more Gauss–Newton step
            $\delta x$ from $\hat x$: $d^2 = \delta x^\top \hat S^{-1} \delta x$.
        n_iter: Levenberg–Marquardt steps taken.
        converged: Rodgers' criterion $d^2 < \epsilon\, n$ holds and
            $\hat x$ is finite. Levenberg–Marquardt's own tolerance may
            still be unmet (``n_iter`` at ``max_steps``) once the step
            left is this small in posterior units.
        prior_mean: $x_a$, kept for ``as_observation_operator``.
        retrieval_noise_cov: $S_m = G S_\varepsilon G^\top$ with the gain
            $G = \hat S K^\top S_\varepsilon^{-1}$, which equals $A \hat S$:
            the measurement-noise part of the retrieval error. It is the
            $R$ to pair with ``as_observation_operator``, whose $A$ already
            carries the smoothing error (Rodgers 2000, §3.2).
    """

    x_hat: Float[Array, " n"]
    S_hat: lx.AbstractLinearOperator
    averaging_kernel: Float[Array, "n n"]
    dofs: Float[Array, ""]
    information: Float[Array, ""]
    chi2: Float[Array, ""]
    d2: Float[Array, ""]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]
    prior_mean: Float[Array, " n"]
    retrieval_noise_cov: lx.AbstractLinearOperator

    def as_observation_operator(self) -> CallableObs:
        r"""The retrieval as an observation of the state, $x \mapsto x_a + A (x - x_a)$.

        Returns:
            A [`CallableObs`][vardax.CallableObs] to assimilate $\hat x$
            with $R = S_m$, ``retrieval_noise_cov``. Not $\hat S$: that
            also counts the smoothing error $(A - I)(x - x_a)$, which the
            operator models through $A$, so using it would underweight the
            retrieval (Rodgers 2000, §3.2).
        """
        prior_mean = self.prior_mean
        kernel = self.averaging_kernel

        def retrieve(state: Float[Array, " n"]) -> Float[Array, " n"]:
            # x_a + A (x − x_a)
            return prior_mean + kernel @ (state - prior_mean)

        return CallableObs(fn=retrieve)


def optimal_estimation(
    forward: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    y: Float[Array, " m"],
    x_a: Float[Array, " n"],
    S_a: lx.AbstractLinearOperator,
    S_e: lx.AbstractLinearOperator,
    *,
    max_steps: int = 20,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    convergence_tol: float = 1e-2,
) -> OEResult:
    r"""Rodgers optimal estimation of $x$ from $y = F(x) + \varepsilon$.

    Algorithm:
        1. Whiten both terms of $J$ with Cholesky factors,
           $S_\varepsilon = L_\varepsilon L_\varepsilon^\top$ and
           $S_a = L_a L_a^\top$ (closed form for diagonal operators).
        2. Minimise $\|[L_\varepsilon^{-1}(y - F(x));\ L_a^{-1}(x - x_a)]\|^2$
           from $x_a$ by Levenberg–Marquardt with ``ImplicitAdjoint``, so
           $\hat x$ is differentiable in $y$ and in the parameters
           ``forward`` closes over.
        3. At $\hat x$, form $K$ by ``jax.jacfwd`` and the diagnostics
           $\hat S$, $A$, $d_s$, $H$ and $\chi^2$.
        4. Report Rodgers' (§5.6.3) $d^2 = g^\top \hat S g$ of one more
           Gauss–Newton step, $g = \nabla \tfrac12 J(\hat x)$, and call the
           retrieval converged when $d^2 <$ ``convergence_tol``
           $\cdot\, n$.

    A pixel swath is ``eqx.filter_vmap`` over pixels; the solver does not
    raise on failure, so check ``converged``.

    Args:
        forward: $F$, the forward model, ``(n,) -> (m,)``.
        y: Measurements, shape ``(m,)``.
        x_a: Prior mean, shape ``(n,)``.
        S_a: Prior covariance, a PSD ``(n, n)`` operator.
        S_e: Measurement-error covariance, a PSD ``(m, m)`` operator.
        max_steps: Levenberg–Marquardt step cap.
        rtol: Solver relative tolerance.
        atol: Solver absolute tolerance.
        convergence_tol: $\epsilon$ in $d^2 < \epsilon\, n$.

    Returns:
        The retrieval and its diagnostics.

    Examples:
        A linear forward is solved exactly, and its DOFS lie in $[0, n]$:

        >>> import jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> K = jnp.array([[1.0, 0.5], [0.0, 1.0], [1.0, 1.0]])
        >>> S_a = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> S_e = lx.DiagonalLinearOperator(jnp.full(3, 0.1))
        >>> result = vdx.optimal_estimation(
        ...     lambda x: K @ x, jnp.array([1.0, 2.0, 3.0]), jnp.zeros(2), S_a, S_e
        ... )
        >>> bool(result.converged), result.x_hat.shape
        (True, (2,))
        >>> bool(0.0 <= float(result.dofs) <= 2.0)
        True
    """
    n_state = x_a.shape[0]
    whiten_e, logdet_e = _whitener(S_e)
    whiten_a, logdet_a = _whitener(S_a)
    del logdet_e

    def residual(
        state: Float[Array, " n"], measurements: Float[Array, " m"]
    ) -> tuple[Array, Array]:
        # [L_ε⁻¹ (y − F(x)); L_a⁻¹ (x − x_a)]
        return whiten_e(measurements - forward(state)), whiten_a(state - x_a)

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    solution = optx.least_squares(
        residual,
        solver,
        x_a,
        args=y,
        adjoint=optx.ImplicitAdjoint(),
        max_steps=max_steps,
        throw=False,
    )
    x_hat = solution.value

    # K = ∂F/∂x at x̂, and its whitened form K̃ = L_ε⁻¹ K
    jacobian = jax.jacfwd(forward)(x_hat)
    whitened_jacobian = jax.vmap(whiten_e, in_axes=1, out_axes=1)(jacobian)
    # L_a⁻¹, so that S_a⁻¹ = L_a⁻ᵀ L_a⁻¹
    prior_root_inverse = jax.vmap(whiten_a, in_axes=1, out_axes=1)(
        jnp.eye(n_state, dtype=x_hat.dtype)
    )
    # Kᵀ S_ε⁻¹ K
    data_precision = whitened_jacobian.T @ whitened_jacobian
    # Ŝ⁻¹ = Kᵀ S_ε⁻¹ K + S_a⁻¹
    precision = data_precision + prior_root_inverse.T @ prior_root_inverse
    precision_factor = jnp.linalg.cholesky(precision)
    identity = jnp.eye(n_state, dtype=x_hat.dtype)
    covariance = jax.scipy.linalg.cho_solve((precision_factor, True), identity)
    covariance = 0.5 * (covariance + covariance.T)
    # A = Ŝ Kᵀ S_ε⁻¹ K
    averaging_kernel = covariance @ data_precision
    # S_m = G S_ε Gᵀ = Ŝ Kᵀ S_ε⁻¹ K Ŝ = A Ŝ, with G = Ŝ Kᵀ S_ε⁻¹
    retrieval_noise = averaging_kernel @ covariance
    retrieval_noise = 0.5 * (retrieval_noise + retrieval_noise.T)
    # H = ½ ln det(S_a Ŝ⁻¹) = ½ (ln det S_a + ln det Ŝ⁻¹)
    logdet_precision = 2.0 * jnp.sum(jnp.log(jnp.diag(precision_factor)))
    information = 0.5 * (logdet_a + logdet_precision)

    data_residual, prior_residual = residual(x_hat, y)
    # J(x̂) = ‖r̃_ε‖² + ‖r̃_a‖²
    chi2 = jnp.sum(data_residual**2) + jnp.sum(prior_residual**2)
    # g = ∇ ½ J = −K̃ᵀ r̃_ε + L_a⁻ᵀ r̃_a; δx = −Ŝ g; d² = δxᵀ Ŝ⁻¹ δx = gᵀ Ŝ g
    gradient = (
        -whitened_jacobian.T @ data_residual + prior_root_inverse.T @ prior_residual
    )
    d2 = gradient @ covariance @ gradient
    return OEResult(
        x_hat=x_hat,
        S_hat=lx.MatrixLinearOperator(covariance, lx.positive_semidefinite_tag),
        averaging_kernel=averaging_kernel,
        dofs=jnp.trace(averaging_kernel),
        information=information,
        chi2=chi2,
        d2=d2,
        n_iter=jnp.asarray(solution.stats["num_steps"]),
        converged=jnp.all(jnp.isfinite(x_hat)) & (d2 < convergence_tol * n_state),
        prior_mean=x_a,
        retrieval_noise_cov=lx.MatrixLinearOperator(
            retrieval_noise, lx.positive_semidefinite_tag
        ),
    )


def _whitener(
    covariance: lx.AbstractLinearOperator,
) -> tuple[Callable[[Array], Array], Float[Array, ""]]:
    """v ↦ L⁻¹ v with L Lᵀ = S, and ln det S; closed form for a diagonal S."""
    if lx.is_diagonal(covariance):
        diagonal = lx.diagonal(covariance)
        root = jnp.sqrt(diagonal)

        def whiten_diagonal(vector: Array) -> Array:
            # L⁻¹ v = v / √diag(S)
            return vector / root

        return whiten_diagonal, jnp.sum(jnp.log(diagonal))
    factor = jnp.linalg.cholesky(covariance.as_matrix())

    def whiten(vector: Array) -> Array:
        # L⁻¹ v by forward substitution
        return jax.scipy.linalg.solve_triangular(factor, vector, lower=True)

    return whiten, 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
