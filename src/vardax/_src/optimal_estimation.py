"""Rodgers optimal estimation: the MAP retrieval x̂ of y = h(x) + ε, with diagnostics.

The retrieval observes the true state through x̂ ≈ μ_x + J_x̂ (x − μ_x), which
``OEResult.as_observation_operator`` returns for assimilating it downstream.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import lineax as lx
import optimistix as optx

from vardax._src.obs_operators.callable import CallableObs


class OEResult(eqx.Module):
    """The output of [`optimal_estimation`][vardax.optimal_estimation].

        Σ_x|y = (Hᵀ Σ_ε⁻¹ H + Σ_x⁻¹)⁻¹,   G = Σ_x|y Hᵀ Σ_ε⁻¹,   J_x̂ = G H
        d_s = tr J_x̂,   𝕀[x; y] = ½ ln det(Σ_x Σ_x|y⁻¹)

    Attributes:
        x_hat: The retrieval x̂.
        S_hat: Its posterior covariance Σ_x|y, a PSD operator.
        averaging_kernel: The averaging kernel J_x̂ = G H.
        dofs: Degrees of freedom for signal, d_s = tr J_x̂.
        information: Shannon information content 𝕀[x; y], in nats.
        chi2: The cost 𝒥(x̂), both terms, without the ½.
        d2: Rodgers' convergence measure of one more Gauss–Newton step δx from x̂:
            d² = δxᵀ Σ_x|y⁻¹ δx.
        n_iter: Levenberg–Marquardt steps taken.
        converged: Rodgers' criterion d² < ``convergence_tol`` · D_x holds and x̂ is
            finite. Levenberg–Marquardt's own tolerance may still be unmet
            (``n_iter`` at ``max_steps``) once the step left is this small in
            posterior units.
        prior_mean: μ_x, kept for ``as_observation_operator``.
        retrieval_noise_cov: Σ_x̂,ε = G Σ_ε Gᵀ, which equals J_x̂ Σ_x|y: the
            measurement-noise part of the retrieval error. It is the
            observation-error covariance to pair with ``as_observation_operator``,
            whose J_x̂ already carries the smoothing error (Rodgers 2000, §3.2).

    Examples:
        A scalar retrieval with H = Σ_x = Σ_ε = 1 halves the prior variance:

        >>> import jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> one = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> result = vdx.optimal_estimation(
        ...     lambda x: x, jnp.ones(1), jnp.zeros(1), one, one
        ... )
        >>> bool(jnp.allclose(result.S_hat.as_matrix(), 0.5))
        True

        One observation of one state carries half a degree of freedom, and the
        retrieval noise is Σ_x̂,ε = ½ · 1 · ½ = ¼:

        >>> bool(jnp.allclose(result.dofs, 0.5))
        True
        >>> bool(jnp.allclose(result.retrieval_noise_cov.as_matrix(), 0.25))
        True

    References:
        Rodgers (2000), *Inverse Methods for Atmospheric Sounding*, chapters 2–5.
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
        """The retrieval as an observation of the state: x ↦ μ_x + J_x̂ (x − μ_x).

        Returns:
            A [`CallableObs`][vardax.CallableObs] to assimilate x̂ with observation
            error Σ_x̂,ε, ``retrieval_noise_cov``. Not Σ_x|y: that also counts the
            smoothing error (J_x̂ − I)(x − μ_x), which the operator models through
            J_x̂, so using it would underweight the retrieval (Rodgers 2000, §3.2).

        Examples:
            The prior mean maps to itself, whatever the averaging kernel:

            >>> import jax.numpy as jnp, lineax as lx
            >>> import vardax as vdx
            >>> one = lx.DiagonalLinearOperator(jnp.ones(1))
            >>> result = vdx.optimal_estimation(
            ...     lambda x: x, jnp.ones(1), jnp.full(1, 0.5), one, one
            ... )
            >>> obs_op = result.as_observation_operator()
            >>> bool(jnp.allclose(obs_op(jnp.full(1, 0.5)), 0.5))
            True

            A state away from the prior is damped by J_x̂ = ½:

            >>> bool(jnp.allclose(obs_op(jnp.full(1, 2.5)), 1.5))
            True
        """
        prior_mean = self.prior_mean
        kernel = self.averaging_kernel

        def retrieve(state: Float[Array, " n"]) -> Float[Array, " n"]:
            # x − μ_x
            anomaly = state - prior_mean
            # μ_x + J_x̂ (x − μ_x)
            return prior_mean + kernel @ anomaly

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
    """Rodgers optimal estimation of x from y = h(x) + ε.

        𝒥(x) = (y − h(x))ᵀ Σ_ε⁻¹ (y − h(x)) + (x − μ_x)ᵀ Σ_x⁻¹ (x − μ_x)
        x̂ = argmin_x 𝒥(x)

    Algorithm:
        1. Whiten both terms of 𝒥 with Cholesky factors, Σ_ε = L_ε L_εᵀ and
           Σ_x = L_x L_xᵀ, by ``gaussx.cholesky``, which keeps diagonal, Kronecker
           and block-diagonal structure.
        2. Minimise ‖[L_ε⁻¹ (y − h(x)); L_x⁻¹ (x − μ_x)]‖² from μ_x by
           Levenberg–Marquardt with ``ImplicitAdjoint``, so x̂ is differentiable in
           y and in the parameters ``forward`` closes over.
        3. At x̂, form H = J_h by ``jax.jacfwd`` and the diagnostics Σ_x|y, J_x̂, d_s,
           𝕀[x; y] and 𝒥(x̂).
        4. Report Rodgers' (§5.6.3) d² of one more Gauss–Newton step, and call the
           retrieval converged when d² < ``convergence_tol`` · D_x.

    A pixel swath is ``eqx.filter_vmap`` over pixels; the solver does not raise on
    failure, so check ``converged``.

    Args:
        forward: The forward model h, from the state to the measurements.
        y: Measurements y.
        x_a: Prior mean μ_x.
        S_a: Prior covariance Σ_x, a PSD operator.
        S_e: Measurement-error covariance Σ_ε, a PSD operator.
        max_steps: Levenberg–Marquardt step cap.
        rtol: Solver relative tolerance.
        atol: Solver absolute tolerance.
        convergence_tol: Threshold on d² / D_x.

    Returns:
        The retrieval and its diagnostics.

    Examples:
        A linear forward h(x) = H x is solved exactly, and its DOFS lie in [0, D_x]:

        >>> import jax, jax.numpy as jnp, lineax as lx
        >>> import vardax as vdx
        >>> obs_operator = jnp.array([[1.0, 0.5], [0.0, 1.0], [1.0, 1.0]])
        >>> prior_cov = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> noise_cov = lx.DiagonalLinearOperator(jnp.full(3, 0.1))
        >>> obs = jnp.array([1.0, 2.0, 3.0])
        >>> result = vdx.optimal_estimation(
        ...     lambda x: obs_operator @ x, obs, jnp.zeros(2), prior_cov, noise_cov
        ... )
        >>> bool(result.converged), result.x_hat.shape
        (True, (2,))
        >>> bool(0.0 <= float(result.dofs) <= 2.0)
        True

        A nonlinear forward iterates, and the diagnostics are taken at x̂:

        >>> def nonlinear(x):
        ...     return obs_operator @ x + 0.01 * (obs_operator @ x) ** 2
        >>> curved = vdx.optimal_estimation(
        ...     nonlinear, obs, jnp.zeros(2), prior_cov, noise_cov
        ... )
        >>> bool(curved.converged), int(curved.n_iter) > 1
        (True, True)
        >>> curved.averaging_kernel.shape
        (2, 2)

        A swath is a ``filter_vmap`` over pixels:

        >>> import equinox as eqx
        >>> swath = jnp.stack([obs, 2.0 * obs])
        >>> eqx.filter_vmap(
        ...     lambda pixel: vdx.optimal_estimation(
        ...         nonlinear, pixel, jnp.zeros(2), prior_cov, noise_cov
        ...     )
        ... )(swath).x_hat.shape
        (2, 2)

    References:
        Rodgers (2000), *Inverse Methods for Atmospheric Sounding*, chapters 2–5.
    """
    state_size = x_a.shape[0]
    whiten_obs = _whitener(S_e)
    whiten_prior = _whitener(S_a)

    def residual(
        state: Float[Array, " n"], measurements: Float[Array, " m"]
    ) -> tuple[Array, Array]:
        # y − h(x)
        innovation = measurements - forward(state)
        # r̃_ε = L_ε⁻¹ (y − h(x))
        whitened_innovation = whiten_obs(innovation)
        # x − μ_x
        increment = state - x_a
        # r̃_x = L_x⁻¹ (x − μ_x)
        return whitened_innovation, whiten_prior(increment)

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
    # x̂ = argmin_x 𝒥(x)
    retrieval = solution.value

    # H = J_h at x̂
    obs_jacobian = jax.jacfwd(forward)(retrieval)
    # H̃ = L_ε⁻¹ H
    whitened_jacobian = jax.vmap(whiten_obs, in_axes=1, out_axes=1)(obs_jacobian)
    # L_x⁻¹, so that Σ_x⁻¹ = L_x⁻ᵀ L_x⁻¹
    identity = jnp.eye(state_size, dtype=retrieval.dtype)
    prior_root_inverse = jax.vmap(whiten_prior, in_axes=1, out_axes=1)(identity)
    # Σ_x|y, J_x̂, d_s, 𝕀[x; y] and Σ_x̂,ε at x̂, by gaussx's square-root formulation
    diagnostics = gaussx.information_diagnostics(obs_jacobian, S_e, S_a)
    posterior_covariance = diagnostics.posterior_cov

    data_residual, prior_residual = residual(retrieval, y)
    # ‖r̃_ε‖²
    data_cost = jnp.sum(data_residual**2)
    # ‖r̃_x‖²
    prior_cost = jnp.sum(prior_residual**2)
    # 𝒥(x̂) = ‖r̃_ε‖² + ‖r̃_x‖²
    cost = data_cost + prior_cost
    # H̃ᵀ
    jacobian_adjoint = einx.id("obs state -> state obs", whitened_jacobian)
    # L_x⁻ᵀ
    root_inverse_adjoint = einx.id("row state -> state row", prior_root_inverse)
    # −H̃ᵀ r̃_ε; @, not einx.dot: the float bits differ
    data_gradient = -jacobian_adjoint @ data_residual
    # L_x⁻ᵀ r̃_x
    prior_gradient = root_inverse_adjoint @ prior_residual
    # ∇½𝒥 = −H̃ᵀ r̃_ε + L_x⁻ᵀ r̃_x; δx = −Σ_x|y ∇½𝒥
    half_cost_gradient = data_gradient + prior_gradient
    # d² = δxᵀ Σ_x|y⁻¹ δx = (∇½𝒥)ᵀ Σ_x|y (∇½𝒥)
    step_measure = half_cost_gradient @ posterior_covariance @ half_cost_gradient
    is_finite = jnp.all(jnp.isfinite(retrieval))
    # d² < convergence_tol · D_x
    is_small_step = step_measure < convergence_tol * state_size
    return OEResult(
        x_hat=retrieval,
        S_hat=lx.MatrixLinearOperator(
            posterior_covariance, lx.positive_semidefinite_tag
        ),
        averaging_kernel=diagnostics.averaging_kernel,
        dofs=diagnostics.dofs,
        information=diagnostics.information_content,
        chi2=cost,
        d2=step_measure,
        n_iter=jnp.asarray(solution.stats["num_steps"]),
        converged=is_finite & is_small_step,
        prior_mean=x_a,
        retrieval_noise_cov=lx.MatrixLinearOperator(
            diagnostics.retrieval_noise_cov, lx.positive_semidefinite_tag
        ),
    )


def _whitener(
    covariance: lx.AbstractLinearOperator,
) -> Callable[[Array], Array]:
    """Return v ↦ L⁻¹ v with L Lᵀ = Σ, keeping Σ's structure.

    ``gaussx.cholesky`` keeps a diagonal, Kronecker or block-diagonal Σ structured
    instead of materialising it.
    """
    # L Lᵀ = Σ, lower-triangular
    factor = gaussx.cholesky(covariance)

    def whiten(vector: Array) -> Array:
        # L⁻¹ v
        return gaussx.solve(factor, vector)

    return whiten
