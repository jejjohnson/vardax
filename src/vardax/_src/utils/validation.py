"""Validation gates for an amortized head: posterior agreement, adjoint and SBC.

A tight posterior centred wrong gives confident wrong answers, so these gates check
the fast model, taken as a plain callable or ``Posterior``, before it is promoted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from vardax._src.errors import MissingInputError, ShapeError
from vardax._src.posterior.container import Posterior
from vardax._src.utils.shapes import as_shape


def assert_posterior_agreement(
    p_fast: Posterior,
    p_oracle: Posterior,
    *,
    tolerance_sigma: float = 1.0,
) -> None:
    """Check that the fast posterior mean lies near the oracle's, per component.

        zᵢ = |μ_fast,i − μ_oracle,i| / σᵢ ≤ ``tolerance_sigma``,   σᵢ² = [Σ_x|y]ᵢᵢ

    with Σ_x|y the oracle's posterior covariance. The test is marginal only. The
    diagonal comes from ``gaussx.diag``, in closed form for a structured operator and
    by materialising a generic one, which is cheap for moderate state size; for very
    large state sizes use Hutchinson estimation upstream and supply the diagonal
    directly via ``Posterior.cov`` materialised as a diagonal operator.

    Algorithm:
        1. Extract the oracle marginal standard deviations σᵢ from ``p_oracle.cov``.
        2. Compute the z-scores zᵢ from the mean difference μ_fast − μ_oracle.
        3. Fail if any zᵢ exceeds ``tolerance_sigma``.

    Args:
        p_fast: Posterior produced by the amortized or fast model, with mean μ_fast.
        p_oracle: Posterior produced by the oracle (e.g. ``StrongFourDVar``
            + [`LaplaceCovariance`][vardax.LaplaceCovariance]), with mean μ_oracle.
        tolerance_sigma: Allowed deviation in units of σᵢ.

    Raises:
        MissingInputError: If ``p_oracle.cov`` is ``None``.
        ShapeError: If the two means have different shapes.
        AssertionError: If any component exceeds the tolerance.

    Examples:
        A fast mean within one oracle standard deviation passes silently:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import vardax as vdx
        >>> cov = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
        >>> oracle = vdx.Posterior(mean=jnp.zeros(3), cov=cov)
        >>> fast = vdx.Posterior(mean=jnp.array([0.5, -0.5, 0.0]))
        >>> vdx.assert_posterior_agreement(fast, oracle, tolerance_sigma=1.0)

        A deviation of three standard deviations fails:

        >>> far = vdx.Posterior(mean=jnp.array([3.0, 0.0, 0.0]))
        >>> vdx.assert_posterior_agreement(
        ...     far, oracle
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        AssertionError: posterior agreement failed: max z = 3.000 > 1.000.

        An oracle without a covariance cannot supply the scale:

        >>> vdx.assert_posterior_agreement(
        ...     fast, fast
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        MissingInputError: assert_posterior_agreement requires p_oracle.cov ...
    """
    if p_oracle.cov is None:
        raise MissingInputError(
            "assert_posterior_agreement requires p_oracle.cov to be set "
            "(used to extract the marginal standard deviations)."
        )
    if p_fast.mean.shape != p_oracle.mean.shape:
        raise ShapeError(
            "p_fast.mean and p_oracle.mean must have the same shape; got "
            f"{p_fast.mean.shape} vs {p_oracle.mean.shape}."
        )
    oracle_std = _marginal_std(p_oracle.cov, p_oracle.mean)
    # zᵢ = |μ_fast,i − μ_oracle,i| / σᵢ, floored so a zero variance cannot divide by 0
    z_score = jnp.abs(p_fast.mean - p_oracle.mean) / jnp.maximum(oracle_std, 1e-12)
    if jnp.any(z_score > tolerance_sigma):
        worst = float(jnp.max(z_score))
        raise AssertionError(
            f"posterior agreement failed: max z = {worst:.3f} > {tolerance_sigma:.3f}."
        )


def assert_adjoint_calibrated(
    fn_fast: Callable[[Array], Array],
    fn_oracle: Callable[[Array], Array],
    y: Float[Array, ...],
    *,
    key: PRNGKeyArray,
    threshold: float = 0.05,
    n_probes: int = 10,
) -> None:
    """Random-vector probe of the Jacobian agreement at ``y``.

        ‖J_fast v − J_oracle v‖ / ‖J_oracle v‖ ≤ ``threshold``

    for ``n_probes`` random unit vectors v. It avoids materialising either Jacobian
    by applying each with ``jax.jvp``, which is cheaper than a dense Jacobian
    comparison and is the operational test used by the six-step cycle.

    Algorithm:
        1. Draw a random unit vector v of the shape of ``y``.
        2. Apply both Jacobians to it with ``jax.jvp``.
        3. Compute the relative error of the two products.
        4. Repeat for ``n_probes`` vectors and fail if the worst error exceeds
           ``threshold``.

    Args:
        fn_fast: Callable from y to the analysis, with Jacobian J_fast (e.g.
            ``lambda y_: amortized(Batch1D(input=y_, mask=mask, target=None))``).
        fn_oracle: Same signature, but the oracle, with Jacobian J_oracle.
        y: Observation y at which the Jacobians are probed.
        key: PRNG key for the probe vectors.
        threshold: Maximum allowed relative error.
        n_probes: Number of probe vectors.

    Raises:
        AssertionError: If any probe exceeds the threshold.

    Examples:
        Two maps with the same Jacobian agree, even if their values differ:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> y = jnp.arange(4.0)
        >>> vdx.assert_adjoint_calibrated(
        ...     lambda v: 2.0 * v + 1.0,
        ...     lambda v: 2.0 * v,
        ...     y,
        ...     key=jax.random.PRNGKey(0),
        ... )

        A fast model whose sensitivity is off by a third fails:

        >>> vdx.assert_adjoint_calibrated(
        ...     lambda v: 2.0 * v, lambda v: 3.0 * v, y, key=jax.random.PRNGKey(0)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        AssertionError: adjoint calibration failed: ...
    """
    keys = jax.random.split(key, n_probes)
    worst_relative_error = 0.0
    for probe_key in keys:
        # v = ξ / ‖ξ‖, ξ ~ 𝒩(0, I)
        probe = jax.random.normal(probe_key, y.shape)
        probe = probe / jnp.linalg.norm(probe)
        # J_fast v
        _, jvp_fast = jax.jvp(fn_fast, (y,), (probe,))
        # J_oracle v
        _, jvp_oracle = jax.jvp(fn_oracle, (y,), (probe,))
        # ‖J_oracle v‖, floored so a zero sensitivity cannot divide by 0
        denominator = jnp.linalg.norm(jvp_oracle) + 1e-12
        # ‖J_fast v − J_oracle v‖ / ‖J_oracle v‖
        relative_error = float(jnp.linalg.norm(jvp_fast - jvp_oracle) / denominator)
        worst_relative_error = max(worst_relative_error, relative_error)
    if worst_relative_error > threshold:
        raise AssertionError(
            f"adjoint calibration failed: max relative error "
            f"{worst_relative_error:.4f} > "
            f"{threshold:.4f} (over {n_probes} probes)."
        )


def simulation_based_calibration(
    sample_posterior: Callable[[Array, PRNGKeyArray, int], Array],
    sample_prior: Callable[[PRNGKeyArray], Array],
    simulate_obs: Callable[[Array, PRNGKeyArray], Array],
    *,
    key: PRNGKeyArray,
    n_runs: int = 200,
    n_samples: int = 200,
) -> Int[Array, " n_runs"]:
    """Rank histogram of true draws within posterior samples (Talts et al. 2018).

        rankₙ = |{e : ‖x⁽ᵉ⁾‖₂ < ‖xₙ‖₂}|,   x⁽ᵉ⁾ ~ q(x ∣ yₙ; ν),   xₙ ~ p(x)

    with n indexing the ``n_runs`` independent runs and e the ``n_samples`` posterior
    samples. A well-calibrated posterior produces uniformly distributed ranks over
    ``[0, n_samples]``; bumps near the edges mean over-confident, a bump at the centre
    means under-confident.

    Algorithm:
        1. For each run, split the key into prior, observation and posterior keys.
        2. Draw xₙ from the prior, simulate yₙ and draw the posterior samples x⁽ᵉ⁾.
        3. Reduce xₙ and every sample to its ℓ₂ norm.
        4. Record how many sample norms fall below the norm of xₙ.

    Args:
        sample_posterior: ``(y, key, n) -> Array`` of posterior samples with
            ``n`` leading rows, conditioned on ``y``. An ``AmortizedPosterior`` plugs
            in through a one-element batch, as in the last example.
        sample_prior: ``key -> x``. Single prior draw xₙ.
        simulate_obs: ``(x, key) -> y``. Forward plus noise, yₙ.
        key: Top-level PRNG key.
        n_runs: Number of (prior, obs, posterior) triples.
        n_samples: Posterior samples per run (defines the rank histogram resolution).

    Returns:
        Integer ranks in ``[0, n_samples]``, one per run.

    Examples:
        Ranks of the truth among samples drawn around the observation lie in
        ``[0, n_samples]``:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> prior = lambda k: jax.random.normal(k, (2,))
        >>> observe = lambda x, k: x + 0.1 * jax.random.normal(k, (2,))
        >>> posterior = lambda y, k, n: y + 0.1 * jax.random.normal(k, (n, 2))
        >>> ranks = vdx.simulation_based_calibration(
        ...     posterior,
        ...     prior,
        ...     observe,
        ...     key=jax.random.PRNGKey(0),
        ...     n_runs=6,
        ...     n_samples=10,
        ... )
        >>> ranks.shape, bool(jnp.all((ranks >= 0) & (ranks <= 10)))
        ((6,), True)

        A posterior that ignores the data puts the truth below every sample, so
        every rank is zero:

        >>> wrong = lambda y, k, n: 100.0 + jax.random.normal(k, (n, 2))
        >>> bad = vdx.simulation_based_calibration(
        ...     wrong,
        ...     prior,
        ...     observe,
        ...     key=jax.random.PRNGKey(0),
        ...     n_runs=6,
        ...     n_samples=10,
        ... )
        >>> bool(jnp.all(bad == 0))
        True

        An [`AmortizedPosterior`][vardax.AmortizedPosterior] plugs in through a
        one-element batch:

        >>> key = jax.random.PRNGKey(1)
        >>> amortized = vdx.AmortizedPosterior(
        ...     encoder=vdx.MLPObsEncoder(input_size=6, context_dim=4, key=key),
        ...     head=vdx.RegressionHead(context_dim=4, state_shape=(2, 3), key=key),
        ...     config=vdx.AmortizedConfig(),
        ... )
        >>> import einx
        >>> def sample_posterior(y, k, n):
        ...     window = einx.id("s... -> 1 s...", y)
        ...     batch = vdx.Batch1D(input=window, mask=jnp.ones_like(window))
        ...     return amortized.sample(batch, k, n)[0]
        >>> ranks = vdx.simulation_based_calibration(
        ...     sample_posterior,
        ...     lambda k: jax.random.normal(k, (2, 3)),
        ...     lambda x, k: x,
        ...     key=jax.random.PRNGKey(2),
        ...     n_runs=4,
        ...     n_samples=5,
        ... )
        >>> ranks.shape
        (4,)

    References:
        Talts, Betancourt, Simpson, Vehtari & Gelman (2018). "Validating Bayesian
        inference algorithms with simulation-based calibration."
    """
    keys = jax.random.split(key, n_runs)
    ranks = []
    for run_key in keys:
        prior_key, obs_key, posterior_key = jax.random.split(run_key, 3)
        # xₙ ~ p(x)
        true_state = sample_prior(prior_key)
        # yₙ = h(xₙ) + ε
        observation = simulate_obs(true_state, obs_key)
        # x⁽ᵉ⁾ ~ q(x ∣ yₙ; ν)
        samples = sample_posterior(observation, posterior_key, n_samples)
        # ‖xₙ‖₂ is the scalar reduction the rank compares
        true_norm = float(jnp.linalg.norm(true_state))
        # ‖x⁽ᵉ⁾‖₂² over every state dimension of each sample
        squared_norms = einx.sum("member [s...]", samples**2)
        # ‖x⁽ᵉ⁾‖₂
        sample_norms = jnp.sqrt(squared_norms)
        # rankₙ = |{e : ‖x⁽ᵉ⁾‖₂ < ‖xₙ‖₂}|
        rank = int(jnp.sum(sample_norms < true_norm))
        ranks.append(rank)
    return jnp.asarray(ranks, dtype=jnp.int32)


def _marginal_std(cov_op: Any, ref: Float[Array, ...]) -> Float[Array, ...]:
    """Extract σᵢ = √[Σ]ᵢᵢ from the operator Σ, shaped like ``ref``.

    Delegates to ``gaussx.diag``, which uses closed forms for structured operators
    (diagonal, Kronecker, block-diagonal) and falls back to materialisation only for
    generic operators, which is fine for the moderate-N regime where this gate is
    meaningful.
    """
    # diag(Σ)
    diagonal = gaussx.diag(cov_op)
    # σᵢ = √max([Σ]ᵢᵢ, 0): the floor absorbs small negative round-off
    marginal_std = jnp.sqrt(jnp.maximum(diagonal, 0.0))
    return as_shape(marginal_std, ref.shape)
