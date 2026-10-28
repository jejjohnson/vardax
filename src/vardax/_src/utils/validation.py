r"""Six-step cycle validation gates (Decision D12 / Epic 8).

Amortized inference is dangerous when miscalibrated — a tight
$q_\phi$ centred wrong gives confident wrong answers. Decision
D12 (chapter 14 of the math reference) requires three gates before any
amortized head is promoted to operational use:

1. **Posterior agreement** — the fast model's MAP lies within
   $\sigma_\text{post}$ of an oracle MAP from a slower method.
2. **Adjoint calibration** — the fast model's sensitivity to
   observations matches the physics-based sensitivity, measured by
   random-vector probing of the Jacobian.
3. **Simulation-based calibration (SBC)** — per Talts et al. 2018,
   rank histograms of true draws within posterior samples are uniform
   for a well-calibrated head.

These utilities are deliberately framework-agnostic: they accept the
fast and oracle models as plain callables / vardax ``Posterior``
objects and return / assert numeric results. They do not depend on the
specific Layer 2 method.
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
    r"""Check that ``p_fast.mean`` lies within ``tolerance_sigma`` standard
    deviations of ``p_oracle.mean``.

    Marginal-only test: each component of the mean must satisfy

    $$
    \frac{|x^*_\text{fast} - x^*_\text{oracle}|}{\sigma_\text{post, oracle}}
    \le \text{tolerance\_sigma}.
    $$

    The oracle marginal $\sigma_i$ is extracted from
    ``p_oracle.cov`` by probing $e_i^T \Sigma e_i$ via one matvec per
    component. Cheap for moderate state size; for very large state
    sizes use Hutchinson estimation upstream and supply the diagonal
    directly via ``Posterior.cov`` materialised as a diagonal operator.

    Algorithm:
        1. Extract the oracle marginal standard deviations $\sigma_i$ from
           ``p_oracle.cov``.
        2. Form the mean difference $\Delta x^* = x^*_\text{fast} - x^*_\text{oracle}$.
        3. Compute the z-scores $z_i = |\Delta x^*_i| / \sigma_i$.
        4. Fail if any $z_i$ exceeds ``tolerance_sigma``.

    Args:
        p_fast: Posterior produced by the amortized / fast model.
        p_oracle: Posterior produced by the oracle (e.g. ``StrongFourDVar``
            + [`LaplaceCovariance`][vardax.LaplaceCovariance]).
        tolerance_sigma: Allowed deviation in units of $\sigma$.

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
    # z = |x*_fast − x*_oracle| / σ_oracle
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
    r"""Random-vector probe of the Jacobian agreement at ``y``.

    Tests

    $$
    \frac{\|J_\text{fast} v - J_\text{oracle} v\|}
         {\|J_\text{oracle} v\|} < \text{threshold}
    $$

    for ``n_probes`` random unit vectors $v$. Avoids
    materialising either Jacobian — uses ``jax.jvp`` to apply each as
    needed. Cheaper than dense Jacobian comparison and is the
    operational test used by the six-step cycle.

    Algorithm:
        1. Draw a random unit vector $v$ of the shape of ``y``.
        2. Apply both Jacobians to it with ``jax.jvp``.
        3. Compute the relative error of the two products.
        4. Repeat for ``n_probes`` vectors and fail if the worst error exceeds
           ``threshold``.

    Args:
        fn_fast: Callable ``y → analysis`` (e.g.
            ``lambda y_: amortized(Batch1D(input=y_, mask=mask, target=None))``).
        fn_oracle: Same signature, but the oracle.
        y: Observation tensor at which the Jacobians are probed.
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
    worst_rel = 0.0
    for probe_key in keys:
        probe = jax.random.normal(probe_key, y.shape)
        probe = probe / jnp.linalg.norm(probe)
        _, jvp_fast = jax.jvp(fn_fast, (y,), (probe,))
        _, jvp_oracle = jax.jvp(fn_oracle, (y,), (probe,))
        denom = jnp.linalg.norm(jvp_oracle) + 1e-12
        # ‖J_fast v − J_oracle v‖ / ‖J_oracle v‖
        relative_error = float(jnp.linalg.norm(jvp_fast - jvp_oracle) / denom)
        worst_rel = max(worst_rel, relative_error)
    if worst_rel > threshold:
        raise AssertionError(
            f"adjoint calibration failed: max relative error {worst_rel:.4f} > "
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
    r"""Per Talts et al. 2018: rank histogram of true draws within
    posterior samples.

    Procedure for each of ``n_runs`` independent draws:

    1. Sample $x^{(j)} \sim p(x)$ from the prior.
    2. Simulate $y^{(j)} = \mathrm{simulate\_obs}(x^{(j)})$.
    3. Draw ``n_samples`` posterior samples from
       $q_\phi(\cdot \mid y^{(j)})$.
    4. Compute the rank of (a flattened scalar reduction of)
       $x^{(j)}$ in the sample set.

    A well-calibrated posterior produces uniformly distributed ranks
    over ``[0, n_samples]``. Bumps near the edges → over-confident;
    centre-mass bump → under-confident.

    Algorithm:
        1. For each run, split the key into prior, observation and posterior keys.
        2. Draw $x$, simulate $y$ and draw the posterior samples.
        3. Reduce $x$ and every sample to its $\ell_2$ norm.
        4. Record how many sample norms fall below the norm of $x$.

    References:
        Talts, Betancourt, Simpson, Vehtari & Gelman (2018). "Validating
        Bayesian inference algorithms with simulation-based calibration."

    Args:
        sample_posterior: ``(y, key, n) -> Array`` of posterior samples with
            ``n`` leading rows, conditioned on ``y``. For an
            ``AmortizedPosterior``, the natural wrapper is
            ``lambda y, k, n: model.sample(Batch(input=y[None], ...), k, n)[0]``.
        sample_prior: ``key -> x``. Single prior draw.
        simulate_obs: ``(x, key) -> y``. Forward + noise.
        key: Top-level PRNG key.
        n_runs: Number of (prior, obs, posterior) triples.
        n_samples: Posterior samples per run (defines the rank
            histogram resolution).

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
        >>> def sample_posterior(y, k, n):
        ...     batch = vdx.Batch1D(input=y[None], mask=jnp.ones_like(y)[None])
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
    """
    keys = jax.random.split(key, n_runs)
    ranks = []
    for run_key in keys:
        key_prior, key_obs, key_post = jax.random.split(run_key, 3)
        x_true = sample_prior(key_prior)
        observation = simulate_obs(x_true, key_obs)
        samples = sample_posterior(observation, key_post, n_samples)
        # Flatten and use the L2 norm as the scalar reduction.
        scalar_true = float(jnp.linalg.norm(x_true))
        # ‖xᵢ‖₂ over every state dimension of each sample
        squared_norms = einx.sum("n [s...]", samples**2)
        scalar_samples = jnp.sqrt(squared_norms)
        rank = int(jnp.sum(scalar_samples < scalar_true))
        ranks.append(rank)
    return jnp.asarray(ranks, dtype=jnp.int32)


def _marginal_std(cov_op: Any, ref: Float[Array, ...]) -> Float[Array, ...]:
    r"""Extract $\sqrt{\mathrm{diag}(\Sigma)}$ from the operator.

    Delegates to ``gaussx.diag``, which uses closed forms for
    structured operators (diagonal, Kronecker, block-diagonal) and
    falls back to materialisation only for generic operators — fine
    for the moderate-N regime where this gate is meaningful.
    """
    diag = gaussx.diag(cov_op)
    # σ = √max(diag Σ, 0)
    marginal_std = jnp.sqrt(jnp.maximum(diag, 0.0))
    return as_shape(marginal_std, ref.shape)
