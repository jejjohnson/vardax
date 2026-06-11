"""Six-step cycle validation gates (Decision D12 / Epic 8).

Amortized inference is dangerous when miscalibrated — a tight
:math:`q_\\phi` centred wrong gives confident wrong answers. Decision
D12 (chapter 14 of the math reference) requires three gates before any
amortized head is promoted to operational use:

1. **Posterior agreement** — the fast model's MAP lies within
   :math:`\\sigma_\\text{post}` of an oracle MAP from a slower method.
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

import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from vardax._src.posterior.container import Posterior


def assert_posterior_agreement(
    p_fast: Posterior,
    p_oracle: Posterior,
    *,
    tolerance_sigma: float = 1.0,
) -> None:
    """Check that ``p_fast.mean`` lies within ``tolerance_sigma`` standard
    deviations of ``p_oracle.mean``.

    Marginal-only test: each component of the mean must satisfy

    .. math::

        \\frac{|x^*_\\text{fast} - x^*_\\text{oracle}|}{\\sigma_\\text{post, oracle}}
        \\le \\text{tolerance\\_sigma}.

    The oracle marginal :math:`\\sigma_i` is extracted from
    ``p_oracle.cov`` by probing ``e_i^T \\Sigma e_i`` via one matvec per
    component. Cheap for moderate state size; for very large state
    sizes use Hutchinson estimation upstream and supply the diagonal
    directly via ``Posterior.cov`` materialised as a diagonal operator.

    Args:
        p_fast: Posterior produced by the amortized / fast model.
        p_oracle: Posterior produced by the oracle (e.g. ``StrongFourDVar``
            + ``LaplaceCovariance``).
        tolerance_sigma: Allowed deviation in units of :math:`\\sigma`.

    Raises:
        AssertionError: If any component exceeds the tolerance.
    """
    if p_oracle.cov is None:
        raise ValueError(
            "assert_posterior_agreement requires p_oracle.cov to be set "
            "(used to extract the marginal standard deviations)."
        )
    if p_fast.mean.shape != p_oracle.mean.shape:
        raise ValueError(
            "p_fast.mean and p_oracle.mean must have the same shape; got "
            f"{p_fast.mean.shape} vs {p_oracle.mean.shape}."
        )
    sigma = _marginal_std(p_oracle.cov, p_oracle.mean)
    z = jnp.abs(p_fast.mean - p_oracle.mean) / jnp.maximum(sigma, 1e-12)
    if jnp.any(z > tolerance_sigma):
        worst = float(jnp.max(z))
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

    Tests

    .. math::

        \\frac{\\|J_\\text{fast} v - J_\\text{oracle} v\\|}
             {\\|J_\\text{oracle} v\\|} < \\text{threshold}

    for ``n_probes`` random unit vectors :math:`v`. Avoids
    materialising either Jacobian — uses ``jax.jvp`` to apply each as
    needed. Cheaper than dense Jacobian comparison and is the
    operational test used by the six-step cycle.

    Args:
        fn_fast: Callable ``y → analysis`` (e.g.
            ``lambda y_: amortized(Batch1D(input=y_, mask=mask, target=None))``).
        fn_oracle: Same signature, but the oracle.
        y: Observation tensor.
        key: PRNG key for the probe vectors.
        threshold: Maximum allowed relative error.
        n_probes: Number of probe vectors.

    Raises:
        AssertionError: If any probe exceeds the threshold.
    """
    keys = jax.random.split(key, n_probes)
    worst_rel = 0.0
    for k in keys:
        v = jax.random.normal(k, y.shape)
        v = v / jnp.linalg.norm(v)
        _, jv_fast = jax.jvp(fn_fast, (y,), (v,))
        _, jv_oracle = jax.jvp(fn_oracle, (y,), (v,))
        denom = jnp.linalg.norm(jv_oracle) + 1e-12
        rel = float(jnp.linalg.norm(jv_fast - jv_oracle) / denom)
        worst_rel = max(worst_rel, rel)
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
) -> Float[Array, " n_runs"]:
    """Per Talts et al. 2018: rank histogram of true draws within
    posterior samples.

    Procedure for each of ``n_runs`` independent draws:

    1. Sample :math:`x^{(j)} \\sim p(x)` from the prior.
    2. Simulate :math:`y^{(j)} = \\mathrm{simulate\\_obs}(x^{(j)})`.
    3. Draw ``n_samples`` posterior samples from
       :math:`q_\\phi(\\cdot \\mid y^{(j)})`.
    4. Compute the rank of (a flattened scalar reduction of)
       :math:`x^{(j)}` in the sample set.

    A well-calibrated posterior produces uniformly distributed ranks
    over ``[0, n_samples]``. Bumps near the edges → over-confident;
    centre-mass bump → under-confident.

    Args:
        sample_posterior: ``(y, key, n) -> Array`` of shape
            ``(n, *state_shape)`` — posterior samples conditioned on
            ``y``. For an ``AmortizedPosterior``, the natural wrapper is
            ``lambda y, k, n: model.sample(Batch(input=y[None], ...), k, n)[0]``.
        sample_prior: ``key -> x``. Single prior draw.
        simulate_obs: ``(x, key) -> y``. Forward + noise.
        key: Top-level PRNG key.
        n_runs: Number of (prior, obs, posterior) triples.
        n_samples: Posterior samples per run (defines the rank
            histogram resolution).

    Returns:
        ``(n_runs,)`` int array of ranks in ``[0, n_samples]``.
    """
    keys = jax.random.split(key, n_runs)
    ranks = []
    for k in keys:
        k_prior, k_obs, k_post = jax.random.split(k, 3)
        x_true = sample_prior(k_prior)
        y = simulate_obs(x_true, k_obs)
        samples = sample_posterior(y, k_post, n_samples)
        # Flatten and use the L2 norm as the scalar reduction.
        scalar_true = float(jnp.linalg.norm(x_true))
        scalar_samples = jnp.linalg.norm(samples.reshape(n_samples, -1), axis=-1)
        rank = int(jnp.sum(scalar_samples < scalar_true))
        ranks.append(rank)
    return jnp.asarray(ranks, dtype=jnp.int32)


def _marginal_std(cov_op: Any, ref: Float[Array, ...]) -> Float[Array, ...]:
    """Extract :math:`\\sqrt{\\mathrm{diag}(\\Sigma)}` from the operator.

    Delegates to :func:`gaussx.diag`, which uses closed forms for
    structured operators (diagonal, Kronecker, block-diagonal) and
    falls back to materialisation only for generic operators — fine
    for the moderate-N regime where this gate is meaningful.
    """
    diag = gaussx.diag(cov_op)
    return jnp.sqrt(jnp.maximum(diag, 0.0)).reshape(ref.shape)
