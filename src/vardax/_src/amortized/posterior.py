"""``AmortizedPosterior`` — the seventh AnalysisStep.

Composes an encoder (``(input, mask) → context``) with a density head
(``context → q_φ(x | y)``). Trained by simulation:

```python
def sample_train_pair(key):
    x = prior_distribution.sample(key)
    y = forward_model(x) + obs_noise.sample(key)
    return Batch1D(input=y, mask=quality_mask, target=x)


for batch in simulation_loader:
    model, opt_state, loss = amortized_train_step(
        model,
        batch,
        optimizer,
        opt_state,
    )
```

Inference (sub-second):

```python
x_map = model(batch)  # MAP / mode
samples = model.sample(batch, key, n=200)  # posterior samples
log_p = model.log_prob(x, batch)  # exact for flow/regression
```
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from vardax._src._types import Batch1D, Batch2D

from .config import AmortizedConfig


class AmortizedPosterior(eqx.Module):
    r"""Amortized variational posterior $q_\phi(x \mid y)$.

    Composes an encoder $c = E(y, m)$ with a density head $q_\phi(x \mid c)$;
    a trained model answers any new observation with one forward pass.

    Algorithm:
        1. Encode each batch element's ``(input, mask)`` into a context vector
           (vmapped over the batch).
        2. Hand the context to the head: ``__call__`` takes its mode,
           ``sample`` draws from it, ``log_prob`` evaluates the density.

    Args:
        encoder: ``eqx.Module`` mapping ``(input, mask)`` to a context
            vector. Vmapped over the batch axis at call time.
        head: ``RegressionHead`` / ``ConditionalFlowHead`` /
            ``ScoreDiffusionHead``. Exposes ``map_estimate(ctx)``,
            ``sample(ctx, key, n)``, ``log_prob(x, ctx)``.
        config: ``AmortizedConfig`` carrying head type and sampling
            defaults.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> key = jax.random.PRNGKey(0)
        >>> posterior = vdx.AmortizedPosterior(
        ...     encoder=vdx.MLPObsEncoder(input_size=12, context_dim=6, key=key),
        ...     head=vdx.RegressionHead(context_dim=6, state_shape=(3, 4), key=key),
        ...     config=vdx.AmortizedConfig(n_samples=7),
        ... )
        >>> ones = jnp.ones((2, 3, 4))
        >>> batch = vdx.Batch1D(input=ones, mask=ones)
        >>> posterior(batch).shape
        (2, 3, 4)

        Samples default to ``config.n_samples`` and can be overridden:

        >>> posterior.sample(batch, jax.random.PRNGKey(1)).shape
        (2, 7, 3, 4)
        >>> posterior.sample(batch, jax.random.PRNGKey(1), n=3).shape
        (2, 3, 3, 4)

        One log-density comes back per batch element, here at the mode:

        >>> posterior.log_prob(posterior(batch), batch).shape
        (2,)
    """

    encoder: Any
    head: Any
    config: AmortizedConfig

    def __call__(self, batch: Batch1D | Batch2D) -> Float[Array, ...]:
        r"""Return the per-sample MAP / mode of $q_\phi(x \mid y)$.

        Args:
            batch: Observations ``(input, mask)`` to condition on.

        Returns:
            The mode of the posterior for each batch element, stacked over the batch.
        """

        def _one(input_sample, mask_sample):
            context = self.encoder(input_sample, mask_sample)
            return self.head.map_estimate(context)

        return jax.vmap(_one)(batch.input, batch.mask)

    def sample(
        self,
        batch: Batch1D | Batch2D,
        key: PRNGKeyArray,
        n: int | None = None,
    ) -> Float[Array, ...]:
        """Draw posterior samples per batch element.

        Args:
            batch: Observations ``(input, mask)`` to condition on.
            key: PRNG key, split into one key per batch element.
            n: Number of samples per element. Defaults to ``config.n_samples``.

        Returns:
            Samples stacked as ``(B, n, *state_shape)``.
        """
        n = self.config.n_samples if n is None else n
        batch_size = batch.input.shape[0]
        keys = jax.random.split(key, batch_size)

        def _one(input_sample, mask_sample, key_sample):
            context = self.encoder(input_sample, mask_sample)
            return self.head.sample(context, key_sample, n)

        return jax.vmap(_one)(batch.input, batch.mask, keys)

    def log_prob(
        self,
        x: Float[Array, ...],
        batch: Batch1D | Batch2D,
    ) -> Float[Array, " B"]:
        r"""Per-sample log-density of ``x`` under ``q_φ(·|y)``.

        For ``ScoreDiffusionHead`` this raises ``NotImplementedError``
        (no closed-form density). Use ``sample`` instead.

        Args:
            x: States whose density is evaluated, one per batch element.
            batch: Observations ``(input, mask)`` to condition on.

        Returns:
            The log-density $\log q_\phi(x \mid y)$ for each batch element.
        """

        def _one(state_sample, input_sample, mask_sample):
            context = self.encoder(input_sample, mask_sample)
            return self.head.log_prob(state_sample, context)

        return jax.vmap(_one)(x, batch.input, batch.mask)

    def as_analysis_step(self) -> _AmortizedAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8).

        Returns:
            A callable ``(forecast, obs, *, obs_op, obs_err_cov) -> analysis`` that
            masks the non-finite entries of ``obs`` and returns the MAP estimate.

        Examples:
            Non-finite observations are treated as unobserved:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> import vardax as vdx
            >>> key = jax.random.PRNGKey(0)
            >>> posterior = vdx.AmortizedPosterior(
            ...     encoder=vdx.MLPObsEncoder(input_size=12, context_dim=6, key=key),
            ...     head=vdx.RegressionHead(context_dim=6, state_shape=(3, 4), key=key),
            ...     config=vdx.AmortizedConfig(),
            ... )
            >>> obs = jnp.ones((1, 3, 4)).at[0, 0, 0].set(jnp.nan)
            >>> step = posterior.as_analysis_step()
            >>> step(obs, obs, obs_op=None, obs_err_cov=None).shape
            (1, 3, 4)
        """
        return _AmortizedAnalysisStep(self)


class _AmortizedAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``AmortizedPosterior``."""

    model: AmortizedPosterior

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        mask = jnp.where(jnp.isfinite(obs), 1.0, 0.0)
        obs_clean = jnp.nan_to_num(obs)
        if obs.ndim == 4:
            batch = Batch2D(input=obs_clean, mask=mask, target=None)
        else:
            batch = Batch1D(input=obs_clean, mask=mask, target=None)
        return self.model(batch)
