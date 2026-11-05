"""``AmortizedPosterior``: an encoder and a density head forming q(x | y; ν).

The encoder maps ``(input, mask)`` to a context c and the head maps c to a
density over the state. Trained by simulation:

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
    """Amortized variational posterior q(x | y; ν).

        c = g_enc(y, m; ν),   q(x | y; ν) = q(x | c; ν)

    A trained model answers any new observation with one forward pass.

    Algorithm:
        1. Encode each batch element's ``(input, mask)`` into a context c,
           vmapped over the batch.
        2. Hand c to the head: ``__call__`` takes its mode, ``sample`` draws
           from it and ``log_prob`` evaluates its log-density.

    Attributes:
        encoder: Observation encoder g_enc mapping ``(input, mask)`` to the
            context c, vmapped over the batch axis at call time.
        head: Density head q(x | c; ν), a ``RegressionHead``,
            ``ConditionalFlowHead`` or ``ScoreDiffusionHead``. Exposes
            ``map_estimate(ctx)``, ``sample(ctx, key, n)`` and
            ``log_prob(x, ctx)``.
        config: ``AmortizedConfig`` carrying the head type and the default
            sample count.

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
        """Return the per-sample mode x̂ = argmaxₓ q(x | y; ν).

        Args:
            batch: Observations ``(input, mask)`` to condition on.

        Returns:
            The mode of the posterior for each batch element, stacked over the batch.
        """

        def _one(input_sample, mask_sample):
            # c = g_enc(y, m; ν), then x̂ = mode of q(x | c; ν)
            context = self.encoder(input_sample, mask_sample)
            return self.head.map_estimate(context)

        return jax.vmap(_one)(batch.input, batch.mask)

    def sample(
        self,
        batch: Batch1D | Batch2D,
        key: PRNGKeyArray,
        n: int | None = None,
    ) -> Float[Array, ...]:
        """Draw posterior samples x ∼ q(x | y; ν) per batch element.

        Args:
            batch: Observations ``(input, mask)`` to condition on.
            key: PRNG key, split into one key per batch element.
            n: Number of samples N_E per element. Defaults to ``config.n_samples``.

        Returns:
            Samples stacked as ``(B, n, *state_shape)``.
        """
        n = self.config.n_samples if n is None else n
        batch_size = batch.input.shape[0]
        # One independent key per batch element
        keys = jax.random.split(key, batch_size)

        def _one(input_sample, mask_sample, key_sample):
            # c = g_enc(y, m; ν), then x ∼ q(x | c; ν)
            context = self.encoder(input_sample, mask_sample)
            return self.head.sample(context, key_sample, n)

        return jax.vmap(_one)(batch.input, batch.mask, keys)

    def log_prob(
        self,
        x: Float[Array, ...],
        batch: Batch1D | Batch2D,
    ) -> Float[Array, " B"]:
        """Per-sample log-density log q(x | y; ν).

        Args:
            x: States whose density is evaluated, one per batch element.
            batch: Observations ``(input, mask)`` to condition on.

        Returns:
            The log-density log q(x | y; ν) for each batch element.
        """

        def _one(state_sample, input_sample, mask_sample):
            # c = g_enc(y, m; ν), then log q(x | c; ν)
            context = self.encoder(input_sample, mask_sample)
            return self.head.log_prob(state_sample, context)

        return jax.vmap(_one)(x, batch.input, batch.mask)

    def as_analysis_step(self) -> _AmortizedAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep``.

            μ_x,t|t = x̂(y_t, m_t),   m_t = 1 where y_t is finite, else 0

        Returns:
            A callable ``(forecast, obs, *, obs_op, obs_err_cov) -> analysis`` that
            masks the non-finite entries of ``obs`` and returns the mode of the
            posterior. The forecast, ``obs_op`` and ``obs_err_cov`` are unused.

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
    """``pipekit_cycle.AnalysisStep`` adapter: the analysis is the posterior mode."""

    model: AmortizedPosterior

    def __call__(
        self,
        forecast: Float[Array, ...],
        obs: Float[Array, ...],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, ...]:
        """Return μ_x,t|t = x̂(y_t, m_t) with non-finite entries of y_t masked."""
        # m = 1 where y is finite, else 0
        mask = jnp.where(jnp.isfinite(obs), 1.0, 0.0)
        # Masked entries are zeroed so the encoder never sees NaN or inf
        obs_clean = jnp.nan_to_num(obs)
        # Four axes mean (batch, time, y, x); otherwise (batch, time, obs)
        if obs.ndim == 4:
            batch = Batch2D(input=obs_clean, mask=mask, target=None)
        else:
            batch = Batch1D(input=obs_clean, mask=mask, target=None)
        # x̂ = mode of q(x | y; ν)
        return self.model(batch)
