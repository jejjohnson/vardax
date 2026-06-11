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

    Attributes:
        encoder: ``eqx.Module`` mapping ``(input, mask)`` to a context
            vector. Vmapped over the batch axis at call time.
        head: ``RegressionHead`` / ``ConditionalFlowHead`` /
            ``ScoreDiffusionHead``. Exposes ``map_estimate(ctx)``,
            ``sample(ctx, key, n)``, ``log_prob(x, ctx)``.
        config: ``AmortizedConfig`` carrying head type and sampling
            defaults.
    """

    encoder: Any
    head: Any
    config: AmortizedConfig

    def __call__(self, batch: Batch1D | Batch2D) -> Float[Array, ...]:
        r"""Return the per-sample MAP / mode of $q_\phi(x \mid y)$."""

        def _one(input_i, mask_i):
            ctx = self.encoder(input_i, mask_i)
            return self.head.map_estimate(ctx)

        return jax.vmap(_one)(batch.input, batch.mask)

    def sample(
        self,
        batch: Batch1D | Batch2D,
        key: PRNGKeyArray,
        n: int | None = None,
    ) -> Float[Array, ...]:
        """Draw posterior samples per batch element.

        Returns an array of shape ``(B, n, *state_shape)``.
        """
        n = self.config.n_samples if n is None else n
        b = batch.input.shape[0]
        keys = jax.random.split(key, b)

        def _one(input_i, mask_i, key_i):
            ctx = self.encoder(input_i, mask_i)
            return self.head.sample(ctx, key_i, n)

        return jax.vmap(_one)(batch.input, batch.mask, keys)

    def log_prob(
        self,
        x: Float[Array, ...],
        batch: Batch1D | Batch2D,
    ) -> Float[Array, " B"]:
        """Per-sample log-density of ``x`` under ``q_φ(·|y)``.

        For ``ScoreDiffusionHead`` this raises ``NotImplementedError``
        (no closed-form density). Use ``sample`` instead.
        """

        def _one(x_i, input_i, mask_i):
            ctx = self.encoder(input_i, mask_i)
            return self.head.log_prob(x_i, ctx)

        return jax.vmap(_one)(x, batch.input, batch.mask)

    def as_analysis_step(self) -> _AmortizedAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
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
