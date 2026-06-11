"""Integration adapters for ``pipekit-train``.

Provides minimal glue so a vardax model + a ``pipekit_train.Loss``
implementation compose into a ``pipekit_train.TrainingLoop`` without
the user reimplementing the inner training step.

Typical usage:

```python
from pipekit_jax import JaxModelOp
from pipekit_train import MSE, EarlyStopping, Checkpoint, TrainingLoop
from vardax import FourDVarNet1D
from vardax.adapters.pipekit_train import VardaxReconLoss

model = FourDVarNet1D(state_dim=N, n_time=T, ..., key=key)
model_op = JaxModelOp(model)

loop = TrainingLoop(
    dataset=my_dataset,
    model_op=model_op,
    loss=VardaxReconLoss(),  # pipekit_train.MSE on (model(batch), batch.target)
    callbacks=[EarlyStopping(patience=10), Checkpoint(registry, every_n=100)],
)
trained_op, state = loop(model_op, TrainerCarryState(...))
```

The adapter is intentionally thin — ``pipekit_train.MSE()`` already
works for the standard reconstruction-loss case. ``VardaxReconLoss``
exists as a convenience that pulls ``batch.target`` out of the vardax
``Batch*`` container automatically, so callers don't have to unwrap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, Float

try:
    from pipekit import Operator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vardax.adapters.pipekit_train requires the 'train' extra: "
        "`uv add 'vardax[train]'` or `uv sync --group train`."
    ) from exc


class VardaxReconLoss(Operator):
    """Reconstruction-loss adapter for ``pipekit_train.TrainingLoop``.

    Equivalent to ``pipekit_train.MSE`` but unwraps the vardax
    ``Batch*.target`` automatically: the loop hands the loss a
    ``(prediction, batch)`` pair, and the adapter computes
    ``mean((prediction - batch.target) ** 2)``.

    Returns ``(loss, {"recon_mse": loss})`` so the training loop sees the
    standard ``(scalar, aux_dict)`` shape.
    """

    def _apply(
        self,
        predicted: Float[Array, ...],
        batch: Any,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        target = getattr(batch, "target", None)
        if target is None:
            raise ValueError(
                "VardaxReconLoss expects a Batch* with a non-None `target`; "
                "got batch with target=None. For operational (unlabelled) "
                "data, evaluate the model directly without a loss adapter."
            )
        loss = jnp.mean((predicted - target) ** 2)
        return loss, {"recon_mse": loss}
