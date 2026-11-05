"""Integration adapters for ``pipekit-train``.

Minimal glue that lets a vardax model and a loss compose into a
``pipekit_train.TrainingLoop``, with ``VardaxReconLoss`` unwrapping ``batch.target``.

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

``pipekit_train.MSE()`` already covers the standard reconstruction loss;
``VardaxReconLoss`` only saves callers from unwrapping the ``Batch*`` container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from vardax._src.errors import MissingInputError

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

        ℒ_MSE(x̂, x) = mean((x̂ − x)²)

    Equivalent to ``pipekit_train.MSE`` but unwraps the vardax ``Batch*.target``
    x: the loop hands the loss a ``(prediction, batch)`` pair with prediction x̂.
    The call returns ``(loss, {"recon_mse": loss})``, the standard
    ``(scalar, aux_dict)`` shape.
    """

    def _apply(
        self,
        predicted: Float[Array, ...],
        batch: Any,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        target = getattr(batch, "target", None)
        if target is None:
            raise MissingInputError(
                "VardaxReconLoss expects a Batch* with a non-None `target`; "
                "got batch with target=None. For operational (unlabelled) "
                "data, evaluate the model directly without a loss adapter."
            )
        # ℒ_MSE = mean((x̂ − x)²)
        loss = jnp.mean((predicted - target) ** 2)
        return loss, {"recon_mse": loss}
