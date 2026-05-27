"""Configuration for ``AmortizedPosterior``."""

from __future__ import annotations

import equinox as eqx


class AmortizedConfig(eqx.Module):
    """Configuration for an ``AmortizedPosterior``.

    Attributes:
        head_type: One of ``"regression"``, ``"flow"``, ``"score"``.
            Determines the density family used by the head. Currently
            only ``"regression"`` is fully implemented;  ``"flow"``
            requires ``gauss_flows`` and ``"score"`` is pending the
            diffrax reverse-SDE pipeline. Set on construction; the head
            instance is responsible for matching the type.
        n_samples: Default number of posterior samples to draw in
            ``AmortizedPosterior.sample()`` when ``n`` is not provided.
    """

    head_type: str = eqx.field(static=True, default="regression")
    n_samples: int = eqx.field(static=True, default=64)

    def __post_init__(self) -> None:
        valid = {"regression", "flow", "score"}
        if self.head_type not in valid:
            raise ValueError(
                f"AmortizedConfig.head_type must be one of {sorted(valid)}; "
                f"got {self.head_type!r}."
            )
