"""Configuration of an amortized posterior: the head family and sample count."""

from __future__ import annotations

from typing import Any

import equinox as eqx

from vardax._src.errors import ConfigurationError


class AmortizedConfig(eqx.Module):
    """Configuration of an ``AmortizedPosterior`` q(x | y; ν).

    Attributes:
        head_type: Density family of the head, one of ``"regression"``,
            ``"flow"`` or ``"score"``. Only ``"regression"`` is fully
            implemented: ``"flow"`` requires ``gauss_flows`` and ``"score"``
            is pending the diffrax reverse-SDE pipeline. The head instance is
            responsible for matching the type.
        n_samples: Number of posterior samples N_E that
            ``AmortizedPosterior.sample()`` draws when ``n`` is not provided.

    Raises:
        ConfigurationError: If ``head_type`` is not one of the three values.

    Examples:
        The defaults give a regression head and 64 samples:

        >>> import vardax as vdx
        >>> config = vdx.AmortizedConfig()
        >>> config.head_type, config.n_samples
        ('regression', 64)

        Both fields can be set:

        >>> vdx.AmortizedConfig(head_type="flow", n_samples=10).n_samples
        10

        An unknown head type is rejected:

        >>> vdx.AmortizedConfig(head_type="bogus")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: AmortizedConfig.head_type must be one of ...
    """

    head_type: str = eqx.field(static=True, default="regression")
    n_samples: int = eqx.field(static=True, default=64)

    def __post_init__(self) -> None:
        # Reject an unknown family at construction, before any head is built.
        valid = {"regression", "flow", "score"}
        if self.head_type not in valid:
            raise ConfigurationError(
                f"AmortizedConfig.head_type must be one of {sorted(valid)}; "
                f"got {self.head_type!r}."
            )

    def get_config(self) -> dict[str, Any]:
        """Return the fields as a plain dictionary."""
        return {
            "head_type": self.head_type,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AmortizedConfig:
        """Rebuild a configuration from ``get_config()`` output."""
        return cls(**config)
