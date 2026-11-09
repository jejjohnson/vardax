"""One deprecation warning for the adjoint aliases, attributed to the caller's line."""

from __future__ import annotations

from pathlib import Path
import warnings

import equinox as eqx
import jaxtyping

import vardax

# Frames inside equinox's constructor machinery, jaxtyping's runtime checks and
# vardax itself are skipped, so the warning names the line that built the alias
# and is shown once per line.
_SKIPPED = tuple(
    str(Path(module.__file__).parent) for module in (eqx, jaxtyping, vardax)
)


def warn_renamed(old: str, new: str) -> None:
    """Warn that ``vardax.<old>`` is now ``argminax.adjoint.<new>``.

    Args:
        old: The deprecated vardax name.
        new: Its replacement in ``argminax.adjoint``.
    """
    warnings.warn(
        f"vardax.{old} is deprecated and will be removed in the next release; "
        f"use argminax.adjoint.{new}, which works under any optimistix solve.",
        DeprecationWarning,
        skip_file_prefixes=_SKIPPED,
    )
