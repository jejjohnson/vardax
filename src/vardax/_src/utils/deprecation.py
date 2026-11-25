"""One deprecation warning, attributed to the caller's line rather than vardax's."""

from __future__ import annotations

from pathlib import Path
import warnings

import equinox as eqx
import jaxtyping

import vardax

# Frames inside equinox's constructor machinery, jaxtyping's runtime checks and
# vardax itself are skipped, so the warning names the line that called vardax
# and is shown once per line.
_SKIPPED = tuple(
    str(Path(module.__file__).parent) for module in (eqx, jaxtyping, vardax)
)


def warn_deprecated(message: str) -> None:
    """Emit a ``DeprecationWarning`` at the first frame outside vardax.

    Args:
        message: What is deprecated, and what replaces it.
    """
    warnings.warn(message, DeprecationWarning, skip_file_prefixes=_SKIPPED)
