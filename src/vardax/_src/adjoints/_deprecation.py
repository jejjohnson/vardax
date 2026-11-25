"""One deprecation warning for the adjoint aliases, attributed to the caller's line."""

from __future__ import annotations

from vardax._src.utils.deprecation import warn_deprecated


def warn_renamed(old: str, new: str) -> None:
    """Warn that ``vardax.<old>`` is now ``argminax.adjoint.<new>``.

    Args:
        old: The deprecated vardax name.
        new: Its replacement in ``argminax.adjoint``.
    """
    warn_deprecated(
        f"vardax.{old} is deprecated and will be removed in the next release; "
        f"use argminax.adjoint.{new}, which works under any optimistix solve."
    )
