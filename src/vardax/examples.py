"""Public ``vardax.examples`` API surface — notebook-demo helpers.

Re-exports from ``vardax._src.examples``. These helpers exist for the
tutorial notebooks under ``docs/notebooks/`` and are not part of the
library API proper; see that module's docstring.
"""

from vardax._src.examples import fit_demo

__all__ = ["fit_demo"]
