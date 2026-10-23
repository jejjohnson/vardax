"""Public ``vardax.examples`` API surface — demo helpers and reference residuals.

Re-exports from ``vardax._src.examples`` (the notebook training loop, not
part of the library API proper; see that module's docstring) and from
``vardax._src.utils.residuals`` (reference PDE residuals for
[`ResidualPenalty`][vardax.ResidualPenalty]).
"""

from vardax._src.examples import fit_demo
from vardax._src.utils.residuals import (
    advection_diffusion_2d,
    navier_stokes_vorticity_2d,
)

__all__ = ["advection_diffusion_2d", "fit_demo", "navier_stokes_vorticity_2d"]
