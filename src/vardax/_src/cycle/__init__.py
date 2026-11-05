"""pipekit-cycle integration.

Thin convenience constructors that wire a vardax model into
``pipekit_cycle.DACycle`` / ``pipekit_cycle.SmootherCycle``. Vardax does
not reimplement cycle orchestration; these factories just pull the
``.as_analysis_step()`` adapter off the model and hand it to pipekit.
"""

from .da_cycle import VarDACycle, VarSmootherCycle

__all__ = ["VarDACycle", "VarSmootherCycle"]
