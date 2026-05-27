"""Convenience factories for ``pipekit_cycle.DACycle`` / ``SmootherCycle``.

Per Decision D8, vardax models satisfy ``pipekit_cycle.AnalysisStep`` via
``.as_analysis_step()``. The factories here are pure glue: they take a
vardax model (any of the seven Layer 2 classes) plus a forward and obs
operator, and return a configured pipekit-cycle object ready to call.

Examples:

    >>> import vardax as vdx
    >>> cycle = vdx.cycle.VarDACycle(
    ...     forward=somax_model,
    ...     obs_op=vdx.obs_operators.MaskedIdentity(),
    ...     model=vdx.IncrementalFourDVar(...),
    ...     n_steps=24,
    ... )
    >>> analysis, state = cycle(initial_state, vdx.cycle.DAState())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import pipekit_cycle as _pc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vardax.cycle requires pipekit-cycle, which is a required dependency. "
        "Reinstall vardax to pick it up."
    ) from exc

if TYPE_CHECKING:
    from pipekit_cycle import DACycle, SmootherCycle


def VarDACycle(
    forward: Any,
    obs_op: Any,
    model: Any,
    *,
    obs_source: Any | None = None,
    n_steps: int = 1,
    save_history: bool = True,
) -> DACycle:
    """Build a ``pipekit_cycle.DACycle`` from a vardax model.

    Pulls ``model.as_analysis_step()`` and hands it to pipekit-cycle.
    The same orchestration code works for any of the seven Layer 2
    classes — `OptimalInterpolation`, `ThreeDVar`, `StrongFourDVar`,
    `WeakFourDVar`, `IncrementalFourDVar`, `FourDVarNet`,
    `AmortizedPosterior`.

    Args:
        forward: Forward model satisfying ``pipekit_cycle.ForwardModel``.
        obs_op: Observation operator satisfying
            ``pipekit_cycle.ObservationOperator``.
        model: Vardax Layer 2 model exposing ``.as_analysis_step()``.
        obs_source: Optional ``pipekit.Operator`` invoked as
            ``obs_source(step_index)`` to load observations per cycle.
            If ``None``, the analysis step is skipped and the forecast
            propagates forward.
        n_steps: Number of forecast-analysis cycles per call.
        save_history: Append ``(forecast, analysis)`` pairs to
            ``cycle.history`` each cycle.

    Returns:
        Configured ``pipekit_cycle.DACycle`` ready to call.

    Raises:
        AttributeError: If ``model`` does not expose
            ``.as_analysis_step()``.
    """
    if not hasattr(model, "as_analysis_step"):
        raise AttributeError(
            "VarDACycle requires a model exposing `.as_analysis_step()`; "
            f"got {type(model).__name__}."
        )
    return _pc.DACycle(
        forward_model=forward,
        obs_op=obs_op,
        analysis_step=model.as_analysis_step(),
        obs_source=obs_source,
        n_steps=n_steps,
        save_history=save_history,
    )


def VarSmootherCycle(
    forward: Any,
    obs_op: Any,
    model: Any,
    *,
    window: int,
    stride: int = 1,
    obs_source: Any | None = None,
) -> SmootherCycle:
    """Build a ``pipekit_cycle.SmootherCycle`` from a vardax model.

    Retrospective windowed smoothing: the model analyses each
    ``window``-step window, sliding by ``stride`` between consecutive
    windows.

    Args:
        forward: Forward model.
        obs_op: Observation operator.
        model: Vardax Layer 2 model exposing ``.as_analysis_step()``.
        window: Number of forecast steps per smoother window.
        stride: Step between consecutive window starts.
        obs_source: Optional observation loader.

    Returns:
        Configured ``pipekit_cycle.SmootherCycle``.
    """
    if not hasattr(model, "as_analysis_step"):
        raise AttributeError(
            "VarSmootherCycle requires a model exposing `.as_analysis_step()`; "
            f"got {type(model).__name__}."
        )
    return _pc.SmootherCycle(
        forward_model=forward,
        obs_op=obs_op,
        analysis_step=model.as_analysis_step(),
        window=window,
        stride=stride,
        obs_source=obs_source,
    )
