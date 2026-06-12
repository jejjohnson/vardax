"""Map pipekit adjoint specs onto optimistix adjoints.

The shared vocabulary lives in ``pipekit_cycle.adjoints`` (frozen
dataclasses named after their diffrax counterparts); vardax's inner
solvers consume ``optimistix.AbstractAdjoint`` instances. This module
is the inner-solve-layer interpreter — the counterpart of
``pipekit_jax.to_diffrax_adjoint`` at the dynamics layer.

Specs are matched structurally by class name, so the identically-named
classes from other libraries are accepted interchangeably.
"""

from __future__ import annotations

from typing import Any

import optimistix as optx

from .k_step import KStepAdjoint


def to_optimistix_adjoint(spec: Any) -> optx.AbstractAdjoint:
    """Map an adjoint spec onto an ``optimistix.AbstractAdjoint``.

    Mapping:

    - ``ImplicitAdjoint()`` → ``optx.ImplicitAdjoint()`` — exact at a
      converged fixed point, O(1) memory, one Hessian linear solve.
    - ``RecursiveCheckpointAdjoint(checkpoints)`` →
      ``optx.RecursiveCheckpointAdjoint(checkpoints)`` — exact,
      recomputing.
    - ``TruncatedAdjoint(k)`` →
      [`KStepAdjoint(k)`][vardax.adjoints.KStepAdjoint] — warmup under
      ``stop_gradient``, then ``k`` differentiable steps (``k=1`` is
      [`OneStepAdjoint`][vardax.adjoints.OneStepAdjoint]).
    - ``DirectAdjoint`` / ``BacksolveAdjoint`` → ``ValueError``: the
      first is the plain unrolled default (pass
      ``optx.RecursiveCheckpointAdjoint()`` or nothing), the second
      only exists at the dynamics layer.

    Args:
        spec: A spec from ``pipekit_cycle.adjoints`` or any
            structurally identical object. A ready-made
            ``optx.AbstractAdjoint`` passes through unchanged.

    Returns:
        The corresponding optimistix adjoint instance.

    Raises:
        ValueError: for layer-inappropriate or unrecognised specs.

    Examples:
        >>> import optimistix as optx
        >>> from pipekit_cycle.adjoints import TruncatedAdjoint
        >>> from vardax.adjoints import to_optimistix_adjoint
        >>> to_optimistix_adjoint(TruncatedAdjoint(k=3))
        KStepAdjoint(k=3)
        >>> isinstance(
        ...     to_optimistix_adjoint(optx.ImplicitAdjoint()), optx.ImplicitAdjoint
        ... )
        True
    """
    if isinstance(spec, optx.AbstractAdjoint):
        return spec
    name = type(spec).__name__
    if name == "ImplicitAdjoint":
        return optx.ImplicitAdjoint()
    if name == "RecursiveCheckpointAdjoint":
        return optx.RecursiveCheckpointAdjoint(
            checkpoints=getattr(spec, "checkpoints", None)
        )
    if name == "TruncatedAdjoint":
        return KStepAdjoint(k=getattr(spec, "k", 1))
    if name in ("DirectAdjoint", "BacksolveAdjoint"):
        raise ValueError(
            f"{name} does not apply at the inner-solve layer: DirectAdjoint is "
            "the plain unrolled default (use optx.RecursiveCheckpointAdjoint() "
            "or omit solver_adjoint), and BacksolveAdjoint only exists at the "
            "dynamics layer (see pipekit_jax.DiffraxForwardModel)."
        )
    raise ValueError(f"Unrecognised adjoint spec: {spec!r}")
