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

from vardax._src.errors import UnsupportedInputError

from .k_step import KStepAdjoint


def to_optimistix_adjoint(spec: Any) -> optx.AbstractAdjoint:
    """Map an adjoint spec onto an ``optimistix.AbstractAdjoint``.

    Algorithm:
        1. Return ``spec`` unchanged if it already is an
           ``optimistix.AbstractAdjoint``.
        2. Otherwise match it by class name against the mapping below.
        3. Raise for names that have no inner-solve counterpart.

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
    - ``DirectAdjoint`` / ``BacksolveAdjoint`` →
      [`UnsupportedInputError`][vardax.UnsupportedInputError]: the
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
        UnsupportedInputError: For layer-inappropriate or unrecognised specs.

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

        A pipekit ``TruncatedAdjoint(k=1)`` is the one-step adjoint.

        >>> import vardax
        >>> adjoint = to_optimistix_adjoint(TruncatedAdjoint(k=1))
        >>> isinstance(adjoint, vardax.OneStepAdjoint)
        False
        >>> isinstance(adjoint, vardax.KStepAdjoint)
        True

        A spec with no inner-solve meaning is rejected.

        >>> from pipekit_cycle.adjoints import DirectAdjoint
        >>> to_optimistix_adjoint(DirectAdjoint())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        UnsupportedInputError: DirectAdjoint does not apply at the inner-solve layer
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
        raise UnsupportedInputError(
            f"{name} does not apply at the inner-solve layer: DirectAdjoint is "
            "the plain unrolled default (use optx.RecursiveCheckpointAdjoint() "
            "or omit solver_adjoint), and BacksolveAdjoint only exists at the "
            "dynamics layer (see pipekit_jax.DiffraxForwardModel)."
        )
    raise UnsupportedInputError(f"Unrecognised adjoint spec: {spec!r}")
