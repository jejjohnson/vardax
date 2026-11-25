"""Rollouts of the forward model f over a window, with optional checkpointing.

Every 4D-Var model steps f through this one helper, so that ``checkpoint=`` and
``forward_adjoint=`` behave the same everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from integrix.adjoint import checkpointed_scan
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vardax._src.errors import ConfigurationError, UnsupportedInputError


def checkpoint_every(checkpoint: bool | int) -> int:
    """Number of steps per rematerialised block; ``0`` means no checkpointing.

    Args:
        checkpoint: ``False`` (store every intermediate), ``True`` (one block per
            step) or an integer k ≥ 1 (one block per k steps).

    Returns:
        The block length k, or ``0`` when checkpointing is off.

    Raises:
        ConfigurationError: If ``checkpoint`` is neither a bool nor an int ≥ 1.

    Examples:
        >>> checkpoint_every(False), checkpoint_every(True), checkpoint_every(5)
        (0, 1, 5)
        >>> checkpoint_every(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: checkpoint must be a bool or an int >= 1, got 0
    """
    # bool is a subclass of int, so test it first
    if isinstance(checkpoint, bool):
        return int(checkpoint)
    if not isinstance(checkpoint, int) or checkpoint < 1:
        raise ConfigurationError(
            f"checkpoint must be a bool or an int >= 1, got {checkpoint}"
        )
    return checkpoint


def resolve_forward(forward: Any, forward_adjoint: Any) -> Any:
    """Return the forward model f to roll out, with ``forward_adjoint`` applied.

    A forward that integrates with ``diffrax`` can expose
    ``with_adjoint(adjoint) -> forward`` (the
    [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint] protocol). A per-step
    forward cannot take a diffrax adjoint, so asking for one is an error rather than a
    silent no-op; use ``checkpoint=`` instead.

    Args:
        forward: The forward model f, a ``ForwardModel``.
        forward_adjoint: A ``diffrax.AbstractAdjoint``, or ``None`` to keep the
            forward's own adjoint.

    Returns:
        ``forward`` itself, or ``forward.with_adjoint(forward_adjoint)``.

    Raises:
        UnsupportedInputError: If ``forward_adjoint`` is set and ``forward`` has no
            ``with_adjoint`` method.

    Examples:
        >>> class PerStep:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> fwd = PerStep()
        >>> resolve_forward(fwd, None) is fwd
        True
        >>> resolve_forward(fwd, "backsolve")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        UnsupportedInputError: forward_adjoint was set but PerStep has no with_adjoint
    """
    if forward_adjoint is None:
        return forward
    with_adjoint = getattr(forward, "with_adjoint", None)
    if with_adjoint is None:
        raise UnsupportedInputError(
            f"forward_adjoint was set but {type(forward).__name__} has no "
            "with_adjoint(adjoint) method, so the adjoint cannot reach its "
            "integrator. Per-step forwards are differentiated by reverse-mode "
            "autodiff; bound their memory with checkpoint=True (or an int) "
            "instead."
        )
    return with_adjoint(forward_adjoint)


def rollout(
    forward: Any,
    initial_state: Float[Array, " N"],
    n_steps: int,
    *,
    forcing: Float[Array, "T N"] | None = None,
    checkpoint: bool | int = False,
) -> Float[Array, "T_plus_1 N"]:
    """Step the forward model f over a window, returning T + 1 states.

        xₜ = f(xₜ₋₁) + ηₜ,   t = 1, …, T,   with ηₜ = 0 when ``forcing`` is ``None``

    With ``checkpoint=False`` it is a single plain ``lax.scan``: reverse mode stores
    every intermediate of every step. The trajectory itself is always returned, since
    the observation cost reads every state; what checkpointing removes is the per-step
    work arrays of the step function (Runge–Kutta stages, nonlinear terms).

    Algorithm:
        1. With ``checkpoint=k``, run the steps through
           ``integrix.adjoint.checkpointed_scan``: ⌊T/k⌋ blocks of k steps, plus one
           shorter tail block when k does not divide T. Each block is a scan over its
           steps wrapped in ``jax.checkpoint``, so reverse mode stores only the state
           entering each block and recomputes the steps inside it.
        2. Prepend x₀ to the stacked states.

    Args:
        forward: Forward model f, a ``ForwardModel`` with ``dt`` and
            ``step(state, dt)``.
        initial_state: Initial state x₀.
        n_steps: Number of steps T.
        forcing: Optional additive model error ηₜ per step (weak 4D-Var).
        checkpoint: ``False``, ``True`` (k = 1) or a block length k ≥ 1.

    Returns:
        The trajectory x₀, x₁, …, x_T.

    Raises:
        ConfigurationError: If ``checkpoint`` is neither a bool nor an int ≥ 1.

    Examples:
        Decay f(x) = x / 2 over three steps, with and without checkpointing:

        >>> import jax.numpy as jnp
        >>> class Halve:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        >>> plain = rollout(Halve(), jnp.ones(2), 3)
        >>> plain.shape
        (4, 2)
        >>> blocked = rollout(Halve(), jnp.ones(2), 3, checkpoint=2)
        >>> bool(jnp.array_equal(plain, blocked))
        True

        Additive forcing, as in weak-constraint 4D-Var:

        >>> forced = rollout(Halve(), jnp.zeros(2), 2, forcing=jnp.ones((2, 2)))
        >>> forced[:, 0].tolist()
        [0.0, 1.0, 1.5]
    """
    dt = forward.dt
    block_length = checkpoint_every(checkpoint)

    def step(
        state: Float[Array, " N"], model_error: Float[Array, " N"] | None
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        # xₜ = f(xₜ₋₁), without ηₜ yet
        next_state = forward.step(state, dt)
        if model_error is not None:
            # xₜ = f(xₜ₋₁) + ηₜ
            next_state = next_state + model_error
        return next_state, next_state

    if block_length == 0:
        _, states = jax.lax.scan(step, initial_state, forcing, length=n_steps)
    else:
        # ⌊T/k⌋ checkpointed blocks of k steps, then a checkpointed tail of T mod k
        _, states = checkpointed_scan(
            step, initial_state, forcing, every=block_length, length=n_steps
        )
    # [x₀; x₁, …, x_T]; jnp, not einx: einx cannot size the T = 0 axis
    return jnp.concatenate([jnp.expand_dims(initial_state, 0), states])


def accepts_mask(obs_op: Any) -> bool:
    """Whether ``obs_op.__call__`` takes a ``mask`` keyword.

    Args:
        obs_op: The observation operator h.

    Returns:
        ``True`` when h can be called with ``mask=``, ``False`` otherwise, including
        for a callable whose signature cannot be read.

    Examples:
        ``MaskedIdentity`` takes the mask, and a plain callable does not:

        >>> import vardax
        >>> accepts_mask(vardax.MaskedIdentity())
        True
        >>> class Plain:
        ...     def __call__(self, x):
        ...         return x
        >>> accepts_mask(Plain())
        False
    """
    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


def observed_trajectory(
    model: Any, mask: Float[Array, "T_plus_1 M"]
) -> Callable[[Float[Array, " N"]], Float[Array, "T_plus_1 M"]]:
    """The masked predicted observations of a strong-constraint 4D-Var window.

        ŷₜ(x₀) = mₜ ⊙ h(xₜ),   xₜ = f(xₜ₋₁),   t = 0, …, T

    The result is differentiable, so ``jax.linearize`` and ``jax.vjp`` apply its
    Jacobian and adjoint without forming a matrix.

    Args:
        model: A model with ``obs_op`` and ``_rollout(initial_state, n_steps)``.
        mask: Observation mask mₜ; its length sets the window, T + 1.

    Returns:
        The map x₀ ↦ ŷ(x₀), stacked over time.

    Examples:
        With f(x) = x / 2, the second state is halved and the mask zeroes one entry:

        >>> import jax, jax.numpy as jnp, lineax as lx, vardax
        >>> class Halve:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return 0.5 * x
        >>> eye = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> model = vardax.StrongFourDVar(
        ...     forward=Halve(),
        ...     obs_op=vardax.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(2),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> mask = jnp.array([[1.0, 1.0], [1.0, 0.0]])
        >>> observed = observed_trajectory(model, mask)
        >>> observed(jnp.ones(2)).tolist()
        [[1.0, 1.0], [0.5, 0.0]]

        A one-state window has no rollout steps:

        >>> observed_trajectory(model, jnp.ones((1, 2)))(jnp.ones(2)).shape
        (1, 2)
    """
    # the window holds T + 1 times, so T steps
    num_steps = mask.shape[0] - 1
    obs_op = model.obs_op

    def observe(state: Float[Array, " N"], step_mask: Float[Array, " M"]):
        # h(xₜ)
        if accepts_mask(obs_op):
            predicted = obs_op(state, mask=step_mask)
        else:
            predicted = obs_op(state)
        # ŷₜ = mₜ ⊙ h(xₜ)
        return step_mask * predicted

    def observed(initial_state: Float[Array, " N"]) -> Float[Array, "T_plus_1 M"]:
        # xₜ = f(xₜ₋₁)
        trajectory = model._rollout(initial_state, num_steps)
        return jax.vmap(observe)(trajectory, mask)

    return observed
