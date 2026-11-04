r"""Forward rollouts for the 4DVar models, with optional checkpointing.

One helper steps a ``ForwardModel`` over a window for every 4DVar model,
so that memory control (``checkpoint=``) and diffrax adjoints
(``forward_adjoint=``) behave the same everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vardax._src.errors import ConfigurationError, UnsupportedInputError


def checkpoint_every(checkpoint: bool | int) -> int:
    """Number of steps per rematerialised block; ``0`` means no checkpointing.

    Args:
        checkpoint: ``False`` (store every intermediate), ``True`` (one
            block per step) or an integer ``k ≥ 1`` (one block per ``k``
            steps).

    Returns:
        The block length ``k``, or ``0`` when checkpointing is off.

    Raises:
        ValueError: If ``checkpoint`` is an integer below 1.

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
    """Return the forward to roll out, with ``forward_adjoint`` applied.

    A forward that integrates with ``diffrax`` can expose
    ``with_adjoint(adjoint) -> forward`` (the
    [`SupportsForwardAdjoint`][vardax.SupportsForwardAdjoint] protocol).
    A per-step forward cannot take a diffrax adjoint, so asking for one
    is an error rather than a silent no-op; use ``checkpoint=`` instead.

    Args:
        forward: The model's ``ForwardModel``.
        forward_adjoint: A ``diffrax.AbstractAdjoint``, or ``None`` to keep
            the forward's own adjoint.

    Returns:
        ``forward`` itself, or ``forward.with_adjoint(forward_adjoint)``.

    Raises:
        ValueError: If ``forward_adjoint`` is set and ``forward`` has no
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
    r"""Step a forward model over a window, returning ``T + 1`` states.

    xₜ₊₁ = M(xₜ) + ηₜ,  t = 0 … T − 1,  with ηₜ = 0 when ``forcing`` is None.

    Algorithm:
        1. Group the ``T`` steps into blocks of ``k`` (``checkpoint=k``),
           plus one shorter tail block when ``k`` does not divide ``T``.
        2. Scan over blocks; each block is a scan over its steps wrapped
           in ``jax.checkpoint``, so reverse mode stores only the state
           entering each block and recomputes the steps inside it.
        3. Prepend x₀ to the stacked states.

    With ``checkpoint=False`` it is a single plain ``lax.scan``: reverse
    mode stores every intermediate of every step. The trajectory itself
    is always returned, since the observation cost reads every state;
    what checkpointing removes is the per-step work arrays of the step
    function (Runge–Kutta stages, nonlinear terms).

    Args:
        forward: ``ForwardModel`` with ``dt`` and ``step(state, dt)``.
        initial_state: Initial state x₀.
        n_steps: Number of steps T.
        forcing: Optional additive model error ηₜ per step (weak 4DVar).
        checkpoint: ``False``, ``True`` (k = 1) or a block length k ≥ 1.

    Returns:
        The trajectory ``[x₀, x₁, …, x_T]``.

    Examples:
        Decay xₜ₊₁ = 0.5 xₜ over three steps, with and without
        checkpointing:

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

        Additive forcing, as in weak-constraint 4DVar:

        >>> forced = rollout(Halve(), jnp.zeros(2), 2, forcing=jnp.ones((2, 2)))
        >>> forced[:, 0].tolist()
        [0.0, 1.0, 1.5]
    """
    dt = forward.dt
    block_length = checkpoint_every(checkpoint)

    def step(
        state: Float[Array, " N"], model_error: Float[Array, " N"] | None
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        # xₜ₊₁ = M(xₜ) (+ ηₜ)
        next_state = forward.step(state, dt)
        if model_error is not None:
            next_state = next_state + model_error
        return next_state, next_state

    def scan_steps(
        state: Float[Array, " N"],
        block_forcing: Float[Array, "k N"] | None,
        length: int,
    ) -> tuple[Float[Array, " N"], Float[Array, "k N"]]:
        return jax.lax.scan(step, state, block_forcing, length=length)

    if block_length == 0:
        _, states = scan_steps(initial_state, forcing, n_steps)
        # [x₀; x₁, …, x_T]; jnp, not einx: einx cannot size the T = 0 axis
        return jnp.concatenate([jnp.expand_dims(initial_state, 0), states])

    n_blocks, tail_length = divmod(n_steps, block_length)
    checkpointed_block = jax.checkpoint(scan_steps, static_argnums=(2,))

    # --- full blocks ---
    head_forcing = None
    if forcing is not None:
        head = forcing[: n_blocks * block_length]
        # jnp, not einx: n_blocks is 0 when k exceeds T
        head_forcing = head.reshape(n_blocks, block_length, *forcing.shape[1:])

    def scan_block(
        state: Float[Array, " N"], block_forcing: Float[Array, "k N"] | None
    ) -> tuple[Float[Array, " N"], Float[Array, "k N"]]:
        return checkpointed_block(state, block_forcing, block_length)

    state, blocked_states = jax.lax.scan(
        scan_block, initial_state, head_forcing, length=n_blocks
    )
    # jnp, not einx: n_blocks is 0 when k exceeds T
    head_states = blocked_states.reshape(n_blocks * block_length, *state.shape)

    # --- tail block (k does not divide T) ---
    pieces = [jnp.expand_dims(initial_state, 0), head_states]
    if tail_length > 0:
        tail_forcing = None if forcing is None else forcing[n_blocks * block_length :]
        _, tail_states = checkpointed_block(state, tail_forcing, tail_length)
        pieces.append(tail_states)
    return jnp.concatenate(pieces)


def accepts_mask(obs_op: Any) -> bool:
    """Whether ``obs_op.__call__`` takes a ``mask`` keyword."""
    try:
        return "mask" in inspect.signature(obs_op.__call__).parameters
    except (TypeError, ValueError):
        return False


def observed_trajectory(
    model: Any, mask: Float[Array, "T_plus_1 M"]
) -> Callable[[Float[Array, " N"]], Float[Array, "T_plus_1 M"]]:
    """The window's observation map of a strong-constraint 4DVar model.

    G(x₀) = [mₜ ⊙ Hₜ(Mₜ(x₀))]ₜ₌₀…T, the masked predicted observations,
    stacked over time. Its tangent G' and adjoint G'ᵀ (via ``jax.linearize``
    and ``jax.vjp``) apply ∑ₜ H'ₜ M'ₜ without forming a Jacobian.

    Args:
        model: A model with ``obs_op`` and ``_rollout(initial_state, n_steps)``.
        mask: Observation mask mₜ; its length sets the window, T + 1.

    Returns:
        The map x₀ ↦ G(x₀).
    """
    n_steps = mask.shape[0] - 1
    obs_op = model.obs_op

    def observe(state: Float[Array, " N"], step_mask: Float[Array, " M"]):
        # mₜ ⊙ Hₜ(xₜ)
        if accepts_mask(obs_op):
            predicted = obs_op(state, mask=step_mask)
        else:
            predicted = obs_op(state)
        return step_mask * predicted

    def observed(initial_state: Float[Array, " N"]) -> Float[Array, "T_plus_1 M"]:
        trajectory = model._rollout(initial_state, n_steps)
        return jax.vmap(observe)(trajectory, mask)

    return observed
