"""4DVarNet iterative solver.

Implements the gradient-descent-like solver that unrolls fixed iterations of
the variational cost minimisation, guided by the learned gradient modulator.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._types import Batch1D, Batch2D, LSTMState1D, LSTMState2D

GradMode = Literal["unrolled", "implicit", "one_step"]
"""Differentiation strategy for the 4DVarNet solver.

- ``"unrolled"``: backprop through all ``K`` solver steps (O(K) memory).
- ``"implicit"``:  fixed-point / implicit differentiation (O(1) memory,
  requires the solver to have converged to a fixed point).
- ``"one_step"``:  one-step differentiation (Bolte et al., NeurIPS 2023);
  O(1) memory, only the last solver step is differentiated.
"""

# ---------------------------------------------------------------------------
# Solver state containers
# ---------------------------------------------------------------------------


class SolverState1D(NamedTuple):
    """Mutable solver state for 1-D problems.

    Attributes:
        x: Current state estimate of shape ``(B, T, N)``.
        lstm: Current LSTM hidden/cell state for the gradient modulator.
        step: Current iteration index.
    """

    x: Float[Array, "B T N"]
    lstm: LSTMState1D
    step: int


class SolverState2D(NamedTuple):
    """Mutable solver state for 2-D problems.

    Attributes:
        x: Current state estimate of shape ``(B, T, H, W)``.
        lstm: Current LSTM hidden/cell state for the gradient modulator.
        step: Current iteration index.
    """

    x: Float[Array, "B T H W"]
    lstm: LSTMState2D
    step: int


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def init_solver_state_1d(
    batch: Batch1D,
    hidden_dim: int,
) -> SolverState1D:
    """Initialise a 1-D solver state from a batch.

    Args:
        batch: Input batch.  The initial state is set to the masked input
            (zeros where unobserved).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.

    Returns:
        Zero-initialised :class:`SolverState1D`.
    """
    b, _, n = batch.input.shape
    x0 = batch.input * batch.mask
    lstm = LSTMState1D.zeros(b, hidden_dim, n)
    return SolverState1D(x=x0, lstm=lstm, step=0)


def init_solver_state_2d(
    batch: Batch2D,
    hidden_dim: int,
) -> SolverState2D:
    """Initialise a 2-D solver state from a batch.

    Args:
        batch: Input batch.  The initial state is set to the masked input.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.

    Returns:
        Zero-initialised :class:`SolverState2D`.
    """
    b, _, h, w = batch.input.shape
    x0 = batch.input * batch.mask
    lstm = LSTMState2D.zeros(b, hidden_dim, h, w)
    return SolverState2D(x=x0, lstm=lstm, step=0)


# ---------------------------------------------------------------------------
# Single solver step
# ---------------------------------------------------------------------------


def solver_step_1d(
    solver_state: SolverState1D,
    batch: Batch1D,
    prior_fn: Any,
    grad_mod_fn: Any,
    alpha: float = 1.0,
    prior_weight: float = 1.0,
) -> SolverState1D:
    """Perform a single 1-D solver iteration.

    Computes the gradient of the variational cost, then passes it through
    the learned gradient modulator to obtain a state update.

    Args:
        solver_state: Current solver state.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior`` (prior autoencoder forward pass).
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor :math:`\\lambda` for the prior cost term.

    Returns:
        Updated :class:`SolverState1D`.
    """
    x = solver_state.x

    def cost_fn(x_):
        x_prior = prior_fn(x_)
        obs_diff = batch.mask * (x_ - batch.input)
        j_obs = jnp.sum(obs_diff**2)
        j_prior = prior_weight * jnp.sum((x_ - x_prior) ** 2)
        return j_obs + j_prior

    grad = jax.grad(cost_fn)(x)
    update, new_lstm = grad_mod_fn(grad, x, solver_state.lstm)
    x_new = x - alpha * update

    return SolverState1D(x=x_new, lstm=new_lstm, step=solver_state.step + 1)


def solver_step_2d(
    solver_state: SolverState2D,
    batch: Batch2D,
    prior_fn: Any,
    grad_mod_fn: Any,
    alpha: float = 1.0,
    prior_weight: float = 1.0,
) -> SolverState2D:
    """Perform a single 2-D solver iteration.

    Args:
        solver_state: Current solver state.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor :math:`\\lambda` for the prior cost term.

    Returns:
        Updated :class:`SolverState2D`.
    """
    x = solver_state.x

    def cost_fn(x_):
        x_prior = prior_fn(x_)
        obs_diff = batch.mask * (x_ - batch.input)
        j_obs = jnp.sum(obs_diff**2)
        j_prior = prior_weight * jnp.sum((x_ - x_prior) ** 2)
        return j_obs + j_prior

    grad = jax.grad(cost_fn)(x)
    update, new_lstm = grad_mod_fn(grad, x, solver_state.lstm)
    x_new = x - alpha * update

    return SolverState2D(x=x_new, lstm=new_lstm, step=solver_state.step + 1)


# ---------------------------------------------------------------------------
# Full unrolled solver
# ---------------------------------------------------------------------------


def solve_4dvarnet_1d(
    batch: Batch1D,
    prior_fn: Any,
    grad_mod_fn: Any,
    n_steps: int,
    hidden_dim: int,
    alpha: float = 1.0,
) -> Float[Array, "B T N"]:
    """Run the full 1-D 4DVarNet solver for ``n_steps`` iterations.

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Number of gradient-descent steps to unroll.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.

    Returns:
        Final state estimate of shape ``(B, T, N)``.
    """
    state = init_solver_state_1d(batch, hidden_dim)
    for _ in range(n_steps):
        state = solver_step_1d(state, batch, prior_fn, grad_mod_fn, alpha)
    return state.x


def solve_4dvarnet_2d(
    batch: Batch2D,
    prior_fn: Any,
    grad_mod_fn: Any,
    n_steps: int,
    hidden_dim: int,
    alpha: float = 1.0,
) -> Float[Array, "B T H W"]:
    """Run the full 2-D 4DVarNet solver for ``n_steps`` iterations.

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Number of gradient-descent steps to unroll.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.

    Returns:
        Final state estimate of shape ``(B, T, H, W)``.
    """
    state = init_solver_state_2d(batch, hidden_dim)
    for _ in range(n_steps):
        state = solver_step_2d(state, batch, prior_fn, grad_mod_fn, alpha)
    return state.x


# ---------------------------------------------------------------------------
# Fixed-point solver
# ---------------------------------------------------------------------------


def fp_solver_step_1d(
    x: Float[Array, "B T N"],
    batch: Batch1D,
    prior_fn: Any,
) -> Float[Array, "B T N"]:
    """Perform a single 1-D fixed-point projection step.

    Applies the prior projection then re-inserts observations at observed
    locations:

    .. math::

        x \\leftarrow \\varphi(x), \\quad
        x \\leftarrow m \\odot y + (1 - m) \\odot x

    Args:
        x: Current state estimate of shape ``(B, T, N)``.
        batch: Observed data batch containing ``input`` (observations ``y``)
            and ``mask`` (``m``).
        prior_fn: Callable ``x -> x_prior`` (prior autoencoder forward pass).

    Returns:
        Updated state estimate of shape ``(B, T, N)``.
    """
    x_phi = prior_fn(x)
    return batch.mask * batch.input + (1 - batch.mask) * x_phi


def solve_4dvarnet_1d_fixedpoint(
    batch: Batch1D,
    prior_fn: Any,
    n_fp_steps: int,
) -> Float[Array, "B T N"]:
    """Run ``n_fp_steps`` fixed-point projection steps using :func:`jax.lax.scan`.

    Initialises the state from the masked observations, then iterates the
    fixed-point update :func:`fp_solver_step_1d` for ``n_fp_steps`` steps.

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        n_fp_steps: Number of fixed-point iterations.

    Returns:
        Final state estimate of shape ``(B, T, N)``.
    """
    x0 = batch.input * batch.mask

    def scan_fn(carry: Float[Array, "B T N"], _: None) -> tuple:
        x_new = fp_solver_step_1d(carry, batch, prior_fn)
        return x_new, None

    x_final, _ = jax.lax.scan(scan_fn, x0, None, length=n_fp_steps)
    return x_final


# ---------------------------------------------------------------------------
# One-step differentiation solver
# ---------------------------------------------------------------------------


def one_step_solve_4dvarnet_1d(
    batch: Batch1D,
    prior_fn: Any,
    grad_mod_fn: Any,
    n_steps: int,
    hidden_dim: int,
    alpha: float = 1.0,
    prior_weight: float = 1.0,
) -> Float[Array, "B T N"]:
    """Solve 4DVarNet-1D using one-step differentiation (Bolte et al., 2023).

    Runs ``n_steps - 1`` solver iterations with ``jax.lax.stop_gradient``
    applied to the iterate, then performs a single final step through which
    gradients flow.  This gives O(1) memory cost (matching implicit
    differentiation) while being as simple to implement as unrolled backprop.

    Reference:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Total number of solver iterations (warmup = n_steps - 1,
            then 1 differentiable step).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor :math:`\\lambda` for the prior cost term.

    Returns:
        Final state estimate of shape ``(B, T, N)``.
    """
    # --- warmup: run n_steps-1 steps without tracking gradients ---
    state = init_solver_state_1d(batch, hidden_dim)
    warmup_steps = max(n_steps - 1, 0)
    for _ in range(warmup_steps):
        state = solver_step_1d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    # detach the iterate so earlier steps don't contribute to the gradient
    state = SolverState1D(
        x=jax.lax.stop_gradient(state.x),
        lstm=jax.lax.stop_gradient(state.lstm),
        step=state.step,
    )

    # --- one differentiable step ---
    if n_steps >= 1:
        state = solver_step_1d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    return state.x


def one_step_solve_4dvarnet_2d(
    batch: Batch2D,
    prior_fn: Any,
    grad_mod_fn: Any,
    n_steps: int,
    hidden_dim: int,
    alpha: float = 1.0,
    prior_weight: float = 1.0,
) -> Float[Array, "B T H W"]:
    """Solve 4DVarNet-2D using one-step differentiation (Bolte et al., 2023).

    Runs ``n_steps - 1`` solver iterations with ``jax.lax.stop_gradient``
    applied to the iterate, then performs a single final step through which
    gradients flow.  This gives O(1) memory cost (matching implicit
    differentiation) while being as simple to implement as unrolled backprop.

    Reference:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Total number of solver iterations (warmup = n_steps - 1,
            then 1 differentiable step).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor :math:`\\lambda` for the prior cost term.

    Returns:
        Final state estimate of shape ``(B, T, H, W)``.
    """
    # --- warmup: run n_steps-1 steps without tracking gradients ---
    state = init_solver_state_2d(batch, hidden_dim)
    warmup_steps = max(n_steps - 1, 0)
    for _ in range(warmup_steps):
        state = solver_step_2d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    # detach the iterate so earlier steps don't contribute to the gradient
    state = SolverState2D(
        x=jax.lax.stop_gradient(state.x),
        lstm=jax.lax.stop_gradient(state.lstm),
        step=state.step,
    )

    # --- one differentiable step ---
    if n_steps >= 1:
        state = solver_step_2d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    return state.x
