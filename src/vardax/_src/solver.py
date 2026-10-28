"""4DVarNet iterative solver.

Implements the gradient-descent-like solver that unrolls fixed iterations of
the variational cost minimisation, guided by the learned gradient modulator.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._types import Batch1D, Batch2D, LSTMState1D, LSTMState2D

# v0.4 removes the `GradMode` enum (`"unrolled" | "implicit" | "one_step"`).
# Differentiation strategy is now selected via ``solver_adjoint`` on the
# model — an ``optimistix.AbstractAdjoint`` instance. See
# ``vardax.adjoints`` (Decision D15).

# ---------------------------------------------------------------------------
# Solver state containers
# ---------------------------------------------------------------------------


class SolverState1D(eqx.Module):
    """Mutable solver state for 1-D problems.

    Attributes:
        x: Current state estimate of shape ``(B, T, N)``.
        lstm: Current LSTM hidden/cell state for the gradient modulator.
        step: Current iteration index.
    """

    x: Float[Array, "B T N"]
    lstm: LSTMState1D
    step: int


class SolverState2D(eqx.Module):
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
        Zero-initialised [`SolverState1D`][vardax.SolverState1D].
    """
    batch_size, _, n_space = batch.input.shape
    # u₀ = m ⊙ y
    initial_state = batch.input * batch.mask
    lstm = LSTMState1D.zeros(batch_size, hidden_dim, n_space)
    return SolverState1D(x=initial_state, lstm=lstm, step=0)


def init_solver_state_2d(
    batch: Batch2D,
    hidden_dim: int,
) -> SolverState2D:
    """Initialise a 2-D solver state from a batch.

    Args:
        batch: Input batch.  The initial state is set to the masked input.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.

    Returns:
        Zero-initialised [`SolverState2D`][vardax.SolverState2D].
    """
    batch_size, _, height, width = batch.input.shape
    # u₀ = m ⊙ y
    initial_state = batch.input * batch.mask
    lstm = LSTMState2D.zeros(batch_size, hidden_dim, height, width)
    return SolverState2D(x=initial_state, lstm=lstm, step=0)


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
    r"""Perform a single 1-D solver iteration.

    Computes the gradient of the variational cost, then passes it through
    the learned gradient modulator to obtain a state update.

    Args:
        solver_state: Current solver state.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior`` (prior autoencoder forward pass).
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.

    Returns:
        Updated [`SolverState1D`][vardax.SolverState1D].
    """
    state = solver_state.x

    def cost_fn(current_state):
        prior_state = prior_fn(current_state)
        obs_diff = batch.mask * (current_state - batch.input)
        # J_obs = ‖m ⊙ (u − y)‖²
        obs_cost = jnp.sum(obs_diff**2)
        # J_prior = λ ‖u − φ(u)‖²
        prior_cost = prior_weight * jnp.sum((current_state - prior_state) ** 2)
        return obs_cost + prior_cost

    grad = jax.grad(cost_fn)(state)
    update, new_lstm = grad_mod_fn(grad, state, solver_state.lstm)
    # u ← u − α · update
    new_state = state - alpha * update

    return SolverState1D(x=new_state, lstm=new_lstm, step=solver_state.step + 1)


def solver_step_2d(
    solver_state: SolverState2D,
    batch: Batch2D,
    prior_fn: Any,
    grad_mod_fn: Any,
    alpha: float = 1.0,
    prior_weight: float = 1.0,
) -> SolverState2D:
    r"""Perform a single 2-D solver iteration.

    Args:
        solver_state: Current solver state.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.

    Returns:
        Updated [`SolverState2D`][vardax.SolverState2D].
    """
    state = solver_state.x

    def cost_fn(current_state):
        prior_state = prior_fn(current_state)
        obs_diff = batch.mask * (current_state - batch.input)
        # J_obs = ‖m ⊙ (u − y)‖²
        obs_cost = jnp.sum(obs_diff**2)
        # J_prior = λ ‖u − φ(u)‖²
        prior_cost = prior_weight * jnp.sum((current_state - prior_state) ** 2)
        return obs_cost + prior_cost

    grad = jax.grad(cost_fn)(state)
    update, new_lstm = grad_mod_fn(grad, state, solver_state.lstm)
    # u ← u − α · update
    new_state = state - alpha * update

    return SolverState2D(x=new_state, lstm=new_lstm, step=solver_state.step + 1)


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
    r"""Perform a single 1-D fixed-point projection step.

    Applies the prior projection then re-inserts observations at observed
    locations:

    $$
    x \leftarrow \varphi(x), \quad
    x \leftarrow m \odot y + (1 - m) \odot x
    $$

    Args:
        x: Current state estimate of shape ``(B, T, N)``.
        batch: Observed data batch containing ``input`` (observations ``y``)
            and ``mask`` (``m``).
        prior_fn: Callable ``x -> x_prior`` (prior autoencoder forward pass).

    Returns:
        Updated state estimate of shape ``(B, T, N)``.
    """
    prior_state = prior_fn(x)
    # u ← m ⊙ y + (1 − m) ⊙ φ(u)
    return batch.mask * batch.input + (1 - batch.mask) * prior_state


def solve_4dvarnet_1d_fixedpoint(
    batch: Batch1D,
    prior_fn: Any,
    n_fp_steps: int,
) -> Float[Array, "B T N"]:
    """Run ``n_fp_steps`` fixed-point projection steps using ``jax.lax.scan``.

    Initialises the state from the masked observations, then iterates the
    fixed-point update [`fp_solver_step_1d`][vardax.fp_solver_step_1d] for
    ``n_fp_steps`` steps.

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        n_fp_steps: Number of fixed-point iterations.

    Returns:
        Final state estimate of shape ``(B, T, N)``.
    """
    # u₀ = m ⊙ y
    initial_state = batch.input * batch.mask

    def scan_fn(carry: Float[Array, "B T N"], _: None) -> tuple:
        new_state = fp_solver_step_1d(carry, batch, prior_fn)
        return new_state, None

    final_state, _ = jax.lax.scan(scan_fn, initial_state, None, length=n_fp_steps)
    return final_state


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
    k: int = 1,
) -> Float[Array, "B T N"]:
    r"""Solve 4DVarNet-1D using k-step differentiation (Bolte et al., 2023).

    Runs ``n_steps - k`` solver iterations with ``jax.lax.stop_gradient``
    applied to the iterate, then performs ``k`` final steps through which
    gradients flow.  This gives O(k) memory cost (k=1 matches implicit
    differentiation) while being as simple to implement as unrolled backprop.

    Reference:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Total number of solver iterations (warmup = n_steps - k,
            then k differentiable steps).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.
        k: Number of trailing differentiable steps (clipped to n_steps).

    Returns:
        Final state estimate of shape ``(B, T, N)``.
    """
    # --- warmup: run n_steps-k steps without tracking gradients ---
    state = init_solver_state_1d(batch, hidden_dim)
    live_steps = min(max(k, 1), n_steps) if n_steps >= 1 else 0
    warmup_steps = max(n_steps - live_steps, 0)
    for _ in range(warmup_steps):
        state = solver_step_1d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    # detach the iterate so earlier steps don't contribute to the
    # gradient — but only when there was a warmup to detach: with
    # k >= n_steps the solve is fully differentiable, including
    # gradients with respect to the batch through the initial state.
    if warmup_steps > 0:
        state = SolverState1D(
            x=jax.lax.stop_gradient(state.x),
            lstm=jax.lax.stop_gradient(state.lstm),
            step=state.step,
        )

    # --- k differentiable steps ---
    for _ in range(live_steps):
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
    k: int = 1,
) -> Float[Array, "B T H W"]:
    r"""Solve 4DVarNet-2D using k-step differentiation (Bolte et al., 2023).

    Runs ``n_steps - k`` solver iterations with ``jax.lax.stop_gradient``
    applied to the iterate, then performs ``k`` final steps through which
    gradients flow.  This gives O(k) memory cost (k=1 matches implicit
    differentiation) while being as simple to implement as unrolled backprop.

    Reference:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Args:
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Total number of solver iterations (warmup = n_steps - k,
            then k differentiable steps).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.
        k: Number of trailing differentiable steps (clipped to n_steps).

    Returns:
        Final state estimate of shape ``(B, T, H, W)``.
    """
    # --- warmup: run n_steps-k steps without tracking gradients ---
    state = init_solver_state_2d(batch, hidden_dim)
    live_steps = min(max(k, 1), n_steps) if n_steps >= 1 else 0
    warmup_steps = max(n_steps - live_steps, 0)
    for _ in range(warmup_steps):
        state = solver_step_2d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    # detach the iterate so earlier steps don't contribute to the
    # gradient — but only when there was a warmup to detach: with
    # k >= n_steps the solve is fully differentiable, including
    # gradients with respect to the batch through the initial state.
    if warmup_steps > 0:
        state = SolverState2D(
            x=jax.lax.stop_gradient(state.x),
            lstm=jax.lax.stop_gradient(state.lstm),
            step=state.step,
        )

    # --- k differentiable steps ---
    for _ in range(live_steps):
        state = solver_step_2d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    return state.x
