"""4DVarNet iterative solver: x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾ on 𝒥 = 𝒥_o + β_p 𝒥_p.

Unrolls N_iter iterations of the variational cost minimisation, with the descent
direction Δx given by the learned gradient modulator g_mod.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._types import Batch1D, Batch2D, LSTMState1D, LSTMState2D

# ---------------------------------------------------------------------------
# Solver state containers
# ---------------------------------------------------------------------------


class SolverState1D(eqx.Module):
    """Carry of the 1-D solver loop: iterate, ConvLSTM carry and iteration index.

        (x⁽ⁱ⁾, c⁽ⁱ⁾, i),   c = (c_hid, c_cell)
        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)

    Attributes:
        x: Current iterate x⁽ⁱ⁾.
        lstm: Current ConvLSTM carry c⁽ⁱ⁾ = (c_hid, c_cell) of the gradient modulator.
        step: Current iteration index i.

    Examples:
        A state built by hand:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> state = vdx.SolverState1D(
        ...     x=jnp.zeros((2, 3, 4)), lstm=vdx.LSTMState1D.zeros(2, 5, 4), step=0
        ... )
        >>> state.x.shape, state.lstm.hidden.shape, state.step
        ((2, 3, 4), (2, 5, 4), 0)

        It is an Equinox module, so it is a pytree of x, hidden, cell and the step:

        >>> import jax
        >>> len(jax.tree_util.tree_leaves(state))
        4
    """

    x: Float[Array, "B T N"]
    lstm: LSTMState1D
    step: int


class SolverState2D(eqx.Module):
    """Carry of the 2-D solver loop: iterate, ConvLSTM carry and iteration index.

        (x⁽ⁱ⁾, c⁽ⁱ⁾, i),   c = (c_hid, c_cell)
        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)

    Attributes:
        x: Current iterate x⁽ⁱ⁾.
        lstm: Current ConvLSTM carry c⁽ⁱ⁾ = (c_hid, c_cell) of the gradient modulator.
        step: Current iteration index i.

    Examples:
        A state built by hand:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> state = vdx.SolverState2D(
        ...     x=jnp.zeros((1, 3, 4, 5)),
        ...     lstm=vdx.LSTMState2D.zeros(1, 2, 4, 5),
        ...     step=0,
        ... )
        >>> state.x.shape, state.lstm.cell.shape
        ((1, 3, 4, 5), (1, 2, 4, 5))
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

        x⁽⁰⁾ = m ⊙ y,   c⁽⁰⁾ = (0, 0),   i = 0

    Args:
        batch: Input batch (y, m). The initial iterate is the masked input, zero
            where unobserved.
        hidden_dim: Number of hidden channels of the ConvLSTM gradient modulator.

    Returns:
        Initial [`SolverState1D`][vardax.SolverState1D] holding x⁽⁰⁾, a zero carry
        c⁽⁰⁾ and ``step=0``.

    Examples:
        A typical call, with everything observed:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch1D(input=jnp.ones((2, 3, 4)), mask=jnp.ones((2, 3, 4)))
        >>> state = vdx.init_solver_state_1d(batch, hidden_dim=5)
        >>> state.x.shape == (2, 3, 4), state.step
        (True, 0)

        Unobserved entries start at zero:

        >>> empty = vdx.Batch1D(input=jnp.ones((2, 3, 4)), mask=jnp.zeros((2, 3, 4)))
        >>> bool(jnp.all(vdx.init_solver_state_1d(empty, hidden_dim=5).x == 0))
        True
    """
    batch_size, _, n_space = batch.input.shape
    # x⁽⁰⁾ = m ⊙ y
    initial_state = batch.input * batch.mask
    # c⁽⁰⁾ = (0, 0)
    carry = LSTMState1D.zeros(batch_size, hidden_dim, n_space)
    return SolverState1D(x=initial_state, lstm=carry, step=0)


def init_solver_state_2d(
    batch: Batch2D,
    hidden_dim: int,
) -> SolverState2D:
    """Initialise a 2-D solver state from a batch.

        x⁽⁰⁾ = m ⊙ y,   c⁽⁰⁾ = (0, 0),   i = 0

    Args:
        batch: Input batch (y, m). The initial iterate is the masked input, zero
            where unobserved.
        hidden_dim: Number of hidden channels of the ConvLSTM gradient modulator.

    Returns:
        Initial [`SolverState2D`][vardax.SolverState2D] holding x⁽⁰⁾, a zero carry
        c⁽⁰⁾ and ``step=0``.

    Examples:
        A typical call, with everything observed:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch2D(
        ...     input=jnp.ones((1, 3, 4, 5)), mask=jnp.ones((1, 3, 4, 5))
        ... )
        >>> state = vdx.init_solver_state_2d(batch, hidden_dim=5)
        >>> state.x.shape == (1, 3, 4, 5), state.step
        (True, 0)

        Unobserved entries start at zero:

        >>> empty = vdx.Batch2D(
        ...     input=jnp.ones((1, 3, 4, 5)), mask=jnp.zeros((1, 3, 4, 5))
        ... )
        >>> bool(jnp.all(vdx.init_solver_state_2d(empty, hidden_dim=5).x == 0))
        True
    """
    batch_size, _, height, width = batch.input.shape
    # x⁽⁰⁾ = m ⊙ y
    initial_state = batch.input * batch.mask
    # c⁽⁰⁾ = (0, 0)
    carry = LSTMState2D.zeros(batch_size, hidden_dim, height, width)
    return SolverState2D(x=initial_state, lstm=carry, step=0)


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

        𝒥(x) = 𝒥_o(x) + β_p 𝒥_p(x)
        𝒥_o = ‖m ⊙ (x − y)‖²,   𝒥_p = ‖x − x̂‖²,   x̂ = g_ae(x; θ)
        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)

    The costs are sums of squares over all entries, not means.

    Algorithm:
        1. Evaluate the observation term 𝒥_o and the prior term β_p 𝒥_p at the
           current iterate.
        2. Differentiate their sum with ``jax.grad`` to obtain ∇ₓ𝒥(x⁽ⁱ⁾).
        3. Pass (∇ₓ𝒥, x⁽ⁱ⁾, c⁽ⁱ⁾) through ``grad_mod_fn`` to get Δx⁽ⁱ⁾ and c⁽ⁱ⁺¹⁾.
        4. Set x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾ and increment ``step``.

    Args:
        solver_state: Current solver state (x⁽ⁱ⁾, c⁽ⁱ⁾, i).
        batch: Observed data batch (y, m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ) (autoencoder
            forward pass).
        grad_mod_fn: Callable ``(∇ₓ𝒥, x, c) -> (Δx, c′)``, the gradient modulator
            g_mod.
        alpha: Step size α on the state.
        prior_weight: Weight β_p of the prior cost term 𝒥_p.

    Returns:
        Updated [`SolverState1D`][vardax.SolverState1D]: iterate x⁽ⁱ⁺¹⁾, carry
        c⁽ⁱ⁺¹⁾ and ``step`` incremented by one.

    Examples:
        With an identity prior and a pass-through modulator, one step with
        ``alpha=0.5`` moves a zero state halfway to the observations (the
        observation gradient is −2):

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch1D(input=jnp.ones((2, 3, 4)), mask=jnp.ones((2, 3, 4)))
        >>> state = vdx.SolverState1D(
        ...     x=jnp.zeros((2, 3, 4)), lstm=vdx.LSTMState1D.zeros(2, 5, 4), step=0
        ... )
        >>> passthrough = lambda grad, x, lstm: (grad, lstm)
        >>> new = vdx.solver_step_1d(
        ...     state, batch, vdx.IdentityPrior(), passthrough, alpha=0.5
        ... )
        >>> bool(jnp.allclose(new.x, 1.0)), new.step
        (True, 1)

        Where nothing is observed and the prior is the identity there is no
        gradient, so the state does not move:

        >>> blind = vdx.Batch1D(input=jnp.ones((2, 3, 4)), mask=jnp.zeros((2, 3, 4)))
        >>> new = vdx.solver_step_1d(state, blind, vdx.IdentityPrior(), passthrough)
        >>> bool(jnp.allclose(new.x, 0.0))
        True
    """
    iterate = solver_state.x

    def cost_fn(x):
        # x̂ = g_ae(x; θ)
        reconstruction = prior_fn(x)
        # m ⊙ (x − y)
        obs_residual = batch.mask * (x - batch.input)
        # 𝒥_o = ‖m ⊙ (x − y)‖²
        obs_cost = jnp.sum(obs_residual**2)
        # x − x̂
        prior_residual = x - reconstruction
        # 𝒥_p = ‖x − x̂‖²
        prior_cost = jnp.sum(prior_residual**2)
        # β_p 𝒥_p
        weighted_prior_cost = prior_weight * prior_cost
        # 𝒥 = 𝒥_o + β_p 𝒥_p
        return obs_cost + weighted_prior_cost

    # ∇ₓ𝒥(x⁽ⁱ⁾)
    cost_gradient = jax.grad(cost_fn)(iterate)
    # (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)
    state_update, new_carry = grad_mod_fn(cost_gradient, iterate, solver_state.lstm)
    # x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾
    new_iterate = iterate - alpha * state_update

    return SolverState1D(x=new_iterate, lstm=new_carry, step=solver_state.step + 1)


def solver_step_2d(
    solver_state: SolverState2D,
    batch: Batch2D,
    prior_fn: Any,
    grad_mod_fn: Any,
    alpha: float = 1.0,
    prior_weight: float = 1.0,
) -> SolverState2D:
    """Perform a single 2-D solver iteration.

        𝒥(x) = 𝒥_o(x) + β_p 𝒥_p(x)
        𝒥_o = ‖m ⊙ (x − y)‖²,   𝒥_p = ‖x − x̂‖²,   x̂ = g_ae(x; θ)
        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)

    The costs are sums of squares over all entries, not means.

    Algorithm:
        1. Evaluate the observation term 𝒥_o and the prior term β_p 𝒥_p at the
           current iterate.
        2. Differentiate their sum with ``jax.grad`` to obtain ∇ₓ𝒥(x⁽ⁱ⁾).
        3. Pass (∇ₓ𝒥, x⁽ⁱ⁾, c⁽ⁱ⁾) through ``grad_mod_fn`` to get Δx⁽ⁱ⁾ and c⁽ⁱ⁺¹⁾.
        4. Set x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾ and increment ``step``.

    Args:
        solver_state: Current solver state (x⁽ⁱ⁾, c⁽ⁱ⁾, i).
        batch: Observed data batch (y, m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ) (autoencoder
            forward pass).
        grad_mod_fn: Callable ``(∇ₓ𝒥, x, c) -> (Δx, c′)``, the gradient modulator
            g_mod.
        alpha: Step size α on the state.
        prior_weight: Weight β_p of the prior cost term 𝒥_p.

    Returns:
        Updated [`SolverState2D`][vardax.SolverState2D]: iterate x⁽ⁱ⁺¹⁾, carry
        c⁽ⁱ⁺¹⁾ and ``step`` incremented by one.

    Examples:
        With an identity prior and a pass-through modulator, one step with
        ``alpha=0.5`` moves a zero state halfway to the observations (the
        observation gradient is −2):

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch2D(
        ...     input=jnp.ones((1, 3, 4, 5)), mask=jnp.ones((1, 3, 4, 5))
        ... )
        >>> state = vdx.SolverState2D(
        ...     x=jnp.zeros((1, 3, 4, 5)),
        ...     lstm=vdx.LSTMState2D.zeros(1, 2, 4, 5),
        ...     step=0,
        ... )
        >>> passthrough = lambda grad, x, lstm: (grad, lstm)
        >>> new = vdx.solver_step_2d(
        ...     state, batch, vdx.IdentityPrior(), passthrough, alpha=0.5
        ... )
        >>> bool(jnp.allclose(new.x, 1.0)), new.step
        (True, 1)

        Where nothing is observed and the prior is the identity there is no
        gradient, so the state does not move:

        >>> blind = vdx.Batch2D(
        ...     input=jnp.ones((1, 3, 4, 5)), mask=jnp.zeros((1, 3, 4, 5))
        ... )
        >>> new = vdx.solver_step_2d(state, blind, vdx.IdentityPrior(), passthrough)
        >>> bool(jnp.allclose(new.x, 0.0))
        True
    """
    iterate = solver_state.x

    def cost_fn(x):
        # x̂ = g_ae(x; θ)
        reconstruction = prior_fn(x)
        # m ⊙ (x − y)
        obs_residual = batch.mask * (x - batch.input)
        # 𝒥_o = ‖m ⊙ (x − y)‖²
        obs_cost = jnp.sum(obs_residual**2)
        # x − x̂
        prior_residual = x - reconstruction
        # 𝒥_p = ‖x − x̂‖²
        prior_cost = jnp.sum(prior_residual**2)
        # β_p 𝒥_p
        weighted_prior_cost = prior_weight * prior_cost
        # 𝒥 = 𝒥_o + β_p 𝒥_p
        return obs_cost + weighted_prior_cost

    # ∇ₓ𝒥(x⁽ⁱ⁾)
    cost_gradient = jax.grad(cost_fn)(iterate)
    # (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)
    state_update, new_carry = grad_mod_fn(cost_gradient, iterate, solver_state.lstm)
    # x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾
    new_iterate = iterate - alpha * state_update

    return SolverState2D(x=new_iterate, lstm=new_carry, step=solver_state.step + 1)


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

        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)
        x⁽⁰⁾ = m ⊙ y,   i = 0, …, N_iter − 1

    The cost 𝒥 is that of [`solver_step_1d`][vardax.solver_step_1d], with its
    default prior weight β_p = 1.

    Algorithm:
        1. Initialise the state with
           [`init_solver_state_1d`][vardax.init_solver_state_1d].
        2. Repeat N_iter times: apply [`solver_step_1d`][vardax.solver_step_1d] to
           the state.
        3. Return the final iterate x^(N_iter).

    Args:
        batch: Observed data batch (y, m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        grad_mod_fn: Callable ``(∇ₓ𝒥, x, c) -> (Δx, c′)``, the gradient modulator
            g_mod.
        n_steps: Number of gradient-descent iterations N_iter to unroll.
        hidden_dim: Number of hidden channels of the ConvLSTM gradient modulator.
        alpha: Step size α on the state.

    Returns:
        Final iterate x^(N_iter) after ``n_steps`` iterations.

    Examples:
        Fully observed data is already a minimiser for an identity prior, so
        the solver returns it unchanged:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch1D(input=jnp.ones((2, 3, 4)), mask=jnp.ones((2, 3, 4)))
        >>> passthrough = lambda grad, x, lstm: (grad, lstm)
        >>> out = vdx.solve_4dvarnet_1d(
        ...     batch,
        ...     vdx.IdentityPrior(),
        ...     passthrough,
        ...     n_steps=3,
        ...     hidden_dim=5,
        ...     alpha=0.5,
        ... )
        >>> out.shape == (2, 3, 4), bool(jnp.allclose(out, 1.0))
        (True, True)

        With ``n_steps=0`` the result is the masked input:

        >>> out = vdx.solve_4dvarnet_1d(
        ...     batch, vdx.IdentityPrior(), passthrough, n_steps=0, hidden_dim=5
        ... )
        >>> bool(jnp.allclose(out, batch.input * batch.mask))
        True

        The learned pieces of a [`FourDVarNet1D`][vardax.FourDVarNet1D] plug in
        directly and reproduce its unrolled forward pass:

        >>> import jax
        >>> model = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=2,
        ...     alpha=0.2,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> out = vdx.solve_4dvarnet_1d(
        ...     batch,
        ...     model.prior,
        ...     model.grad_mod,
        ...     n_steps=2,
        ...     hidden_dim=3,
        ...     alpha=0.2,
        ... )
        >>> bool(jnp.allclose(out, model(batch), atol=1e-5))
        True
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

        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)
        x⁽⁰⁾ = m ⊙ y,   i = 0, …, N_iter − 1

    The cost 𝒥 is that of [`solver_step_2d`][vardax.solver_step_2d], with its
    default prior weight β_p = 1.

    Algorithm:
        1. Initialise the state with
           [`init_solver_state_2d`][vardax.init_solver_state_2d].
        2. Repeat N_iter times: apply [`solver_step_2d`][vardax.solver_step_2d] to
           the state.
        3. Return the final iterate x^(N_iter).

    Args:
        batch: Observed data batch (y, m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        grad_mod_fn: Callable ``(∇ₓ𝒥, x, c) -> (Δx, c′)``, the gradient modulator
            g_mod.
        n_steps: Number of gradient-descent iterations N_iter to unroll.
        hidden_dim: Number of hidden channels of the ConvLSTM gradient modulator.
        alpha: Step size α on the state.

    Returns:
        Final iterate x^(N_iter) after ``n_steps`` iterations.

    Examples:
        Fully observed data is already a minimiser for an identity prior, so
        the solver returns it unchanged:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch2D(
        ...     input=jnp.ones((1, 3, 4, 5)), mask=jnp.ones((1, 3, 4, 5))
        ... )
        >>> passthrough = lambda grad, x, lstm: (grad, lstm)
        >>> out = vdx.solve_4dvarnet_2d(
        ...     batch,
        ...     vdx.IdentityPrior(),
        ...     passthrough,
        ...     n_steps=3,
        ...     hidden_dim=5,
        ...     alpha=0.5,
        ... )
        >>> out.shape == (1, 3, 4, 5), bool(jnp.allclose(out, 1.0))
        (True, True)

        With ``n_steps=0`` the result is the masked input:

        >>> out = vdx.solve_4dvarnet_2d(
        ...     batch, vdx.IdentityPrior(), passthrough, n_steps=0, hidden_dim=5
        ... )
        >>> bool(jnp.allclose(out, batch.input * batch.mask))
        True

        The learned pieces of a [`FourDVarNet2D`][vardax.FourDVarNet2D] plug in
        directly and reproduce its unrolled forward pass:

        >>> import jax
        >>> model = vdx.FourDVarNet2D(
        ...     n_time=3,
        ...     height=4,
        ...     width=5,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=2,
        ...     alpha=0.2,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> out = vdx.solve_4dvarnet_2d(
        ...     batch,
        ...     model.prior,
        ...     model.grad_mod,
        ...     n_steps=2,
        ...     hidden_dim=3,
        ...     alpha=0.2,
        ... )
        >>> bool(jnp.allclose(out, model(batch), atol=1e-5))
        True
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

        x̂ = g_ae(x; θ)
        x ← m ⊙ y + (1 − m) ⊙ x̂

    Algorithm:
        1. Project the iterate with the prior, x̂ = g_ae(x; θ).
        2. Overwrite observed entries with the data and keep the projection
           elsewhere, m ⊙ y + (1 − m) ⊙ x̂.

    Args:
        x: Current iterate x.
        batch: Observed data batch holding ``input`` (observations y) and ``mask``
            (m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ) (autoencoder
            forward pass).

    Returns:
        Updated iterate m ⊙ y + (1 − m) ⊙ x̂.

    Examples:
        Observed entries are reinserted, unobserved ones come from the prior:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch1D(
        ...     input=jnp.ones((1, 2, 3)),
        ...     mask=jnp.array([[[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
        ... )
        >>> prior = lambda x: 5.0 * jnp.ones_like(x)
        >>> out = vdx.fp_solver_step_1d(jnp.zeros((1, 2, 3)), batch, prior)
        >>> out[0].tolist()
        [[1.0, 5.0, 1.0], [5.0, 5.0, 1.0]]

        A fully observed batch returns the data whatever the prior says:

        >>> full = vdx.Batch1D(input=jnp.ones((1, 2, 3)), mask=jnp.ones((1, 2, 3)))
        >>> bool(jnp.all(vdx.fp_solver_step_1d(jnp.zeros((1, 2, 3)), full, prior) == 1))
        True
    """
    # x̂ = g_ae(x; θ)
    reconstruction = prior_fn(x)
    # m ⊙ y
    observed = batch.mask * batch.input
    # (1 − m) ⊙ x̂
    unobserved = (1 - batch.mask) * reconstruction
    # x ← m ⊙ y + (1 − m) ⊙ x̂
    return observed + unobserved


def solve_4dvarnet_1d_fixedpoint(
    batch: Batch1D,
    prior_fn: Any,
    n_fp_steps: int,
) -> Float[Array, "B T N"]:
    """Run ``n_fp_steps`` fixed-point projection steps using ``jax.lax.scan``.

        x⁽ⁱ⁺¹⁾ = m ⊙ y + (1 − m) ⊙ x̂⁽ⁱ⁾,   x̂⁽ⁱ⁾ = g_ae(x⁽ⁱ⁾; θ),   x⁽⁰⁾ = m ⊙ y

    This is the projection used by the implicit-adjoint path.

    Algorithm:
        1. Initialise x⁽⁰⁾ = m ⊙ y.
        2. Scan [`fp_solver_step_1d`][vardax.fp_solver_step_1d] for ``n_fp_steps``
           steps.
        3. Return the final iterate.

    Args:
        batch: Observed data batch (y, m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        n_fp_steps: Number of fixed-point iterations.

    Returns:
        Final iterate x⁽ⁿ⁾ after ``n_fp_steps`` projections.

    Examples:
        With an identity prior the unobserved entries stay at zero:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> batch = vdx.Batch1D(
        ...     input=jnp.ones((1, 2, 3)),
        ...     mask=jnp.array([[[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
        ... )
        >>> out = vdx.solve_4dvarnet_1d_fixedpoint(batch, vdx.IdentityPrior(), 4)
        >>> bool(jnp.allclose(out, batch.input * batch.mask))
        True

        Zero steps return the masked input, and a constant prior fills the
        gaps after one step:

        >>> zero = vdx.solve_4dvarnet_1d_fixedpoint(batch, vdx.IdentityPrior(), 0)
        >>> bool(jnp.allclose(zero, batch.input * batch.mask))
        True
        >>> out = vdx.solve_4dvarnet_1d_fixedpoint(batch, lambda x: 2 + 0 * x, 1)
        >>> out[0, 0].tolist()
        [1.0, 2.0, 1.0]
    """
    # x⁽⁰⁾ = m ⊙ y
    initial_state = batch.input * batch.mask

    def scan_fn(iterate: Float[Array, "B T N"], _: None) -> tuple:
        # x⁽ⁱ⁺¹⁾ = m ⊙ y + (1 − m) ⊙ g_ae(x⁽ⁱ⁾; θ)
        new_iterate = fp_solver_step_1d(iterate, batch, prior_fn)
        return new_iterate, None

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
    """Solve 4DVarNet-1D using k-step differentiation (Bolte et al., 2023).

        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)
        x^(N_iter − k) held fixed, so ∂x^(N_iter)/∂θ spans the last k iterations

    This is the iteration of [`solve_4dvarnet_1d`][vardax.solve_4dvarnet_1d], but
    only the last k steps carry gradients. This gives O(k) memory cost (k = 1
    matches implicit differentiation) while being as simple to implement as
    unrolled backprop.

    Algorithm:
        1. Initialise the state from the masked observations, x⁽⁰⁾ = m ⊙ y.
        2. Run N_iter − k warm-up steps and apply ``jax.lax.stop_gradient`` to the
           iterate and the carry (skipped when there is no warm-up).
        3. Run the remaining k steps through which gradients flow.
        4. Return the final iterate x^(N_iter).

    Args:
        batch: Observed data batch (y, m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        grad_mod_fn: Callable ``(∇ₓ𝒥, x, c) -> (Δx, c′)``, the gradient modulator
            g_mod.
        n_steps: Total number of solver iterations N_iter (warm-up N_iter − k, then
            k differentiable steps).
        hidden_dim: Number of hidden channels of the ConvLSTM gradient modulator.
        alpha: Step size α on the state.
        prior_weight: Weight β_p of the prior cost term 𝒥_p.
        k: Number of trailing differentiable steps (clipped to N_iter).

    Returns:
        Final iterate x^(N_iter) after ``n_steps`` iterations.

    References:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Examples:
        The forward value equals the fully unrolled solve; only the gradient
        path differs:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> from argminax.adjoint import OneStep
        >>> model = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=3,
        ...     alpha=0.2,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> batch = vdx.Batch1D(input=jnp.ones((2, 3, 4)), mask=jnp.ones((2, 3, 4)))
        >>> kwargs = dict(n_steps=3, hidden_dim=3, alpha=0.2)
        >>> full = vdx.solve_4dvarnet_1d(batch, model.prior, model.grad_mod, **kwargs)
        >>> one = vdx.one_step_solve_4dvarnet_1d(
        ...     batch, model.prior, model.grad_mod, k=1, **kwargs
        ... )
        >>> bool(jnp.allclose(full, one, atol=1e-5))
        True

        ``k >= n_steps`` disables the warm-up, so the solve is the same
        value again:

        >>> big = vdx.one_step_solve_4dvarnet_1d(
        ...     batch, model.prior, model.grad_mod, k=10, **kwargs
        ... )
        >>> bool(jnp.allclose(full, big, atol=1e-5))
        True

        It is what [`FourDVarNet1D`][vardax.FourDVarNet1D] runs under a
        [`OneStepAdjoint`][vardax.OneStepAdjoint]:

        >>> fast = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=3,
        ...     alpha=0.2,
        ...     solver_adjoint=OneStep(),
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> bool(jnp.allclose(fast(batch), one, atol=1e-5))
        True
    """
    # --- warm-up: N_iter − k steps whose gradients are cut ---
    state = init_solver_state_1d(batch, hidden_dim)
    live_steps = min(max(k, 1), n_steps) if n_steps >= 1 else 0
    warmup_steps = max(n_steps - live_steps, 0)
    for _ in range(warmup_steps):
        state = solver_step_1d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    # Detach the iterate so earlier steps do not contribute to the gradient, but
    # only when there was a warm-up to detach: with k >= N_iter the solve is fully
    # differentiable, including gradients with respect to the batch through the
    # initial state.
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
    """Solve 4DVarNet-2D using k-step differentiation (Bolte et al., 2023).

        x⁽ⁱ⁺¹⁾ = x⁽ⁱ⁾ − α Δx⁽ⁱ⁾,   (Δx⁽ⁱ⁾, c⁽ⁱ⁺¹⁾) = g_mod(∇ₓ𝒥(x⁽ⁱ⁾), x⁽ⁱ⁾, c⁽ⁱ⁾)
        x^(N_iter − k) held fixed, so ∂x^(N_iter)/∂θ spans the last k iterations

    This is the iteration of [`solve_4dvarnet_2d`][vardax.solve_4dvarnet_2d], but
    only the last k steps carry gradients. This gives O(k) memory cost (k = 1
    matches implicit differentiation) while being as simple to implement as
    unrolled backprop.

    Algorithm:
        1. Initialise the state from the masked observations, x⁽⁰⁾ = m ⊙ y.
        2. Run N_iter − k warm-up steps and apply ``jax.lax.stop_gradient`` to the
           iterate and the carry (skipped when there is no warm-up).
        3. Run the remaining k steps through which gradients flow.
        4. Return the final iterate x^(N_iter).

    Args:
        batch: Observed data batch (y, m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        grad_mod_fn: Callable ``(∇ₓ𝒥, x, c) -> (Δx, c′)``, the gradient modulator
            g_mod.
        n_steps: Total number of solver iterations N_iter (warm-up N_iter − k, then
            k differentiable steps).
        hidden_dim: Number of hidden channels of the ConvLSTM gradient modulator.
        alpha: Step size α on the state.
        prior_weight: Weight β_p of the prior cost term 𝒥_p.
        k: Number of trailing differentiable steps (clipped to N_iter).

    Returns:
        Final iterate x^(N_iter) after ``n_steps`` iterations.

    References:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Examples:
        The forward value equals the fully unrolled solve; only the gradient
        path differs:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> from argminax.adjoint import OneStep
        >>> model = vdx.FourDVarNet2D(
        ...     n_time=3,
        ...     height=4,
        ...     width=5,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=3,
        ...     alpha=0.2,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> batch = vdx.Batch2D(
        ...     input=jnp.ones((1, 3, 4, 5)), mask=jnp.ones((1, 3, 4, 5))
        ... )
        >>> kwargs = dict(n_steps=3, hidden_dim=3, alpha=0.2)
        >>> full = vdx.solve_4dvarnet_2d(batch, model.prior, model.grad_mod, **kwargs)
        >>> one = vdx.one_step_solve_4dvarnet_2d(
        ...     batch, model.prior, model.grad_mod, k=1, **kwargs
        ... )
        >>> bool(jnp.allclose(full, one, atol=1e-5))
        True

        ``k >= n_steps`` disables the warm-up, so the solve is the same
        value again:

        >>> big = vdx.one_step_solve_4dvarnet_2d(
        ...     batch, model.prior, model.grad_mod, k=10, **kwargs
        ... )
        >>> bool(jnp.allclose(full, big, atol=1e-5))
        True

        It is what [`FourDVarNet2D`][vardax.FourDVarNet2D] runs under a
        [`OneStepAdjoint`][vardax.OneStepAdjoint]:

        >>> fast = vdx.FourDVarNet2D(
        ...     n_time=3,
        ...     height=4,
        ...     width=5,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=3,
        ...     alpha=0.2,
        ...     solver_adjoint=OneStep(),
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> bool(jnp.allclose(fast(batch), one, atol=1e-5))
        True
    """
    # --- warm-up: N_iter − k steps whose gradients are cut ---
    state = init_solver_state_2d(batch, hidden_dim)
    live_steps = min(max(k, 1), n_steps) if n_steps >= 1 else 0
    warmup_steps = max(n_steps - live_steps, 0)
    for _ in range(warmup_steps):
        state = solver_step_2d(state, batch, prior_fn, grad_mod_fn, alpha, prior_weight)

    # Detach the iterate so earlier steps do not contribute to the gradient, but
    # only when there was a warm-up to detach: with k >= N_iter the solve is fully
    # differentiable, including gradients with respect to the batch through the
    # initial state.
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
