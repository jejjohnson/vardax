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
    r"""Carry of the 1-D solver loop: iterate, ConvLSTM memory and step counter.

    The state is the triple $(u_k, (h_k, c_k), k)$ threaded through the unrolled
    iteration $u_{k+1} = u_k - \alpha\, \mathcal{G}(\nabla J(u_k), u_k, h_k, c_k)$.

    Attributes:
        x: Current state estimate $u_k$.
        lstm: Current LSTM hidden/cell state $(h_k, c_k)$ of the gradient modulator.
        step: Current iteration index $k$.

    Examples:
        A state built by hand:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> state = vdx.SolverState1D(
        ...     x=jnp.zeros((2, 3, 4)), lstm=vdx.LSTMState1D.zeros(2, 5, 4), step=0
        ... )
        >>> state.x.shape, state.lstm.hidden.shape, state.step
        ((2, 3, 4), (2, 5, 4), 0)

        It is an Equinox module, so it is a pytree (x, hidden, cell and the step counter):

        >>> import jax
        >>> len(jax.tree_util.tree_leaves(state))
        4
    """

    x: Float[Array, "B T N"]
    lstm: LSTMState1D
    step: int


class SolverState2D(eqx.Module):
    r"""Carry of the 2-D solver loop: iterate, ConvLSTM memory and step counter.

    The state is the triple $(u_k, (h_k, c_k), k)$ threaded through the unrolled
    iteration $u_{k+1} = u_k - \alpha\, \mathcal{G}(\nabla J(u_k), u_k, h_k, c_k)$.

    Attributes:
        x: Current state estimate $u_k$.
        lstm: Current LSTM hidden/cell state $(h_k, c_k)$ of the gradient modulator.
        step: Current iteration index $k$.

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
    r"""Initialise a 1-D solver state from a batch.

    The iterate starts at the masked observations, $u_0 = m \odot y$, with a zero
    ConvLSTM memory and step counter zero.

    Args:
        batch: Input batch $(y, m)$.  The initial state is set to the masked
            input (zeros where unobserved).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.

    Returns:
        Initial [`SolverState1D`][vardax.SolverState1D] holding $u_0$, a zero LSTM
        state and ``step=0``.

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
    # u₀ = m ⊙ y
    initial_state = batch.input * batch.mask
    lstm = LSTMState1D.zeros(batch_size, hidden_dim, n_space)
    return SolverState1D(x=initial_state, lstm=lstm, step=0)


def init_solver_state_2d(
    batch: Batch2D,
    hidden_dim: int,
) -> SolverState2D:
    r"""Initialise a 2-D solver state from a batch.

    The iterate starts at the masked observations, $u_0 = m \odot y$, with a zero
    ConvLSTM memory and step counter zero.

    Args:
        batch: Input batch $(y, m)$.  The initial state is set to the masked
            input (zeros where unobserved).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.

    Returns:
        Initial [`SolverState2D`][vardax.SolverState2D] holding $u_0$, a zero LSTM
        state and ``step=0``.

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

    With the variational cost

    $$
    J(u) = \|m \odot (u - y)\|^2 + \lambda \|u - \varphi(u)\|^2
    $$

    one iteration moves the state along the learned, modulated gradient,
    $u \leftarrow u - \alpha\, \mathcal{G}(\nabla J(u), u, h)$.

    Algorithm:
        1. Evaluate the observation term $\|m \odot (u - y)\|^2$ and the prior
           term $\lambda \|u - \varphi(u)\|^2$ at the current state.
        2. Differentiate the sum with ``jax.grad`` to obtain $\nabla J(u)$.
        3. Pass $(\nabla J, u, h)$ through ``grad_mod_fn`` to get the update and
           the new LSTM state.
        4. Set $u \leftarrow u - \alpha \cdot \text{update}$ and increment ``step``.

    Args:
        solver_state: Current solver state $(u, h, c, k)$.
        batch: Observed data batch $(y, m)$.
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(u)$ (prior
            autoencoder forward pass).
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``, the
            gradient modulator $\mathcal{G}$.
        alpha: Step-size scaling factor $\alpha$.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.

    Returns:
        Updated [`SolverState1D`][vardax.SolverState1D]: new iterate, new LSTM state
        and ``step`` incremented by one.

    Examples:
        With an identity prior and a pass-through modulator, one step with
        ``alpha=0.5`` moves a zero state halfway to the observations (the
        observation gradient is $-2$):

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

    With the variational cost

    $$
    J(u) = \|m \odot (u - y)\|^2 + \lambda \|u - \varphi(u)\|^2
    $$

    one iteration moves the state along the learned, modulated gradient,
    $u \leftarrow u - \alpha\, \mathcal{G}(\nabla J(u), u, h)$.

    Algorithm:
        1. Evaluate the observation term $\|m \odot (u - y)\|^2$ and the prior
           term $\lambda \|u - \varphi(u)\|^2$ at the current state.
        2. Differentiate the sum with ``jax.grad`` to obtain $\nabla J(u)$.
        3. Pass $(\nabla J, u, h)$ through ``grad_mod_fn`` to get the update and
           the new LSTM state.
        4. Set $u \leftarrow u - \alpha \cdot \text{update}$ and increment ``step``.

    Args:
        solver_state: Current solver state $(u, h, c, k)$.
        batch: Observed data batch $(y, m)$.
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(u)$ (prior
            autoencoder forward pass).
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``, the
            gradient modulator $\mathcal{G}$.
        alpha: Step-size scaling factor $\alpha$.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.

    Returns:
        Updated [`SolverState2D`][vardax.SolverState2D]: new iterate, new LSTM state
        and ``step`` incremented by one.

    Examples:
        With an identity prior and a pass-through modulator, one step with
        ``alpha=0.5`` moves a zero state halfway to the observations (the
        observation gradient is $-2$):

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
    r"""Run the full 1-D 4DVarNet solver for ``n_steps`` iterations.

    Unrolls $u_{k+1} = u_k - \alpha\, \mathcal{G}(\nabla J(u_k), u_k, h_k)$ from
    $u_0 = m \odot y$, with $J$ the variational cost of
    [`solver_step_1d`][vardax.solver_step_1d].

    Algorithm:
        1. Initialise the state with
           [`init_solver_state_1d`][vardax.init_solver_state_1d].
        2. Repeat ``n_steps`` times: apply
           [`solver_step_1d`][vardax.solver_step_1d] to the state.
        3. Return the final iterate.

    Args:
        batch: Observed data batch $(y, m)$.
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(u)$.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Number of gradient-descent steps to unroll.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor $\alpha$.

    Returns:
        Final state estimate $u_{n}$ after ``n_steps`` iterations.

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
    r"""Run the full 2-D 4DVarNet solver for ``n_steps`` iterations.

    Unrolls $u_{k+1} = u_k - \alpha\, \mathcal{G}(\nabla J(u_k), u_k, h_k)$ from
    $u_0 = m \odot y$, with $J$ the variational cost of
    [`solver_step_2d`][vardax.solver_step_2d].

    Algorithm:
        1. Initialise the state with
           [`init_solver_state_2d`][vardax.init_solver_state_2d].
        2. Repeat ``n_steps`` times: apply
           [`solver_step_2d`][vardax.solver_step_2d] to the state.
        3. Return the final iterate.

    Args:
        batch: Observed data batch $(y, m)$.
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(u)$.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Number of gradient-descent steps to unroll.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor $\alpha$.

    Returns:
        Final state estimate $u_{n}$ after ``n_steps`` iterations.

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
    r"""Perform a single 1-D fixed-point projection step.

    Applies the prior projection then re-inserts observations at observed
    locations:

    $$
    x \leftarrow \varphi(x), \quad
    x \leftarrow m \odot y + (1 - m) \odot x
    $$

    Algorithm:
        1. Project the state with the prior, $\varphi(x)$.
        2. Overwrite observed entries with the data and keep the projection
           elsewhere, $m \odot y + (1 - m) \odot \varphi(x)$.

    Args:
        x: Current state estimate $x$.
        batch: Observed data batch containing ``input`` (observations $y$)
            and ``mask`` ($m$).
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(x)$ (prior
            autoencoder forward pass).

    Returns:
        Updated state estimate $m \odot y + (1 - m) \odot \varphi(x)$.

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
    prior_state = prior_fn(x)
    # u ← m ⊙ y + (1 − m) ⊙ φ(u)
    return batch.mask * batch.input + (1 - batch.mask) * prior_state


def solve_4dvarnet_1d_fixedpoint(
    batch: Batch1D,
    prior_fn: Any,
    n_fp_steps: int,
) -> Float[Array, "B T N"]:
    r"""Run ``n_fp_steps`` fixed-point projection steps using ``jax.lax.scan``.

    Iterates $x_{k+1} = m \odot y + (1 - m) \odot \varphi(x_k)$ from the masked
    observations, the projection used by the implicit-adjoint path.

    Algorithm:
        1. Initialise $x_0 = m \odot y$.
        2. Scan [`fp_solver_step_1d`][vardax.fp_solver_step_1d] for
           ``n_fp_steps`` steps.
        3. Return the final iterate.

    Args:
        batch: Observed data batch $(y, m)$.
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(x)$.
        n_fp_steps: Number of fixed-point iterations.

    Returns:
        Final state estimate $x_{n}$ after ``n_fp_steps`` projections.

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

    Runs the same iteration as [`solve_4dvarnet_1d`][vardax.solve_4dvarnet_1d],
    $u_{j+1} = u_j - \alpha\, \mathcal{G}(\nabla J(u_j), u_j, h_j)$, but
    only the last ``k`` steps carry gradients.  This gives O(k) memory cost
    (k=1 matches implicit differentiation) while being as simple to
    implement as unrolled backprop.

    Algorithm:
        1. Initialise the state from the masked observations.
        2. Run ``n_steps - k`` warm-up steps and apply ``jax.lax.stop_gradient``
           to the iterate and LSTM state (skipped when there is no warm-up).
        3. Run the remaining ``k`` steps through which gradients flow.
        4. Return the final iterate.

    References:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Args:
        batch: Observed data batch $(y, m)$.
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(u)$.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Total number of solver iterations (warmup = n_steps - k,
            then k differentiable steps).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor $\alpha$.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.
        k: Number of trailing differentiable steps (clipped to n_steps).

    Returns:
        Final state estimate after ``n_steps`` iterations.

    Examples:
        The forward value equals the fully unrolled solve; only the gradient
        path differs:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
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
        ...     solver_adjoint=vdx.OneStepAdjoint(),
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> bool(jnp.allclose(fast(batch), one, atol=1e-5))
        True
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

    Runs the same iteration as [`solve_4dvarnet_2d`][vardax.solve_4dvarnet_2d],
    $u_{j+1} = u_j - \alpha\, \mathcal{G}(\nabla J(u_j), u_j, h_j)$, but
    only the last ``k`` steps carry gradients.  This gives O(k) memory cost
    (k=1 matches implicit differentiation) while being as simple to
    implement as unrolled backprop.

    Algorithm:
        1. Initialise the state from the masked observations.
        2. Run ``n_steps - k`` warm-up steps and apply ``jax.lax.stop_gradient``
           to the iterate and LSTM state (skipped when there is no warm-up).
        3. Run the remaining ``k`` steps through which gradients flow.
        4. Return the final iterate.

    References:
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation of
        iterative algorithms." https://arxiv.org/abs/2305.13768

    Args:
        batch: Observed data batch $(y, m)$.
        prior_fn: Callable ``x -> x_prior`` giving $\varphi(u)$.
        grad_mod_fn: Callable ``(grad, x, lstm) -> (update, new_lstm)``.
        n_steps: Total number of solver iterations (warmup = n_steps - k,
            then k differentiable steps).
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        alpha: Step-size scaling factor $\alpha$.
        prior_weight: Weighting factor $\lambda$ for the prior cost term.
        k: Number of trailing differentiable steps (clipped to n_steps).

    Returns:
        Final state estimate after ``n_steps`` iterations.

    Examples:
        The forward value equals the fully unrolled solve; only the gradient
        path differs:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
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
        ...     solver_adjoint=vdx.OneStepAdjoint(),
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> bool(jnp.allclose(fast(batch), one, atol=1e-5))
        True
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
