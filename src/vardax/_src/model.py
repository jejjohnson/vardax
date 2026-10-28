"""End-to-end 4DVarNet models.

Composes the prior, gradient modulator, and solver into a single
``eqx.Module`` that can be trained end-to-end via ``optax`` +
``eqx.filter_value_and_grad``.

In v0.4 the differentiation strategy is selected via a
``solver_adjoint: optimistix.AbstractAdjoint`` constructor slot
(Decision D15). The ``grad_mode`` enum from v0.1.x is gone:

- ``optimistix.RecursiveCheckpointAdjoint()`` → unrolled backprop with
  recursive checkpointing (the default)
- ``vardax.adjoints.OneStepAdjoint()`` → Bolte et al. 2023, O(1) memory
- ``optimistix.ImplicitAdjoint()`` → fixed-point projection + IFT

The dispatch in ``FourDVarNet*.__call__`` checks the adjoint's type
to pick the corresponding inner-solver path.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import optimistix as optx

from ._types import Batch1D, Batch2D, LSTMState1D, LSTMState2D
from .adjoints import KStepAdjoint
from .grad_mod import ConvLSTMGradMod1D, ConvLSTMGradMod2D
from .priors import BilinAEPrior1D, BilinAEPrior2D


def _default_adjoint() -> optx.AbstractAdjoint:
    """Default adjoint for the FourDVarNet inner solver."""
    return optx.RecursiveCheckpointAdjoint()


class FourDVarNet1D(eqx.Module):
    r"""End-to-end 4DVarNet model for 1-D spatiotemporal reconstruction.

    Minimises the variational cost

    $$
    J(x) = \|\mathbf{m} \odot (x - y)\|^2 + \lambda \|x - \varphi(x)\|^2
    $$

    using ``n_solver_steps`` learned gradient steps modulated by a ConvLSTM,
    with the differentiation strategy selected by ``solver_adjoint``.

    Algorithm:
        1. Start from the masked observations, $x_0 = m \odot y$, with a zero
           ConvLSTM state.
        2. For each of ``n_solver_steps`` iterations, compute
           $\nabla J(x_k)$, pass it through the ConvLSTM gradient modulator
           $\mathcal{G}$ and update $x_{k+1} = x_k - \alpha\, \mathcal{G}(\cdot)$.
        3. Return $x_K$.  A ``KStepAdjoint`` runs the one-step solver and an
           ``ImplicitAdjoint`` the fixed-point projection instead; any other
           adjoint unrolls with ordinary backprop.

    Args:
        state_dim: Spatial size $N$ of the state.
        n_time: Number of time steps $T$ in a window.
        latent_dim: Dimensionality of the prior's latent code.
        hidden_dim: Number of hidden channels of the ConvLSTM.
        n_solver_steps: Number of solver iterations to unroll.
        alpha: Gradient step-size $\alpha$.
        prior_weight: Weight $\lambda$ for the prior cost term.
        solver_adjoint: ``optimistix.AbstractAdjoint`` selecting the
            differentiation strategy. Defaults to
            ``RecursiveCheckpointAdjoint`` (standard backprop). Use
            ``OneStepAdjoint()`` for O(1)-memory training (Bolte et al.
            2023) or ``ImplicitAdjoint()`` for fixed-point projection.
        key: PRNG key for the prior and gradient-modulator weights.

    Attributes:
        prior: BilinAEPrior1D learned prior.
        grad_mod: ConvLSTMGradMod1D learned gradient modulator.

    Examples:
        A tiny model reconstructs a batch with the shape of its input:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> model = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=2,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> batch = vdx.Batch1D(input=jnp.ones((2, 3, 4)), mask=jnp.ones((2, 3, 4)))
        >>> model(batch).shape
        (2, 3, 4)

        Selecting the one-step adjoint changes only the gradient path, not
        the forward value:

        >>> fast = vdx.FourDVarNet1D(
        ...     state_dim=4,
        ...     n_time=3,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=2,
        ...     solver_adjoint=vdx.OneStepAdjoint(),
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> bool(jnp.allclose(fast(batch), model(batch), atol=1e-5))
        True

        Compose it with [`train_step`][vardax.train_step] for one optimiser
        update:

        >>> import optax
        >>> target = vdx.Batch1D(
        ...     input=batch.input, mask=batch.mask, target=jnp.zeros((2, 3, 4))
        ... )
        >>> optimizer = optax.adam(1e-3)
        >>> opt_state = optimizer.init(model)
        >>> model, opt_state, loss = vdx.train_step(model, target, optimizer, opt_state)
        >>> loss.shape
        ()
    """

    n_solver_steps: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    prior_weight: float = eqx.field(static=True)
    solver_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    prior: BilinAEPrior1D
    grad_mod: ConvLSTMGradMod1D

    def __init__(
        self,
        state_dim: int,
        n_time: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        n_solver_steps: int = 15,
        alpha: float = 0.2,
        prior_weight: float = 1.0,
        solver_adjoint: optx.AbstractAdjoint | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.n_solver_steps = n_solver_steps
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.solver_adjoint = solver_adjoint or _default_adjoint()
        key_prior, key_grad = jax.random.split(key)
        self.prior = BilinAEPrior1D(
            state_dim=state_dim,
            latent_dim=latent_dim,
            n_time=n_time,
            key=key_prior,
        )
        self.grad_mod = ConvLSTMGradMod1D(
            state_channels=n_time,
            hidden_dim=hidden_dim,
            key=key_grad,
        )

    def __call__(self, batch: Batch1D) -> Float[Array, "B T N"]:
        """Run the solver and return the final state estimate.

        Dispatches on ``solver_adjoint`` to pick the differentiation path.

        Args:
            batch: Observed batch $(y, m)$.

        Returns:
            Reconstructed state $x_K$ with the shape of ``batch.input``.
        """
        if isinstance(self.solver_adjoint, KStepAdjoint):
            return self._call_one_step(batch)
        if isinstance(self.solver_adjoint, optx.ImplicitAdjoint):
            return self._call_implicit(batch)
        # Default: optimistix.RecursiveCheckpointAdjoint() and anything
        # else falls through to the unrolled backprop path.
        return self._call_unrolled(batch)

    def _call_unrolled(self, batch: Batch1D) -> Float[Array, "B T N"]:
        batch_size, _, n_space = batch.input.shape
        # u₀ = m ⊙ y
        state = batch.input * batch.mask
        lstm = LSTMState1D.zeros(batch_size, self.grad_mod.hidden_dim, n_space)

        for _ in range(self.n_solver_steps):

            def cost_fn(current_state):
                obs_diff = batch.mask * (current_state - batch.input)
                # J_obs = ‖m ⊙ (u − y)‖²
                obs_cost = jnp.sum(obs_diff**2)
                # J_prior = λ ‖u − φ(u)‖²
                prior_cost = self.prior_weight * jnp.sum(
                    (current_state - self.prior(current_state)) ** 2
                )
                return obs_cost + prior_cost

            grad = jax.grad(cost_fn)(state)
            update, lstm = self.grad_mod(grad, state, lstm)
            # u ← u − α · update
            state = state - self.alpha * update

        return state

    def _call_one_step(self, batch: Batch1D) -> Float[Array, "B T N"]:
        from .solver import one_step_solve_4dvarnet_1d

        return one_step_solve_4dvarnet_1d(
            batch,
            self.prior,
            self.grad_mod,
            n_steps=self.n_solver_steps,
            hidden_dim=self.grad_mod.hidden_dim,
            alpha=self.alpha,
            prior_weight=self.prior_weight,
            k=getattr(self.solver_adjoint, "k", 1),
        )

    def _call_implicit(self, batch: Batch1D) -> Float[Array, "B T N"]:
        # Uses the fixed-point projection solver (prior only; grad_mod
        # and alpha are not used in this mode).
        from .solver import solve_4dvarnet_1d_fixedpoint

        return solve_4dvarnet_1d_fixedpoint(
            batch, self.prior, n_fp_steps=self.n_solver_steps
        )

    def as_analysis_step(self) -> _FourDVarNet1DAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8).

        Returns:
            A callable ``(forecast, obs, *, obs_op, obs_err_cov) -> analysis`` that
            masks the non-finite entries of ``obs`` and runs this model on them.

        Examples:
            Non-finite observations become unobserved entries:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> import vardax as vdx
            >>> model = vdx.FourDVarNet1D(
            ...     state_dim=4,
            ...     n_time=3,
            ...     latent_dim=2,
            ...     hidden_dim=3,
            ...     n_solver_steps=1,
            ...     key=jax.random.PRNGKey(0),
            ... )
            >>> step = model.as_analysis_step()
            >>> obs = jnp.ones((1, 3, 4)).at[0, 0, 0].set(jnp.nan)
            >>> step(obs, obs, obs_op=None, obs_err_cov=None).shape
            (1, 3, 4)
        """
        return _FourDVarNet1DAnalysisStep(self)


class _FourDVarNet1DAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``FourDVarNet1D``."""

    model: FourDVarNet1D

    def __call__(
        self,
        forecast: Float[Array, "B T N"],
        obs: Float[Array, "B T N"],
        *,
        obs_op: Any,  # pipekit_cycle.ObservationOperator
        obs_err_cov: Any,
    ) -> Float[Array, "B T N"]:
        mask = jnp.where(jnp.isfinite(obs), 1.0, 0.0)
        obs_clean = jnp.nan_to_num(obs)
        batch = Batch1D(input=obs_clean, mask=mask, target=None)
        return self.model(batch)


class FourDVarNet2D(eqx.Module):
    r"""End-to-end 4DVarNet model for 2-D spatiotemporal reconstruction.

    Minimises the same variational cost as [`FourDVarNet1D`][vardax.FourDVarNet1D],

    $$
    J(x) = \|\mathbf{m} \odot (x - y)\|^2 + \lambda \|x - \varphi(x)\|^2
    $$

    over fields with two spatial axes.

    Algorithm:
        1. Start from the masked observations, $x_0 = m \odot y$, with a zero
           ConvLSTM state.
        2. For each of ``n_solver_steps`` iterations, compute
           $\nabla J(x_k)$, pass it through the ConvLSTM gradient modulator
           $\mathcal{G}$ and update $x_{k+1} = x_k - \alpha\, \mathcal{G}(\cdot)$.
        3. Return $x_K$.  A ``KStepAdjoint`` runs the one-step solver; any
           other supported adjoint unrolls with ordinary backprop.

    Args:
        n_time: Number of time steps $T$ in a window.
        height: Spatial height $H$.
        width: Spatial width $W$.
        latent_dim: Dimensionality of the prior's latent code.
        hidden_dim: Number of hidden channels of the ConvLSTM.
        n_solver_steps: Number of solver iterations to unroll.
        alpha: Gradient step-size $\alpha$.
        prior_weight: Weight $\lambda$ for the prior cost term.
        solver_adjoint: ``optimistix.AbstractAdjoint`` selecting the
            differentiation strategy. Defaults to
            ``RecursiveCheckpointAdjoint`` (standard backprop).
            ``ImplicitAdjoint`` is not yet implemented for 2-D models.
        key: PRNG key for the prior and gradient-modulator weights.

    Attributes:
        prior: BilinAEPrior2D learned prior.
        grad_mod: ConvLSTMGradMod2D learned gradient modulator.

    Examples:
        A tiny model reconstructs a batch with the shape of its input:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> model = vdx.FourDVarNet2D(
        ...     n_time=3,
        ...     height=4,
        ...     width=5,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=2,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> batch = vdx.Batch2D(
        ...     input=jnp.ones((1, 3, 4, 5)), mask=jnp.ones((1, 3, 4, 5))
        ... )
        >>> model(batch).shape
        (1, 3, 4, 5)

        Selecting the one-step adjoint changes only the gradient path, not
        the forward value:

        >>> fast = vdx.FourDVarNet2D(
        ...     n_time=3,
        ...     height=4,
        ...     width=5,
        ...     latent_dim=2,
        ...     hidden_dim=3,
        ...     n_solver_steps=2,
        ...     solver_adjoint=vdx.OneStepAdjoint(),
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> bool(jnp.allclose(fast(batch), model(batch), atol=1e-5))
        True
    """

    n_solver_steps: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    prior_weight: float = eqx.field(static=True)
    solver_adjoint: optx.AbstractAdjoint = eqx.field(static=True)
    prior: BilinAEPrior2D
    grad_mod: ConvLSTMGradMod2D

    def __init__(
        self,
        n_time: int,
        height: int,
        width: int,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        n_solver_steps: int = 15,
        alpha: float = 0.2,
        prior_weight: float = 1.0,
        solver_adjoint: optx.AbstractAdjoint | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.n_solver_steps = n_solver_steps
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.solver_adjoint = solver_adjoint or _default_adjoint()
        key_prior, key_grad = jax.random.split(key)
        self.prior = BilinAEPrior2D(
            latent_dim=latent_dim,
            n_time=n_time,
            height=height,
            width=width,
            key=key_prior,
        )
        self.grad_mod = ConvLSTMGradMod2D(
            state_channels=n_time,
            hidden_dim=hidden_dim,
            key=key_grad,
        )

    def __call__(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        """Run the solver and return the final state estimate.

        Dispatches on ``solver_adjoint`` to pick the differentiation path.

        Args:
            batch: Observed batch $(y, m)$.

        Returns:
            Reconstructed state $x_K$ with the shape of ``batch.input``.
        """
        if isinstance(self.solver_adjoint, KStepAdjoint):
            return self._call_one_step(batch)
        if isinstance(self.solver_adjoint, optx.ImplicitAdjoint):
            return self._call_implicit(batch)
        return self._call_unrolled(batch)

    def _call_unrolled(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        batch_size, _, height, width = batch.input.shape
        # u₀ = m ⊙ y
        state = batch.input * batch.mask
        lstm = LSTMState2D.zeros(batch_size, self.grad_mod.hidden_dim, height, width)

        for _ in range(self.n_solver_steps):

            def cost_fn(current_state):
                obs_diff = batch.mask * (current_state - batch.input)
                # J_obs = ‖m ⊙ (u − y)‖²
                obs_cost = jnp.sum(obs_diff**2)
                # J_prior = λ ‖u − φ(u)‖²
                prior_cost = self.prior_weight * jnp.sum(
                    (current_state - self.prior(current_state)) ** 2
                )
                return obs_cost + prior_cost

            grad = jax.grad(cost_fn)(state)
            update, lstm = self.grad_mod(grad, state, lstm)
            # u ← u − α · update
            state = state - self.alpha * update

        return state

    def _call_one_step(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        from .solver import one_step_solve_4dvarnet_2d

        return one_step_solve_4dvarnet_2d(
            batch,
            self.prior,
            self.grad_mod,
            n_steps=self.n_solver_steps,
            hidden_dim=self.grad_mod.hidden_dim,
            alpha=self.alpha,
            prior_weight=self.prior_weight,
            k=getattr(self.solver_adjoint, "k", 1),
        )

    def _call_implicit(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        raise NotImplementedError(
            "ImplicitAdjoint for FourDVarNet2D is not yet implemented; "
            "use RecursiveCheckpointAdjoint or OneStepAdjoint."
        )

    def as_analysis_step(self) -> _FourDVarNet2DAnalysisStep:
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8).

        Returns:
            A callable ``(forecast, obs, *, obs_op, obs_err_cov) -> analysis`` that
            masks the non-finite entries of ``obs`` and runs this model on them.
        """
        return _FourDVarNet2DAnalysisStep(self)


class _FourDVarNet2DAnalysisStep(eqx.Module):
    """``pipekit_cycle.AnalysisStep`` adapter for ``FourDVarNet2D``."""

    model: FourDVarNet2D

    def __call__(
        self,
        forecast: Float[Array, "B T H W"],
        obs: Float[Array, "B T H W"],
        *,
        obs_op: Any,
        obs_err_cov: Any,
    ) -> Float[Array, "B T H W"]:
        mask = jnp.where(jnp.isfinite(obs), 1.0, 0.0)
        obs_clean = jnp.nan_to_num(obs)
        batch = Batch2D(input=obs_clean, mask=mask, target=None)
        return self.model(batch)
