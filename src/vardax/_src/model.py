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

    Attributes:
        n_solver_steps: Number of solver iterations to unroll.
        alpha: Gradient step-size.
        prior_weight: Weight $\lambda$ for the prior cost term.
        solver_adjoint: ``optimistix.AbstractAdjoint`` selecting the
            differentiation strategy. Defaults to
            ``RecursiveCheckpointAdjoint`` (standard backprop). Use
            ``OneStepAdjoint()`` for O(1)-memory training (Bolte et al.
            2023) or ``ImplicitAdjoint()`` for fixed-point projection.
        prior: BilinAEPrior1D learned prior.
        grad_mod: ConvLSTMGradMod1D learned gradient modulator.
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
        k_prior, k_grad = jax.random.split(key)
        self.prior = BilinAEPrior1D(
            state_dim=state_dim,
            latent_dim=latent_dim,
            n_time=n_time,
            key=k_prior,
        )
        self.grad_mod = ConvLSTMGradMod1D(
            state_channels=n_time,
            hidden_dim=hidden_dim,
            key=k_grad,
        )

    def __call__(self, batch: Batch1D) -> Float[Array, "B T N"]:
        """Run the solver and return the final state estimate.

        Dispatches on ``solver_adjoint`` to pick the differentiation
        path.
        """
        if isinstance(self.solver_adjoint, KStepAdjoint):
            return self._call_one_step(batch)
        if isinstance(self.solver_adjoint, optx.ImplicitAdjoint):
            return self._call_implicit(batch)
        # Default: optimistix.RecursiveCheckpointAdjoint() and anything
        # else falls through to the unrolled backprop path.
        return self._call_unrolled(batch)

    def _call_unrolled(self, batch: Batch1D) -> Float[Array, "B T N"]:
        b, _, n = batch.input.shape
        x = batch.input * batch.mask
        lstm = LSTMState1D.zeros(b, self.grad_mod.hidden_dim, n)

        for _ in range(self.n_solver_steps):

            def cost_fn(x_):
                obs_diff = batch.mask * (x_ - batch.input)
                j_obs = jnp.sum(obs_diff**2)
                j_prior = self.prior_weight * jnp.sum((x_ - self.prior(x_)) ** 2)
                return j_obs + j_prior

            grad = jax.grad(cost_fn)(x)
            update, lstm = self.grad_mod(grad, x, lstm)
            x = x - self.alpha * update

        return x

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
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
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
    """End-to-end 4DVarNet model for 2-D spatiotemporal reconstruction.

    Attributes:
        n_solver_steps: Number of solver iterations to unroll.
        alpha: Gradient step-size.
        prior_weight: Weight for the prior cost term.
        solver_adjoint: ``optimistix.AbstractAdjoint`` selecting the
            differentiation strategy. Defaults to
            ``RecursiveCheckpointAdjoint`` (standard backprop).
            ``ImplicitAdjoint`` is not yet implemented for 2-D models.
        prior: BilinAEPrior2D learned prior.
        grad_mod: ConvLSTMGradMod2D learned gradient modulator.
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
        k_prior, k_grad = jax.random.split(key)
        self.prior = BilinAEPrior2D(
            latent_dim=latent_dim,
            n_time=n_time,
            height=height,
            width=width,
            key=k_prior,
        )
        self.grad_mod = ConvLSTMGradMod2D(
            state_channels=n_time,
            hidden_dim=hidden_dim,
            key=k_grad,
        )

    def __call__(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        """Run the solver and return the final state estimate.

        Dispatches on ``solver_adjoint`` to pick the differentiation
        path.
        """
        if isinstance(self.solver_adjoint, KStepAdjoint):
            return self._call_one_step(batch)
        if isinstance(self.solver_adjoint, optx.ImplicitAdjoint):
            return self._call_implicit(batch)
        return self._call_unrolled(batch)

    def _call_unrolled(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        b, _, h, w = batch.input.shape
        x = batch.input * batch.mask
        lstm = LSTMState2D.zeros(b, self.grad_mod.hidden_dim, h, w)

        for _ in range(self.n_solver_steps):

            def cost_fn(x_):
                obs_diff = batch.mask * (x_ - batch.input)
                j_obs = jnp.sum(obs_diff**2)
                j_prior = self.prior_weight * jnp.sum((x_ - self.prior(x_)) ** 2)
                return j_obs + j_prior

            grad = jax.grad(cost_fn)(x)
            update, lstm = self.grad_mod(grad, x, lstm)
            x = x - self.alpha * update

        return x

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
        """Adapt to ``pipekit_cycle.AnalysisStep`` (Decision D8)."""
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
