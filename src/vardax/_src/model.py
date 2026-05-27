"""End-to-end 4DVarNet models.

Composes the prior, gradient modulator, and solver into a single Flax NNX
module that can be trained end-to-end.
"""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._types import Batch1D, Batch2D, LSTMState1D, LSTMState2D
from .grad_mod import ConvLSTMGradMod1D, ConvLSTMGradMod2D
from .priors import BilinAEPrior1D, BilinAEPrior2D
from .solver import GradMode


class FourDVarNet1D(nnx.Module):
    """End-to-end 4DVarNet model for 1-D spatiotemporal reconstruction.

    The model minimises the variational cost

    .. math::

        J(x) = \\|\\mathbf{m} \\odot (x - y)\\|^2 + \\lambda \\|x - \\varphi(x)\\|^2

    using ``n_solver_steps`` learned gradient steps, where :math:`\\varphi` is
    the bilinear autoencoder prior and the gradient steps are modulated by a
    ConvLSTM.

    Attributes:
        state_dim: Spatial dimension ``N`` of the state.
        n_time: Number of time steps ``T``.
        latent_dim: Latent dimension of the bilinear autoencoder prior.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        n_solver_steps: Number of solver iterations to unroll.
        alpha: Gradient step-size.
        prior_weight: Weight :math:`\\lambda` for the prior cost term.
        grad_mode: Differentiation and solver strategy (``"unrolled"``,
            ``"one_step"``, or ``"implicit"``).  In ``"implicit"`` mode the
            forward pass uses a fixed-point projection based only on the
            prior; the ConvLSTM gradient modulator, ``alpha``, and
            ``prior_weight`` are not used in the solver, and only the
            implicit fixed point is differentiated through.
    """

    def __init__(
        self,
        state_dim: int,
        n_time: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        n_solver_steps: int = 15,
        alpha: float = 0.2,
        prior_weight: float = 1.0,
        grad_mode: GradMode = "unrolled",
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_solver_steps = n_solver_steps
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.grad_mode = grad_mode
        self.prior = BilinAEPrior1D(
            state_dim=state_dim,
            latent_dim=latent_dim,
            n_time=n_time,
            rngs=rngs,
        )
        self.grad_mod = ConvLSTMGradMod1D(
            state_channels=n_time,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

    def __call__(self, batch: Batch1D) -> Float[Array, "B T N"]:
        """Run the solver and return the final state estimate.

        Args:
            batch: Input batch with ``input``, ``mask``, and ``target`` fields.

        Returns:
            Reconstructed state of shape ``(B, T, N)``.
        """
        if self.grad_mode == "unrolled":
            return self._call_unrolled(batch)
        elif self.grad_mode == "one_step":
            return self._call_one_step(batch)
        elif self.grad_mode == "implicit":
            return self._call_implicit(batch)
        else:
            raise ValueError(f"Unknown grad_mode: {self.grad_mode!r}")

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
        )

    def _call_implicit(self, batch: Batch1D) -> Float[Array, "B T N"]:
        # Uses the fixed-point projection solver (prior only; grad_mod and
        # alpha are not used in this mode).  A full implicit-differentiation
        # implementation that incorporates the gradient modulator requires
        # jaxopt or a similar fixed-point solver library.
        from .solver import solve_4dvarnet_1d_fixedpoint

        return solve_4dvarnet_1d_fixedpoint(
            batch, self.prior, n_fp_steps=self.n_solver_steps
        )


class FourDVarNet2D(nnx.Module):
    """End-to-end 4DVarNet model for 2-D spatiotemporal reconstruction.

    Attributes:
        n_time: Number of time steps ``T``.
        height: Spatial height ``H``.
        width: Spatial width ``W``.
        latent_dim: Latent dimension of the bilinear autoencoder prior.
        hidden_dim: Hidden dimension of the ConvLSTM gradient modulator.
        n_solver_steps: Number of solver iterations to unroll.
        alpha: Gradient step-size.
        prior_weight: Weight for the prior cost term.
        grad_mode: Differentiation and solver strategy (``"unrolled"``,
            ``"one_step"``, or ``"implicit"``).  In ``"implicit"`` mode the
            forward pass uses a fixed-point projection based only on the
            prior; the ConvLSTM gradient modulator, ``alpha``, and
            ``prior_weight`` are not used in the solver, and only the
            implicit fixed point is differentiated through.  Note that
            ``"implicit"`` is not yet implemented for 2-D models.
    """

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
        grad_mode: GradMode = "unrolled",
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_solver_steps = n_solver_steps
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.grad_mode = grad_mode
        self.prior = BilinAEPrior2D(
            latent_dim=latent_dim,
            n_time=n_time,
            height=height,
            width=width,
            rngs=rngs,
        )
        self.grad_mod = ConvLSTMGradMod2D(
            state_channels=n_time,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

    def __call__(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        """Run the solver and return the final state estimate.

        Args:
            batch: Input batch with ``input``, ``mask``, and ``target`` fields.

        Returns:
            Reconstructed state of shape ``(B, T, H, W)``.
        """
        if self.grad_mode == "unrolled":
            return self._call_unrolled(batch)
        elif self.grad_mode == "one_step":
            return self._call_one_step(batch)
        elif self.grad_mode == "implicit":
            return self._call_implicit(batch)
        else:
            raise ValueError(f"Unknown grad_mode: {self.grad_mode!r}")

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
        )

    def _call_implicit(self, batch: Batch2D) -> Float[Array, "B T H W"]:
        # Note: the fixed-point projection solver uses only the prior (no
        # gradient modulator).  A full implicit-diff implementation with a
        # gradient-modulated solver requires jaxopt.
        raise NotImplementedError(
            "Implicit differentiation for FourDVarNet2D is not yet implemented."
        )
