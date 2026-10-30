r"""Linear-Gaussian Kalman filter and RTS smoother over irregular times.

A control that evolves as a Gauss–Markov process between observation
times $t_1, \ldots, t_T$ (an emission rate across satellite overpasses,
a bias drifting between scenes):

$$
x_k = F_k x_{k-1} + \eta_k, \quad \eta_k \sim \mathcal{N}(0, \Sigma_{\eta,k}),
\qquad
y_k = H_k x_k + \varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, \Sigma_{\varepsilon,k}),
$$

with $x_1 \sim \mathcal{N}(\mu_1, \Sigma_1)$. ``KalmanSmoother`` returns
the exact smoothed marginals $p(x_k \mid y_{1:T})$, through gaussx's
``kalman_filter`` and ``rts_smoother``. $F_k$ and $\Sigma_{\eta,k}$ act
over the actual gap $t_k - t_{k-1}$, so irregular times need no
regridding; ``from_sde`` builds them from a gaussx SDE kernel.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D
from vardax._src.errors import ShapeError
from vardax._src.posterior.container import Posterior


class KalmanSmoother(eqx.Module):
    r"""Kalman filter and Rauch–Tung–Striebel smoother for time-varying controls.

    Algorithm:
        1. Prepend $F_1 = I$, $\Sigma_{\eta,1} = 0$, so that gaussx's
           predict-then-update filter observes $x_1$ under the prior
           $\mathcal{N}(\mu_1, \Sigma_1)$ itself.
        2. Filter forward: predict through $F_k$, then update with
           $y_k$. Masked observation entries are marginalised out
           exactly (their rows of $H_k$ are dropped), and a time with
           every entry masked only predicts.
        3. Smooth backward with the RTS recurrence
           $\mu_{k \mid T} = \mu_{k \mid k} + G_k (\mu_{k+1 \mid T} - \mu_{k+1 \mid k})$,
           $G_k = \Sigma_{k \mid k} F_{k+1}^\top \Sigma_{k+1 \mid k}^{-1}$.

    Attributes:
        transition: $F_k$ from time $k - 1$ to $k$, shape
            ``(T - 1, D_x, D_x)``, or ``(D_x, D_x)`` for every gap.
        transition_cov: $\Sigma_{\eta,k}$, the same shapes.
        obs_op: $H_k$, shape ``(T, D_y, D_x)``, or ``(D_y, D_x)`` for
            every time.
        obs_cov: $\Sigma_{\varepsilon,k}$, shape ``(T, D_y, D_y)`` or
            ``(D_y, D_y)``.
        prior_mean: $\mu_1$, shape ``(D_x,)``.
        prior_cov: $\Sigma_1$, shape ``(D_x, D_x)``.

    Examples:
        A random walk observed at three times, the middle one missing;
        the smoother fills the gap from both neighbours:

        >>> import jax.numpy as jnp
        >>> import vardax as vdx
        >>> smoother = vdx.KalmanSmoother(
        ...     transition=jnp.eye(1),
        ...     transition_cov=jnp.eye(1),
        ...     obs_op=jnp.eye(1),
        ...     obs_cov=0.01 * jnp.eye(1),
        ...     prior_mean=jnp.zeros(1),
        ...     prior_cov=jnp.eye(1),
        ... )
        >>> obs = jnp.array([[[1.0], [0.0], [3.0]]])
        >>> mask = jnp.array([[[1.0], [0.0], [1.0]]])
        >>> means = smoother(vdx.Batch1D(input=obs, mask=mask))
        >>> means.shape
        (1, 3, 1)
        >>> round(float(means[0, 1, 0]), 2)
        2.0
    """

    transition: Float[Array, "*gaps D D"]
    transition_cov: Float[Array, "*gaps D D"]
    obs_op: Float[Array, "*times M D"]
    obs_cov: Float[Array, "*times M M"]
    prior_mean: Float[Array, " D"]
    prior_cov: Float[Array, "D D"]

    @classmethod
    def from_sde(
        cls,
        sde: gaussx.SDEKernel,
        times: Float[Array, " T"],
        obs_op: Float[Array, "*times M D"],
        obs_cov: Float[Array, "*times M M"],
        prior_mean: Float[Array, " D"] | None = None,
    ) -> KalmanSmoother:
        r"""Discretise a gaussx SDE kernel over the gaps of ``times``.

        $F_k = e^{A \Delta t_k}$ and
        $\Sigma_{\eta,k} = P_\infty - F_k P_\infty F_k^\top$ on
        $\Delta t_k = t_k - t_{k-1}$, with the prior at $t_1$ the
        kernel's initial (stationary) covariance.

        Args:
            sde: A gaussx SDE kernel, e.g. ``gaussx.MaternSDE``.
            times: Observation times $t_1 < \ldots < t_T$, shape ``(T,)``.
            obs_op: $H_k$, shape ``(T, D_y, D_x)`` or ``(D_y, D_x)``.
            obs_cov: $\Sigma_{\varepsilon,k}$, shape ``(T, D_y, D_y)`` or
                ``(D_y, D_y)``.
            prior_mean: $\mu_1$. Default ``None``: zero.

        Returns:
            The smoother for those times.
        """
        transition, transition_cov = sde.discretise_sequence(jnp.diff(times))
        prior_cov = sde.initial_covariance()
        if prior_mean is None:
            prior_mean = jnp.zeros(prior_cov.shape[0], prior_cov.dtype)
        return cls(
            transition=transition,
            transition_cov=transition_cov,
            obs_op=obs_op,
            obs_cov=obs_cov,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )

    def __call__(self, batch: Batch1D) -> Float[Array, "S T D"]:
        """Smoothed means $\\mu_{k \\mid T}$ of every sequence.

        Args:
            batch: ``Batch1D`` with ``input`` and ``mask`` of shape
                ``(S, T, D_y)``; masked entries may be ``NaN``.

        Returns:
            The smoothed means, shape ``(S, T, D_x)``.
        """

        def smooth_one(obs: Array, mask: Array) -> Array:
            means, _ = self._smooth(obs, mask)
            return means

        return jax.vmap(smooth_one)(batch.input, batch.mask)

    def filter(
        self, batch: Batch1D
    ) -> tuple[Float[Array, "T D"], Float[Array, "T D D"]]:
        """Filtered means and covariances $p(x_k \\mid y_{1:k})$ of one sequence.

        Args:
            batch: ``Batch1D`` holding one sequence, ``(1, T, D_y)``.

        Returns:
            ``(means, covs)``, shapes ``(T, D_x)`` and ``(T, D_x, D_x)``.

        Raises:
            ShapeError: If the batch holds several sequences.
        """
        obs, mask = _one_sequence(batch)
        state = self._filter(obs, mask)
        return state.filtered_means, state.filtered_covs

    def posterior(self, batch: Batch1D) -> Posterior:
        """Smoothed marginals $\\mathcal{N}(\\mu_{k \\mid T}, \\Sigma_{k \\mid T})$ of one sequence.

        Args:
            batch: ``Batch1D`` holding one sequence, ``(1, T, D_y)``.

        Returns:
            [`Posterior`][vardax.Posterior] with ``mean`` of shape
            ``(T, D_x)`` and ``cov`` the block-diagonal operator of the
            $T$ marginal covariances, ``(T D_x, T D_x)``. Cross-time
            covariances are not kept.

        Raises:
            ShapeError: If the batch holds several sequences.
        """
        obs, mask = _one_sequence(batch)
        means, covs = self._smooth(obs, mask)
        blocks = [
            lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag) for cov in covs
        ]
        return Posterior(
            mean=means,
            cov=gaussx.BlockDiag(*blocks),
            samples=None,
            provenance={"adapter": "KalmanSmoother", "structure": "marginals"},
        )

    def _sequences(self, n_times: int) -> tuple[Array, Array, Array, Array]:
        """F, Σ_η over T steps (F₁ = I, Σ_η,₁ = 0), and H, Σ_ε over T times."""
        n_state = self.prior_mean.shape[0]
        transition = _over(self.transition, n_times - 1)
        transition_cov = _over(self.transition_cov, n_times - 1)
        identity = jnp.eye(n_state, dtype=transition.dtype)[None]
        # x₁ is observed under the prior itself: F₁ = I, Σ_η,₁ = 0
        transition = jnp.concatenate([identity, transition])
        transition_cov = jnp.concatenate([jnp.zeros_like(identity), transition_cov])
        return (
            transition,
            transition_cov,
            _over(self.obs_op, n_times),
            _over(self.obs_cov, n_times),
        )

    def _filter(self, obs: Array, mask: Array) -> Any:
        transition, transition_cov, obs_op, obs_cov = self._sequences(obs.shape[0])
        return gaussx.kalman_filter(
            transition,
            obs_op,
            transition_cov,
            obs_cov,
            obs,
            self.prior_mean,
            self.prior_cov,
            mask=mask != 0,
            solver=gaussx.DenseSolver(),
        )

    def _smooth(self, obs: Array, mask: Array) -> tuple[Array, Array]:
        state = self._filter(obs, mask)
        transition, *_ = self._sequences(obs.shape[0])
        return gaussx.rts_smoother(state, transition, solver=gaussx.DenseSolver())


def _over(stack: Array, length: int) -> Array:
    """``stack`` with a leading axis of ``length``, broadcasting one matrix."""
    if stack.ndim == 2:
        return jnp.broadcast_to(stack, (length, *stack.shape))
    if stack.shape[0] != length:
        raise ShapeError(
            f"Expected {length} matrices along the time axis; got {stack.shape[0]}."
        )
    return stack


def _one_sequence(batch: Batch1D) -> tuple[Array, Array]:
    """``(T, D_y)`` observations and mask of a one-sequence batch."""
    if batch.input.shape[0] != 1:
        raise ShapeError(
            f"Takes one sequence; the batch holds {batch.input.shape[0]}. Slice "
            "it: jax.tree.map(lambda a: a[i:i + 1], batch)."
        )
    return batch.input[0], batch.mask[0]
