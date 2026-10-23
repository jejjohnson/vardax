r"""Neural-field 4D-Var: the whole window as one continuous field.

The analysis is a space-time field $u(t, s) = f_\theta(t, s)$, fitted to
scattered observations with the physics as a soft constraint and no
time-stepping rollout (Oh 2025). With observations $y_k$ at $(t_k, s_k)$
with error standard deviations $\sigma_k$, a PDE operator $\mathcal N$
($\mathcal N[u] = 0$ for the true flow), collocation points $c_j$ and
weights $\lambda, \alpha \ge 0$,

$$
J(\theta) = \tfrac{1}{2}\sum_k \frac{\big(y_k - f_\theta(t_k, s_k)\big)^2}
    {\sigma_k^2}
  + \frac{\lambda}{2 N_c}\sum_j \big\lVert \mathcal N[f_\theta](c_j)
    \big\rVert^2
  + \frac{\alpha}{2}\lVert\theta\rVert^2 .
$$

There is no background term: the network's spectral bias plays the role
of $B$. The pieces are the merged building blocks: the observation term
is [`point_obs_cost`][vardax.point_obs_cost] of a
[`FieldPointObs`][vardax.FieldPointObs], the physics term a
[`ResidualPenalty`][vardax.ResidualPenalty], the weight decay and the
control/frozen split a
[`NeuralFieldParameterization`][vardax.NeuralFieldParameterization], and
the minimisation an [`OptaxMinimiser`][vardax.OptaxMinimiser] (or an
[`OptimistixMinimiser`][vardax.OptimistixMinimiser]).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from vardax._src._types import PointObsBatch
from vardax._src.costs import ResidualPenalty
from vardax._src.minimisers import OptaxMinimiser, OptimistixMinimiser
from vardax._src.obs_operators.point import FieldPointObs, point_obs_cost
from vardax._src.parameterization import NeuralFieldParameterization


def _evaluate(field: Any, coords: Float[Array, "N P"]) -> Array:
    """``field`` at every row of ``coords``; a single channel is squeezed."""
    values = jax.vmap(field)(coords)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values


class SpaceTimeFieldResult(eqx.Module):
    r"""A fitted space-time field and its minimisation diagnostics.

    Attributes:
        field: The analysis $f_{\theta^\star}$, a single-example callable
            of one space-time point ``(t, *s)`` of shape ``(1 + D,)``.
        t0: Start of the window, $t_0$, used by `initial_condition`.
        cost: $J(\theta^\star)$.
        grad_norm: $\lVert\nabla J(\theta^\star)\rVert_2$; a first-order
            optimiser stopped by its step budget leaves it non-zero.
        num_steps: Optimiser steps taken.

    An [`ensemble`][vardax.SpaceTimeFieldVar.ensemble] returns one result
    whose array leaves carry a leading member axis; evaluate it with
    ``eqx.filter_vmap(lambda r: r.initial_condition(grid))(result)``.
    """

    field: Any
    t0: float = eqx.field(static=True)
    cost: Float[Array, ""]
    grad_norm: Float[Array, ""]
    num_steps: Int[Array, ""]

    def evaluate(self, coords: Float[Array, "N P"]) -> Float[Array, "N ..."]:
        """Evaluate the field at space-time points.

        Args:
            coords: Points ``(t, *s)``, shape ``(N, 1 + D)``.

        Returns:
            ``(N,)`` for a single-channel field, else ``(N, K)``.
        """
        return _evaluate(self.field, jnp.asarray(coords))

    def initial_condition(self, grid: Float[Array, "N D"]) -> Float[Array, "N ..."]:
        r"""The analysis at the window start, $u(t_0, s_j)$, on a grid.

        This is the initial state for a forecast with a ``ForwardModel``.

        Args:
            grid: Spatial points $s_j$, shape ``(N, D)``.

        Returns:
            ``(N,)`` for a single-channel field, else ``(N, K)``.
        """
        grid = jnp.asarray(grid)
        if grid.ndim != 2:
            raise ValueError(f"grid must be 2-D (N, D); got shape {grid.shape}.")
        time = jnp.full((grid.shape[0], 1), self.t0, dtype=grid.dtype)
        return _evaluate(self.field, jnp.concatenate([time, grid], axis=-1))


class SpaceTimeFieldVar(eqx.Module):
    r"""Neural-field 4D-Var: fit one continuous field $u(t, s) = f_\theta(t, s)$.

    Minimises, over the network weights $\theta$,

    $$
    J(\theta) = \tfrac{1}{2}\sum_k m_k \frac{\big(y_k - f_\theta(t_k, s_k)
        \big)^2}{\sigma_k^2}
      + \frac{\lambda}{2 N_c}\sum_j \big\lVert \mathcal N[f_\theta](c_j)
        \big\rVert^2
      + \frac{\alpha}{2}\lVert\theta\rVert^2 ,
    $$

    with the observations of a [`PointObsBatch`][vardax.PointObsBatch]
    ($m_k = 0$ on padding rows, which contribute nothing, ``NaN``
    included), the physics term of an optional
    [`ResidualPenalty`][vardax.ResidualPenalty] and weight decay $\alpha$.
    There is no background covariance and no rollout: the whole window
    is the field, which can be queried and differentiated anywhere
    (Oh 2025).

    Coordinates put **time first**: the field maps a point
    ``(t, *s)`` of shape ``(1 + D,)`` to ``(K,)`` (or a scalar), the
    observation coordinates are ``(N_obs, 1 + D)`` in the same order, and
    so are the collocation points, as the residuals of
    `vardax.examples` expect. The observed quantity is the field's single
    output (`FieldPointObs`); a multi-channel field needs its own wrapper.

    The penalty's collocation points are fixed during a fit: a
    ``ResidualPenalty`` with a ``sampler`` draws them once, from ``key``,
    at the start of the analysis, so the cost the minimiser sees is
    deterministic (its patience test, best-iterate tracking and implicit
    adjoint all assume that). Draw more points, or refit with new keys,
    for a better estimate of the expected residual.

    Attributes:
        field: The initial network, a single-example ``eqx.Module``
            ``(1 + D,) -> (K,)`` such as `geonnax.SIREN` or
            `geonnax.NeuralFieldMLP`. Its parameters are the initial
            control.
        residual: Optional [`ResidualPenalty`][vardax.ResidualPenalty]
            (the physics term); ``None`` fits the observations alone.
        weight_decay: $\alpha \ge 0$.
        minimiser: An [`OptaxMinimiser`][vardax.OptaxMinimiser] (the
            default, Adam with learning rate $10^{-3}$) or an
            [`OptimistixMinimiser`][vardax.OptimistixMinimiser].
        filter_spec: Which leaves of ``field`` are the control, as in
            [`NeuralFieldParameterization`][vardax.NeuralFieldParameterization]:
            exclude frozen arrays (random Fourier frequencies) so the weight
            decay does not shrink them. Default ``eqx.is_inexact_array``.
        t0: Start of the window, $t_0$ (default ``0.0``).

    Examples:
        A field linear in fixed features, $f_w(t, x) = w_0 + w_1 x$, with no
        physics is Bayesian linear regression: the analysis is the ridge
        solution, here $w^\star = (0, 1)$ to within the small weight decay.

        >>> import equinox as eqx, jax, jax.numpy as jnp, vardax
        >>> class Line(eqx.Module):
        ...     w: jax.Array
        ...
        ...     def __call__(self, s):
        ...         return (self.w[0] + self.w[1] * s[1])[None]
        >>> obs = vardax.PointObsBatch(
        ...     coords=jnp.array([[0.0, -1.0], [0.5, 0.0], [1.0, 1.0], [0.0, 9.0]]),
        ...     values=jnp.array([-1.0, 0.0, 1.0, jnp.nan]),  # last row: padding
        ...     sigma=jnp.ones(4),
        ...     valid=jnp.array([True, True, True, False]),
        ... )
        >>> ana = vardax.SpaceTimeFieldVar(
        ...     field=Line(jnp.zeros(2)),
        ...     weight_decay=1e-6,
        ...     minimiser=vardax.OptimistixMinimiser(),
        ... )
        >>> result = ana(obs)
        >>> result.initial_condition(jnp.array([[0.5], [2.0]])).round(3)
        Array([0.5, 2. ], dtype=float32)

    References:
        Oh, Y. (2025). On the effect of neural field reparameterization for
        4DVAR. arXiv:2509.21751.
    """

    field: Any
    residual: ResidualPenalty | None
    weight_decay: float = eqx.field(static=True)
    minimiser: OptaxMinimiser | OptimistixMinimiser = eqx.field(static=True)
    filter_spec: Any = eqx.field(static=True)
    t0: float = eqx.field(static=True)

    def __init__(
        self,
        field: Any,
        residual: ResidualPenalty | None = None,
        weight_decay: float = 0.0,
        minimiser: OptaxMinimiser | OptimistixMinimiser | None = None,
        *,
        filter_spec: Any = eqx.is_inexact_array,
        t0: float = 0.0,
    ) -> None:
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")
        if residual is not None and not isinstance(residual, ResidualPenalty):
            raise TypeError(
                "residual must be a vardax.ResidualPenalty or None; got "
                f"{type(residual).__name__}."
            )
        if minimiser is None:
            minimiser = OptaxMinimiser()
        if not isinstance(minimiser, OptaxMinimiser | OptimistixMinimiser):
            raise TypeError(
                "minimiser must be a vardax.OptaxMinimiser or "
                f"vardax.OptimistixMinimiser; got {type(minimiser).__name__}."
            )
        self.field = field
        self.residual = residual
        self.weight_decay = weight_decay
        self.minimiser = minimiser
        self.filter_spec = filter_spec
        self.t0 = t0

    @property
    def _parameterization(self) -> NeuralFieldParameterization:
        # Only the control split, `as_field` and the weight decay are used:
        # the space-time field is evaluated at the observation and
        # collocation points, never on a grid, so the grid is empty.
        return NeuralFieldParameterization(
            self.field,
            jnp.zeros((0, 1)),
            self.weight_decay,
            filter_spec=self.filter_spec,
        )

    def _fixed_residual(self, key: PRNGKeyArray | None) -> ResidualPenalty | None:
        """The residual penalty with its collocation points drawn and fixed."""
        residual = self.residual
        if residual is None or residual.sampler is None:
            return residual
        if key is None:
            raise ValueError(
                "The residual penalty has a sampler: pass key= to draw its "
                "collocation points."
            )
        return ResidualPenalty(
            residual_fn=residual.residual_fn,
            collocation=residual.sampler(key),
            weight=residual.weight,
        )

    def __call__(
        self, obs: PointObsBatch, *, key: PRNGKeyArray | None = None
    ) -> SpaceTimeFieldResult:
        """Fit the field to one sample of point observations.

        Args:
            obs: The observations, one sample (``values`` of shape
                ``(N_obs,)``, ``coords`` ``(N_obs, 1 + D)`` with time first;
                ``time_index`` and ``n_times`` are not used). For a batch
                of samples, ``eqx.filter_vmap`` over this call.
            key: Draws the collocation points when the residual penalty
                has a ``sampler``; unused otherwise.

        Returns:
            A [`SpaceTimeFieldResult`][vardax.SpaceTimeFieldResult].

        Raises:
            TypeError: If ``obs`` is not a `PointObsBatch`.
            ValueError: If ``obs`` carries a batch axis, or the penalty
                needs a key and none is given.
        """
        if not isinstance(obs, PointObsBatch):
            raise TypeError(
                f"obs must be a vardax.PointObsBatch; got {type(obs).__name__}."
            )
        if obs.values.ndim != 1:
            raise ValueError(
                "SpaceTimeFieldVar fits one sample: obs.values must be "
                f"(N_obs,), got shape {obs.values.shape}; use eqx.filter_vmap "
                "for a batch."
            )
        param = self._parameterization
        residual = self._fixed_residual(key)
        field_obs = FieldPointObs()

        def cost(
            theta: PyTree[Array],
            args: tuple[PointObsBatch, ResidualPenalty | None],
        ) -> Float[Array, ""]:
            batch, res = args
            field = param.as_field(theta)
            # ½ Σₖ mₖ (yₖ - f_θ(tₖ, sₖ))² / σₖ² + alpha/2 ‖θ‖²: ()
            j = point_obs_cost(field_obs(field, batch), batch) + param.penalty(theta)
            if res is not None:
                # λ/(2 N_c) Σⱼ ‖N[f_θ](cⱼ)‖²: ()
                j = j + res(field)
            return j

        out = self.minimiser.run(cost, param.init_control(), (obs, residual))
        return SpaceTimeFieldResult(
            field=param.as_field(out.value),
            t0=self.t0,
            cost=out.cost,
            grad_norm=out.grad_norm,
            num_steps=out.num_steps,
        )

    def ensemble(
        self,
        obs: PointObsBatch,
        field_init: Any,
        n_members: int,
        *,
        key: PRNGKeyArray,
    ) -> SpaceTimeFieldResult:
        r"""A deep ensemble: refit from ``n_members`` random initialisations.

        Each member re-initialises the field with ``field_init`` and draws
        its own collocation points (when the penalty has a sampler), then
        fits as `__call__` does; the members run under ``eqx.filter_vmap``.
        The spread of the members is a heuristic uncertainty: small where
        the observations pin the field down, large away from them. A
        $\theta$-space Laplace push-forward $J \Sigma_\theta J^\top$
        ($J = \partial u / \partial\theta$) is not implemented.

        Args:
            obs: The observations, one sample.
            field_init: ``key -> field``, building a network with the
                same structure as ``self.field`` (e.g.
                ``lambda k: geonnax.SIREN.init(2, 32, 1, depth=3, key=k)``).
            n_members: Number of members, at least 1.
            key: Split into one initialisation key and one collocation key
                per member.

        Returns:
            A [`SpaceTimeFieldResult`][vardax.SpaceTimeFieldResult] whose
            array leaves carry a leading axis of size ``n_members``.

        Raises:
            ValueError: If ``n_members < 1``.
        """
        if n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}.")

        def member(k: PRNGKeyArray) -> SpaceTimeFieldResult:
            k_init, k_colloc = jax.random.split(k)
            model = eqx.tree_at(lambda m: m.field, self, field_init(k_init))
            return model(obs, key=k_colloc)

        return eqx.filter_vmap(member)(jax.random.split(key, n_members))

    def as_analysis_step(
        self,
        grid: Float[Array, "N D"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> _SpaceTimeFieldAnalysisStep:
        r"""Adapt to ``pipekit_cycle.AnalysisStep``.

        The step fits the field to the cycle's observations. With a
        ``grid`` it returns the analysis at the window start on it,
        $u(t_0, s_j)$ (`SpaceTimeFieldResult.initial_condition`), the
        state a ``ForwardModel`` steps on; without one it returns the
        [`SpaceTimeFieldResult`][vardax.SpaceTimeFieldResult] itself.

        Args:
            grid: Spatial points $s_j$ of the returned state, ``(N, D)``,
                or ``None``.
            key: Collocation key for a penalty with a ``sampler``; every
                call of the step uses it.

        Returns:
            The adapter.
        """
        if grid is not None:
            grid = jnp.asarray(grid)
            if grid.ndim != 2:
                raise ValueError(f"grid must be 2-D (N, D); got shape {grid.shape}.")
        return _SpaceTimeFieldAnalysisStep(self, grid, key)


class _SpaceTimeFieldAnalysisStep(eqx.Module):
    r"""``pipekit_cycle.AnalysisStep`` adapter for ``SpaceTimeFieldVar``.

    ``obs`` is a [`PointObsBatch`][vardax.PointObsBatch] (one sample),
    whose $\sigma_k$ are the observation errors: ``obs_err_cov`` must be
    ``None`` and ``obs_op`` ``None`` or a `FieldPointObs`. There is no
    background term, so ``forecast`` is not used in the cost; with a grid
    and a ``forecast``, the analysis $u(t_0, \cdot)$ on the grid is
    reshaped to the forecast's shape.
    """

    model: SpaceTimeFieldVar
    grid: Float[Array, "N D"] | None
    key: PRNGKeyArray | None

    def __call__(
        self,
        forecast: Array | None,
        obs: PointObsBatch,
        *,
        obs_op: Any = None,
        obs_err_cov: Any = None,
    ) -> Array | SpaceTimeFieldResult:
        if obs_op is not None and not isinstance(obs_op, FieldPointObs):
            raise TypeError(
                "SpaceTimeFieldVar observes its field at the points: obs_op "
                f"must be None or a FieldPointObs, not {type(obs_op).__name__}."
            )
        if obs_err_cov is not None:
            raise ValueError(
                "A PointObsBatch carries its observation errors in sigma: "
                "pass obs_err_cov=None."
            )
        result = self.model(obs, key=self.key)
        if self.grid is None:
            return result
        xa = result.initial_condition(self.grid)
        if forecast is None:
            return xa
        forecast = jnp.asarray(forecast)
        if forecast.size != xa.size:
            raise ValueError(
                f"forecast has shape {forecast.shape}, but the analysis on the "
                f"grid has {xa.size} values."
            )
        return xa.reshape(forecast.shape).astype(forecast.dtype)


__all__ = ["SpaceTimeFieldResult", "SpaceTimeFieldVar"]
