"""Mean-square costs for 4DVarNet-style variational data assimilation.

𝒥_o and 𝒥_b are masked or plain mean squares, not Gaussian negative log-densities,
and 𝒥_res is the weak-PDE-constraint penalty of a continuous field.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from vardax._src.errors import ConfigurationError, MissingInputError, ShapeError

from ._types import Batch1D


def _masked_mean_square(
    state: Float[Array, ...], obs: Float[Array, ...], mask: Float[Array, ...]
) -> Float[Array, ""]:
    """Masked mean square mean((m ⊙ (x − y))²) shared by the observation costs."""
    # m ⊙ (x − y)
    departure = mask * (state - obs)
    # mean((m ⊙ (x − y))²)
    return jnp.mean(departure**2)


def obs_cost_1d(
    state: Float[Array, "B T N"],
    obs: Float[Array, "B T N"],
    mask: Float[Array, "B T N"],
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    """Observation cost for 1-D data: the masked mean-squared error.

        𝒥_o = mean((m ⊙ (x − y))²)

    The mean runs over all elements, so it is the masked squared error divided by
    the total element count, not by the number of observed locations.

    Algorithm:
        1. Optionally replace ``NaN`` entries of y by zero.
        2. Form the masked departure m ⊙ (x − y).
        3. Return the mean of its square.

    Args:
        state: Current state estimate x.
        obs: Observations y.
        mask: Binary observation mask m.
            A value of ``1`` indicates an observed location.
        nan_to_num: When ``True``, replace ``NaN`` entries in ``obs`` with
            zero (via :func:`jax.numpy.nan_to_num`) *before* masking. Use this
            for real-world geophysical products (e.g. satellite altimetry)
            where gaps are stored as ``NaN`` rather than an explicit binary
            mask. Without it, a ``NaN`` observation poisons both the cost and
            its gradient even at masked-out locations, since ``0 * NaN`` is
            ``NaN`` in JAX. Ported from ``mfourdvar``'s ``ObsOperator.loss``.

    Returns:
        Scalar observation cost 𝒥_o.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import obs_cost_1d
        >>> state = jnp.ones((1, 1, 4))
        >>> obs = jnp.zeros((1, 1, 4))
        >>> mask = jnp.ones((1, 1, 4))
        >>> float(obs_cost_1d(state, obs, mask))
        1.0

        Masked-out locations do not contribute, even when the observation
        there is ``NaN`` (with ``nan_to_num=True``).

        >>> obs = jnp.array([[[0.0, jnp.nan, 0.0, 0.0]]])
        >>> mask = jnp.array([[[1.0, 0.0, 1.0, 1.0]]])
        >>> float(obs_cost_1d(state, obs, mask, nan_to_num=True))
        0.75

        The cost is differentiable in the state.

        >>> import jax
        >>> jax.grad(obs_cost_1d)(state, jnp.zeros((1, 1, 4)), mask).shape
        (1, 1, 4)
    """
    if nan_to_num:
        # a NaN at a masked-out location would still poison 0 ⊙ NaN
        obs = jnp.nan_to_num(obs)
    return _masked_mean_square(state, obs, mask)


def obs_cost_2d(
    state: Float[Array, "B T H W"],
    obs: Float[Array, "B T H W"],
    mask: Float[Array, "B T H W"],
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    """Observation cost for 2-D data: the masked mean-squared error.

        𝒥_o = mean((m ⊙ (x − y))²)

    The mask m is ``1`` where observed and ``0`` elsewhere, as in
    [`obs_cost_1d`][vardax.obs_cost_1d].

    Algorithm:
        1. Optionally replace ``NaN`` entries of y by zero.
        2. Form the masked departure m ⊙ (x − y).
        3. Return the mean of its square.

    Args:
        state: Current state estimate x.
        obs: Observations y.
        mask: Binary observation mask m.
        nan_to_num: When ``True``, replace ``NaN`` entries in ``obs`` with
            zero before masking (see [`obs_cost_1d`][vardax.obs_cost_1d]
            for the rationale).

    Returns:
        Scalar observation cost 𝒥_o.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import obs_cost_2d
        >>> state = jnp.ones((1, 1, 2, 2))
        >>> obs = jnp.zeros((1, 1, 2, 2))
        >>> float(obs_cost_2d(state, obs, jnp.ones((1, 1, 2, 2))))
        1.0

        With nothing observed the cost is zero.

        >>> float(obs_cost_2d(state, obs, jnp.zeros((1, 1, 2, 2))))
        0.0
    """
    if nan_to_num:
        # a NaN at a masked-out location would still poison 0 ⊙ NaN
        obs = jnp.nan_to_num(obs)
    return _masked_mean_square(state, obs, mask)


def prior_cost(
    state: Float[Array, ...],
    prior_reconstruction: Float[Array, ...],
) -> Float[Array, ""]:
    """Prior cost of a learned autoencoder: the reconstruction mean square.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)
        𝒥_p = mean((x − x̂)²)

    Algorithm:
        1. Take the reconstruction x̂ = g_ae(x; θ) of the state.
        2. Return the mean square of the departure x − x̂.

    Args:
        state: Current state estimate x.
        prior_reconstruction: Autoencoder reconstruction x̂ of the state.

    Returns:
        Scalar prior cost 𝒥_p.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import prior_cost
        >>> x = jnp.ones((2, 3))
        >>> float(prior_cost(x, x))
        0.0

        A reconstruction that is off by two everywhere costs four.

        >>> float(prior_cost(x, x + 2.0))
        4.0
    """
    # x − x̂
    departure = state - prior_reconstruction
    # 𝒥_p = mean((x − x̂)²)
    return jnp.mean(departure**2)


# ---------------------------------------------------------------------------
# Structured variational cost utilities
# ---------------------------------------------------------------------------


def variational_cost(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> Float[Array, ""]:
    """Variational cost, the weighted sum of observation and prior mean squares.

        𝒥(x) = α_o 𝒥_o + α_p 𝒥_p
        𝒥_o = mean((m ⊙ (x − y))²),   𝒥_p = mean((x − x̂)²),   x̂ = g_ae(x; θ)

    Algorithm:
        1. Form the masked departure m ⊙ (x − y) from the batch.
        2. Form the prior departure x − x̂ with ``prior_fn``.
        3. Return the weighted sum of the two mean squares.

    Args:
        x: Current state estimate x.
        batch: Observed data batch with ``input`` (y) and ``mask`` (m).
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        alpha_obs: Weight α_o of the observation term (default ``0.5``).
        alpha_prior: Weight α_p of the prior term (default ``0.5``).

    Returns:
        Scalar cost 𝒥.

    Examples:
        With the trivial [`IdentityPrior`][vardax.IdentityPrior] the
        prior term vanishes, leaving the weighted observation MSE.

        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.Batch1D(input=jnp.zeros((1, 2, 4)), mask=jnp.ones((1, 2, 4)))
        >>> x = jnp.ones((1, 2, 4))
        >>> float(vardax.variational_cost(x, batch, vardax.IdentityPrior()))
        0.5

        Setting ``alpha_obs=0`` removes the observation term.

        >>> cost = vardax.variational_cost(
        ...     x, batch, vardax.IdentityPrior(), alpha_obs=0.0
        ... )
        >>> float(cost)
        0.0

        The gradient has the shape of the state.

        >>> import jax
        >>> g = jax.grad(vardax.variational_cost)(x, batch, vardax.IdentityPrior())
        >>> g.shape
        (1, 2, 4)
    """
    # 𝒥_o = mean((m ⊙ (x − y))²)
    observation_cost = _masked_mean_square(x, batch.input, batch.mask)
    # x̂ = g_ae(x; θ)
    reconstruction = prior_fn(x)
    # 𝒥_p = mean((x − x̂)²)
    prior_departure = x - reconstruction
    prior_term_cost = jnp.mean(prior_departure**2)
    # 𝒥 = α_o 𝒥_o + α_p 𝒥_p
    return alpha_obs * observation_cost + alpha_prior * prior_term_cost


def variational_cost_grad(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> Float[Array, ...]:
    """Gradient of [`variational_cost`][vardax.variational_cost] w.r.t. ``x``.

        g = ∇ₓ 𝒥(x)

    Algorithm:
        1. Differentiate [`variational_cost`][vardax.variational_cost]
           with respect to its first argument ``x``.
        2. Return the resulting array.

    Args:
        x: Current state estimate x.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        alpha_obs: Weight α_o of the observation term.
        alpha_prior: Weight α_p of the prior term.

    Returns:
        The gradient ∇ₓ 𝒥, shaped like ``x``.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.Batch1D(input=jnp.zeros((1, 2, 4)), mask=jnp.ones((1, 2, 4)))
        >>> x = jnp.ones((1, 2, 4))
        >>> g = vardax.variational_cost_grad(x, batch, vardax.IdentityPrior())
        >>> g.shape
        (1, 2, 4)

        It matches ``jax.grad`` of the cost.

        >>> import jax
        >>> ref = jax.grad(vardax.variational_cost)(x, batch, vardax.IdentityPrior())
        >>> bool(jnp.allclose(g, ref))
        True
    """
    return jax.grad(variational_cost)(x, batch, prior_fn, alpha_obs, alpha_prior)


def decomposed_loss(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> dict[str, Float[Array, ""]]:
    """Decomposed variational loss: the weighted terms alongside their sum.

        𝒥 = α_o 𝒥_o + α_p 𝒥_p

    The terms 𝒥_o and 𝒥_p are those of [`variational_cost`][vardax.variational_cost].
    The dictionary matches the ``ModelLoss`` pattern from the legacy codebase.

    Algorithm:
        1. Compute the weighted observation term α_o 𝒥_o.
        2. Compute the weighted prior term α_p 𝒥_p.
        3. Return both and their sum.

    Args:
        x: Current state estimate x.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x̂``, the learned prior g_ae(·; θ).
        alpha_obs: Weight α_o of the observation term.
        alpha_prior: Weight α_p of the prior term.

    Returns:
        Dictionary with keys ``"obs"``, ``"prior"``, and ``"total"``.

    Examples:
        >>> import jax.numpy as jnp, vardax
        >>> batch = vardax.Batch1D(input=jnp.zeros((1, 2, 4)), mask=jnp.ones((1, 2, 4)))
        >>> x = jnp.ones((1, 2, 4))
        >>> parts = vardax.decomposed_loss(x, batch, vardax.IdentityPrior())
        >>> sorted(parts)
        ['obs', 'prior', 'total']
        >>> float(parts["obs"]), float(parts["prior"])
        (0.5, 0.0)

        The total equals [`variational_cost`][vardax.variational_cost].

        >>> total = vardax.variational_cost(x, batch, vardax.IdentityPrior())
        >>> bool(jnp.allclose(parts["total"], total))
        True
    """
    # α_o 𝒥_o = α_o mean((m ⊙ (x − y))²)
    observation_cost = _masked_mean_square(x, batch.input, batch.mask)
    obs = alpha_obs * observation_cost
    # x̂ = g_ae(x; θ)
    reconstruction = prior_fn(x)
    # α_p 𝒥_p = α_p mean((x − x̂)²)
    prior_departure = x - reconstruction
    prior = alpha_prior * jnp.mean(prior_departure**2)
    # 𝒥 = α_o 𝒥_o + α_p 𝒥_p
    return {"obs": obs, "prior": prior, "total": obs + prior}


# ---------------------------------------------------------------------------
# Strong-constraint variational cost (ported from mfourdvar)
# ---------------------------------------------------------------------------


def background_cost(
    initial_state: Float[Array, ...],
    background: Float[Array, ...],
) -> Float[Array, ""]:
    """Background cost: the mean square of the departure from the first guess.

        𝒥_b = mean((x₀ − μ_x)²)

    Penalises departure of the initial state x₀ (the control variable in
    strong-constraint 4DVar) from the background / first guess μ_x. It is a mean
    square like the other functional costs in this module, not a Gaussian
    negative log-density.

    Algorithm:
        1. Form the departure x₀ − μ_x.
        2. Return the mean of its square.

    Args:
        initial_state: Initial state estimate x₀.
        background: Background state μ_x.

    Returns:
        Scalar background cost 𝒥_b.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import background_cost
        >>> x0 = jnp.ones(4)
        >>> float(background_cost(x0, x0))
        0.0

        A uniform departure of 0.5 costs a quarter.

        >>> float(background_cost(x0, x0 - 0.5))
        0.25
    """
    # x₀ − μ_x
    departure = initial_state - background
    # 𝒥_b = mean((x₀ − μ_x)²)
    return jnp.mean(departure**2)


def strong_variational_cost(
    initial_state: Float[Array, ...],
    ts: Float[Array, " T"],
    batch: Batch1D,
    forward_fn: Callable[..., Any],
    *,
    background: Float[Array, ...] | None = None,
    alpha_obs: float = 0.5,
    alpha_bg: float = 0.5,
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    """Strong-constraint variational cost of the initial state.

        𝒥(x₀) = α_o 𝒥_o + α_b 𝒥_b
        𝒥_o = mean((m ⊙ (f_{1:T}(x₀) − y))²),   𝒥_b = mean((x₀ − μ_x)²)

    The dynamical model is enforced as a **hard** constraint: the initial
    state ``initial_state`` is propagated through the dynamics
    ``forward_fn(initial_state, ts)``
    and only the resulting trajectory is scored against the observations.
    This differs from the weak / soft-constraint
    [`variational_cost`][vardax.variational_cost], where the model appears
    as an *additive* prior penalty and the whole state field is free.

    Ported from ``mfourdvar``'s ``StrongVarCost``, adapted to a functional
    form that mirrors [`variational_cost`][vardax.variational_cost].

    Algorithm:
        1. Propagate x₀ through ``forward_fn(initial_state, ts)`` to get the
           trajectory f_{1:T}(x₀).
        2. Score the trajectory against the observations with the mask.
        3. Add the background term, defaulting to zero when ``background`` is
           omitted.
        4. Return the weighted sum.

    Args:
        initial_state: Initial state x₀ (the control variable), whatever
            ``forward_fn`` expects.
        ts: Time coordinates t passed to ``forward_fn``.
        batch: Observed data batch supplying ``input`` (y) and ``mask`` (m),
            matching ``forward_fn``'s trajectory output.
        forward_fn: Callable ``(initial_state, ts) -> f_{1:T}(x₀)`` producing a state
            trajectory the same shape as ``batch.input``. A
            [`DynTrajectory`][vardax.DynTrajectory] (optionally ``vmap``-ed
            over the batch axis) is the canonical choice.
        background: Background state μ_x for the background term. Defaults to
            ``initial_state`` (i.e. no background penalty).
        alpha_obs: Weight α_o of the observation term (default ``0.5``).
        alpha_bg: Weight α_b of the background term (default ``0.5``).
        nan_to_num: When ``True``, replace ``NaN`` entries in the
            observations with zero before masking (see
            [`obs_cost_1d`][vardax.obs_cost_1d]).

    Returns:
        Scalar strong-constraint variational cost 𝒥.

    Examples:
        A persistence model repeats the initial state over the window.

        >>> import jax.numpy as jnp, vardax
        >>> ts = jnp.arange(3.0)
        >>> forward = lambda x0, ts: jnp.broadcast_to(x0[:, None], (1, 3, 4))
        >>> batch = vardax.Batch1D(input=jnp.zeros((1, 3, 4)), mask=jnp.ones((1, 3, 4)))
        >>> x0 = jnp.ones((1, 4))
        >>> float(vardax.strong_variational_cost(x0, ts, batch, forward))
        0.5

        A background away from ``initial_state`` adds the background term.

        >>> cost = vardax.strong_variational_cost(
        ...     x0, ts, batch, forward, background=x0 - 1.0
        ... )
        >>> float(cost)
        1.0

        The cost is differentiable in the control.

        >>> import jax
        >>> g = jax.grad(vardax.strong_variational_cost)(x0, ts, batch, forward)
        >>> g.shape
        (1, 4)
    """
    if background is None:
        background = initial_state
    # f_{1:T}(x₀): the hard constraint, only the rollout meets the observations
    trajectory = forward_fn(initial_state, ts)
    obs = batch.input
    if nan_to_num:
        obs = jnp.nan_to_num(obs)
    # 𝒥_o = mean((m ⊙ (f_{1:T}(x₀) − y))²)
    observation_cost = _masked_mean_square(trajectory, obs, batch.mask)
    # 𝒥_b = mean((x₀ − μ_x)²)
    background_term_cost = background_cost(initial_state, background)
    # 𝒥 = α_o 𝒥_o + α_b 𝒥_b
    return alpha_obs * observation_cost + alpha_bg * background_term_cost


# ---------------------------------------------------------------------------
# Physics residual penalty (weak PDE constraint on a continuous field)
# ---------------------------------------------------------------------------


class ResidualPenalty(eqx.Module):
    """PDE residual of a continuous field at collocation points, as a cost.

        𝒥_res(θ) = β_res / (2 N_c) ∑ⱼ ‖r(cⱼ; θ)‖²
        r(s; θ) = ``residual_fn(field, s)``

    Here r(s; θ) is the PDE residual of the field x(·; θ) (a callable of one point,
    e.g. a neural field) at the point s, cⱼ are the N_c collocation points and
    β_res is the ``weight``. Added to an observation cost it imposes the physics as
    a weak constraint, without a time-stepping model. Residuals are callables built
    on `geonnax.diffops` (see `vardax.examples` for advection–diffusion and
    vorticity-form Navier–Stokes); vardax does not import a PDE library.

    With a ``sampler``, the points are redrawn from a key on every call. If each
    point is drawn from a distribution p, the resampled cost is an unbiased
    estimate of ½ β_res 𝔼_{c ∼ p} ‖r(c; θ)‖², so its gradient is a stochastic
    gradient of that expected penalty.

    The field must be smooth enough for the derivatives the residual takes
    (twice differentiable for advection–diffusion, four times for the
    streamfunction form of Navier–Stokes) at every collocation point. A raw
    `jnp.linalg.norm` or `jnp.sqrt` of a quantity that vanishes at a point
    gives NaN gradients there; guard it with the double-`where` idiom
    (`jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, 1.0)), 0.0)`).

    ``residual_fn`` and ``sampler`` are static fields, so they must be
    hashable (functions and closures are); ``collocation`` and ``weight``
    are array leaves, so changing the weight (e.g. annealing β_res)
    does not recompile a jitted cost.

    Algorithm:
        1. Obtain the points cⱼ, from ``collocation`` or by calling
           ``sampler(key)``.
        2. Evaluate the residual r(cⱼ; θ) at every point.
        3. Return β_res / (2 N_c) times the sum of squared residual norms.

    Raises:
        MissingInputError: If neither ``collocation`` nor ``sampler`` is
            given, or if a sampler is set and a call omits ``key``.
        ShapeError: If ``collocation`` is not ``(N_c, D)`` or ``weight`` is
            not a scalar.
        ConfigurationError: If ``weight`` is negative.

    Attributes:
        residual_fn: ``(field, s) -> (R,)``, the residual r(s; θ) of the field
            x(·; θ) at one point ``s`` of shape ``(D,)``, where ``field`` maps
            ``(D,) -> (K,)``. A scalar return is treated as ``(1,)``.
        collocation: Fixed collocation points cⱼ of shape ``(N_c, D)``. May be
            ``None`` when ``sampler`` is given.
        weight: Penalty weight β_res ≥ 0, stored as a 0-d array.
        sampler: Optional ``key -> (N_c, D)``; when given, the points are
            resampled on every call and ``collocation`` is not used.

    Examples:
        A residual that is the field itself gives
        𝒥_res = ½ β_res ‖x‖² for a constant field:

        >>> import jax.numpy as jnp
        >>> import vardax
        >>> pen = vardax.ResidualPenalty(
        ...     residual_fn=lambda f, s: f(s),
        ...     collocation=jnp.zeros((5, 3)),
        ...     weight=0.5,
        ... )
        >>> float(pen(lambda s: jnp.array([1.0, 2.0])))
        1.25

        With a sampler, pass a key on every call:

        >>> import jax
        >>> pen = vardax.ResidualPenalty(
        ...     residual_fn=lambda f, s: f(s),
        ...     collocation=None,
        ...     weight=1.0,
        ...     sampler=lambda key: jax.random.uniform(key, (64, 3)),
        ... )
        >>> float(pen(lambda s: s[:1], key=jax.random.PRNGKey(0))) > 0
        True

        A penalty with neither collocation points nor a sampler is rejected.

        >>> vardax.ResidualPenalty(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     residual_fn=lambda f, s: f(s), collocation=None, weight=1.0
        ... )
        Traceback (most recent call last):
        ...
        MissingInputError: ResidualPenalty needs collocation points or a sampler.
    """

    residual_fn: Callable[..., Float[Array, " R"]] = eqx.field(static=True)
    collocation: Float[Array, "N D"] | None
    weight: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    sampler: Callable[[PRNGKeyArray], Float[Array, "N D"]] | None = eqx.field(
        static=True, default=None
    )

    def __check_init__(self) -> None:
        if self.collocation is None and self.sampler is None:
            raise MissingInputError(
                "ResidualPenalty needs collocation points or a sampler."
            )
        if self.collocation is not None and jnp.ndim(self.collocation) != 2:
            raise ShapeError(
                "collocation must have shape (N_c, D); got shape "
                f"{jnp.shape(self.collocation)}."
            )
        if jnp.ndim(self.weight) != 0:
            raise ShapeError(
                f"weight must be a scalar; got shape {jnp.shape(self.weight)}."
            )
        try:
            value = float(self.weight)
        except jax.errors.ConcretizationTypeError:
            return  # built under a transform: the sign is not known yet
        if value < 0:
            raise ConfigurationError(f"weight must be non-negative; got {value}.")

    def __call__(
        self,
        field: Callable[[Float[Array, " D"]], Float[Array, " K"]],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Evaluate the penalty for ``field``.

        Args:
            field: The field, a callable of one point ``(D,) -> (K,)``.
            key: PRNG key for the sampler. Required when ``sampler`` is set;
                ignored otherwise.

        Returns:
            The scalar penalty 𝒥_res.

        Raises:
            MissingInputError: If a sampler is set and ``key`` is ``None``.
            ShapeError: If the points are not of shape ``(N_c, D)``.
        """
        if self.sampler is not None:
            if key is None:
                raise MissingInputError("ResidualPenalty with a sampler needs a key.")
            points = self.sampler(key)
        else:
            points = self.collocation
        assert points is not None  # guaranteed by __check_init__
        if points.ndim != 2:
            raise ShapeError(
                f"collocation points must have shape (N_c, D); got {points.shape}."
            )

        # __check_init__ skips the sign check when the weight is traced
        residual_weight = eqx.error_if(
            self.weight, self.weight < 0, "weight must be non-negative"
        )
        # r(cⱼ; θ) for every point: (N_c, R)
        residuals = jax.vmap(
            lambda point: jnp.atleast_1d(self.residual_fn(field, point))
        )(points)
        # β_res / (2 N_c)
        scale = residual_weight / (2 * points.shape[0])
        # ∑ⱼ ‖r(cⱼ; θ)‖²
        squared_norms = jnp.sum(residuals**2)
        # 𝒥_res = β_res / (2 N_c) ∑ⱼ ‖r(cⱼ; θ)‖²
        return scale * squared_norms
