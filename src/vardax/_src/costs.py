"""Cost functions for 4DVarNet variational data assimilation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from vardax._src.errors import ConfigurationError, MissingInputError, ShapeError

from ._types import Batch1D


def obs_cost_1d(
    state: Float[Array, "B T N"],
    obs: Float[Array, "B T N"],
    mask: Float[Array, "B T N"],
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    r"""Observation cost for 1-D data.

    Computes the masked mean-squared error between the state and observations:

    $$ J_{obs} = \operatorname{mean}\big((m \odot (x - y))^2\big) $$

    with state $x$, observations $y$ and observation mask $m$ (``1`` where
    observed, ``0`` elsewhere). The mean runs over all elements, so it is
    the masked squared error divided by the total element count, not by
    the number of observed locations.

    Algorithm:
        1. Optionally replace ``NaN`` entries of ``obs`` by zero.
        2. Form the masked departure $m \odot (x - y)$.
        3. Return the mean of its square.

    Args:
        state: Current state estimate $x$.
        obs: Observations $y$.
        mask: Binary observation mask $m$.
            A value of ``1`` indicates an observed location.
        nan_to_num: When ``True``, replace ``NaN`` entries in ``obs`` with
            zero (via :func:`jax.numpy.nan_to_num`) *before* masking. Use this
            for real-world geophysical products (e.g. satellite altimetry)
            where gaps are stored as ``NaN`` rather than an explicit binary
            mask. Without it, a ``NaN`` observation poisons both the cost and
            its gradient even at masked-out locations, since ``0 * NaN`` is
            ``NaN`` in JAX. Ported from ``mfourdvar``'s ``ObsOperator.loss``.

    Returns:
        Scalar observation cost.

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
        obs = jnp.nan_to_num(obs)
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def obs_cost_2d(
    state: Float[Array, "B T H W"],
    obs: Float[Array, "B T H W"],
    mask: Float[Array, "B T H W"],
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    r"""Observation cost for 2-D data.

    Computes the masked mean-squared error between the state and observations:

    $$ J_{obs} = \operatorname{mean}\big((m \odot (x - y))^2\big) $$

    with state $x$, observations $y$ and observation mask $m$ (``1`` where
    observed, ``0`` elsewhere), as in [`obs_cost_1d`][vardax.obs_cost_1d].

    Algorithm:
        1. Optionally replace ``NaN`` entries of ``obs`` by zero.
        2. Form the masked departure $m \odot (x - y)$.
        3. Return the mean of its square.

    Args:
        state: Current state estimate $x$.
        obs: Observations $y$.
        mask: Binary observation mask $m$.
        nan_to_num: When ``True``, replace ``NaN`` entries in ``obs`` with
            zero before masking (see [`obs_cost_1d`][vardax.obs_cost_1d]
            for the rationale).

    Returns:
        Scalar observation cost.

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
        obs = jnp.nan_to_num(obs)
    diff = mask * (state - obs)
    return jnp.mean(diff**2)


def prior_cost(
    state: Float[Array, ...],
    prior_reconstruction: Float[Array, ...],
) -> Float[Array, ""]:
    r"""Prior cost based on learned autoencoder reconstruction.

    Computes the mean-squared error between the state and its reconstruction
    through the learned prior (autoencoder):

    $$ J_{prior} = \operatorname{mean}\big((x - \varphi(x))^2\big) $$

    Algorithm:
        1. Take the reconstruction $\varphi(x)$ of the state.
        2. Return the mean squared difference $x - \varphi(x)$.

    Args:
        state: Current state estimate $x$.
        prior_reconstruction: Autoencoder reconstruction $\varphi(x)$ of
            the state.

    Returns:
        Scalar prior cost.

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
    return jnp.mean((state - prior_reconstruction) ** 2)


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
    r"""Compute the variational cost $U(x)$.

    $$ U(x) = \alpha_{obs} J_o + \alpha_{prior} J_p $$

    with $J_o = \operatorname{mean}\big((m \odot (x - y))^2\big)$ and
    $J_p = \operatorname{mean}\big((x - \varphi(x))^2\big)$.

    Algorithm:
        1. Form the masked departure $m \odot (x - y)$ from the batch.
        2. Form the prior departure $x - \varphi(x)$ with ``prior_fn``.
        3. Return the weighted sum of the two mean squares.

    Args:
        x: Current state estimate.
        batch: Observed data batch with ``input`` (``y``) and ``mask``
            (``m``).
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term (default ``0.5``).
        alpha_prior: Weight for the prior term (default ``0.5``).

    Returns:
        Scalar cost value.

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
    obs_diff = batch.mask * (x - batch.input)
    # J_o = mean((m ⊙ (x − y))²)
    observation_term_cost = jnp.mean(obs_diff**2)
    # J_p = mean((x − φ(x))²)
    prior_term_cost = jnp.mean((x - prior_fn(x)) ** 2)
    return alpha_obs * observation_term_cost + alpha_prior * prior_term_cost


def variational_cost_grad(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> Float[Array, ...]:
    r"""Gradient of [`variational_cost`][vardax.variational_cost] w.r.t. ``x``.

    $$ g = \nabla_x U(x) $$

    Algorithm:
        1. Differentiate [`variational_cost`][vardax.variational_cost]
           with respect to its first argument ``x``.
        2. Return the resulting array.

    Args:
        x: Current state estimate.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term.
        alpha_prior: Weight for the prior term.

    Returns:
        The gradient $\nabla_x U(x)$, shaped like ``x``.

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
    r"""Compute the decomposed variational loss.

    Returns individual observation and prior components alongside the
    total, matching the ``ModelLoss`` pattern from the legacy codebase.

    $$ U = \alpha_{obs} J_o + \alpha_{prior} J_p $$

    with $J_o = \operatorname{mean}\big((m \odot (x - y))^2\big)$ and
    $J_p = \operatorname{mean}\big((x - \varphi(x))^2\big)$ as in
    [`variational_cost`][vardax.variational_cost].

    Algorithm:
        1. Compute the weighted observation term.
        2. Compute the weighted prior term.
        3. Return both and their sum.

    Args:
        x: Current state estimate.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term.
        alpha_prior: Weight for the prior term.

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
    obs_diff = batch.mask * (x - batch.input)
    obs = alpha_obs * jnp.mean(obs_diff**2)
    prior = alpha_prior * jnp.mean((x - prior_fn(x)) ** 2)
    return {"obs": obs, "prior": prior, "total": obs + prior}


# ---------------------------------------------------------------------------
# Strong-constraint variational cost (ported from mfourdvar)
# ---------------------------------------------------------------------------


def background_cost(
    initial_state: Float[Array, ...],
    background: Float[Array, ...],
) -> Float[Array, ""]:
    r"""Background cost $\operatorname{mean}\big((x_0 - x_b)^2\big)$.

    Penalises departure of the initial state $x_0$ (the control variable
    in strong-constraint 4DVar) from the background / first guess $x_b$.
    Uses the same mean-squared convention as the other functional costs in
    this module.

    $$ J_b = \operatorname{mean}\big((x_0 - x_b)^2\big) $$

    Algorithm:
        1. Form the departure $x_0 - x_b$.
        2. Return the mean of its square.

    Args:
        initial_state: Initial state estimate $x_0$.
        background: Background state $x_b$.

    Returns:
        Scalar background cost.

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
    return jnp.mean((initial_state - background) ** 2)


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
    r"""Strong-constraint variational cost $U(x_0)$.

    $$ U(x_0) = \alpha_{obs} J_o + \alpha_{bg} J_b $$

    with $J_o = \operatorname{mean}\big((m \odot (\varphi(x_0) - y))^2\big)$
    and $J_b = \operatorname{mean}\big((x_0 - x_b)^2\big)$.

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
        1. Propagate $x_0$ through ``forward_fn(initial_state, ts)`` to get the
           trajectory $\varphi(x_0)$.
        2. Score the trajectory against the observations with the mask.
        3. Add the background term, defaulting to zero when ``background`` is
           omitted.
        4. Return the weighted sum.

    Args:
        initial_state: Initial state $x_0$ (the control variable), whatever
            ``forward_fn`` expects.
        ts: Time coordinates $t$ passed to ``forward_fn``.
        batch: Observed data batch supplying ``input`` (``y``) and ``mask``
            (``m``), matching ``forward_fn``'s trajectory output.
        forward_fn: Callable ``(initial_state, ts) -> trajectory`` producing a state
            trajectory the same shape as ``batch.input``. A
            [`DynTrajectory`][vardax.DynTrajectory] (optionally ``vmap``-ed
            over the batch axis) is the canonical choice.
        background: Background state for the background term. Defaults to
            ``initial_state``
            (i.e. no background penalty).
        alpha_obs: Weight for the observation term (default ``0.5``).
        alpha_bg: Weight for the background term (default ``0.5``).
        nan_to_num: When ``True``, replace ``NaN`` entries in the
            observations with zero before masking (see
            [`obs_cost_1d`][vardax.obs_cost_1d]).

    Returns:
        Scalar strong-constraint variational cost.

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
    trajectory = forward_fn(initial_state, ts)
    obs = batch.input
    if nan_to_num:
        obs = jnp.nan_to_num(obs)
    # J_o = mean((m ⊙ (φ(x₀) − y))²)
    observation_term_cost = jnp.mean((batch.mask * (trajectory - obs)) ** 2)
    # J_b = mean((x₀ − x_b)²)
    background_term_cost = background_cost(initial_state, background)
    return alpha_obs * observation_term_cost + alpha_bg * background_term_cost


# ---------------------------------------------------------------------------
# Physics residual penalty (weak PDE constraint on a continuous field)
# ---------------------------------------------------------------------------


class ResidualPenalty(eqx.Module):
    r"""PDE residual of a continuous field at collocation points, as a cost.

    For a field $f_\theta$ (a callable of one point, e.g. a neural field),
    a PDE operator $\mathcal N$ supplied as ``residual_fn`` and collocation
    points $c_1, \dots, c_{N_c}$,

    $$
    J_{\text{res}}(\theta) = \frac{\lambda}{2 N_c} \sum_{j=1}^{N_c}
        \big\lVert \mathcal N[f_\theta](c_j) \big\rVert^2 .
    $$

    Added to an observation cost it imposes the physics as a weak
    constraint, without a time-stepping model. Residuals are callables built
    on `geonnax.diffops` (see `vardax.examples` for advection–diffusion and
    vorticity-form Navier–Stokes); vardax does not import a PDE library.

    With a ``sampler``, the points are redrawn from a key on every call. If
    each point is drawn from a distribution $p$, the resampled cost is an
    unbiased estimate of
    $\tfrac{\lambda}{2}\,\mathbb E_{c \sim p} \lVert \mathcal N[f_\theta](c) \rVert^2$,
    so its gradient is a stochastic gradient of that expected penalty.

    The field must be smooth enough for the derivatives the residual takes
    (twice differentiable for advection–diffusion, four times for the
    streamfunction form of Navier–Stokes) at every collocation point. A raw
    `jnp.linalg.norm` or `jnp.sqrt` of a quantity that vanishes at a point
    gives NaN gradients there; guard it with the double-`where` idiom
    (`jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, 1.0)), 0.0)`).

    ``residual_fn`` and ``sampler`` are static fields, so they must be
    hashable (functions and closures are); ``collocation`` and ``weight``
    are array leaves, so changing the weight (e.g. annealing $\lambda$)
    does not recompile a jitted cost.

    Algorithm:
        1. Obtain the points $c_j$, from ``collocation`` or by calling
           ``sampler(key)``.
        2. Evaluate the residual $\mathcal N[f_\theta](c_j)$ at every point.
        3. Return $\lambda / (2 N_c)$ times the sum of squared residuals.

    Raises:
        MissingInputError: If neither ``collocation`` nor ``sampler`` is
            given, or if a sampler is set and a call omits ``key``.
        ShapeError: If ``collocation`` is not ``(N_c, D)`` or ``weight`` is
            not a scalar.
        ConfigurationError: If ``weight`` is negative.

    Attributes:
        residual_fn: ``(field, s) -> (R,)``, the residual $\mathcal N[f](s)$
            at one point ``s`` of shape ``(D,)``, where ``field`` maps
            ``(D,) -> (K,)``. A scalar return is treated as ``(1,)``.
        collocation: Fixed collocation points of shape ``(N_c, D)``. May be
            ``None`` when ``sampler`` is given.
        weight: Penalty weight $\lambda \ge 0$, stored as a 0-d array.
        sampler: Optional ``key -> (N_c, D)``; when given, the points are
            resampled on every call and ``collocation`` is not used.

    Examples:
        A residual that is the field itself gives
        $J = \tfrac{\lambda}{2} \lVert f \rVert^2$ for a constant field:

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
            The scalar penalty $J_{\text{res}}$.

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
        weight = eqx.error_if(
            self.weight, self.weight < 0, "weight must be non-negative"
        )
        # N[f](c_j) for every point: (N_c, R)
        residuals = jax.vmap(
            lambda point: jnp.atleast_1d(self.residual_fn(field, point))
        )(points)
        # J = λ/(2 N_c) Σⱼ ‖N[f](c_j)‖²: ()
        return weight / (2 * points.shape[0]) * jnp.sum(residuals**2)
