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

    $$
    J_{obs} = \frac{1}{|\Omega|} \sum_{i \in \Omega} (x_i - y_i)^2
    $$

    where $\Omega$ is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, N)``.
        obs: Observations of shape ``(B, T, N)``.
        mask: Binary observation mask of shape ``(B, T, N)``.
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

    $$
    J_{obs} = \frac{1}{|\Omega|} \sum_{i \in \Omega} (x_i - y_i)^2
    $$

    where $\Omega$ is the set of observed locations (``mask == 1``).

    Args:
        state: Current state estimate of shape ``(B, T, H, W)``.
        obs: Observations of shape ``(B, T, H, W)``.
        mask: Binary observation mask of shape ``(B, T, H, W)``.
        nan_to_num: When ``True``, replace ``NaN`` entries in ``obs`` with
            zero before masking (see :func:`obs_cost_1d` for the rationale).

    Returns:
        Scalar observation cost.
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

    $$
    J_{prior} = \|x - \varphi(x)\|^2
    $$

    Args:
        state: Current state estimate of arbitrary shape.
        prior_reconstruction: Autoencoder reconstruction of the state,
            same shape as ``state``.

    Returns:
        Scalar prior cost.
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

    $$
    U(x) = \alpha_{obs} \|m \odot (x - y)\|^2
          + \alpha_{prior} \|x - \varphi(x)\|^2
    $$

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
    """
    obs_diff = batch.mask * (x - batch.input)
    j_obs = jnp.mean(obs_diff**2)
    j_prior = jnp.mean((x - prior_fn(x)) ** 2)
    return alpha_obs * j_obs + alpha_prior * j_prior


def variational_cost_grad(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> Float[Array, ...]:
    """Gradient of [`variational_cost`][vardax.variational_cost] w.r.t. ``x``.

    Args:
        x: Current state estimate.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term.
        alpha_prior: Weight for the prior term.

    Returns:
        Gradient array with the same shape as ``x``.
    """
    return jax.grad(variational_cost)(x, batch, prior_fn, alpha_obs, alpha_prior)


def decomposed_loss(
    x: Float[Array, ...],
    batch: Batch1D,
    prior_fn: Callable[..., Any],
    alpha_obs: float = 0.5,
    alpha_prior: float = 0.5,
) -> dict[str, Float[Array, ""]]:
    """Compute the decomposed variational loss.

    Returns individual observation and prior components alongside the
    total, matching the ``ModelLoss`` pattern from the legacy codebase.

    Args:
        x: Current state estimate.
        batch: Observed data batch.
        prior_fn: Callable ``x -> x_prior``.
        alpha_obs: Weight for the observation term.
        alpha_prior: Weight for the prior term.

    Returns:
        Dictionary with keys ``"obs"``, ``"prior"``, and ``"total"``.
    """
    obs_diff = batch.mask * (x - batch.input)
    obs = alpha_obs * jnp.mean(obs_diff**2)
    prior = alpha_prior * jnp.mean((x - prior_fn(x)) ** 2)
    return {"obs": obs, "prior": prior, "total": obs + prior}


# ---------------------------------------------------------------------------
# Strong-constraint variational cost (ported from mfourdvar)
# ---------------------------------------------------------------------------


def background_cost(
    x0: Float[Array, ...],
    xb: Float[Array, ...],
) -> Float[Array, ""]:
    r"""Background cost $\|x_0 - x_b\|^2$.

    Penalises departure of the initial state $x_0$ (the control variable
    in strong-constraint 4DVar) from the background / first guess $x_b$.
    Uses the same mean-squared convention as the other functional costs in
    this module.

    Args:
        x0: Initial state estimate of arbitrary shape.
        xb: Background state, same shape as ``x0``.

    Returns:
        Scalar background cost.
    """
    return jnp.mean((x0 - xb) ** 2)


def strong_variational_cost(
    x0: Float[Array, ...],
    ts: Float[Array, " T"],
    batch: Batch1D,
    forward_fn: Callable[..., Any],
    *,
    xb: Float[Array, ...] | None = None,
    alpha_obs: float = 0.5,
    alpha_bg: float = 0.5,
    nan_to_num: bool = False,
) -> Float[Array, ""]:
    r"""Strong-constraint variational cost $U(x_0)$.

    $$
    U(x_0) = \alpha_{obs}\,\|m \odot (\varphi(x_0) - y)\|^2
           + \alpha_{bg}\,\|x_0 - x_b\|^2 .
    $$

    The dynamical model is enforced as a **hard** constraint: the initial
    state ``x0`` is propagated through the dynamics ``forward_fn(x0, ts)``
    and only the resulting trajectory is scored against the observations.
    This differs from the weak / soft-constraint
    [`variational_cost`][vardax.variational_cost], where the model appears
    as an *additive* prior penalty and the whole state field is free.

    Ported from ``mfourdvar``'s ``StrongVarCost``, adapted to a functional
    form that mirrors [`variational_cost`][vardax.variational_cost].

    Args:
        x0: Initial state (control variable). Whatever shape ``forward_fn``
            expects — e.g. ``(B, N)`` for a batched 1-D system.
        ts: Time coordinates of shape ``(T,)`` passed to ``forward_fn``.
        batch: Observed data batch supplying ``input`` (``y``) and ``mask``
            (``m``), each of shape matching ``forward_fn``'s trajectory output
            (e.g. ``(B, T, N)``).
        forward_fn: Callable ``(x0, ts) -> trajectory`` producing a state
            trajectory the same shape as ``batch.input``. A
            [`DynTrajectory`][vardax.DynTrajectory] (optionally ``vmap``-ed
            over the batch axis) is the canonical choice.
        xb: Background state for the background term. Defaults to ``x0``
            (i.e. no background penalty).
        alpha_obs: Weight for the observation term (default ``0.5``).
        alpha_bg: Weight for the background term (default ``0.5``).
        nan_to_num: When ``True``, replace ``NaN`` entries in the
            observations with zero before masking (see
            [`obs_cost_1d`][vardax.obs_cost_1d]).

    Returns:
        Scalar strong-constraint variational cost.
    """
    if xb is None:
        xb = x0
    traj = forward_fn(x0, ts)
    obs = batch.input
    if nan_to_num:
        obs = jnp.nan_to_num(obs)
    j_obs = jnp.mean((batch.mask * (traj - obs)) ** 2)
    j_bg = background_cost(x0, xb)
    return alpha_obs * j_obs + alpha_bg * j_bg


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
    unbiased estimate of $\tfrac{\lambda}{2}\,\mathbb E_{c \sim p}
    \lVert \mathcal N[f_\theta](c) \rVert^2$, so its gradient is a
    stochastic gradient of that expected penalty.

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
            ValueError: If a sampler is set and ``key`` is ``None``, or the
                points are not of shape ``(N_c, D)``.
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
        r = jax.vmap(lambda c: jnp.atleast_1d(self.residual_fn(field, c)))(points)
        # J = λ/(2 N_c) Σⱼ ‖N[f](c_j)‖²: ()
        return weight / (2 * points.shape[0]) * jnp.sum(r**2)
