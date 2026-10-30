r"""Gauss–Newton posterior precision for every analysis family, matrix-free.

The Laplace and Gauss–Newton adapters both need the precision of the
analysis at the MAP. What enters it depends on how the observations
depend on the control:

- **Single time** (``OptimalInterpolation``, ``ThreeDVar``):
  P*⁻¹ = B⁻¹ + H'ᵀ R⁻¹ H'.
- **Strong 4DVar** (``StrongFourDVar``, ``IncrementalFourDVar``,
  ``ReducedBasisFourDVar`` without a basis): the window's observations
  reach x₀ through the model, so with G(x₀) = [mₜ ⊙ Hₜ(Mₜ(x₀))]ₜ,
  P*⁻¹ = B⁻¹ + G'ᵀ R⁻¹ G' = B⁻¹ + ∑ₜ (H'ₜ M'ₜ)ᵀ R⁻¹ (H'ₜ M'ₜ).
- **Weak 4DVar** (``WeakFourDVar``): the control is z = (x₀, η₁ … η_T);
  its precision is blockdiag(B⁻¹, Q⁻¹ …) + G_z'ᵀ R⁻¹ G_z', and the
  covariance of x₀ is the x₀ block of its inverse.
- **Linear control** (``parameterization=LinearParameterization(basis,
  x_b)``, or ``ReducedBasisFourDVar(basis=...)``): x = x* + Φθ with
  pen(θ) = ½ θᵀ Q⁻¹ θ, so the precision lives in coefficient space,
  P_θ = (Q⁻¹ + Φᵀ G'ᵀ R⁻¹ G' Φ)⁻¹, and the state covariance is
  P_x = Φ P_θ Φᵀ. Single time and strong 4DVar only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gaussx
import jax
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.errors import MissingInputError, ShapeError, UnsupportedInputError
from vardax._src.models._rollout import accepts_mask, observed_trajectory
from vardax._src.utils.shapes import as_shape


def inverse_operator(
    operator: lx.AbstractLinearOperator, solver: lx.AbstractLinearSolver
) -> lx.AbstractLinearOperator:
    """Lazy PSD inverse — delegates to ``gaussx.inv``.

    Structured operators (diagonal, Kronecker, block-diagonal, low-rank)
    invert in closed form; everything else solves with ``solver``, never
    materialising the inverse. The result is re-wrapped as a
    ``FunctionLinearOperator`` so lineax can linearise it inside a composed
    precision that is itself solved (gaussx's ``InverseOperator`` does not
    register ``lx.linearise``).

    Args:
        operator: Symmetric positive-definite operator A.
        solver: Linear solver for operators without a closed-form inverse.

    Returns:
        A⁻¹ as a lazy operator.
    """
    tagged = lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
    inverse = gaussx.inv(tagged, solver=solver)
    return lx.FunctionLinearOperator(inverse.mv, operator.in_structure())


def has_rollout(model: Any) -> bool:
    """Whether ``model`` assimilates a window through a forward model."""
    return hasattr(model, "forward") and (
        hasattr(model, "_rollout") or hasattr(model, "_rollout_with_error")
    )


def window_mask(batch: Any) -> Float[Array, "T_plus_1 N"]:
    """The ``(T + 1, N)`` observation mask of a single-sample window batch.

    Args:
        batch: The ``Batch1D`` the analysis was computed from, either
            unbatched ``(T + 1, N)`` or with a leading sample axis of size 1.

    Returns:
        The mask mₜ of that one window.

    Raises:
        ValueError: If the batch is a point batch, is missing, or holds
            more than one sample.
    """
    if isinstance(batch, PointObsBatch):
        raise UnsupportedInputError(
            "The posterior adapters do not support PointObsBatch windows yet; "
            "pass a gridded Batch1D."
        )
    if not isinstance(batch, Batch1D):
        raise UnsupportedInputError(
            "A 4DVar posterior needs the window's Batch1D (for its length and "
            f"mask); got {type(batch).__name__}."
        )
    mask = batch.mask
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ShapeError(
                f"The batch holds {mask.shape[0]} windows but the analysis is "
                "one state; pass that sample's batch, e.g. "
                "jax.tree.map(lambda a: a[i:i + 1], batch)."
            )
        mask = mask[0]
    return mask


def _model_covariance(
    given: lx.AbstractLinearOperator | None, model: Any, name: str
) -> lx.AbstractLinearOperator:
    """The adapter's covariance if given, else the model's of the same name."""
    if given is not None:
        return given
    from_model = getattr(model, name, None)
    if from_model is None:
        raise MissingInputError(
            f"No {name} given to the adapter and {type(model).__name__} has "
            f"none; pass {name}= to the adapter."
        )
    return from_model


def _check_state_space_control(model: Any) -> None:
    # a reduced-basis or parameterized control has its own prior, not B
    if getattr(model, "basis", None) is not None or (
        getattr(model, "parameterization", None) is not None
    ):
        raise UnsupportedInputError(
            "The posterior adapters build the state-space posterior with B; "
            f"this {type(model).__name__} has a basis= or parameterization= "
            "control, whose prior is not B. Use EnsembleCovariance, or the "
            "posterior of the control coefficients."
        )


def _linear_control(model: Any) -> Any:
    """The `ReducedBasis` of a linear control, ``None`` for a state control.

    Args:
        model: The underlying analysis model.

    Returns:
        ``parameterization.basis`` for a
        [`LinearParameterization`][vardax.LinearParameterization], the
        model's ``basis`` for ``ReducedBasisFourDVar(basis=...)``, or
        ``None`` when the control is the state itself.

    Raises:
        ValueError: For any other parameterization: its decode is not
            affine, so Φ P_θ Φᵀ is not its posterior covariance.
    """
    from vardax._src.parameterization import LinearParameterization

    parameterization = getattr(model, "parameterization", None)
    if parameterization is None:
        return getattr(model, "basis", None)
    if isinstance(parameterization, LinearParameterization):
        return parameterization.basis
    raise UnsupportedInputError(
        "The posterior adapters support a state control or a "
        f"LinearParameterization; this {type(model).__name__} has a "
        f"{type(parameterization).__name__}, whose decode is not affine. Use "
        "EnsembleCovariance, or the posterior of the control coefficients."
    )


def _masked_tangent(
    tangent: lx.AbstractLinearOperator, batch: Any
) -> lx.AbstractLinearOperator:
    """M H' for a single-time ``Batch1D``; H' unchanged without one.

    A batch of several fully observed windows is accepted (its mask is all
    ones); several windows with missing entries raise, since the mask of
    the analysed sample is ambiguous. Under tracing that check cannot run
    and the mask is not applied.

    ``OptimalInterpolation`` and ``ThreeDVar`` observe M ⊙ H(x), but
    ``obs_op.linearize`` knows no mask (``MaskedIdentity.linearize`` is the
    identity). The mask is composed as its own operator, so ``tangent``
    keeps its own transpose (a materialised Jacobian stays a dense
    product).
    """
    if not isinstance(batch, Batch1D):
        return tangent
    if batch.mask.ndim == 3 and batch.mask.shape[0] != 1:
        # several windows: only a fully observed batch says nothing about
        # this sample's mask (the single-time adapters always accepted it)
        if isinstance(batch.mask, jax.core.Tracer) or bool(jnp.all(batch.mask == 1)):
            return tangent
        raise ShapeError(
            f"The batch holds {batch.mask.shape[0]} windows with missing "
            "observations but the analysis is one state; pass that sample's "
            "batch, e.g. jax.tree.map(lambda a: a[i:i + 1], batch)."
        )
    mask = window_mask(batch)
    obs_structure = tangent.out_structure()
    if mask.shape == (1, *obs_structure.shape):
        # an (S, N) batch of (N,) states: drop the sample axis, as __call__ does
        mask = mask[0]
    if mask.shape != obs_structure.shape:
        raise ShapeError(
            f"The batch mask has shape {mask.shape} but the observation "
            f"operator returns {obs_structure.shape}."
        )
    mask_op = lx.FunctionLinearOperator(
        lambda observed: mask * observed, obs_structure, lx.symmetric_tag
    )
    return mask_op @ tangent


def _marginalises_masked_obs(model: Any) -> bool:
    """Whether ``model``'s cost weights masked residuals by (R_oo)⁻¹.

    ``ReducedBasisFourDVar`` solves each time against
    Mₜ R Mₜ + (I − Mₜ), so a missing observation drops out of the
    likelihood exactly even when R is correlated; the other rollout
    models apply R⁻¹ to the zeroed residual.
    """
    from vardax._src.models.reduced_basis_fourdvar import ReducedBasisFourDVar

    return isinstance(model, ReducedBasisFourDVar)


def _basis_operator(
    basis: Any, analysis: Float[Array, ...]
) -> lx.FunctionLinearOperator:
    """Φ: θ ↦ the state increment Φ(0) θ, shaped like ``analysis``."""
    coefficients = jax.ShapeDtypeStruct((basis.nbasis,), analysis.dtype)
    return lx.FunctionLinearOperator(
        lambda theta: as_shape(basis.operg(0.0, theta), analysis.shape).astype(
            analysis.dtype
        ),
        coefficients,
    )


def _state_covariance(
    basis_op: lx.AbstractLinearOperator, coefficient_cov: lx.AbstractLinearOperator
) -> lx.AbstractLinearOperator:
    """P_x = Φ P_θ Φᵀ, lazily."""
    adjoint = basis_op.transpose()

    def covariance_mv(direction: Array) -> Array:
        # Φ P_θ Φᵀ v
        return basis_op.mv(coefficient_cov.mv(adjoint.mv(direction)))

    return lx.FunctionLinearOperator(
        covariance_mv, basis_op.out_structure(), lx.positive_semidefinite_tag
    )


def gauss_newton_covariance(
    analysis: Any,
    model: Any,
    batch: Any,
    prior_cov_op: lx.AbstractLinearOperator | None,
    obs_cov_op: lx.AbstractLinearOperator | None,
    solver: lx.AbstractLinearSolver,
) -> tuple[Float[Array, ...], lx.AbstractLinearOperator, str]:
    r"""Mean and lazy Gauss–Newton covariance of the analysis.

    Algorithm:
        1. Without a rollout (single time), linearise ``obs_op`` at the
           analysis and apply the batch mask: P*⁻¹ = B⁻¹ + (M H')ᵀ R⁻¹ (M H').
        2. Strong 4DVar: build G(x₀) from the model's own rollout and form
           v ↦ B⁻¹ v + G'ᵀ R⁻¹ G' v, with R⁻¹ applied at each time.
        3. Weak 4DVar: the same over z = (x₀, η), with Q⁻¹ on each ηₜ; the
           covariance of x₀ solves P_z u = (v, 0) and keeps u's x₀ part.
        4. Invert lazily with ``solver``.

        A linear control x = x* + Φθ (``LinearParameterization`` or a
        ``basis=``) replaces B⁻¹ by Q⁻¹ and G' by G'Φ in steps 1–2, and
        returns P_x = Φ P_θ Φᵀ.

    Args:
        analysis: The analysis of one window: x₀*, or (x₀*, η*) for weak
            4DVar.
        model: The analysis model or its ``.as_analysis_step()`` wrapper.
        batch: The single window the analysis was computed from (4DVar).
        prior_cov_op: B, or ``None`` for the model's ``prior_cov_op``.
        obs_cov_op: R at one time, or ``None`` for the model's
            ``obs_cov_op``.
        solver: Linear solver for the lazy inverses.

    Returns:
        ``(mean, covariance, structure)``, where ``structure`` names the
        branch taken (``"single_time"``, ``"strong_4dvar"`` or
        ``"weak_4dvar"``).

    Raises:
        AttributeError: If the model exposes no ``obs_op``.
        ValueError: For point batches, multi-sample batches, a
            nonlinear parameterization, a parameterized weak 4DVar, or a
            covariance neither given nor on the model.
    """
    underlying = getattr(model, "model", model)
    if getattr(underlying, "obs_op", None) is None:
        raise AttributeError(
            "The posterior adapters require the model to expose `obs_op`; "
            "got a model without one."
        )
    rollout = has_rollout(underlying)
    if rollout and hasattr(underlying, "_rollout_with_error"):
        _check_state_space_control(underlying)
    basis = _linear_control(underlying)
    if basis is None:
        prior_cov_op = _model_covariance(prior_cov_op, underlying, "prior_cov_op")
        prior_precision = inverse_operator(prior_cov_op, solver)
    obs_cov_op = _model_covariance(obs_cov_op, underlying, "obs_cov_op")
    obs_precision = inverse_operator(obs_cov_op, solver)
    if basis is not None:
        # x = x* + Φθ: the precision lives in θ, with prior Q⁻¹
        basis_op = _basis_operator(basis, analysis)
        coefficient_prior = lx.FunctionLinearOperator(
            basis.prior_inv, basis_op.in_structure(), lx.positive_semidefinite_tag
        )

    # --- 1. single time ---
    if not rollout:
        # H' at the analysis; the cost observes M ⊙ H(x), so the tangent is
        # M H' (missing entries carry no information)
        tangent = _masked_tangent(underlying.obs_op.linearize(analysis), batch)
        if basis is None:
            # P*⁻¹ = H'ᵀ R⁻¹ H' + B⁻¹
            precision = tangent.transpose() @ obs_precision @ tangent + prior_precision
            tagged = lx.TaggedLinearOperator(precision, lx.positive_semidefinite_tag)
            return analysis, gaussx.inv(tagged, solver=solver), "single_time"
        # P_θ⁻¹ = Φᵀ H'ᵀ R⁻¹ H' Φ + Q⁻¹
        reduced = tangent @ basis_op
        precision = reduced.transpose() @ obs_precision @ reduced + coefficient_prior
        tagged = lx.TaggedLinearOperator(precision, lx.positive_semidefinite_tag)
        coefficient_cov = gaussx.inv(tagged, solver=solver)
        return analysis, _state_covariance(basis_op, coefficient_cov), "single_time"

    mask = window_mask(batch)

    if _marginalises_masked_obs(underlying):
        # The cost solves each time against R̃ₜ = Mₜ R Mₜ + (I - Mₜ), whose
        # inverse on the observed block is (R_oo)⁻¹: weight with the same.
        def apply_obs_precision(
            observed_vector: Float[Array, "T_plus_1 N"],
        ) -> Float[Array, "T_plus_1 N"]:
            def solve(step_vector: Array, step_mask: Array) -> Array:
                mask_op = lx.DiagonalLinearOperator(step_mask)
                marginal = lx.TaggedLinearOperator(
                    mask_op @ obs_cov_op @ mask_op
                    + lx.DiagonalLinearOperator(1.0 - step_mask),
                    lx.positive_semidefinite_tag,
                )
                return lx.linear_solve(marginal, step_vector, solver=solver).value

            return jax.vmap(solve)(observed_vector, mask)

    else:

        def apply_obs_precision(
            observed_vector: Float[Array, "T_plus_1 N"],
        ) -> Float[Array, "T_plus_1 N"]:
            # R⁻¹ at each time
            return jax.vmap(obs_precision.mv)(observed_vector)

    # --- 3. weak 4DVar: augmented control ---
    if hasattr(underlying, "_rollout_with_error"):
        return _weak_covariance(
            analysis,
            underlying,
            mask,
            prior_precision,
            apply_obs_precision,
            solver,
        )

    # --- 2. strong 4DVar ---
    observed = observed_trajectory(underlying, mask)
    if basis is not None:
        # G'Φ at x₀*: linearise θ ↦ G(x₀* + Φθ) at θ = 0
        def observed_coefficients(theta: Array) -> Array:
            return observed(analysis + basis_op.mv(theta))

        zero = jnp.zeros(basis_op.in_structure().shape, analysis.dtype)
        coefficient_cov = _strong_covariance(
            zero,
            observed_coefficients,
            coefficient_prior,
            apply_obs_precision,
            solver,
        )
        covariance = _state_covariance(basis_op, coefficient_cov)
        return analysis, covariance, "strong_4dvar"
    covariance = _strong_covariance(
        analysis, observed, prior_precision, apply_obs_precision, solver
    )
    return analysis, covariance, "strong_4dvar"


def _gauss_newton_precision(
    observed: Callable[[Any], Array],
    point: Any,
    prior_mv: Callable[[Any], Any],
    apply_obs_precision: Callable[[Array], Array],
) -> lx.AbstractLinearOperator:
    """v ↦ prior_mv(v) + G'ᵀ R⁻¹ G' v at ``point``, matrix-free."""
    _, observed_tangent = jax.linearize(observed, point)
    _, observed_adjoint = jax.vjp(observed, point)

    def precision_mv(direction: Any) -> Any:
        # G' v
        observed_direction = observed_tangent(direction)
        # R⁻¹ G' v
        weighted = apply_obs_precision(observed_direction)
        # G'ᵀ R⁻¹ G' v
        (data_term,) = observed_adjoint(weighted)
        # prior precision v + G'ᵀ R⁻¹ G' v
        return jax.tree.map(jnp.add, prior_mv(direction), data_term)

    structure = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), point
    )
    return lx.FunctionLinearOperator(
        precision_mv, structure, lx.positive_semidefinite_tag
    )


def _strong_covariance(
    initial_state: Float[Array, " N"],
    observed: Callable[[Array], Array],
    prior_precision: lx.AbstractLinearOperator,
    apply_obs_precision: Callable[[Array], Array],
    solver: lx.AbstractLinearSolver,
) -> lx.AbstractLinearOperator:
    """(B⁻¹ + G'ᵀ R⁻¹ G')⁻¹ at x₀*, lazily."""
    precision = _gauss_newton_precision(
        observed, initial_state, prior_precision.mv, apply_obs_precision
    )
    return gaussx.inv(precision, solver=solver)


def _weak_covariance(
    analysis: Any,
    model: Any,
    mask: Float[Array, "T_plus_1 N"],
    prior_precision: lx.AbstractLinearOperator,
    apply_obs_precision: Callable[[Array], Array],
    solver: lx.AbstractLinearSolver,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator, str]:
    """x₀ block of the inverse weak-constraint Gauss–Newton precision."""
    if not isinstance(analysis, tuple) or len(analysis) != 2:
        raise UnsupportedInputError(
            "WeakFourDVar's posterior needs analysis=(x0, eta), the pair the "
            "model returns for one window."
        )
    initial_state, model_error = analysis

    def observed(control: tuple[Array, Array]) -> Float[Array, "T_plus_1 N"]:
        # G_z(x₀, η) = [mₜ ⊙ Hₜ(xₜ)]ₜ,  xₜ = M(xₜ₋₁) + ηₜ
        state, errors = control
        trajectory = model._rollout_with_error(state, errors)

        def observe(step_state, step_mask):
            if accepts_mask(model.obs_op):
                return step_mask * model.obs_op(step_state, mask=step_mask)
            return step_mask * model.obs_op(step_state)

        return jax.vmap(observe)(trajectory, mask)

    if model_error.shape[0] == 0:
        # one-state window: no model-error steps, so z = x₀
        def observed_state(state: Float[Array, " N"]) -> Float[Array, "1 N"]:
            return observed((state, model_error))

        covariance = _strong_covariance(
            initial_state, observed_state, prior_precision, apply_obs_precision, solver
        )
        return initial_state, covariance, "weak_4dvar"

    model_error_precision = inverse_operator(model.model_err_cov_op, solver)

    def prior_mv(direction: tuple[Array, Array]) -> tuple[Array, Array]:
        # blockdiag(B⁻¹, Q⁻¹, …, Q⁻¹) v
        state_direction, error_direction = direction
        prior_state = prior_precision.mv(state_direction)
        prior_error = jax.vmap(model_error_precision.mv)(error_direction)
        return prior_state, prior_error

    control = (initial_state, model_error)
    precision = _gauss_newton_precision(
        observed, control, prior_mv, apply_obs_precision
    )

    def covariance_mv(direction: Float[Array, " N"]) -> Float[Array, " N"]:
        # solve P_z u = (v, 0); Cov(x₀) v = u_{x₀}
        right_hand_side = (direction, jnp.zeros_like(model_error))
        solution = lx.linear_solve(precision, right_hand_side, solver=solver)
        return solution.value[0]

    structure = jax.ShapeDtypeStruct(initial_state.shape, initial_state.dtype)
    covariance = lx.FunctionLinearOperator(
        covariance_mv, structure, lx.positive_semidefinite_tag
    )
    return initial_state, covariance, "weak_4dvar"


def model_posterior(model: Any, batch: Any, adapter: Any = None) -> Any:
    """Analyse one window and return its [`Posterior`][vardax.Posterior].

    The ``.posterior(batch)`` method of the variational models: run the
    analysis on ``batch``, then hand the analysis, the model and the batch
    to ``adapter``.

    Args:
        model: An analysis model with ``__call__(batch)``.
        batch: A ``Batch1D`` holding one window.
        adapter: A [`PosteriorAdapter`][vardax.PosteriorAdapter]. Default
            ``None``: [`LaplaceCovariance`][vardax.LaplaceCovariance] with
            the model's own $B$ and $R$.

    Returns:
        The posterior of that window.

    Raises:
        ValueError: If ``batch`` holds more than one window or is a point
            batch.
    """
    from vardax._src.posterior.laplace import LaplaceCovariance

    if isinstance(batch, PointObsBatch):
        raise UnsupportedInputError(
            "posterior() does not support PointObsBatch windows yet; pass a "
            "gridded Batch1D."
        )
    n_windows = batch.input.shape[0]
    if n_windows != 1:
        raise ShapeError(
            f"posterior() takes one window; the batch holds {n_windows}. Slice "
            "it: jax.tree.map(lambda a: a[i:i + 1], batch)."
        )
    adapter = LaplaceCovariance() if adapter is None else adapter
    analysis = model(batch)
    if isinstance(analysis, tuple):
        # weak 4DVar returns (x0, eta), each with a leading window axis
        window_analysis = tuple(part[0] for part in analysis)
    else:
        window_analysis = analysis[0]
    return adapter(window_analysis, model, batch)
