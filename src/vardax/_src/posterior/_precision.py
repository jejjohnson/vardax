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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.models._rollout import accepts_mask, observed_trajectory


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
        raise ValueError(
            "The posterior adapters do not support PointObsBatch windows yet; "
            "pass a gridded Batch1D."
        )
    if not isinstance(batch, Batch1D):
        raise ValueError(
            "A 4DVar posterior needs the window's Batch1D (for its length and "
            f"mask); got {type(batch).__name__}."
        )
    mask = batch.mask
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError(
                f"The batch holds {mask.shape[0]} windows but the analysis is "
                "one state; pass that sample's batch, e.g. "
                "jax.tree.map(lambda a: a[i:i + 1], batch)."
            )
        mask = mask[0]
    return mask


def _check_state_space_control(model: Any) -> None:
    # a reduced-basis or parameterized control has its own prior, not B
    if getattr(model, "basis", None) is not None or (
        getattr(model, "parameterization", None) is not None
    ):
        raise ValueError(
            "The posterior adapters build the state-space posterior with B; "
            f"this {type(model).__name__} has a basis= or parameterization= "
            "control, whose prior is not B. Use EnsembleCovariance, or the "
            "posterior of the control coefficients."
        )


def gauss_newton_covariance(
    analysis: Any,
    model: Any,
    batch: Any,
    prior_cov_op: lx.AbstractLinearOperator,
    obs_cov_op: lx.AbstractLinearOperator,
    solver: lx.AbstractLinearSolver,
) -> tuple[Float[Array, ...], lx.AbstractLinearOperator, str]:
    r"""Mean and lazy Gauss–Newton covariance of the analysis.

    Algorithm:
        1. Without a rollout (single time), linearise ``obs_op`` at the
           analysis: P*⁻¹ = B⁻¹ + H'ᵀ R⁻¹ H'.
        2. Strong 4DVar: build G(x₀) from the model's own rollout and form
           v ↦ B⁻¹ v + G'ᵀ R⁻¹ G' v, with R⁻¹ applied at each time.
        3. Weak 4DVar: the same over z = (x₀, η), with Q⁻¹ on each ηₜ; the
           covariance of x₀ solves P_z u = (v, 0) and keeps u's x₀ part.
        4. Invert lazily with ``solver``.

    Args:
        analysis: The analysis of one window: x₀*, or (x₀*, η*) for weak
            4DVar.
        model: The analysis model or its ``.as_analysis_step()`` wrapper.
        batch: The single window the analysis was computed from (4DVar).
        prior_cov_op: B.
        obs_cov_op: R, for one time.
        solver: Linear solver for the lazy inverses.

    Returns:
        ``(mean, covariance, structure)``, where ``structure`` names the
        branch taken (``"single_time"``, ``"strong_4dvar"`` or
        ``"weak_4dvar"``).

    Raises:
        AttributeError: If the model exposes no ``obs_op``.
        ValueError: For point batches, multi-sample batches, or a basis /
            parameterized control.
    """
    underlying = getattr(model, "model", model)
    if getattr(underlying, "obs_op", None) is None:
        raise AttributeError(
            "The posterior adapters require the model to expose `obs_op`; "
            "got a model without one."
        )
    prior_precision = inverse_operator(prior_cov_op, solver)
    obs_precision = inverse_operator(obs_cov_op, solver)

    # --- 1. single time ---
    if not has_rollout(underlying):
        # H' at the analysis
        tangent = underlying.obs_op.linearize(analysis)
        # P*⁻¹ = H'ᵀ R⁻¹ H' + B⁻¹
        precision = tangent.transpose() @ obs_precision @ tangent + prior_precision
        tagged = lx.TaggedLinearOperator(precision, lx.positive_semidefinite_tag)
        return analysis, gaussx.inv(tagged, solver=solver), "single_time"

    _check_state_space_control(underlying)
    mask = window_mask(batch)

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
        raise ValueError(
            "WeakFourDVar's posterior needs analysis=(x0, eta), the pair the "
            "model returns for one window."
        )
    initial_state, model_error = analysis

    def observed(control: tuple[Array, Array]) -> Float[Array, "T_plus_1 N"]:
        # G_z(x₀, η) = [mₜ ⊙ Hₜ(xₜ)]ₜ,  xₜ = M(xₜ₋₁) + ηₜ
        state, errors = control
        trajectory = model._rollout_with_error(state, errors)

        def observe(state_t, mask_t):
            if accepts_mask(model.obs_op):
                return mask_t * model.obs_op(state_t, mask=mask_t)
            return mask_t * model.obs_op(state_t)

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
