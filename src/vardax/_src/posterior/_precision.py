"""Gauss–Newton posterior precision Λ_x|y of every analysis family, matrix-free.

Λ_x|y = Jᵀ Λ̃_ε J + Λ_x, with J the observation Jacobian of the control and Λ̃_ε the
masked observation precision; the Laplace and Gauss–Newton adapters invert it lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import gaussx
import jax
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from vardax._src._types import Batch1D, PointObsBatch
from vardax._src.errors import MissingInputError, ShapeError, UnsupportedInputError
from vardax._src.models._rollout import (
    accepts_mask,
    observed_trajectory,
)
from vardax._src.obs_operators.tangent import obs_tangent
from vardax._src.parameterization import LinearParameterization
from vardax._src.utils.covariance import inverse_operator
from vardax._src.utils.shapes import as_shape


def has_rollout(model: Any) -> bool:
    """Whether ``model`` assimilates a window through a forward model f.

    Args:
        model: The underlying analysis model.

    Returns:
        ``True`` for the 4D-Var models, ``False`` for the single-time ones.

    Examples:
        A model with only an observation operator has no rollout:

        >>> from types import SimpleNamespace
        >>> has_rollout(SimpleNamespace(obs_op=None))
        False

        A strong 4D-Var model rolls its forward model out:

        >>> has_rollout(SimpleNamespace(forward=None, _rollout=None))
        True

        So does a weak one, through its model-error rollout:

        >>> has_rollout(SimpleNamespace(forward=None, _rollout_with_error=None))
        True
    """
    return hasattr(model, "forward") and (
        hasattr(model, "_rollout") or hasattr(model, "_rollout_with_error")
    )


def window_mask(batch: Any) -> Float[Array, "T_plus_1 N"]:
    """The observation mask mₜ of a single-sample window batch.

    Args:
        batch: The ``Batch1D`` the analysis was computed from, either unbatched or
            with a leading sample axis of size 1.

    Returns:
        The mask mₜ of that one window, time axis first.

    Raises:
        UnsupportedInputError: If the batch is a point batch or not a ``Batch1D``.
        ShapeError: If the batch holds more than one sample.

    Examples:
        The leading sample axis of size 1 is dropped:

        >>> import jax.numpy as jnp
        >>> from vardax._src._types import Batch1D
        >>> window = Batch1D(input=jnp.zeros((1, 2, 3)), mask=jnp.ones((1, 2, 3)))
        >>> window_mask(window).shape
        (2, 3)

        An unbatched window keeps its shape:

        >>> unbatched = Batch1D(input=jnp.zeros((2, 3)), mask=jnp.ones((2, 3)))
        >>> window_mask(unbatched).shape
        (2, 3)

        Several windows are ambiguous, since the analysis is one state:

        >>> pair = Batch1D(input=jnp.zeros((2, 2, 3)), mask=jnp.ones((2, 2, 3)))
        >>> window_mask(pair)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapeError: The batch holds 2 windows but the analysis is one state; ...
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
    """Raise for a reduced-basis or parameterized control, whose prior is not Σ_x."""
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
        [`LinearParameterization`][vardax.LinearParameterization], the model's
        ``basis`` for ``ReducedBasisFourDVar(basis=...)``, or ``None`` when the
        control is the state itself.

    Raises:
        UnsupportedInputError: For any other parameterization: its decode is not
            affine, so Φ Σ_z|y Φᵀ is not its posterior covariance.
    """
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


def _single_time_mask(
    batch: Any, obs_structure: jax.ShapeDtypeStruct
) -> Float[Array, "*obs"] | None:
    """The mask m of a single-time ``Batch1D``, or ``None`` without one.

    A batch of several fully observed windows is accepted, since its mask is all
    ones; several windows with missing entries raise, since the mask of the analysed
    sample is ambiguous. Under tracing that check cannot run and the mask is not
    applied.

    ``OptimalInterpolation`` and ``ThreeDVar`` observe m ⊙ h(x), but
    ``obs_op.linearize`` knows no mask (``MaskedIdentity.linearize`` is the identity),
    so the caller composes the mask as its own operator.

    Args:
        batch: The batch of the analysis, or anything else for no mask.
        obs_structure: The output structure of the observation Jacobian J_h.

    Returns:
        The mask m on the shape of the observation, or ``None``.

    Raises:
        ShapeError: If several windows have missing entries, or the mask does not
            match the observation.
        UnsupportedInputError: If the batch is a point batch.
    """
    if not isinstance(batch, Batch1D):
        return None
    if batch.mask.ndim == 3 and batch.mask.shape[0] != 1:
        # several windows: only a fully observed batch says nothing about
        # this sample's mask (the single-time adapters always accepted it)
        if isinstance(batch.mask, jax.core.Tracer) or bool(jnp.all(batch.mask == 1)):
            return None
        raise ShapeError(
            f"The batch holds {batch.mask.shape[0]} windows with missing "
            "observations but the analysis is one state; pass that sample's "
            "batch, e.g. jax.tree.map(lambda a: a[i:i + 1], batch)."
        )
    mask = window_mask(batch)
    if mask.shape == (1, *obs_structure.shape):
        # an (S, N) batch of (N,) states: drop the sample axis, as __call__ does
        mask = mask[0]
    if mask.shape != obs_structure.shape:
        raise ShapeError(
            f"The batch mask has shape {mask.shape} but the observation "
            f"operator returns {obs_structure.shape}."
        )
    return mask


def _single_time_precision(
    obs_jacobian: lx.AbstractLinearOperator,
    masked_jacobian: lx.AbstractLinearOperator,
    mask: Float[Array, "*obs"] | None,
    obs_precision: lx.AbstractLinearOperator,
    prior_precision: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Λ_x|y = Jᵀ Λ̃_ε J + Λ_x, a true diagonal when J_h = I and Σ_x, Σ_ε are diagonal.

    Args:
        obs_jacobian: J_h, without the mask.
        masked_jacobian: J = M J_h.
        mask: The mask m of a single-time batch, or ``None``.
        obs_precision: Λ̃_ε.
        prior_precision: Λ_x.

    Returns:
        The precision Λ_x|y.
    """
    if (
        isinstance(obs_jacobian, lx.IdentityLinearOperator)
        and lx.is_diagonal(obs_precision)
        and lx.is_diagonal(prior_precision)
    ):
        return _diagonal_precision(mask, obs_precision, prior_precision)
    # Jᵀ
    adjoint = masked_jacobian.transpose()
    # Jᵀ Λ̃_ε J
    data_precision = adjoint @ obs_precision @ masked_jacobian
    # Λ_x|y = Jᵀ Λ̃_ε J + Λ_x
    precision = data_precision + prior_precision
    return lx.TaggedLinearOperator(precision, lx.positive_semidefinite_tag)


def _diagonal_precision(
    mask: Float[Array, "*obs"] | None,
    obs_precision: lx.AbstractLinearOperator,
    prior_precision: lx.AbstractLinearOperator,
) -> lx.DiagonalLinearOperator:
    """diag(Λ_x|y) = m² ⊙ diag(Λ̃_ε) + diag(Λ_x), as a diagonal operator.

    lineax's ``diagonal`` materialises a composition, so the sum is formed here from
    each term's closed-form diagonal, and Σ_x|y inverts in closed form too.

    Args:
        mask: The mask of a single-time batch, or ``None``.
        obs_precision: Λ̃_ε, diagonal.
        prior_precision: Λ_x, diagonal, on the same structure.

    Returns:
        The precision, in the shape of the state.
    """
    shape = prior_precision.in_structure().shape
    # jnp.reshape, not einx, here and below: einx cannot size a zero-length state
    # diag(Λ̃_ε)
    data_diagonal = jnp.reshape(gaussx.diag(obs_precision), shape)
    if mask is not None:
        # m²
        mask_squared = jnp.reshape(mask, shape) ** 2
        # m² ⊙ diag(Λ̃_ε)
        data_diagonal = mask_squared * data_diagonal
    # diag(Λ_x)
    prior_diagonal = jnp.reshape(gaussx.diag(prior_precision), shape)
    # diag(Λ_x|y) = m² ⊙ diag(Λ̃_ε) + diag(Λ_x)
    return lx.DiagonalLinearOperator(data_diagonal + prior_diagonal)


def _masked_tangent(
    obs_jacobian: lx.AbstractLinearOperator, mask: Float[Array, "*obs"] | None
) -> lx.AbstractLinearOperator:
    """J = M J_h for a mask m, J_h unchanged without one.

    The mask is composed as its own operator, so J_h keeps its own transpose (a
    materialised Jacobian stays a dense product).
    """
    if mask is None:
        return obs_jacobian
    # M = diag(m): diagonal, so a diagonal precision stays visible to gaussx.diag; in
    # the Jacobian's dtype, as lineax composes only matching structures
    mask_operator = lx.DiagonalLinearOperator(
        mask.astype(obs_jacobian.out_structure().dtype)
    )
    # J = M J_h
    return mask_operator @ obs_jacobian


def _whitened_control(model: Any) -> bool:
    """Whether ``model`` solves over the whitened control x₀ = μ_x + L z, L Lᵀ = Σ_x.

    ``IncrementalFourDVar`` with ``config.cvt`` never applies Λ_x, so its posterior is
    built over z as well, where the prior precision is I.
    """
    # lazy: models and posterior import each other at module level, so this stays
    # independent of which package initialises first
    from vardax._src.models.incremental_fourdvar import IncrementalFourDVar

    return isinstance(model, IncrementalFourDVar) and model.config.cvt


def _jac_mode(model: Any) -> Literal["fwd", "bwd"]:
    """``config.jac`` of an ``IncrementalFourDVar``, else ``"fwd"``."""
    config = getattr(model, "config", None)
    return getattr(config, "jac", "fwd")


def _basis_operator(
    basis: Any, analysis: Float[Array, ...]
) -> lx.FunctionLinearOperator:
    """Φ: z ↦ the state increment Φ(0) z, shaped like ``analysis``."""
    coefficient_structure = jax.ShapeDtypeStruct((basis.nbasis,), analysis.dtype)

    def basis_mv(coefficients: Array) -> Array:
        # δx = Φ(0) z
        increment = basis.operg(0.0, coefficients)
        shaped_increment = as_shape(increment, analysis.shape)
        return shaped_increment.astype(analysis.dtype)

    return lx.FunctionLinearOperator(basis_mv, coefficient_structure)


def _state_covariance(
    basis_operator: lx.AbstractLinearOperator,
    coefficient_cov: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Σ_x|y = Φ Σ_z|y Φᵀ, lazily."""
    # Φᵀ
    adjoint = basis_operator.transpose()

    def covariance_mv(direction: Array) -> Array:
        # Φᵀ v
        coefficient_direction = adjoint.mv(direction)
        # Σ_z|y Φᵀ v
        coefficient_image = coefficient_cov.mv(coefficient_direction)
        # Φ Σ_z|y Φᵀ v
        return basis_operator.mv(coefficient_image)

    return lx.FunctionLinearOperator(
        covariance_mv, basis_operator.out_structure(), lx.positive_semidefinite_tag
    )


def gauss_newton_covariance(
    analysis: Any,
    model: Any,
    batch: Any,
    prior_cov_op: lx.AbstractLinearOperator | None,
    obs_cov_op: lx.AbstractLinearOperator | None,
    solver: lx.AbstractLinearSolver,
) -> tuple[
    Float[Array, ...],
    lx.AbstractLinearOperator,
    str,
    Any,
    lx.AbstractLinearOperator | None,
]:
    """Mean and lazy Gauss–Newton covariance of the analysis.

        Σ_x|y = Λ_x|y⁻¹,   Λ_x|y = Jᵀ Λ̃_ε J + Λ_x
        Λ̃_ε,t = Σ̃_ε,t⁻¹,   Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ)

    What enters J depends on how the observations reach the control, with J_h the
    Jacobian of h and J_f,t = ∂xₜ/∂x₀ that of the rollout of f:

    - Single time (``OptimalInterpolation``, ``ThreeDVar``): J = M J_h.
    - Strong 4D-Var (``StrongFourDVar``, ``IncrementalFourDVar``,
      ``ReducedBasisFourDVar`` without a basis): Jₜ = Mₜ J_h,t J_f,t, so
      Λ_x|y = ∑ₜ Jₜᵀ Λ̃_ε,t Jₜ + Λ_x.
    - Weak 4D-Var (``WeakFourDVar``): the control is w = (x₀, η₁, …, η_T), with
      Λ_w|y = J_wᵀ Λ̃_ε J_w + blockdiag(Λ_x, Λ_η, …, Λ_η), and Σ_x|y is the x₀ block of
      its inverse.
    - Linear control (``LinearParameterization``, or
      ``ReducedBasisFourDVar(basis=...)``): x = x⋆ + Φ z with ℛ(z) = ½ zᵀ Σ_z⁻¹ z, so
      Σ_z|y = (Φᵀ Jᵀ Λ̃_ε J Φ + Σ_z⁻¹)⁻¹ and Σ_x|y = Φ Σ_z|y Φᵀ. Single time and strong
      4D-Var only.

    Algorithm:
        1. Without a rollout (single time), linearise h at the analysis and apply
           the batch mask: J = M J_h.
        2. Strong 4D-Var: build the observed window from the model's own rollout
           and form v ↦ Jᵀ Λ̃_ε J v + Λ_x v, with Λ̃_ε,t applied at each time.
        3. Weak 4D-Var: the same over w = (x₀, η₁, …, η_T), with Λ_η on each ηₜ; the
           covariance of x₀ solves Λ_w|y u = (v, 0) and keeps u's x₀ part.
        4. Invert lazily with ``solver``.

        A linear control x = x⋆ + Φ z (``LinearParameterization`` or a ``basis=``)
        replaces Λ_x by Σ_z⁻¹ and J by J Φ in steps 1–2, and returns
        Σ_x|y = Φ Σ_z|y Φᵀ. ``IncrementalFourDVar`` with ``config.cvt`` takes the same
        path with Φ = L (L Lᵀ = Σ_x) and Σ_z = I, unless the adapter is given its own
        ``prior_cov_op``.

    Args:
        analysis: The analysis of one window: x₀⋆, or (x₀⋆, η⋆) for weak 4D-Var.
        model: The analysis model or its ``.as_analysis_step()`` wrapper.
        batch: The single window the analysis was computed from (4D-Var).
        prior_cov_op: Prior covariance Σ_x, or ``None`` for the model's
            ``prior_cov_op``.
        obs_cov_op: Observation-error covariance Σ_ε at one time, or ``None`` for the
            model's ``obs_cov_op``.
        solver: Linear solver for the lazy inverses.

    Returns:
        ``(mean, covariance, structure, coefficients, precision)``, where
        ``structure`` names the branch taken (``"single_time"``, ``"strong_4dvar"`` or
        ``"weak_4dvar"``), ``coefficients`` is ``(Φ, Σ_z|y)`` for a linear control,
        else ``None``, and ``precision`` is the operator ``covariance`` inverts, when
        it is a plain inverse (not for a linear control or the x₀ block of a weak
        4D-Var).

    Raises:
        AttributeError: If the model exposes no ``obs_op``.
        UnsupportedInputError: For a 4D-Var model given a point batch or a batch that
            is not a ``Batch1D``, a nonlinear parameterization, a parameterized weak
            4D-Var or one with ``model_err_basis``.
        ShapeError: For a 4D-Var model given several windows, or a single-time model
            given several windows with missing entries.
        MissingInputError: For a covariance neither given nor on the model.

    Examples:
        One time, Σ_x = Σ_ε = I and J_h = I: Λ_x|y = 2 I is an actual diagonal.

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from types import SimpleNamespace
        >>> from vardax._src.obs_operators.masked import MaskedIdentity
        >>> eye = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> solver = lx.CG(atol=1e-6, rtol=1e-6)
        >>> model = SimpleNamespace(obs_op=MaskedIdentity())
        >>> result = gauss_newton_covariance(
        ...     jnp.zeros(2), model, None, eye, eye, solver
        ... )
        >>> result[2], type(result[4]).__name__
        ('single_time', 'DiagonalLinearOperator')
        >>> bool(jnp.allclose(result[1].mv(jnp.ones(2)), 0.5))
        True

        A missing covariance names the argument to pass:

        >>> gauss_newton_covariance(
        ...     jnp.zeros(2), model, None, None, eye, solver
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        MissingInputError: No prior_cov_op given to the adapter and ...
    """
    underlying = getattr(model, "model", model)
    if getattr(underlying, "obs_op", None) is None:
        raise AttributeError(
            "The posterior adapters require the model to expose `obs_op`; "
            "got a model without one."
        )
    uses_rollout = has_rollout(underlying)
    if uses_rollout and hasattr(underlying, "_rollout_with_error"):
        _check_state_space_control(underlying)
    basis = _linear_control(underlying)
    basis_operator = None
    if basis is not None:
        # x = x⋆ + Φ z: the precision lives in z, with prior precision Σ_z⁻¹
        basis_operator = _basis_operator(basis, analysis)
        coefficient_prior = lx.FunctionLinearOperator(
            basis.prior_inv,
            basis_operator.in_structure(),
            lx.positive_semidefinite_tag,
        )
    elif prior_cov_op is None and _whitened_control(underlying):
        # x₀ = x⋆ + L z with L Lᵀ = Σ_x, the control the inner loop solves over:
        # Φ = L and Σ_z = I, so Λ_x is never applied
        basis_operator = underlying._prior_sqrt()
        coefficient_prior = lx.IdentityLinearOperator(basis_operator.in_structure())
    else:
        prior_cov_op = _model_covariance(prior_cov_op, underlying, "prior_cov_op")
        # Λ_x = Σ_x⁻¹
        prior_precision = inverse_operator(prior_cov_op, solver)
    obs_cov_op = _model_covariance(obs_cov_op, underlying, "obs_cov_op")
    # Λ_ε = Σ_ε⁻¹
    obs_precision = inverse_operator(obs_cov_op, solver)

    # --- 1. single time ---
    if not uses_rollout:
        # J_h at the analysis
        obs_jacobian = obs_tangent(underlying.obs_op, analysis)
        mask = _single_time_mask(batch, obs_jacobian.out_structure())
        # J = M J_h: the cost observes m ⊙ h(x), and missing entries carry no
        # information
        masked_jacobian = _masked_tangent(obs_jacobian, mask)
        if mask is not None:
            # Λ̃_ε = (M Σ_ε M + (I − M))⁻¹, as the cost: masked entries drop out
            masked_covariance = gaussx.masked_obs_covariance(obs_cov_op, mask)
            obs_precision = inverse_operator(masked_covariance, solver)
        if basis_operator is None:
            # Λ_x|y = Jᵀ Λ̃_ε J + Λ_x
            state_precision = _single_time_precision(
                obs_jacobian, masked_jacobian, mask, obs_precision, prior_precision
            )
            # Σ_x|y = Λ_x|y⁻¹
            covariance = gaussx.inv(state_precision, solver=solver)
            return analysis, covariance, "single_time", None, state_precision
        # Σ_z|y = (Φᵀ Jᵀ Λ̃_ε J Φ + Σ_z⁻¹)⁻¹
        coefficient_cov = _single_time_coefficient_cov(
            masked_jacobian, basis_operator, obs_precision, coefficient_prior, solver
        )
        # Σ_x|y = Φ Σ_z|y Φᵀ
        covariance = _state_covariance(basis_operator, coefficient_cov)
        coefficients = (basis_operator, coefficient_cov)
        return analysis, covariance, "single_time", coefficients, None

    mask = window_mask(batch)

    # vₜ ↦ Λ̃_ε,t vₜ over the window
    apply_obs_precision = _masked_obs_precision(obs_cov_op, mask, solver)

    # --- 3. weak 4D-Var: augmented control ---
    if hasattr(underlying, "_rollout_with_error"):
        if getattr(underlying, "model_err_basis", None) is not None:
            raise UnsupportedInputError(
                "The posterior adapters do not support a WeakFourDVar with "
                "model_err_basis yet; its control is (x0, ω), not (x0, η)."
            )
        mean, covariance, structure, coefficients = _weak_covariance(
            analysis,
            underlying,
            mask,
            prior_precision,
            apply_obs_precision,
            solver,
        )
        return mean, covariance, structure, coefficients, None

    # --- 2. strong 4D-Var ---
    observed_window = observed_trajectory(underlying, mask)
    if basis_operator is not None:

        def observed_coefficients(coefficients: Array) -> Array:
            # x₀ = x₀⋆ + Φ z
            initial_state = analysis + basis_operator.mv(coefficients)
            # J Φ is the Jacobian of z ↦ [mₜ ⊙ h(xₜ)]ₜ at z = 0
            return observed_window(initial_state)

        zero_coefficients = jnp.zeros(
            basis_operator.in_structure().shape, analysis.dtype
        )
        # Σ_z|y = (Φᵀ Jᵀ Λ̃_ε J Φ + Σ_z⁻¹)⁻¹
        coefficient_cov = _strong_covariance(
            zero_coefficients,
            observed_coefficients,
            coefficient_prior,
            apply_obs_precision,
            solver,
            _jac_mode(underlying),
        )
        # Σ_x|y = Φ Σ_z|y Φᵀ
        covariance = _state_covariance(basis_operator, coefficient_cov)
        coefficients = (basis_operator, coefficient_cov)
        return analysis, covariance, "strong_4dvar", coefficients, None
    # Λ_x|y = Jᵀ Λ̃_ε J + Λ_x
    precision = gaussx.gauss_newton_operator(
        observed_window,
        analysis,
        noise_precision=apply_obs_precision,
        prior_precision=prior_precision.mv,
        jac=_jac_mode(underlying),
    )
    # Σ_x|y = Λ_x|y⁻¹
    covariance = gaussx.inv(precision, solver=solver)
    return analysis, covariance, "strong_4dvar", None, precision


def _single_time_coefficient_cov(
    masked_jacobian: lx.AbstractLinearOperator,
    basis_operator: lx.AbstractLinearOperator,
    obs_precision: lx.AbstractLinearOperator,
    coefficient_prior: lx.AbstractLinearOperator,
    solver: lx.AbstractLinearSolver,
) -> lx.AbstractLinearOperator:
    """Σ_z|y = (Φᵀ Jᵀ Λ̃_ε J Φ + Σ_z⁻¹)⁻¹ of a single-time linear control, lazily.

    Args:
        masked_jacobian: J = M J_h.
        basis_operator: Φ.
        obs_precision: Λ̃_ε.
        coefficient_prior: Σ_z⁻¹.
        solver: Linear solver for the lazy inverse.

    Returns:
        The coefficient-space posterior covariance Σ_z|y.
    """
    # J Φ
    reduced_jacobian = masked_jacobian @ basis_operator
    # (J Φ)ᵀ
    reduced_adjoint = reduced_jacobian.transpose()
    # Φᵀ Jᵀ Λ̃_ε J Φ
    data_precision = reduced_adjoint @ obs_precision @ reduced_jacobian
    # Λ_z|y = Φᵀ Jᵀ Λ̃_ε J Φ + Σ_z⁻¹
    coefficient_precision = data_precision + coefficient_prior
    tagged_precision = lx.TaggedLinearOperator(
        coefficient_precision, lx.positive_semidefinite_tag
    )
    # Σ_z|y = Λ_z|y⁻¹
    return gaussx.inv(tagged_precision, solver=solver)


def _masked_obs_precision(
    obs_cov_op: lx.AbstractLinearOperator,
    mask: Float[Array, "T_plus_1 N"],
    solver: lx.AbstractLinearSolver,
) -> Callable[[Array], Array]:
    """Return the map vₜ ↦ Λ̃_ε,t vₜ over a window, with Λ̃_ε,t = Σ̃_ε,t⁻¹.

    Args:
        obs_cov_op: Observation-error covariance Σ_ε at one time.
        mask: The window's mask mₜ.
        solver: Linear solver for the lazy inverses.

    Returns:
        The map, applied over the time axis.
    """

    def apply_obs_precision(
        observed_vector: Float[Array, "T_plus_1 N"],
    ) -> Float[Array, "T_plus_1 N"]:
        def solve(step_vector: Array, step_mask: Array) -> Array:
            # Σ̃_ε,t = Mₜ Σ_ε Mₜ + (I − Mₜ), as every model's cost
            masked_covariance = gaussx.masked_obs_covariance(obs_cov_op, step_mask)
            # Λ̃_ε,t vₜ
            return inverse_operator(masked_covariance, solver).mv(step_vector)

        return jax.vmap(solve)(observed_vector, mask)

    return apply_obs_precision


def _strong_covariance(
    initial_state: Float[Array, " N"],
    observed: Callable[[Array], Array],
    prior_precision: lx.AbstractLinearOperator,
    apply_obs_precision: Callable[[Array], Array],
    solver: lx.AbstractLinearSolver,
    jac: Literal["fwd", "bwd"] = "fwd",
) -> lx.AbstractLinearOperator:
    """(Jᵀ Λ̃_ε J + Λ_x)⁻¹ at x₀⋆, lazily."""
    # Jᵀ Λ̃_ε J + Λ_x
    precision = gaussx.gauss_newton_operator(
        observed,
        initial_state,
        noise_precision=apply_obs_precision,
        prior_precision=prior_precision.mv,
        jac=jac,
    )
    # Σ_x|y = Λ_x|y⁻¹
    return gaussx.inv(precision, solver=solver)


def _weak_covariance(
    analysis: Any,
    model: Any,
    mask: Float[Array, "T_plus_1 N"],
    prior_precision: lx.AbstractLinearOperator,
    apply_obs_precision: Callable[[Array], Array],
    solver: lx.AbstractLinearSolver,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator, str, None]:
    """x₀ block of the inverse weak-constraint Gauss–Newton precision Λ_w|y.

    Args:
        analysis: The pair (x₀⋆, η⋆) of one window.
        model: The ``WeakFourDVar``.
        mask: The window's mask mₜ.
        prior_precision: Λ_x.
        apply_obs_precision: vₜ ↦ Λ̃_ε,t vₜ over the window.
        solver: Linear solver for the lazy inverses.

    Returns:
        ``(x₀⋆, Σ_x|y, "weak_4dvar", None)``.

    Raises:
        UnsupportedInputError: If ``analysis`` is not a pair.
    """
    if not isinstance(analysis, tuple) or len(analysis) != 2:
        raise UnsupportedInputError(
            "WeakFourDVar's posterior needs analysis=(x0, eta), the pair the "
            "model returns for one window."
        )
    initial_state, model_error = analysis

    def observed(control: tuple[Array, Array]) -> Float[Array, "T_plus_1 N"]:
        state, errors = control
        # xₜ = f(xₜ₋₁) + ηₜ
        trajectory = model._rollout_with_error(state, errors)

        def observe(step_state, step_mask):
            # mₜ ⊙ h(xₜ)
            if accepts_mask(model.obs_op):
                return step_mask * model.obs_op(step_state, mask=step_mask)
            return step_mask * model.obs_op(step_state)

        return jax.vmap(observe)(trajectory, mask)

    if model_error.shape[0] == 0:

        def observed_state(state: Float[Array, " N"]) -> Float[Array, "1 N"]:
            # one-state window: no model-error steps, so w = x₀
            return observed((state, model_error))

        covariance = _strong_covariance(
            initial_state, observed_state, prior_precision, apply_obs_precision, solver
        )
        return initial_state, covariance, "weak_4dvar", None

    # Λ_η = Σ_η⁻¹
    model_error_precision = inverse_operator(model.model_err_cov_op, solver)

    def prior_mv(direction: tuple[Array, Array]) -> tuple[Array, Array]:
        state_direction, error_direction = direction
        # Λ_x v_x
        prior_state = prior_precision.mv(state_direction)
        # Λ_η vₜ, at each step
        prior_error = jax.vmap(model_error_precision.mv)(error_direction)
        # blockdiag(Λ_x, Λ_η, …, Λ_η) v
        return prior_state, prior_error

    control = (initial_state, model_error)
    # Λ_w|y = J_wᵀ Λ̃_ε J_w + blockdiag(Λ_x, Λ_η, …, Λ_η)
    precision = gaussx.gauss_newton_operator(
        observed,
        control,
        noise_precision=apply_obs_precision,
        prior_precision=prior_mv,
    )

    def covariance_mv(direction: Float[Array, " N"]) -> Float[Array, " N"]:
        right_hand_side = (direction, jnp.zeros_like(model_error))
        # solve Λ_w|y u = (v, 0)
        solution = lx.linear_solve(precision, right_hand_side, solver=solver)
        # Σ_x|y v = u_x₀
        return solution.value[0]

    structure = jax.ShapeDtypeStruct(initial_state.shape, initial_state.dtype)
    covariance = lx.FunctionLinearOperator(
        covariance_mv, structure, lx.positive_semidefinite_tag
    )
    return initial_state, covariance, "weak_4dvar", None


def model_posterior(model: Any, batch: Any, adapter: Any = None) -> Any:
    """Analyse one window and return its [`Posterior`][vardax.Posterior].

    The ``.posterior(batch)`` method of the variational models: run the analysis on
    ``batch``, then hand the analysis, the model and the batch to ``adapter``.

    Args:
        model: An analysis model with ``__call__(batch)``.
        batch: A ``Batch1D`` holding one window.
        adapter: A [`PosteriorAdapter`][vardax.PosteriorAdapter]. Default ``None``:
            [`LaplaceCovariance`][vardax.LaplaceCovariance] with the model's own Σ_x
            and Σ_ε.

    Returns:
        The posterior of that window.

    Raises:
        ShapeError: If ``batch`` holds more than one window.
        UnsupportedInputError: If ``batch`` is a point batch.

    Examples:
        A one-state window of a 4D-Var model, with Σ_x = Σ_ε = I and h = I:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import vardax as vdx
        >>> class Hold:
        ...     dt = 1.0
        ...
        ...     def step(self, x, dt):
        ...         return x
        >>> eye = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> model = vdx.StrongFourDVar(
        ...     forward=Hold(),
        ...     obs_op=vdx.MaskedIdentity(),
        ...     prior_mean=jnp.zeros(2),
        ...     prior_cov_op=eye,
        ...     obs_cov_op=eye,
        ... )
        >>> window = vdx.Batch1D(input=jnp.ones((1, 1, 2)), mask=jnp.ones((1, 1, 2)))
        >>> post = model_posterior(model, window)
        >>> bool(jnp.allclose(post.mean, 0.5, atol=1e-3))
        True
        >>> bool(jnp.allclose(post.cov.mv(jnp.ones(2)), 0.5, atol=1e-3))
        True

        Pass another adapter to change the approximation:

        >>> gauss_newton = vdx.GaussNewtonHessian()
        >>> model_posterior(model, window, gauss_newton).provenance["adapter"]
        'GaussNewtonHessian'

        Two windows raise, since the posterior is of one analysis:

        >>> pair = vdx.Batch1D(input=jnp.ones((2, 1, 2)), mask=jnp.ones((2, 1, 2)))
        >>> model_posterior(model, pair)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapeError: posterior() takes one window; the batch holds 2. ...
    """
    # lazy: laplace imports this module
    from vardax._src.posterior.laplace import LaplaceCovariance

    if isinstance(batch, PointObsBatch):
        raise UnsupportedInputError(
            "posterior() does not support PointObsBatch windows yet; pass a "
            "gridded Batch1D."
        )
    num_windows = batch.input.shape[0]
    if num_windows != 1:
        raise ShapeError(
            f"posterior() takes one window; the batch holds {num_windows}. Slice "
            "it: jax.tree.map(lambda a: a[i:i + 1], batch)."
        )
    adapter = LaplaceCovariance() if adapter is None else adapter
    analysis = model(batch)
    if isinstance(analysis, tuple):
        # weak 4D-Var returns (x₀, η), each with a leading window axis
        window_analysis = tuple(part[0] for part in analysis)
    else:
        window_analysis = analysis[0]
    return adapter(window_analysis, model, batch)
