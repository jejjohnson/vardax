"""The package error hierarchy: one base, named subclasses, builtin-compatible."""

import jax
import jax.numpy as jnp
import pytest

import vardax as vdx

VALUE_ERRORS = [
    vdx.ConfigurationError,
    vdx.ShapeError,
    vdx.GridError,
    vdx.ObservationError,
    vdx.DataError,
    vdx.MissingInputError,
    vdx.UnsupportedInputError,
]


@pytest.mark.parametrize("error", [*VALUE_ERRORS, vdx.InputTypeError])
def test_every_error_derives_from_the_package_base(error):
    assert issubclass(error, vdx.VardaxError)


@pytest.mark.parametrize("error", VALUE_ERRORS)
def test_value_errors_stay_value_errors(error):
    # callers that caught ValueError before the hierarchy keep working
    assert issubclass(error, ValueError)


def test_input_type_error_stays_a_type_error():
    assert issubclass(vdx.InputTypeError, TypeError)


def test_invalid_hyperparameter_raises_configuration_error():
    with pytest.raises(vdx.ConfigurationError, match="max_steps"):
        vdx.OptimistixMinimiser(max_steps=0)


@pytest.mark.unchecked_types
def test_malformed_design_matrix_raises_shape_error():
    with pytest.raises(vdx.ShapeError, match="design_matrix must be 2-D"):
        vdx.linear_basis(jnp.ones(3))


def test_non_increasing_grid_raises_grid_error():
    y_axis = jnp.array([0.0, 1.0, 2.0])
    x_axis = jnp.array([0.0, 0.0, 1.0])
    with pytest.raises(vdx.GridError, match="strictly increasing"):
        vdx.interp_obs_from_coords((y_axis, x_axis), jnp.zeros((1, 2)))


def test_sampled_penalty_without_key_raises_missing_input_error():
    penalty = vdx.ResidualPenalty(
        residual_fn=lambda field, point: field(point),
        collocation=None,
        weight=1.0,
        sampler=lambda key: jax.random.uniform(key, (4, 1)),
    )
    with pytest.raises(vdx.MissingInputError, match="needs a key"):
        penalty(lambda point: point)


def test_negative_traced_penalty_weight_is_caught_at_call_time():
    # under jit the constructor cannot see the sign, so __call__ checks it
    @jax.jit
    def penalty_value(weight):
        penalty = vdx.ResidualPenalty(
            residual_fn=lambda field, point: field(point),
            collocation=jnp.ones((3, 1)),
            weight=weight,
        )
        return penalty(lambda point: point)

    assert jnp.isfinite(penalty_value(jnp.asarray(1.0)))
    with pytest.raises(Exception, match="weight must be non-negative"):
        penalty_value(jnp.asarray(-1.0))
