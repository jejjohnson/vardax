"""Zero-length axes stay valid: einx cannot size them, so these paths use jnp."""

import jax.numpy as jnp
import numpy as np

import vardax as vdx


def test_time_patches_of_a_single_time_is_empty():
    assert vdx.time_patches(jnp.arange(1.0)).shape == (0, 2)


def test_interp_obs_with_no_observations_returns_empty():
    grid_coords = (np.arange(4.0), np.arange(5.0))
    obs_op = vdx.interp_obs_from_coords(grid_coords, np.zeros((0, 2)))
    assert obs_op(jnp.ones((4, 5))).shape == (0,)
