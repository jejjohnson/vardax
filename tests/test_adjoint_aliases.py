"""The vardax adjoint names are deprecated aliases of argminax's (vardax#278).

They warn once, naming the argminax class, and otherwise behave exactly as before:
the FourDVarNet dispatch treats the alias and the argminax class the same.
"""

from __future__ import annotations

import warnings

from argminax.adjoint import KStep, OneStep
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import vardax as vdx


@pytest.mark.parametrize(
    ("build", "replacement"),
    [
        (lambda: vdx.KStepAdjoint(k=2), "argminax.adjoint.KStep"),
        (vdx.OneStepAdjoint, "argminax.adjoint.OneStep"),
    ],
)
def test_aliases_warn_once_naming_argminax(build, replacement):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert replacement in str(deprecations[0].message)
    # attributed to this file, the line that built the alias
    assert deprecations[0].filename == __file__


def test_aliases_are_the_argminax_classes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        k_step = vdx.KStepAdjoint(k=3)
        one_step = vdx.OneStepAdjoint()
    assert isinstance(k_step, KStep)
    assert isinstance(one_step, KStep)
    assert k_step.k == 3
    assert one_step.k == 1


def test_mapping_returns_the_alias_quietly():
    adjoints = pytest.importorskip("pipekit_cycle.adjoints")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mapped = vdx.to_optimistix_adjoint(adjoints.TruncatedAdjoint(k=4))
    assert isinstance(mapped, vdx.KStepAdjoint)
    assert mapped.k == 4


@pytest.mark.slow
def test_alias_and_argminax_class_give_the_same_model():
    rng = jax.random.PRNGKey(0)
    batch = vdx.Batch1D(
        input=jax.random.normal(rng, (2, 3, 8)),
        mask=(jax.random.uniform(jax.random.PRNGKey(1), (2, 3, 8)) > 0.3).astype(
            jnp.float32
        ),
    )

    def build(adjoint):
        return vdx.FourDVarNet1D(
            state_dim=8,
            n_time=3,
            latent_dim=4,
            hidden_dim=8,
            n_solver_steps=3,
            solver_adjoint=adjoint,
            key=rng,
        )

    def loss(model):
        return jnp.mean((model(batch) - batch.input) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alias = build(vdx.OneStepAdjoint())
    native = build(OneStep())
    assert jnp.array_equal(alias(batch), native(batch))
    alias_grads = eqx.filter_grad(loss)(alias)
    native_grads = eqx.filter_grad(loss)(native)
    # the adjoints are different classes, so compare the array leaves
    alias_leaves = jax.tree_util.tree_leaves(eqx.filter(alias_grads, eqx.is_array))
    native_leaves = jax.tree_util.tree_leaves(eqx.filter(native_grads, eqx.is_array))
    assert len(alias_leaves) == len(native_leaves)
    assert all(
        jnp.array_equal(a, b) for a, b in zip(alias_leaves, native_leaves, strict=True)
    )
