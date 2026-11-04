"""Ratchet for the einx convention on dense arrays.

Array code goes through einx rather than ``jnp.einsum``, ``jnp.reshape``,
``jnp.transpose``, a method ``.reshape(``, an ``axis=`` keyword or a ``[:, None]``
broadcast. Older code still has them, so this test caps each count at its current
value: new code may not add any, and converting old sites lowers the count. Lower
the ceiling in the same PR when you convert some.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Ceilings, as counted on the PR that introduced this test. Only ever lower them.
CEILINGS = {
    "src/vardax": {
        "method .reshape/.ravel": 9,
        "axis= keyword": 2,
        "None-index broadcast": 1,
        "jnp shape functions": 8,
    },
    "tests": {
        "method .reshape/.ravel": 51,
        "axis= keyword": 21,
        "None-index broadcast": 156,
        "jnp shape functions": 3,
    },
}

_JNP_SHAPE_FUNCTIONS = frozenset(
    {"einsum", "expand_dims", "moveaxis", "ravel", "reshape", "swapaxes", "transpose"}
)


def _has_none(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Tuple):
        return any(_has_none(elt) for elt in node.elts)
    return False


def _count(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for file in path.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    on_jnp = isinstance(func.value, ast.Name) and func.value.id == "jnp"
                    if func.attr in ("reshape", "ravel") and not on_jnp:
                        counts["method .reshape/.ravel"] += 1
                    if func.attr in _JNP_SHAPE_FUNCTIONS and on_jnp:
                        counts["jnp shape functions"] += 1
                counts["axis= keyword"] += sum(kw.arg == "axis" for kw in node.keywords)
            elif isinstance(node, ast.Subscript) and _has_none(node.slice):
                counts["None-index broadcast"] += 1
    return counts


@pytest.mark.parametrize("tree", sorted(CEILINGS))
def test_no_new_non_einx_array_ops(tree):
    path = ROOT / tree
    if not path.is_dir():
        pytest.skip(f"{tree} is not present")
    counts = _count(path)
    over = {
        kind: (counts[kind], ceiling)
        for kind, ceiling in CEILINGS[tree].items()
        if counts[kind] > ceiling
    }
    assert not over, (
        f"{tree} adds non-einx array operations (count, ceiling): {over}. Use einx "
        "(einx.id, einx.sum, einx.dot, einx.add) on dense arrays instead."
    )
