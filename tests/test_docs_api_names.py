"""Every vardax API name used in a docs code block must exist (issue #99).

Docs examples drifted from the package: they called methods, helpers,
modules and keyword arguments that do not exist, so they failed on
copy-paste. This test parses every ```python block in README.md and the
user-facing docs and checks, statically:

- ``from vardax[.module] import name`` and ``import vardax.module``
  resolve;
- ``vardax.X`` / ``vdx.X`` attribute chains resolve;
- keyword arguments passed to a vardax callable are in its signature
  (unless it takes ``**kwargs``);
- attributes read from a variable assigned a vardax class instance in
  the same block (``model = vardax.StrongFourDVar(...); model.posterior``)
  exist on that class.

``docs/design/`` is excluded: it holds design records that describe
modules as planned (``vardax.persist``, ``vardax.cvt``, the latent-DA
layer), not the shipped API.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pathlib
import re
from typing import Any

import pytest

import vardax

# end-to-end: several components (or pipekit / the docs) together
pytestmark = pytest.mark.integration

REPO = pathlib.Path(__file__).parents[1]
DOCS = REPO / "docs"
README = REPO / "README.md"
EXCLUDED = ("design",)
ALIASES = {"vardax": vardax, "vdx": vardax}
_MISSING = object()


def _pages() -> list[pathlib.Path]:
    pages = sorted(DOCS.rglob("*.md"))
    pages = [p for p in pages if p.relative_to(DOCS).parts[0] not in EXCLUDED]
    return [README, *pages]


def _blocks(page: pathlib.Path) -> list[tuple[int, str]]:
    text = page.read_text()
    blocks = []
    for match in re.finditer(r"```python\n(.*?)```", text, flags=re.S):
        line = text.count("\n", 0, match.start()) + 2
        blocks.append((line, match.group(1)))
    return blocks


def _resolve_chain(node: ast.AST, bound: dict[str, Any]) -> Any:
    """Resolve ``a.b.c`` against vardax-bound names; ``None`` if unrelated."""
    if isinstance(node, ast.Name):
        return bound.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolve_chain(node.value, bound)
        if owner is None or owner is _MISSING:
            return owner
        value = getattr(owner, node.attr, _MISSING)
        if value is _MISSING and inspect.ismodule(owner):
            # a submodule not yet imported into its parent
            try:
                value = importlib.import_module(f"{owner.__name__}.{node.attr}")
            except ImportError:
                value = _MISSING
        return value
    return None


def _has_attribute(cls: type, name: str) -> bool:
    if hasattr(cls, name):
        return True
    if dataclasses.is_dataclass(cls):
        return name in {f.name for f in dataclasses.fields(cls)}
    return False


def _bind_imports(tree: ast.AST, problems: list[str]) -> dict[str, Any]:
    """Names the block binds to vardax objects; unresolved imports → problems."""
    bound: dict[str, Any] = dict(ALIASES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "vardax":
                    continue
                try:
                    module = importlib.import_module(alias.name)
                except ImportError:
                    problems.append(f"import {alias.name}: no such module")
                    continue
                bound[alias.asname or "vardax"] = module if alias.asname else vardax
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] != "vardax":
                continue
            try:
                module = importlib.import_module(node.module)
            except ImportError:
                problems.append(f"from {node.module} import ...: no such module")
                continue
            for alias in node.names:
                value = getattr(module, alias.name, _MISSING)
                if value is _MISSING:
                    problems.append(f"from {node.module} import {alias.name}: missing")
                bound[alias.asname or alias.name] = value
    return bound


def _check_keywords(node: ast.Call, bound: dict[str, Any]) -> list[str]:
    """Keyword arguments of a call to a vardax callable must be parameters."""
    target = _resolve_chain(node.func, bound)
    if target is None or target is _MISSING or not callable(target):
        return []
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    return [
        f"{ast.unparse(node.func)}({keyword.arg}=...): no such parameter"
        for keyword in node.keywords
        if keyword.arg and keyword.arg not in parameters
    ]


def _instances(tree: ast.AST, bound: dict[str, Any]) -> dict[str, type]:
    """Variables assigned an instance of a vardax class: ``x = vardax.C(...)``."""
    instances: dict[str, type] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            target = _resolve_chain(node.value.func, bound)
            if inspect.isclass(target):
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        instances[name.id] = target
    return instances


def _check_block(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        if "vardax" in source or "vdx" in source:
            return [f"block does not parse ({error.msg}, line {error.lineno})"]
        return []

    problems: list[str] = []
    bound = _bind_imports(tree, problems)
    instances = _instances(tree, bound)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            problems.extend(_check_keywords(node, bound))
        if not isinstance(node, ast.Attribute):
            continue
        # vardax.X chains
        value = _resolve_chain(node, bound)
        if value is _MISSING and _resolve_chain(node.value, bound) is not _MISSING:
            problems.append(f"{ast.unparse(node)}: no such attribute")
        # attributes of instances of vardax classes
        owner = node.value
        if isinstance(owner, ast.Name) and owner.id in instances:
            cls = instances[owner.id]
            if not _has_attribute(cls, node.attr):
                problems.append(
                    f"{owner.id}.{node.attr}: {cls.__name__} has no "
                    f"attribute {node.attr!r}"
                )
    return problems


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(REPO)))
def test_docs_api_names_resolve(page):
    problems = [
        f"{page.relative_to(REPO)}:{line}: {problem}"
        for line, block in _blocks(page)
        for problem in _check_block(block)
    ]
    assert not problems, "\n".join(problems)


class TestChecker:
    """The checker itself catches each kind of drift #99 found."""

    def test_missing_method(self):
        source = (
            "import vardax\nmodel = vardax.StrongFourDVar(forward=f, obs_op=o, "
            "prior_mean=m, prior_cov_op=b, obs_cov_op=r)\nmodel.not_a_method(batch)\n"
        )
        assert any("not_a_method" in p for p in _check_block(source))

    def test_missing_helper(self):
        assert _check_block("import vardax\nvardax.utils.multi_start(m)\n")

    def test_missing_keyword(self):
        source = "from vardax import IncrementalConfig\nIncrementalConfig(n_outer=2, nope=True)\n"
        assert any("nope" in p for p in _check_block(source))

    def test_missing_module(self):
        assert _check_block("from vardax.nonexistent import X\n")

    def test_valid_block(self):
        source = (
            "import vardax\nfrom vardax import IncrementalConfig\n"
            "IncrementalConfig(n_outer=2, cvt=True)\nvardax.adjoints.OneStepAdjoint\n"
        )
        assert _check_block(source) == []
