"""Docstring quality gates for the rendered API docs.

mkdocstrings renders docstrings through griffe's Google-style parser,
and three classes of mistake silently degrade the built docs:

* Doctest code under a header griffe does not recognise as an examples
  section (e.g. singular ``Example:``) is parsed as a generic
  admonition, and the ``>>>`` lines collapse into a run-together
  paragraph instead of a highlighted code block.
* Sphinx/RST math roles (``:math:``, ``.. math::``) are not rendered
  by MkDocs — equations must use arithmatex ``$...$`` / ``$$...$$``.
* Unbalanced ``$$`` math fences leave raw TeX in the rendered page.

These tests parse every docstring in the package exactly the way the
docs build does and fail on any of the three, so the problems cannot
reach the published site again.
"""

from __future__ import annotations

import griffe
from griffe import Docstring, DocstringSectionKind, Parser
import pytest


def _iter_docstrings(obj, seen):
    """Yield (path, Docstring) for every object defined in vardax."""
    if obj.path in seen:
        return
    seen.add(obj.path)
    if obj.docstring is not None:
        yield obj.path, obj.docstring
    members = getattr(obj, "members", {})
    for member in members.values():
        # Skip aliases (re-exports): pipekit-owned docstrings are
        # tested upstream, and vardax-owned ones are reached at their
        # definition site.
        if member.is_alias:
            continue
        yield from _iter_docstrings(member, seen)


@pytest.fixture(scope="module")
def all_docstrings() -> list[tuple[str, Docstring]]:
    pkg = griffe.load("vardax", search_paths=["src"])
    docstrings = list(_iter_docstrings(pkg, set()))
    assert len(docstrings) > 50  # the walk found the real package
    return docstrings


def test_doctests_parse_as_examples_sections(all_docstrings):
    """Every ``>>>`` snippet must land in a real examples section.

    A doctest inside a text or admonition section means the section
    header is one griffe does not recognise (the ``Example:`` vs
    ``Examples:`` trap) and the code will render as prose.
    """
    offenders = []
    for path, docstring in all_docstrings:
        for section in docstring.parse(Parser.google):
            if section.kind is DocstringSectionKind.examples:
                continue
            if section.kind is DocstringSectionKind.admonition:
                text = section.value.contents
            elif section.kind is DocstringSectionKind.text:
                text = section.value
            else:
                continue
            if ">>>" in str(text):
                offenders.append(f"{path} ({section.kind.value})")
    assert not offenders, (
        "Doctest code outside an examples section — use the plural "
        f"'Examples:' header so mkdocstrings renders it as code: {offenders}"
    )


def test_examples_sections_contain_code(all_docstrings):
    """Parsed examples sections must carry at least one code part."""
    offenders = []
    for path, docstring in all_docstrings:
        for section in docstring.parse(Parser.google):
            if section.kind is not DocstringSectionKind.examples:
                continue
            kinds = [kind for kind, _ in section.value]
            if DocstringSectionKind.examples not in kinds:
                offenders.append(path)
    assert not offenders, f"Examples sections without doctest code: {offenders}"


def test_no_sphinx_math_roles(all_docstrings):
    """Equations must use arithmatex ``$``/``$$`` syntax, not RST roles.

    MkDocs does not render ``:math:`` or ``.. math::`` — they appear
    as literal text in the built pages.
    """
    offenders = [
        path
        for path, docstring in all_docstrings
        if ":math:" in docstring.value or ".. math::" in docstring.value
    ]
    assert not offenders, (
        f"Sphinx math roles do not render under MkDocs — use $...$ "
        f"inline or $$...$$ display blocks instead: {offenders}"
    )


def test_math_fences_are_balanced(all_docstrings):
    """``$$`` math fences must come in pairs, or raw TeX leaks into the docs."""
    offenders = [
        path
        for path, docstring in all_docstrings
        if docstring.value.count("$$") % 2 != 0
    ]
    assert not offenders, f"Unbalanced $$ math fences: {offenders}"
