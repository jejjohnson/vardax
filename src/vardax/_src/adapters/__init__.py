"""Adapters for composing vardax with optional ecosystem libraries.

Each submodule corresponds to one upstream library that vardax knows how
to talk to without taking a hard dependency. The adapter modules are
only importable when the corresponding optional extra is installed.

- ``vardax.adapters.pipekit_train`` — Loss / training-loop integration with
  ``pipekit-train`` (install via ``vardax[train]``).
"""
