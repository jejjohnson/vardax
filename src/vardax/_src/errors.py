"""Package errors: one base class, with subclasses named by what failed.

Each subclass also inherits the builtin it replaces (``ValueError`` or
``TypeError``), so ``except ValueError`` in caller code keeps working.
"""


class VardaxError(Exception):
    """Base class for every error vardax raises on purpose.

    Examples:
        Catch any vardax error with the base class:

        >>> import vardax as vdx
        >>> try:
        ...     vdx.OptimistixMinimiser(max_steps=0)
        ... except vdx.VardaxError as error:
        ...     type(error).__name__
        'ConfigurationError'

        Every subclass is still the builtin it replaced:

        >>> issubclass(vdx.ShapeError, ValueError)
        True
        >>> issubclass(vdx.InputTypeError, TypeError)
        True
    """


class ConfigurationError(VardaxError, ValueError):
    """A hyperparameter or constructor argument has an invalid value."""


class ShapeError(VardaxError, ValueError):
    """An array or batch has a shape the operation cannot use."""


class GridError(VardaxError, ValueError):
    """Grid axes or coordinates are malformed (too few points, not increasing)."""


class ObservationError(VardaxError, ValueError):
    """Observations are inconsistent with the model or the assimilation window."""


class DataError(VardaxError, ValueError):
    """A dataset cannot be processed (no observed values, too few samples)."""


class MissingInputError(VardaxError, ValueError):
    """A required input (a key, a target, a covariance) was not given."""


class UnsupportedInputError(VardaxError, ValueError):
    """The input is valid in general but this operation does not support it."""


class InputTypeError(VardaxError, TypeError):
    """An argument has the wrong type."""
