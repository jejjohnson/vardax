"""Visualization helpers, which take an optional ``ax`` and return figure and axes.

``matplotlib`` is imported inside each function, so it stays a soft dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vardax._src.errors import ConfigurationError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_3d_attractor(
    states: np.ndarray,
    *,
    dim_indices: tuple[int, int, int] = (0, 1, 2),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a 3-D attractor trajectory.

    Args:
        states: State trajectory, indexed ``[time, variable]``, with at least three
            variables.
        dim_indices: Indices of the three variables to plot.
        ax: Optional existing 3-D ``Axes`` to draw on.

    Returns:
        ``(fig, ax)``: the Matplotlib figure and 3-D axes.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> states = np.random.default_rng(0).normal(size=(20, 4))
        >>> fig, ax = plot_3d_attractor(states)
        >>> type(ax).__name__, ax.get_xlabel()
        ('Axes3D', 'x0')

        Any three variables can be chosen:

        >>> fig, ax = plot_3d_attractor(states, dim_indices=(3, 1, 0))
        >>> ax.get_xlabel(), ax.get_zlabel()
        ('x3', 'x0')
    """
    # lazy: matplotlib is a soft dependency
    import matplotlib.pyplot as plt

    x_index, y_index, z_index = dim_indices
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()
        assert fig is not None

    ax.plot(states[:, x_index], states[:, y_index], states[:, z_index], lw=0.5)
    ax.set_xlabel(f"x{x_index}")
    ax.set_ylabel(f"x{y_index}")
    z_name = f"x{z_index}"
    ax.set_zlabel(z_name)  # type: ignore[union-attr]  # ty:ignore[missing-attribute]
    return fig, ax  # type: ignore[return-value]  # ty:ignore[invalid-return-type]


def plot_state_grid(
    states: np.ndarray,
    time: np.ndarray,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Hovmöller-style image of states over time.

    Args:
        states: State trajectory, indexed ``[time, feature]``.
        time: Time coordinate of each row of ``states``.
        ax: Optional existing ``Axes`` to draw on.

    Returns:
        ``(fig, ax)``: the Matplotlib figure and axes.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> states = np.random.default_rng(0).normal(size=(20, 8))
        >>> fig, ax = plot_state_grid(states, np.linspace(0.0, 1.0, 20))
        >>> ax.get_xlabel(), ax.get_ylabel()
        ('time', 'feature index')

        Drawing onto an existing axes returns that axes:

        >>> import matplotlib.pyplot as plt
        >>> _, mine = plt.subplots()
        >>> _, same = plot_state_grid(states, np.linspace(0.0, 1.0, 20), ax=mine)
        >>> same is mine
        True
    """
    # lazy: matplotlib is a soft dependency
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        assert fig is not None

    ax.imshow(
        states.T,
        aspect="auto",
        origin="lower",
        extent=(float(time[0]), float(time[-1]), 0.0, float(states.shape[1])),
    )
    ax.set_xlabel("time")
    ax.set_ylabel("feature index")
    return fig, ax  # type: ignore[return-value]  # ty:ignore[invalid-return-type]


def plot_trajectories(
    states: np.ndarray,
    time: np.ndarray,
    *,
    ax: Axes | None = None,
    orientation: str = "horizontal",
) -> tuple[Figure, Axes]:
    """Overlay time-series of all state variables.

    Args:
        states: State trajectory, indexed ``[time, feature]``.
        time: Time coordinate of each row of ``states``.
        ax: Optional existing ``Axes`` to draw on.
        orientation: ``"horizontal"`` plots time on the x-axis; ``"vertical"`` rotates
            so that time is on the y-axis.

    Returns:
        ``(fig, ax)``: the Matplotlib figure and axes.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> states = np.random.default_rng(0).normal(size=(20, 3))
        >>> time = np.linspace(0.0, 1.0, 20)
        >>> fig, ax = plot_trajectories(states, time)
        >>> len(ax.lines), ax.get_xlabel()
        (3, 'time')

        ``orientation="vertical"`` puts time on the y-axis:

        >>> fig, ax = plot_trajectories(states, time, orientation="vertical")
        >>> ax.get_ylabel()
        'time'
    """
    # lazy: matplotlib is a soft dependency
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        assert fig is not None

    num_features = states.shape[1]
    for i in range(num_features):
        if orientation == "vertical":
            ax.plot(states[:, i], time, label=f"x{i}")
            ax.set_xlabel("state")
            ax.set_ylabel("time")
        else:
            ax.plot(time, states[:, i], label=f"x{i}")
            ax.set_xlabel("time")
            ax.set_ylabel("state")

    ax.legend()
    return fig, ax  # type: ignore[return-value]  # ty:ignore[invalid-return-type]


def plot_reconstruction_comparison(
    target: np.ndarray,
    masked_input: np.ndarray,
    reconstruction: np.ndarray,
    *,
    sample_idx: int = 0,
) -> tuple[Figure, np.ndarray]:
    """Side-by-side comparison of target, masked input, and reconstruction.

    Draws three panels (time on the x-axis, feature on the y-axis) for one
    batch element.

    Args:
        target: Ground-truth states, indexed ``[batch, time, feature]``.
        masked_input: Masked / noisy observations, same layout.
        reconstruction: Model reconstruction, same layout.
        sample_idx: Which batch element to visualize.

    Returns:
        ``(fig, axes)``: the Matplotlib figure and the array of three axes.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> import vardax as vdx
        >>> truth = np.random.default_rng(0).normal(size=(2, 6, 4))
        >>> fig, axes = vdx.plot_reconstruction_comparison(truth, truth, truth)
        >>> axes.shape, type(axes[0]).__name__
        ((3,), 'Axes')

        Another batch element can be selected:

        >>> fig, axes = vdx.plot_reconstruction_comparison(
        ...     truth, truth, truth, sample_idx=1
        ... )
        >>> axes[1].get_title()
        'Masked input'
    """
    # lazy: matplotlib is a soft dependency
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    data = [
        (target[sample_idx], "Target"),
        (masked_input[sample_idx], "Masked input"),
        (reconstruction[sample_idx], "Reconstruction"),
    ]
    for ax, (panel, title) in zip(axes, data, strict=False):
        ax.imshow(panel.T, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("feature")
    fig.tight_layout()
    return fig, axes


def plot_l96_grid(
    states: np.ndarray,
    time_coords: np.ndarray,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Hovmöller-style image of Lorenz-96 states over time.

    Args:
        states: State trajectory, indexed ``[time, variable]``.
        time_coords: Time coordinate of each row of ``states``.
        ax: Optional existing ``Axes`` to draw on.

    Returns:
        ``(fig, ax)``: the Matplotlib figure and axes.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> import vardax as vdx
        >>> states = np.random.default_rng(0).normal(size=(20, 8))
        >>> fig, ax = vdx.plot_l96_grid(states, np.linspace(0.0, 1.0, 20))
        >>> type(ax).__name__, ax.get_xlabel()
        ('Axes', 'time')

        Drawing onto an existing axes returns that axes:

        >>> import matplotlib.pyplot as plt
        >>> _, mine = plt.subplots()
        >>> _, same = vdx.plot_l96_grid(states, np.linspace(0.0, 1.0, 20), ax=mine)
        >>> same is mine
        True
    """
    # lazy: matplotlib is a soft dependency
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        assert fig is not None

    ax.imshow(
        states.T,
        aspect="auto",
        origin="lower",
        extent=(
            float(time_coords[0]),
            float(time_coords[-1]),
            0.0,
            float(states.shape[1]),
        ),
    )
    ax.set_xlabel("time")
    ax.set_ylabel("variable index")
    return fig, ax  # type: ignore[return-value]  # ty:ignore[invalid-return-type]


def plot_l96_trajectories(
    states: np.ndarray,
    time_coords: np.ndarray,
    *,
    n_vars: int = 5,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Line plot of selected Lorenz-96 variables over time.

    Args:
        states: State trajectory, indexed ``[time, variable]``.
        time_coords: Time coordinate of each row of ``states``.
        n_vars: Number of evenly-spaced variables to plot (capped at the
            number of variables).
        ax: Optional existing ``Axes`` to draw on.

    Returns:
        ``(fig, ax)``: the Matplotlib figure and axes.

    Raises:
        ConfigurationError: If ``n_vars`` is not positive.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> import vardax as vdx
        >>> states = np.random.default_rng(0).normal(size=(20, 8))
        >>> time = np.linspace(0.0, 1.0, 20)
        >>> fig, ax = vdx.plot_l96_trajectories(states, time, n_vars=3)
        >>> len(ax.lines)
        3

        Asking for more variables than exist plots all of them:

        >>> fig, ax = vdx.plot_l96_trajectories(states, time, n_vars=50)
        >>> len(ax.lines)
        8

        A non-positive count is rejected:

        >>> vdx.plot_l96_trajectories(
        ...     states, time, n_vars=0
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: n_vars must be a positive integer, got 0.
    """
    # lazy: matplotlib is a soft dependency
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        assert fig is not None

    num_variables = states.shape[1]
    if n_vars <= 0:
        raise ConfigurationError(f"n_vars must be a positive integer, got {n_vars}.")
    num_selected = min(n_vars, num_variables)
    indices = np.unique(np.linspace(0, num_variables - 1, num_selected, dtype=int))
    for i in indices:
        ax.plot(time_coords, states[:, i], label=f"x{i}")
    ax.set_xlabel("time")
    ax.set_ylabel("state")
    ax.legend()
    return fig, ax  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
