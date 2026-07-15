"""Shared PSF plotting helpers for ray-tracing workflows."""

import matplotlib.pyplot as plt

RADIUS_LABEL = "Radius (cm)"
CONTAINED_LIGHT_LABEL = "Contained light %"
X_POSITION_LABEL = "X Position (cm)"
Y_POSITION_LABEL = "Y Position (cm)"


def create_cumulative_psf_figure(
    data,
    radius_key,
    cumulative_key,
    containment_radius_cm,
    psf_diameter_cm=None,
    ax=None,
    **plot_kwargs,
):
    """
    Create a cumulative PSF plot.

    Parameters
    ----------
    data : numpy.ndarray
        Structured array with radius and cumulative PSF columns.
    radius_key : str
        Column name for the radius values.
    cumulative_key : str
        Column name for the cumulative PSF values.
    containment_radius_cm : float
        Radius of the containment marker in cm.
    psf_diameter_cm : float, optional
        Additional PSF diameter marker in cm.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    **plot_kwargs
        Keyword arguments passed to ``Axes.plot``.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.figure

    ax.set_xlabel(RADIUS_LABEL)
    ax.set_ylabel(CONTAINED_LIGHT_LABEL)
    ax.plot(data[radius_key], data[cumulative_key], **plot_kwargs)
    ax.axvline(x=containment_radius_cm, color="b", linestyle="--", linewidth=1)
    if psf_diameter_cm is not None:
        ax.axvline(x=psf_diameter_cm / 2.0, color="r", linestyle="--", linewidth=1)
    return fig, ax


def create_psf_image_figure(
    data,
    containment_radius_cm,
    center,
    ax=None,
    psf_kwargs=None,
    show_reference_axes=True,
    **hist_kwargs,
):
    """
    Create a 2D histogram plot for a PSF image.

    Parameters
    ----------
    data : numpy.ndarray
        Structured array with ``X`` and ``Y`` columns.
    containment_radius_cm : float
        Radius of the containment circle in cm.
    center : tuple
        Center of the PSF circle in cm.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    psf_kwargs : dict, optional
        Keyword arguments passed to ``matplotlib.patches.Circle``.
    show_reference_axes : bool
        Draw dashed x/y reference axes if True.
    **hist_kwargs
        Keyword arguments passed to ``Axes.hist2d``.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.set_xlabel(X_POSITION_LABEL)
    ax.set_ylabel(Y_POSITION_LABEL)
    ax.hist2d(data["X"], data["Y"], **hist_kwargs)
    ax.set_aspect("equal", "box")

    circle = plt.Circle(center, containment_radius_cm, **(psf_kwargs or {}))
    ax.add_artist(circle)

    if show_reference_axes:
        ax.axhline(0, color="k", linestyle="--", zorder=3, linewidth=0.5)
        ax.axvline(0, color="k", linestyle="--", zorder=3, linewidth=0.5)

    return fig, ax
