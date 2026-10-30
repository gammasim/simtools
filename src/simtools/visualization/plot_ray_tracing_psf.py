"""Shared PSF plotting helpers for ray-tracing workflows."""

from pathlib import Path

import astropy.units as u

from simtools.visualization import visualize
from simtools.visualization.matplotlib_backend import pyplot as plt


def create_cumulative_psf_figure(
    data, radius_key, cumulative_key, containment_radius_cm, psf_diameter_cm=None, **plot_kwargs
):
    """
    Create a cumulative PSF plot.

    Parameters
    ----------
    data : numpy.ndarray
        Structured array with radius and cumulative PSF columns.
    radius_key, cumulative_key : str
        Column names of the radius and cumulative PSF values.
    containment_radius_cm : float
        Containment marker radius in cm.
    psf_diameter_cm : float, optional
        PSF diameter marker in cm.
    **plot_kwargs
        Passed to ``Axes.plot``.

    Returns
    -------
    tuple
        Figure and axes.
    """
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_xlabel("Radius (cm)")
    ax.set_ylabel("Contained light %")
    ax.plot(data[radius_key], data[cumulative_key], **plot_kwargs)
    ax.axvline(x=containment_radius_cm, color="b", linestyle="--", linewidth=1)
    if psf_diameter_cm is not None:
        ax.axvline(x=psf_diameter_cm / 2.0, color="r", linestyle="--", linewidth=1)
    return fig, ax


def create_psf_image_figure(
    data, containment_radius, center, psf_kwargs=None, use_current_axes=False, **hist_kwargs
):
    """
    Create a 2D histogram plot of a PSF image with the containment circle.

    Parameters
    ----------
    data : numpy.ndarray
        Structured array with ``X`` and ``Y`` columns.
    containment_radius : float
        Radius of the containment circle, in the same unit as the data.
    center : tuple
        Center of the PSF circle, in the same unit as the data.
    psf_kwargs : dict, optional
        Keyword arguments passed to ``matplotlib.patches.Circle``.
    use_current_axes : bool
        Draw on the current axes instead of a new figure.
    **hist_kwargs
        Passed to ``visualize.plot_hist_2d`` (``image_range`` is an alias of ``range``).

    Returns
    -------
    tuple
        Figure and axes.
    """
    if "image_range" in hist_kwargs:
        hist_kwargs["range"] = hist_kwargs.pop("image_range")
    fig = visualize.plot_hist_2d(data, ax=plt.gca() if use_current_axes else None, **hist_kwargs)
    ax = fig.gca()
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.add_artist(plt.Circle(center, containment_radius, **(psf_kwargs or {})))
    for draw_line in (ax.axhline, ax.axvline):
        draw_line(0, color="k", linestyle="--", zorder=3, linewidth=0.5)
    return fig, ax


def create_annotated_psf_image_figure(
    data,
    off_x,
    off_y,
    psf,
    containment_radius,
    image_range,
    bins=150,
    cmap=None,
    psf_kwargs=None,
    telescope_name=None,
):
    """
    Create a PSF image figure annotated with offset and PSF information.

    Parameters
    ----------
    data : numpy.ndarray
        Structured array with ``X`` and ``Y`` columns, in the display unit.
    off_x, off_y : float
        Off-axis components in degrees.
    psf : astropy.units.Quantity
        PSF diameter; its unit sets the axis labels and annotation.
    containment_radius : astropy.units.Quantity
        Containment radius in the same unit as ``psf``.
    image_range : list
        Range passed to ``Axes.hist2d``.
    bins : int
        Number of histogram bins.
    cmap : matplotlib colormap, optional
        Colormap passed to ``Axes.hist2d``.
    psf_kwargs : dict, optional
        Keyword arguments passed to ``matplotlib.patches.Circle``.
    telescope_name : str, optional
        Telescope name shown in the annotation box.

    Returns
    -------
    matplotlib.figure.Figure
        Created figure.
    """
    unit_str = psf.unit.to_string()
    psf_fmt = ".4f" if psf.unit == u.deg else ".2f"
    fig, _ = plt.subplots(figsize=(8, 6), tight_layout=True)
    _, ax = create_psf_image_figure(
        data,
        containment_radius=containment_radius.to(psf.unit).value,
        center=(0, 0),
        use_current_axes=True,
        image_range=image_range,
        bins=bins,
        cmap=cmap,
        rasterized=True,
        psf_kwargs=psf_kwargs,
    )
    ax.set_xlabel(f"X Position ({unit_str})")
    ax.set_ylabel(f"Y Position ({unit_str})")

    offset_text = f"Offset: ({off_x:+.2f} deg, {off_y:+.2f} deg)"
    psf_text = f"PSF: {psf.value:{psf_fmt}} {unit_str}"
    text = "\n".join(filter(None, [telescope_name, offset_text, psf_text]))
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=10,
        family="monospace",
    )
    return fig


def save_and_close_figure(fig, file_name):
    """Save a figure to an exact path and close it."""
    file_name = Path(file_name)
    visualize.save_figure(
        fig, file_name.with_suffix(""), figure_format=[file_name.suffix[1:]], close=True
    )
