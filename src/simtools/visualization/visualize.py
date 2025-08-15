#!/usr/bin/python3
"""Module for visualization."""

import logging
import re
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.table import QTable
from cycler import cycler
from matplotlib import gridspec

__all__ = [
    "get_colors",
    "get_lines",
    "get_markers",
    "plot_1d",
    "plot_hist_2d",
    "plot_simtel_ctapipe",
    "plot_simtel_event_image",
    "plot_simtel_integrated_pedestal_image",
    "plot_simtel_integrated_signal_image",
    "plot_simtel_peak_timing",
    "plot_simtel_step_traces",
    "plot_simtel_time_traces",
    "plot_simtel_waveform_pcolormesh",
    "plot_table",
    "save_figure",
    "set_style",
]

COLORS = {}
COLORS["classic"] = [
    "#ba2c54",
    "#5B90DC",
    "#FFAB44",
    "#0C9FB3",
    "#57271B",
    "#3B507D",
    "#794D88",
    "#FD6989",
    "#8A978E",
    "#3B507D",
    "#D8153C",
    "#cc9214",
]
COLORS["modified classic"] = [
    "#D6088F",
    "#424D9C",
    "#178084",
    "#AF99DA",
    "#F58D46",
    "#634B5B",
    "#0C9FB3",
    "#7C438A",
    "#328cd6",
    "#8D0F25",
    "#8A978E",
    "#ffcb3d",
]
COLORS["autumn"] = [
    "#A9434D",
    "#4E615D",
    "#3C8DAB",
    "#A4657A",
    "#424D9C",
    "#DC575A",
    "#1D2D38",
    "#634B5B",
    "#56276D",
    "#577580",
    "#134663",
    "#196096",
]
COLORS["purples"] = [
    "#a57bb7",
    "#343D80",
    "#EA60BF",
    "#B7308E",
    "#E099C3",
    "#7C438A",
    "#AF99DA",
    "#4D428E",
    "#56276D",
    "#CC4B93",
    "#DC4E76",
    "#5C4AE4",
]
COLORS["greens"] = [
    "#268F92",
    "#abc14d",
    "#8A978E",
    "#0C9FB3",
    "#BDA962",
    "#B0CB9E",
    "#769168",
    "#5E93A5",
    "#178084",
    "#B7BBAD",
    "#163317",
    "#76A63F",
]

COLORS["default"] = COLORS["classic"]

MARKERS = ["o", "s", "v", "^", "*", "P", "d", "X", "p", "<", ">", "h"]
LINES = [
    (0, ()),  # solid
    (0, (1, 1)),  # densely dotted
    (0, (3, 1, 1, 1)),  # densely dashdotted
    (0, (5, 5)),  # dashed
    (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
    (0, (5, 1)),  # densely dashed
    (0, (1, 5)),  # dotted
    (0, (3, 5, 1, 5)),  # dashdotted
    (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
    (0, (5, 10)),  # loosely dashed
    (0, (1, 10)),  # loosely dotted
    (0, (3, 10, 1, 10)),  # loosely dashdotted
]

_logger = logging.getLogger(__name__)


def _add_unit(title, array):
    """
    Add a unit to "title" (presumably an axis title).

    The unit is extracted from the unit field of the array, in case array is an astropy quantity.
    If a unit is found, it is added to title in the form [unit]. If a unit already is present in
    title (in the same form), a warning is printed and no unit is added. The function assumes
    array not to be empty and returns the modified title.

    Parameters
    ----------
    title: str
    array: astropy.Quantity

    Returns
    -------
    str
        Title with units.
    """
    if title and "[" in title and "]" in title:
        _logger.warning(
            "Tried to add a unit from astropy.unit, "
            "but axis already has an explicit unit. Left axis title as is."
        )
        return title

    if not isinstance(array, u.Quantity) or not str(array[0].unit):
        return title

    unit = str(array[0].unit)
    unit_str = f" [{unit}]"
    if re.search(r"\d", unit_str):
        unit_str = re.sub(r"(\d)", r"^\1", unit_str)
        unit_str = unit_str.replace("[", r"[$").replace("]", r"$]")

    return f"{title}{unit_str}"


def set_style(palette="default", big_plot=False):
    """
    Set the plotting style to homogenize plot style across the framework.

    The function receives the colour palette name and whether it is a big plot or not.\
    The latter sets the fonts and marker to be bigger in case it is a big plot. The available \
    colour palettes are as follows:

    - classic (default): A classic colorful palette with strong colors and contrast.
    - modified classic: Similar to the classic, with slightly different colors.
    - autumn: A slightly darker autumn style colour palette.
    - purples: A pseudo sequential purple colour palette (not great for contrast).
    - greens: A pseudo sequential green colour palette (not great for contrast).

    To use the function, simply call it before plotting anything. The function is made public, so \
    that it can be used outside the visualize module. However, it is highly recommended to create\
    plots only through the visualize module.

    Parameters
    ----------
    palette: str
        Colour palette.
    big_plot: bool
        Flag to set fonts and marker bigger. If True, it sets them bigger.

    Raises
    ------
    KeyError
        if provided palette does not exist.
    """
    if palette not in COLORS:
        raise KeyError(f"palette must be one of {', '.join(COLORS)}")

    fontsize = {"default": 17, "big_plot": 30}
    markersize = {"default": 6, "big_plot": 18}
    plot_size = "big_plot" if big_plot else "default"

    plt.rc("lines", linewidth=2, markersize=markersize[plot_size])
    plt.rc(
        "axes",
        prop_cycle=(
            cycler(color=COLORS[palette]) + cycler(linestyle=LINES) + cycler(marker=MARKERS)
        ),
        titlesize=fontsize[plot_size],
        labelsize=fontsize[plot_size],
        labelpad=5,
        grid=True,
        axisbelow=True,
    )
    plt.rc("xtick", labelsize=fontsize[plot_size])
    plt.rc("ytick", labelsize=fontsize[plot_size])
    plt.rc("legend", loc="best", shadow=False, fontsize="medium")
    plt.rc("font", family="serif", size=fontsize[plot_size])


def get_colors(palette="default"):
    """
    Get the colour list of the palette requested.

    If no palette is provided, the default is returned.

    Parameters
    ----------
    palette: str
        Colour palette.

    Returns
    -------
    list
        Colour list.

    Raises
    ------
    KeyError
        if provided palette does not exist.
    """
    if palette not in COLORS:
        raise KeyError(f"palette must be one of {', '.join(COLORS)}")

    return COLORS[palette]


def get_markers():
    """
    Get the marker list used in this module.

    Returns
    -------
    list
        List with markers.
    """
    return MARKERS


def get_lines():
    """
    Get the line style list used in this module.

    Returns
    -------
    list
        List with line styles.
    """
    return LINES


def filter_plot_kwargs(kwargs):
    """Filter out kwargs that are not valid for plt.plot."""
    valid_keys = {
        "color",
        "linestyle",
        "linewidth",
        "marker",
        "markersize",
        "markerfacecolor",
        "markeredgecolor",
        "markeredgewidth",
        "alpha",
        "label",
        "zorder",
        "dashes",
        "gapcolor",
        "solid_capstyle",
        "solid_joinstyle",
        "dash_capstyle",
        "dash_joinstyle",
        "antialiased",
    }
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def plot_1d(data, **kwargs):
    """
    Produce a high contrast one dimensional plot from multiple data sets.

    A ratio plot can be added at the bottom to allow easy comparison. Additional options,
    such as plot title, plot legend, etc., are given in kwargs. Any option that can be
    changed after plotting (e.g., axes limits, log scale, etc.) should be done using the
    returned plt instance.

    Parameters
    ----------
    data: numpy structured array or a dictionary of structured arrays
          Each structured array has two columns, the first is the x-axis and the second the y-axis.
          The titles of the columns are set as the axes titles.
          The labels of each dataset set are given in the dictionary keys
          and will be used in the legend.
    **kwargs:
        * palette: string
          Choose a colour palette (see set_style for additional information).
        * title: string
          Set a plot title.
        * no_legend: bool
          Do not print a legend for the plot.
        * big_plot: bool
          Increase marker and font sizes (like in a wide light curve).
        * no_markers: bool
          Do not print markers.
        * empty_markers: bool
          Print empty (hollow) markers
        * plot_ratio: bool
          Add a ratio plot at the bottom. The first entry in the data dictionary
          is used as the reference for the ratio.
          If data dictionary is not an OrderedDict, the reference will be random.
        * plot_difference: bool
          Add a difference plot at the bottom. The first entry in the data dictionary
          is used as the reference for the difference.
          If data dictionary is not an OrderedDict, the reference will be random.
        * Any additional kwargs for plt.plot

    Returns
    -------
    pyplot.figure
        Instance of pyplot.figure in which the plot was produced

    Raises
    ------
    ValueError
        if asked to plot a ratio or difference with just one set of data
    """
    kwargs = handle_kwargs(kwargs)
    data_dict, plot_ratio, plot_difference = prepare_data(data, kwargs)
    fig, ax1, gs = setup_plot(kwargs, plot_ratio, plot_difference)
    plot_args = filter_plot_kwargs(kwargs)
    plot_main_data(data_dict, kwargs, plot_args)

    if plot_ratio or plot_difference:
        plot_ratio_difference(ax1, data_dict, plot_ratio, gs, plot_args)

    if not (plot_ratio or plot_difference):
        plt.tight_layout()

    return fig


def handle_kwargs(kwargs):
    """Extract and handle keyword arguments."""
    kwargs_defaults = {
        "palette": "default",
        "big_plot": False,
        "title": "",
        "no_legend": False,
        "no_markers": False,
        "empty_markers": False,
        "plot_ratio": False,
        "plot_difference": False,
        "xscale": "linear",
        "yscale": "linear",
        "xlim": (None, None),
        "ylim": (None, None),
        "xtitle": None,
        "ytitle": None,
    }
    for key, default in kwargs_defaults.items():
        kwargs[key] = kwargs.pop(key, default)

    if kwargs["no_markers"]:
        kwargs["marker"] = "None"
        kwargs["linewidth"] = 4
    if kwargs["empty_markers"]:
        kwargs["markerfacecolor"] = "None"

    return kwargs


def prepare_data(data, kwargs):
    """Prepare data for plotting."""
    if not isinstance(data, dict):
        data_dict = {"_default": data}
    else:
        data_dict = data

    if (kwargs["plot_ratio"] or kwargs["plot_difference"]) and len(data_dict) < 2:
        raise ValueError("Asked to plot a ratio or difference with just one set of data")

    return data_dict, kwargs["plot_ratio"], kwargs["plot_difference"]


def setup_plot(kwargs, plot_ratio, plot_difference):
    """Set up the plot style, figure, and gridspec."""
    set_style(kwargs["palette"], kwargs["big_plot"])

    if plot_ratio or plot_difference:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        fig = plt.figure(figsize=(8, 8))
        gs.update(hspace=0.02)
    else:
        gs = gridspec.GridSpec(1, 1)
        fig = plt.figure(figsize=(8, 6))

    plt.subplot(gs[0])
    ax1 = plt.gca()

    return fig, ax1, gs


def _plot_error_plots(kwargs, data_now, x_col, y_col, x_err_col, y_err_col, color):
    """Plot error plots."""
    if kwargs.get("error_type") == "fill_between" and y_err_col:
        plt.fill_between(
            data_now[x_col],
            data_now[y_col] - data_now[y_err_col],
            data_now[y_col] + data_now[y_err_col],
            alpha=0.2,
            color=color,
        )

    if kwargs.get("error_type") == "errorbar" and (x_err_col or y_err_col):
        plt.errorbar(
            data_now[x_col],
            data_now[y_col],
            xerr=data_now[x_err_col] if x_err_col else None,
            yerr=data_now[y_err_col] if y_err_col else None,
            fmt=".",
            color=color,
        )


def _get_data_columns(data_now):
    """Return data columns depending on availability."""
    columns = data_now.dtype.names
    assert len(columns) >= 2, "Input array must have at least two columns with titles."
    x_col, y_col = columns[:2]
    if len(columns) == 3:
        x_err_col = None
        y_err_col = columns[2]
    elif len(columns) == 4:
        x_err_col = columns[2]
        y_err_col = columns[3]
    else:
        x_err_col = None
        y_err_col = None
    return x_col, y_col, x_err_col, y_err_col


def plot_main_data(data_dict, kwargs, plot_args):
    """Plot the main data."""
    for label, data_now in data_dict.items():
        x_col, y_col, x_err_col, y_err_col = _get_data_columns(data_now)

        x_title = kwargs.get("xtitle", x_col)
        y_title = kwargs.get("ytitle", y_col)

        x_title_unit = _add_unit(x_title, data_now[x_col])
        y_title_unit = _add_unit(y_title, data_now[y_col])

        (line,) = plt.plot(data_now[x_col], data_now[y_col], label=label, **plot_args)
        color = line.get_color()

        _plot_error_plots(kwargs, data_now, x_col, y_col, x_err_col, y_err_col, color)

    plt.xscale(kwargs["xscale"])
    plt.yscale(kwargs["yscale"])
    plt.xlim(kwargs["xlim"])
    plt.ylim(kwargs["ylim"])
    plt.ylabel(y_title_unit)

    if not (kwargs["plot_ratio"] or kwargs["plot_difference"]):
        plt.xlabel(x_title_unit)

    if kwargs["title"]:
        plt.title(kwargs["title"], y=1.02)

    if "_default" not in data_dict and not kwargs["no_legend"]:
        plt.legend()


def plot_ratio_difference(ax1, data_dict, plot_ratio, gs, plot_args):
    """Plot the ratio or difference plot."""
    plt.subplot(gs[1], sharex=ax1)
    plt.plot([], [])  # Advance cycler for consistent colors/styles

    data_ref_name = next(iter(data_dict))
    for label, data_now in data_dict.items():
        if label == data_ref_name:
            continue
        x_title, y_title = data_now.dtype.names[0], data_now.dtype.names[1]
        x_title_unit = _add_unit(x_title, data_now[x_title])

        if plot_ratio:
            y_values = data_now[y_title] / data_dict[data_ref_name][y_title]
            y_title_ratio = f"Ratio to {data_ref_name}"
        else:
            y_values = data_now[y_title] - data_dict[data_ref_name][y_title]
            y_title_ratio = f"Difference to {data_ref_name}"

        plt.plot(data_now[x_title], y_values, **plot_args)

    plt.xlabel(x_title_unit)

    if len(y_title_ratio) > 20 and plot_ratio:
        y_title_ratio = "Ratio"

    plt.ylabel(y_title_ratio)
    ylim = plt.gca().get_ylim()
    nbins = min(int((ylim[1] - ylim[0]) / 0.05 + 1), 6)
    plt.locator_params(axis="y", nbins=nbins)


def plot_table(table, y_title, **kwargs):
    """
    Produce a high contrast one dimensional plot from the data in an astropy.Table.

    A ratio plot can be added at the bottom to allow easy comparison. Additional options, such
    as plot title, plot legend, etc., are given in kwargs. Any option that can be changed after
    plotting (e.g., axes limits, log scale, etc.) should be done using the returned plt instance.

    Parameters
    ----------
    table: astropy.Table or astropy.QTable
           The first column of the table is the x-axis and the second column is the y-axis. Any \
           additional columns will be treated as additional data to plot. The column titles are \
           used in the legend (except for the first column).
    y_title: str
           The y-axis title.

    **kwargs:
        * palette: choose a colour palette (see set_style for additional information).
        * title: set a plot title.
        * no_legend: do not print a legend for the plot.
        * big_plot: increase marker and font sizes (like in a wide light curve).
        * no_markers: do not print markers.
        * empty_markers: print empty (hollow) markers
        * plot_ratio: bool
          Add a ratio plot at the bottom. The first entry in the data dictionary is used as the \
          reference for the ratio. If data dictionary is not an OrderedDict, the reference will be\
          random.
        * plot_difference: bool
          Add a difference plot at the bottom. The first entry in the data dictionary is used as \
          the reference for the difference. If data dictionary is not an OrderedDict, the reference\
          will be random.
        * Any additional kwargs for plt.plot

    Returns
    -------
    pyplot.fig
        Instance of pyplot.fig.

    Raises
    ------
    ValueError
        if table has less than two columns.
    """
    if len(table.keys()) < 2:
        raise ValueError("Table has to have at least two columns")

    x_axis = table.keys()[0]
    data_dict = OrderedDict()
    for column in table.keys()[1:]:
        data_dict[column] = QTable([table[x_axis], table[column]], names=[x_axis, y_title])

    return plot_1d(data_dict, **kwargs)


def plot_hist_2d(data, **kwargs):
    """
    Produce a two dimensional histogram plot.

    Any option that can be changed after plotting (e.g., axes limits, log scale, etc.)
    should be done using the returned plt instance.

    Parameters
    ----------
    data: numpy structured array
          The columns of the structured array are used as the x-axis and y-axis titles.
    **kwargs:
        * title: set a plot title.
        * Any additional kwargs for plt.hist2d

    Returns
    -------
    pyplot.figure
        Instance of pyplot.figure in which the plot was produced.

    """
    cmap = plt.cm.gist_heat_r
    if "title" in kwargs:
        title = kwargs["title"]
        kwargs.pop("title", None)
    else:
        title = ""

    # Set default style since the usual options do not affect 2d plots (for now).
    set_style()

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(figsize=(8, 6))

    ##########################################################################################
    # Plot the data
    ##########################################################################################

    plt.subplot(gs[0])
    assert len(data.dtype.names) == 2, "Input array must have two columns with titles."
    x_title, y_title = data.dtype.names[0], data.dtype.names[1]
    x_title_unit = _add_unit(x_title, data[x_title])
    y_title_unit = _add_unit(y_title, data[y_title])
    plt.hist2d(data[x_title], data[y_title], cmap=cmap, **kwargs)

    plt.xlabel(x_title_unit)
    plt.ylabel(y_title_unit)

    plt.gca().set_aspect("equal", adjustable="datalim")

    if len(title) > 0:
        plt.title(title, y=1.02)

    fig.tight_layout()

    return fig


def plot_simtel_ctapipe(filename, cleaning_args, distance, return_cleaned=False):
    """
    Read in a sim_telarray file and plots reconstructed photoelectrons via ctapipe.

    Parameters
    ----------
    filename : str
        Path to the sim_telarray file.
    cleaning_args : tuple, optional
        Cleaning parameters as (boundary_thresh, picture_thresh, min_number_picture_neighbors).
    distance : astropy Quantity, optional
        Distance to the target.
    return_cleaned : bool, optional
        If True, apply cleaning to the image.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the plot.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import tailcuts_clean
    from ctapipe.io import EventSource
    from ctapipe.visualization import CameraDisplay

    default_cleaning_levels = {
        "CHEC": (2, 4, 2),
        "LSTCam": (3.5, 7, 2),
        "FlashCam": (3.5, 7, 2),
        "NectarCam": (4, 8, 2),
    }

    source = EventSource(filename, max_events=1)
    event = None
    for ev in source:
        event = ev
        break
    if event is None:
        _logger.warning(f"No events found in {filename}")
        return None
    tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if not tel_ids:
        _logger.warning("First event has no R1 telescope data")
        return None
    tel_id = tel_ids[0]

    calib = CameraCalibrator(subarray=source.subarray)
    calib(event)

    geometry = source.subarray.tel[tel_id].camera.geometry
    image = event.dl1.tel[tel_id].image
    cleaned = image.copy()

    if return_cleaned:
        if cleaning_args is None:
            boundary, picture, min_neighbors = default_cleaning_levels[geometry.name]
        else:
            boundary, picture, min_neighbors = cleaning_args
        mask = tailcuts_clean(
            geometry,
            image,
            picture_thresh=picture,
            boundary_thresh=boundary,
            min_number_picture_neighbors=min_neighbors,
        )
        cleaned[~mask] = 0

    fig, ax = plt.subplots(dpi=300)
    title = f"CT{tel_id}, run {event.index.obs_id} event {event.index.event_id}"
    disp = CameraDisplay(geometry, image=cleaned, norm="symlog", ax=ax)
    disp.cmap = "RdBu_r"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)
    ax.set_title(title, pad=20)
    if distance is not None:
        try:
            d_str = f"{distance.to(u.m)}"
        except (AttributeError, TypeError, ValueError):
            d_str = str(distance)
    else:
        d_str = "n/a"
    ax.annotate(
        f"tel type: {source.subarray.tel[tel_id].type.name}\n"
        f"optics: {source.subarray.tel[tel_id].optics.name}\n"
        f"camera: {source.subarray.tel[tel_id].camera_name}\n"
        f"distance: {d_str}",
        xy=(0, 0),
        xytext=(0.1, 1),
        xycoords="axes fraction",
        va="top",
        size=7,
    )
    ax.annotate(
        f"dl1 image,\ntotal $p.e._{{reco}}$: {np.round(np.sum(image))}\n",
        xy=(0, 0),
        xytext=(0.75, 1),
        xycoords="axes fraction",
        va="top",
        ha="left",
        size=7,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _select_event_by_type(source, preferred: str | None):
    """Return the first event from the source.

    The sim_telarray MC files often don't contain reliable trigger event_type
    metadata for flasher/pedestal discrimination, so we keep this simple and
    just return the first available event. If a preferred type is provided, we
    log that filtering is not applied.
    """
    for ev in source:
        if preferred:
            _logger.info(f"Event type filtering ('{preferred}') not applied; returning first event")
        return ev
    _logger.warning("No events available from source")
    return None


def plot_simtel_event_image(
    filename,
    event_type: str | None = None,
    cleaning_args=None,
    distance=None,
    return_cleaned: bool = False,
):
    """
    Plot a single sim_telarray event image filtered by event type (flasher/pedestal).

    Parameters
    ----------
    filename : str
        Path to the sim_telarray file.
    event_type : str | None
        Event type to select: 'flasher' or 'pedestal'. If None, first event is used.
    cleaning_args : tuple | None
        Cleaning parameters as (boundary_thresh, picture_thresh, min_neighbors).
    distance : astropy Quantity | None
        Distance annotation.
    return_cleaned : bool
        If True, apply tailcuts cleaning.

    Returns
    -------
    fig : matplotlib.figure.Figure | None
        The figure or None if no event found.
    """
    # pylint:disable=import-outside-toplevel
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import tailcuts_clean
    from ctapipe.io import EventSource
    from ctapipe.visualization import CameraDisplay

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source, event_type)
    if event is None:
        _logger.warning(f"No event found in {filename} matching type='{event_type}'")
        return None

    calib = CameraCalibrator(subarray=source.subarray)
    calib(event)

    # Prefer DL1 telescopes after calibration; fallback to R1 if needed
    dl1_tel_ids = sorted(getattr(event.dl1, "tel", {}).keys())
    if dl1_tel_ids:
        tel_id = dl1_tel_ids[0]
    else:
        r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
        if not r1_tel_ids:
            _logger.warning("Event has no DL1 or R1 telescope data")
            return None
        tel_id = r1_tel_ids[0]

    geometry = source.subarray.tel[tel_id].camera.geometry
    try:
        image = event.dl1.tel[tel_id].image
    except (AttributeError, KeyError):
        _logger.warning("No DL1 image available for selected telescope")
        return None

    cleaned = image.copy()

    if return_cleaned:
        if cleaning_args is None:
            # defaults per camera (as in plot_simtel_ctapipe)
            defaults = {
                "CHEC": (2, 4, 2),
                "LSTCam": (3.5, 7, 2),
                "FlashCam": (3.5, 7, 2),
                "NectarCam": (4, 8, 2),
            }
            boundary, picture, min_neighbors = defaults.get(geometry.name, (3, 6, 2))
        else:
            boundary, picture, min_neighbors = cleaning_args
        mask = tailcuts_clean(
            geometry,
            image,
            picture_thresh=picture,
            boundary_thresh=boundary,
            min_number_picture_neighbors=min_neighbors,
        )
        cleaned[~mask] = 0

    fig, ax = plt.subplots(dpi=300)
    disp = CameraDisplay(geometry, image=cleaned, norm="symlog", ax=ax)
    disp.cmap = "RdBu_r"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)

    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    title = f"CT{tel_id}, run {event.index.obs_id} event {event.index.event_id} ({et_name})"
    ax.set_title(title, pad=20)

    if distance is not None:
        try:
            d_str = f"{distance.to(u.m)}"
        except (AttributeError, TypeError, ValueError):
            d_str = str(distance)
    else:
        d_str = "n/a"

    ax.annotate(
        f"tel type: {source.subarray.tel[tel_id].type.name}\n"
        f"optics: {source.subarray.tel[tel_id].optics.name}\n"
        f"camera: {source.subarray.tel[tel_id].camera_name}\n"
        f"distance: {d_str}",
        xy=(0, 0),
        xytext=(0.1, 1),
        xycoords="axes fraction",
        va="top",
        size=7,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_simtel_time_traces(
    filename,
    event_type: str | None = None,
    tel_id: int | None = None,
    n_pixels: int = 3,
):
    """
    Plot time traces (R1 waveforms) for a few camera pixels of a selected event.

    Pixels are chosen as the highest-amplitude pixels from the dl1 image (if available),
    otherwise by integrated waveform amplitude.

    Parameters
    ----------
    filename : str
        Path to the sim_telarray file.
    event_type : str | None
        Event type to select: 'flasher' or 'pedestal'. If None, first event is used.
    tel_id : int | None
        Telescope id to plot. If None, use the first available.
    n_pixels : int
        Number of pixel traces to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure | None
        The figure or None if no event/waveforms found.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.calib import CameraCalibrator
    from ctapipe.io import EventSource

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source, event_type)
    if event is None:
        _logger.warning(f"No event found in {filename} matching type='{event_type}'")
        return None

    # Determine telescope id for waveforms (R1), fallback to DL1 if R1 not present
    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        dl1_tel_ids = sorted(getattr(event.dl1, "tel", {}).keys())
        if not dl1_tel_ids:
            _logger.warning("Event has no R1 or DL1 telescope data for traces")
            return None
        tel_id = tel_id or dl1_tel_ids[0]

    # Calibrate to get dl1 image for pixel selection if possible
    calib = CameraCalibrator(subarray=source.subarray)
    try:
        calib(event)
        image = event.dl1.tel[tel_id].image
    except (RuntimeError, ValueError, KeyError, AttributeError):
        image = None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning("No R1 waveforms available in event")
        return None

    # Handle waveform shape (n_chan, n_pix, n_samp) or (n_pix, n_samp)
    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]  # first gain channel
    _, n_samp = w.shape

    # Choose pixel ids
    if image is not None:
        pix_ids = np.argsort(image)[-n_pixels:][::-1]
    else:
        integrals = w.sum(axis=1)
        pix_ids = np.argsort(integrals)[-n_pixels:][::-1]

    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0  # ns
    t = np.arange(n_samp) * dt

    fig, ax = plt.subplots(dpi=300)
    for pid in pix_ids:
        ax.plot(t, w[pid], label=f"pix {int(pid)}", drawstyle="steps-mid")
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("R1 samples [a.u.]")
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    ax.set_title(f"CT{tel_id} waveforms ({et_name})")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return fig


def plot_simtel_waveform_pcolormesh(
    filename,
    event_type: str | None = None,
    tel_id: int | None = None,
    pixel_step: int | None = None,
    vmax: float | None = None,
):
    """
    Pseudocolor image of waveforms (samples vs pixel id) for one event.

    Parameters
    ----------
    filename : str
        Path to the sim_telarray file.
    event_type : str | None
        'flasher' or 'pedestal'. If None, first event is used.
    tel_id : int | None
        Telescope id. If None, use the first available with waveforms.
    pixel_step : int | None
        If set, take every N-th pixel to reduce size.
    vmax : float | None
        Optional color scale upper limit.

    Returns
    -------
    fig : matplotlib.figure.Figure | None
        The figure or None if no waveforms found.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source, event_type)
    if event is None:
        _logger.warning(f"No event found in {filename} matching type='{event_type}'")
        return None

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for waveform plot")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning("No R1 waveforms available in event")
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    if pixel_step and pixel_step > 1:
        pix_idx = np.arange(0, n_pix, pixel_step)
        w_sel = w[pix_idx]
    else:
        pix_idx = np.arange(n_pix)
        w_sel = w

    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0
    t = np.arange(n_samp) * dt

    fig, ax = plt.subplots(dpi=300)
    mesh = ax.pcolormesh(t, pix_idx, w_sel, shading="auto", vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("R1 samples [a.u.]")
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    ax.set_title(f"CT{tel_id} waveform matrix ({et_name})")
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("pixel id")
    fig.tight_layout()
    return fig


def plot_simtel_step_traces(
    filename,
    event_type: str | None = None,
    tel_id: int | None = None,
    pixel_step: int = 100,
    max_pixels: int | None = None,
):
    """
    Plot step-style traces for regularly sampled pixels: pix 0, N, 2N, ...

    Parameters
    ----------
    filename : str
    event_type : str | None
    tel_id : int | None
    pixel_step : int
        Step between pixel ids (default 100).
    max_pixels : int | None
        Maximum number of pixels to draw.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source, event_type)
    if event is None:
        _logger.warning(f"No event found in {filename} matching type='{event_type}'")
        return None

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for traces plot")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning("No R1 waveforms available in event")
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0
    t = np.arange(n_samp) * dt

    pix_ids = np.arange(0, n_pix, max(1, pixel_step))
    if max_pixels is not None:
        pix_ids = pix_ids[:max_pixels]

    fig, ax = plt.subplots(dpi=300)
    for pid in pix_ids:
        ax.plot(t, w[int(pid)], label=f"pix {int(pid)}", drawstyle="steps-mid")
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("R1 samples [a.u.]")
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    ax.set_title(f"CT{tel_id} step traces ({et_name})")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _detect_peaks(trace, peak_width, signal_mod):
    """Return indices of peaks using CWT if available, else find_peaks fallback."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    peaks = []
    try:
        if hasattr(signal_mod, "find_peaks_cwt"):
            peaks = signal_mod.find_peaks_cwt(trace, widths=np.array([peak_width]))
        if not np.any(peaks):
            peaks, _ = signal_mod.find_peaks(trace, prominence=np.max(trace) * 0.1)
    except (ValueError, RuntimeError, TypeError):
        peaks = []
    return np.asarray(peaks, dtype=int) if np.size(peaks) else np.array([], dtype=int)


def _collect_peak_samples(w, sum_threshold, peak_width, signal_mod):
    """Compute peak sample per pixel, return samples, considered pixel ids and count with peaks."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    n_pix, _ = w.shape
    sums = w.sum(axis=1)
    has_signal = sums > float(sum_threshold)
    pix_ids = np.arange(n_pix)[has_signal]
    if pix_ids.size == 0:
        return None, None, 0

    peak_samples = []
    found_count = 0
    for pid in pix_ids:
        trace = w[int(pid)]
        pks = _detect_peaks(trace, peak_width, signal_mod)
        if pks.size:
            found_count += 1
            peak_idx = int(pks[np.argmax(trace[pks])])
        else:
            peak_idx = int(np.argmax(trace))
        peak_samples.append(peak_idx)

    return np.asarray(peak_samples), pix_ids, found_count


def _histogram_edges(n_samp, timing_bins):
    """Return contiguous histogram bin edges for sample indices."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    if timing_bins and timing_bins > 0:
        return np.linspace(-0.5, n_samp - 0.5, int(timing_bins) + 1)
    return np.arange(-0.5, n_samp + 0.5, 1.0)


def _draw_peak_hist(
    ax,
    peak_samples,
    edges,
    mean_sample,
    std_sample,
    tel_id,
    et_name,
    considered,
    found_count,
):
    """Draw contiguous-bar histogram, stats overlays, and annotations."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    counts, edges = np.histogram(peak_samples, bins=edges)
    ax.bar(edges[:-1], counts, width=np.diff(edges), color="#5B90DC", align="edge")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("peak sample")
    ax.set_ylabel("N pixels")
    ax.axvline(
        mean_sample,
        color="#D8153C",
        linestyle="--",
        label=f"mean={mean_sample:.2f}",
    )
    ax.axvspan(
        mean_sample - std_sample,
        mean_sample + std_sample,
        color="#D8153C",
        alpha=0.2,
        label=f"std={std_sample:.2f}",
    )
    ax.set_title(f"CT{tel_id} peak timing ({et_name})")
    ax.text(
        0.98,
        0.95,
        f"considered: {considered}\npeaks found: {found_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.6,
            "linewidth": 0.0,
        },
    )
    ax.legend(fontsize=7)


def plot_simtel_peak_timing(
    filename,
    event_type: str | None = None,
    tel_id: int | None = None,
    sum_threshold: float = 10.0,
    peak_width: int = 8,
    examples: int = 3,
    timing_bins: int | None = None,
    return_stats: bool = False,
):
    """
    Peak finding per pixel; report mean/std of peak sample and plot a histogram.

    Parameters
    ----------
    filename : str
    event_type : str | None
    tel_id : int | None
    sum_threshold : float
        Minimum integrated signal per pixel to consider for peak finding.
    peak_width : int
        Expected peak width (samples).
    examples : int
        Number of example pixels to overlay with detected peaks.
    timing_bins : int | None
        If set, use this many bins for the peak-sample histogram; otherwise use
        one bin per sample (edges at integer samples) to ensure contiguous bars.
    return_stats : bool
        If True, return (fig, stats_dict) where stats includes considered, found, mean, std.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource
    from scipy import signal as _signal

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source, event_type)
    if event is None:
        _logger.warning(f"No event found in {filename} matching type='{event_type}'")
        return None

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for peak timing plot")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning("No R1 waveforms available in event")
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    _, n_samp = w.shape

    # Collect peak samples
    peak_samples, pix_ids, found_count = _collect_peak_samples(
        w, sum_threshold, peak_width, _signal
    )
    if peak_samples is None or pix_ids is None:
        _logger.warning("No pixels exceeded sum_threshold for peak timing")
        return None

    mean_sample = float(np.mean(peak_samples))
    std_sample = float(np.std(peak_samples))

    parts = [
        f"Peak timing over {peak_samples.size} pixels:",
        f"mean={mean_sample:.2f}, std={std_sample:.2f};",
        f"considered={pix_ids.size}, peaks_found={found_count}",
    ]
    msg = " ".join(parts)
    _logger.info(msg)

    # Build figure with histogram + example traces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    # Histogram with contiguous bars
    edges = _histogram_edges(n_samp, timing_bins)
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    _draw_peak_hist(
        ax1,
        peak_samples,
        edges,
        mean_sample,
        std_sample,
        tel_id,
        et_name,
        pix_ids.size,
        found_count,
    )

    # Example traces with peaks
    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0
    t = np.arange(n_samp) * dt

    ex_ids = pix_ids[: max(1, int(examples))]
    for pid in ex_ids:
        trace = w[int(pid)]
        pks = _detect_peaks(trace, peak_width, _signal)
        ax2.plot(t, trace, drawstyle="steps-mid", label=f"pix {int(pid)}")
        if pks.size:
            ax2.scatter(t[pks], trace[pks], s=10)
    ax2.set_xlabel("time [ns]")
    ax2.set_ylabel("R1 samples [a.u.]")
    ax2.legend(fontsize=7)

    fig.tight_layout()

    if return_stats:
        stats = {
            "considered": int(pix_ids.size),
            "found": int(found_count),
            "mean": float(mean_sample),
            "std": float(std_sample),
        }
        return fig, stats
    return fig


def plot_simtel_integrated_signal_image(
    filename,
    event_type: str | None = None,
    tel_id: int | None = None,
    half_width: int = 8,
):
    """Plot camera image of integrated signal per pixel around the flasher peak.

    The integration window is centered on each pixel's peak sample with +/- half_width samples.

    Returns matplotlib.figure.Figure or None if unavailable.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource
    from ctapipe.visualization import CameraDisplay

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source, event_type)
    if event is None:
        _logger.warning(f"No event found in {filename} matching type='{event_type}'")
        return None

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for integrated-signal image")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning("No R1 waveforms available in event")
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    win_len = 2 * int(half_width) + 1
    img = np.zeros(n_pix, dtype=float)
    for pid in range(n_pix):
        trace = w[pid]
        peak_idx = int(np.argmax(trace))
        a = max(0, peak_idx - half_width)
        b = min(n_samp, peak_idx + half_width + 1)
        img[pid] = float(np.sum(trace[a:b]))

    geometry = source.subarray.tel[tel_id].camera.geometry
    fig, ax = plt.subplots(dpi=300)
    disp = CameraDisplay(geometry, image=img, norm="lin", ax=ax)
    disp.cmap = "viridis"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)

    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    ax.set_title(
        f"CT{tel_id} integrated signal (win {win_len}) ({et_name})",
        pad=20,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_simtel_integrated_pedestal_image(
    filename,
    event_type: str | None = None,
    tel_id: int | None = None,
    half_width: int = 8,
    gap: int = 16,
):
    """Plot camera image of integrated pedestal per pixel away from the flasher peak.

    For each pixel, a pedestal window of length (2*half_width+1) samples is chosen starting
    at peak_idx + gap when possible; otherwise a window before the peak is used.

    Returns matplotlib.figure.Figure or None if unavailable.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource
    from ctapipe.visualization import CameraDisplay

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source, event_type)
    if event is None:
        _logger.warning(f"No event found in {filename} matching type='{event_type}'")
        return None

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for integrated-pedestal image")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning("No R1 waveforms available in event")
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    win_len = 2 * int(half_width) + 1
    img = np.zeros(n_pix, dtype=float)
    for pid in range(n_pix):
        trace = w[pid]
        peak_idx = int(np.argmax(trace))
        start = peak_idx + int(gap)
        if start + win_len <= n_samp:
            a = start
            b = start + win_len
        else:
            # fallback before peak
            start = max(0, peak_idx - int(gap) - win_len)
            a = start
            b = min(n_samp, start + win_len)
        if a >= b:
            a = 0
            b = min(n_samp, win_len)
        img[pid] = float(np.sum(trace[a:b]))

    geometry = source.subarray.tel[tel_id].camera.geometry
    fig, ax = plt.subplots(dpi=300)
    disp = CameraDisplay(geometry, image=img, norm="lin", ax=ax)
    disp.cmap = "cividis"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)

    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    ax.set_title(
        f"CT{tel_id} integrated pedestal (win {win_len}, gap {gap}) ({et_name})",
        pad=20,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def save_figure(fig, output_file, figure_format=None, log_title="", dpi="figure"):
    """
    Save figure to output file(s).

    Parameters
    ----------
    fig: plt.figure
        Figure to save.
    output_file: Path, str
        Path to save the figure (without suffix).
    figure_format: list
        List of formats to save the figure.
    title: str
        Title of the figure to be added to the log message.
    """
    figure_format = figure_format or ["pdf", "png"]
    for fmt in figure_format:
        _file = Path(output_file).with_suffix(f".{fmt}")
        fig.savefig(_file, format=fmt, bbox_inches="tight", dpi=dpi)
        logging.info(f"Saved plot {log_title} to {_file}")

    fig.clf()
