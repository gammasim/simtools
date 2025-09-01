#!/usr/bin/python3
"""Module for visualization."""

import logging
import re
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from cycler import cycler
from matplotlib import gridspec

__all__ = [
    "get_colors",
    "get_lines",
    "get_markers",
    "plot_1d",
    "plot_hist_2d",
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


def plot_incident_angles(
    results: QTable | None,
    output_dir: Path,
    label: str,
    logger: logging.Logger | None = None,
) -> None:
    """Plot and save a histogram of the focal-surface incidence angles.

    Parameters
    ----------
    results : QTable or None
        Table containing column ``angle_incidence_focal`` with ``astropy.units``.
    output_dir : Path
        Directory to write the PNG plot into.
    label : str
        Label used to compose the output filename.
    logger : logging.Logger, optional
        Logger to emit warnings; if not provided, a module-level logger is used.
    """
    log = logger or logging.getLogger(__name__)
    if results is None or len(results) == 0:
        log.warning("No results to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(
        results["angle_incidence_focal"].value,
        bins=50,
        alpha=0.9,
        color="royalblue",
        histtype="stepfilled",
        edgecolor="none",
    )
    ax.set_xlabel("Angle of incidence at focal surface (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Incident angle distribution (focal surface)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = Path(output_dir) / f"incident_angles_{label}.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_incident_angles_multi(
    results_by_offset: dict[float, QTable],
    output_dir: Path,
    label: str,
    bin_width_deg: float = 0.1,
    logger: logging.Logger | None = None,
) -> None:
    """Plot overlaid histograms of focal-surface incidence angles for multiple offsets.

    Parameters
    ----------
    results_by_offset : dict(float -> QTable)
        Mapping from off-axis angle in degrees to result table containing column
        ``angle_incidence_focal`` with ``astropy.units``.
    output_dir : Path
        Directory to write the PNG plot into.
    label : str
        Label used to compose the output filename.
    bin_width_deg : float, optional
        Histogram bin width in degrees (default: 0.1 deg).
    logger : logging.Logger, optional
        Logger to emit warnings; if not provided, a module-level logger is used.
    """
    log = logger or logging.getLogger(__name__)
    if not results_by_offset:
        log.warning("No results provided for multi-offset plot")
        return

    arrays = []
    for off, tab in results_by_offset.items():
        if tab is None or len(tab) == 0:
            log.warning(f"Empty results for off-axis={off}")
            continue
        arrays.append(tab["angle_incidence_focal"].to(u.deg).value)
    if not arrays:
        log.warning("No non-empty results to plot")
        return

    all_vals = np.concatenate(arrays)
    vmin = float(np.floor(all_vals.min() / bin_width_deg) * bin_width_deg)
    vmax = float(np.ceil(all_vals.max() / bin_width_deg) * bin_width_deg)
    bins = np.arange(vmin, vmax + bin_width_deg * 0.5, bin_width_deg)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for off in sorted(results_by_offset.keys()):
        tab = results_by_offset[off]
        if tab is None or len(tab) == 0:
            continue
        data = tab["angle_incidence_focal"].to(u.deg).value

        _, _, patches = ax.hist(
            data,
            bins=bins,
            histtype="step",
            linewidth=0.5,
            label=f"off-axis {off:g} deg",
            zorder=3,
        )
        color = patches[0].get_edgecolor() if patches else None

        ax.hist(
            data,
            bins=bins,
            histtype="stepfilled",
            alpha=0.15,
            color=color,
            edgecolor="none",
            label="_nolegend_",
            zorder=1,
        )

        ax.hist(
            data,
            bins=bins,
            histtype="step",
            linewidth=0.5,
            color=color,
            label="_nolegend_",
            zorder=4,
        )

    ax.set_xlabel("Angle of incidence at focal surface (deg) w.r.t. optical axis")
    ax.set_ylabel("Count / Bin")
    ax.set_title("Incident angle distribution vs off-axis angle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_png = Path(output_dir) / f"incident_angles_multi_{label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
