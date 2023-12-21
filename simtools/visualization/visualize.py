#!/usr/bin/python3

import logging
import re
from collections import OrderedDict

import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.table import QTable
from cycler import cycler
from matplotlib import gridspec
from matplotlib.collections import PatchCollection

from simtools.utils import geometry as transf
from simtools.utils import names
from simtools.utils.names import mst, sct
from simtools.visualization import legend_handlers as leg_h

__all__ = [
    "get_colors",
    "get_lines",
    "get_markers",
    "get_telescope_patch",
    "plot_1d",
    "plot_array",
    "plot_hist_2d",
    "plot_table",
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
    A function to add a unit to "title" (presumably an axis title). The unit is extracted from the\
    unit field of the array, in case array is an astropy quantity. If a unit is found, it is added \
    to title in the form [unit]. If a unit already is present in title (in the same form), a \
    warning is printed and no unit is added. The function assumes array not to be empty and returns\
    the modified title.

    Parameters
    ----------
    title: str
    array: astropy.Quantity

    Returns
    -------
    str
        Title with units.
    """

    unit = ""
    if isinstance(array, u.Quantity):
        unit = str(array[0].unit)
        if len(unit) > 0:
            unit = f" [{unit}]"
        if re.search(r"\d", unit):
            unit = re.sub(r"(\d)", r"^\1", unit)
            unit = unit.replace("[", r"$[").replace("]", r"]$")
        if "[" in title and "]" in title:
            _logger.warning(
                "Tried to add a unit from astropy.unit, "
                "but axis already has an explicit unit. Left axis title as is."
            )
            unit = ""

    return f"{title}{unit}"


def set_style(palette="default", big_plot=False):
    """
    A function to set the plotting style as part of an effort to homogenize plot style across the \
    framework. The function receives the colour palette name and whether it is a big plot or not.\
    The latter sets the fonts and marker to be bigger in case it is a big plot. The available \
    colour palettes are as follows:

    - classic (default): A classic colourful palette with strong colours and contrast.
    - modified classic: Similar to the classic, with slightly different colours.
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
    markersize = {"default": 8, "big_plot": 18}
    plot_size = "default"
    if big_plot:
        plot_size = "big_plot"

    plt.rc("lines", linewidth=2, markersize=markersize[plot_size])
    plt.rc(
        "axes",
        prop_cycle=(
            cycler(color=COLORS[palette]) + cycler(linestyle=LINES) + cycler(marker=MARKERS)
        ),
    )
    plt.rc(
        "axes",
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
    Get the colour list of the palette requested. If no palette is provided, the default is \
    returned.

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


def plot_1d(data, **kwargs):
    """
    Produce a high contrast one dimensional plot from multiple data sets. A ratio plot can be \
    added at the bottom to allow easy comparison. Additional options, such as plot title, plot
    legend, etc., are given in kwargs. Any option that can be changed after plotting (e.g., axes\
    limits, log scale, etc.) should be done using the returned plt instance.

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
        * np_legend: bool
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

    palette = kwargs.get("palette", "default")
    kwargs.pop("palette", None)
    big_plot = kwargs.get("big_plot", False)
    kwargs.pop("big_plot", None)
    title = kwargs.get("title", "")
    kwargs.pop("title", None)
    no_legend = kwargs.get("no_legend", False)
    kwargs.pop("no_legend", None)
    no_markers = kwargs.get("no_markers", False)
    kwargs.pop("no_markers", None)
    empty_markers = kwargs.get("empty_markers", False)
    kwargs.pop("empty_markers", None)

    if no_markers:
        kwargs["marker"] = "None"
        kwargs["linewidth"] = 4
    if empty_markers:
        kwargs["markerfacecolor"] = "None"
        kwargs.pop("empty_markers", None)

    set_style(palette, big_plot)

    if not isinstance(data, dict):
        data_dict = {}
        data_dict["_default"] = data
    else:
        data_dict = data

    plot_ratio = kwargs.get("plot_ratio", False)
    kwargs.pop("plot_ratio", None)
    plot_difference = kwargs.get("plot_difference", False)
    kwargs.pop("plot_difference", None)
    if plot_ratio or plot_difference:
        if len(data_dict) < 2:
            raise ValueError("Asked to plot a ratio or difference with just one set of data")

    if plot_ratio or plot_difference:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        fig = plt.figure(figsize=(8, 8))
    else:
        gs = gridspec.GridSpec(1, 1)
        fig = plt.figure(figsize=(8, 6))

    ##########################################################################################
    # Plot the data
    ##########################################################################################

    plt.subplot(gs[0])
    ax1 = plt.gca()

    for label, data_now in data_dict.items():
        assert len(data_now.dtype.names) == 2, "Input array must have two columns with titles."
        x_title, y_title = data_now.dtype.names[0], data_now.dtype.names[1]
        x_title_unit = _add_unit(x_title, data_now[x_title])
        y_title_unit = _add_unit(y_title, data_now[y_title])
        plt.plot(data_now[x_title], data_now[y_title], label=label, **kwargs)

    if plot_ratio or plot_difference:
        gs.update(hspace=0.02)
    else:
        plt.xlabel(x_title_unit)
    plt.ylabel(y_title_unit)

    if len(title) > 0:
        plt.title(title, y=1.02)
    if "_default" not in list(data_dict.keys()) and not no_legend:
        plt.legend()
    if not (plot_ratio or plot_difference):
        plt.tight_layout()

    ##########################################################################################
    # Plot a ratio
    ##########################################################################################

    if plot_ratio or plot_difference:
        plt.subplot(gs[1], sharex=ax1)
        # In order to advance the cycler one color/style,
        # so the colors stay consistent in the ratio, plot null data first.
        plt.plot([], [])

        # Use the first entry as the reference for the ratio.
        # If data_dict is not an OrderedDict, the reference will be random.
        data_ref_name = next(iter(data_dict))
        for label, data_now in data_dict.items():
            if label == data_ref_name:
                continue
            x_title, y_title = data_now.dtype.names[0], data_now.dtype.names[1]
            x_title_unit = _add_unit(x_title, data_now[x_title])
            if plot_ratio:
                y_values = data_now[y_title] / data_dict[data_ref_name][y_title]
            else:
                y_values = data_now[y_title] - data_dict[data_ref_name][y_title]
            plt.plot(data_now[x_title], y_values, **kwargs)

        plt.xlabel(x_title_unit)
        y_title_ratio = f"Ratio to {data_ref_name}"
        if len(y_title_ratio) > 20:
            y_title_ratio = "Ratio"
        if plot_difference:
            y_title_ratio = f"Difference to {data_ref_name}"
        plt.ylabel(y_title_ratio)

        ylim = plt.gca().get_ylim()
        nbins = min(int((ylim[1] - ylim[0]) / 0.05 + 1), 6)
        plt.locator_params(axis="y", nbins=nbins)

    return fig


def plot_table(table, y_title, **kwargs):
    """
    Produce a high contrast one dimensional plot from the data in an astropy.Table. A ratio plot\
    can be added at the bottom to allow easy comparison. Additional options, such as plot title,
    plot legend, etc., are given in kwargs. Any option that can be changed after plotting (e.g.,\
    axes limits, log scale, etc.) should be done using the returned plt instance.

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
    Produce a two dimensional histogram plot. Any option that can be changed after plotting (e.g.,\
    axes limits, log scale, etc.) should be done using the returned plt instance.

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


@u.quantity_input(x=u.m, y=u.m, radius=u.m)
def get_telescope_patch(name, x, y, radius):
    """
    Collect the patch of one telescope to be plotted by plot_array.

    Parameters
    ----------
    name: str
        Name of the telescope (type).
    x: astropy.units.Quantity
        x position of the telescope usually in meters.
    y: astropy.units.Quantity
        y position of the telescope usually in meters.
    radius: astropy.units.Quantity
        Radius of the telescope sphere usually in meters.

    Returns
    -------
    patch
        Instance of mpatches.Circle.
    """
    tel_obj = leg_h.TelescopeHandler()
    valid_name = names.get_telescope_type(name)
    fill_flag = False

    x = x.to(u.m)
    y = y.to(u.m)
    radius = radius.to(u.m)

    if valid_name == mst:
        fill_flag = True
    if valid_name == sct:
        patch = mpatches.Rectangle(
            ((x - radius / 2).value, (y - radius / 2).value),
            width=radius.value,
            height=radius.value,
            fill=False,
            color=tel_obj.colors_dict[sct],
        )
    else:
        patch = mpatches.Circle(
            (x.value, y.value),
            radius=radius.value,
            fill=fill_flag,
            color=tel_obj.colors_dict[valid_name],
        )
    return patch


@u.quantity_input(rotate_angle=u.deg)
def plot_array(telescopes, rotate_angle=0, show_tel_label=False):
    """
    Plot the array of telescopes. The x axis gives the easting direction and y axis gives the
    northing direction.
    Note that in order to convert from the CORSIKA coordinate system to the 'conventional' system
    of North/East, a 90 degree rotation is always applied.
    Rotation of the array elements is possible through the 'rotate_angle' given either in degrees,
    or in radians.
    The direction of rotation of the array elements is counterclockwise.
    The rotation does not change Telescope instance attributes.

    Parameters
    ----------
    telescopes: dict
        Dictionary with the telescope position and names. Coordinates are given in the CORSIKA
        coordinate system, i.e., the x positive axis points toward north
        and the y positive axis points toward west.
    rotate_angle:
        Angle to rotate the plot. For rotate_angle = 0 the resulting plot will have
        the x-axis pointing towards the east, and the y-axis pointing towards the North.
    show_tel_label: bool
        If True it will print the label of the individual telescopes in the plot.
        While it works well for the smaller arrays, it gets crowded for larger arrays.

    Returns
    -------
    plt.figure
        Instance of plt.figure with the array of telescopes plotted.

    """

    fig, ax = plt.subplots(1)
    legend_objects = []
    legend_labels = []
    tel_counters = {one_telescope: 0 for one_telescope in names.all_telescope_class_names}
    if rotate_angle != 0:
        pos_x_rotated, pos_y_rotated = transf.rotate(
            telescopes["position_x"], telescopes["position_y"], rotate_angle
        )
    else:
        pos_x_rotated, pos_y_rotated = telescopes["position_x"], telescopes["position_y"]

    pos_x_rotated, pos_y_rotated = transf.rotate(pos_x_rotated, pos_y_rotated, 90 * u.deg)

    if len(pos_x_rotated) > 30:
        fontsize = 4
        scale = 2
    else:
        fontsize = 8
        scale = 1

    patches = []
    for i_tel, tel_now in enumerate(telescopes):
        for tel_type in tel_counters:
            if tel_type in tel_now["telescope_name"]:
                tel_counters[tel_type] += 1
        i_tel_name = names.get_telescope_type(telescopes[i_tel]["telescope_name"])
        patches.append(
            get_telescope_patch(
                i_tel_name,
                pos_x_rotated[i_tel],
                pos_y_rotated[i_tel],
                scale * telescopes[i_tel]["radius"],
            )
        )
        if show_tel_label:
            ax.text(
                pos_x_rotated[i_tel].value,
                pos_y_rotated[i_tel].value + scale * telescopes[i_tel]["radius"].value,
                telescopes[i_tel]["telescope_name"],
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=fontsize,
            )

    for _, one_telescope in enumerate(names.all_telescope_class_names):
        if tel_counters[one_telescope] > 0:
            legend_objects.append(leg_h.all_telescope_objects[one_telescope]())
            legend_labels.append(f"{one_telescope} ({tel_counters[one_telescope]})")

    plt.gca().add_collection(PatchCollection(patches, match_original=True))

    x_title = "Easting [m]"
    y_title = "Northing [m]"
    plt.axis("square")
    plt.grid(True)
    plt.gca().set_axisbelow(True)
    plt.xlabel(x_title, fontsize=18, labelpad=0)
    plt.ylabel(y_title, fontsize=18, labelpad=0)
    plt.tick_params(axis="both", which="major", labelsize=15)

    legend_handler_map = {
        list(leg_h.legend_handler_map.keys())[i_key]: list(leg_h.legend_handler_map.values())[
            i_key
        ]()
        for i_key, _ in enumerate(leg_h.legend_handler_map)
    }
    plt.legend(
        legend_objects,
        legend_labels,
        handler_map=legend_handler_map,
        prop={"size": 11},
        loc="best",
    )

    plt.tight_layout()

    return fig
