#!/usr/bin/python3

import logging
import re
from collections import OrderedDict

import astropy.units as u
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from astropy.table import QTable
from cycler import cycler

__all__ = ["setStyle", "plot1D", "plotTable"]

COLORS = dict()
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


def _addUnit(title, array):
    """
    A function to add a unit to "title" (presumably an axis title).
    The unit is extracted from the unit field of the array, in case array is an astropy quantity.
    If a unit is found, it is added to title in the form [unit].
    If a unit already is present in title (in the same form),
    a warning is printed and no unit is added.
    The function assumes array not to be empty and returns the modified title.

    Parameters
    ----------
    title: str
    array: astropy.Quantity

    Returns
    -------
    str
        Title with units.
    """
    _logger = logging.getLogger(__name__)

    unit = ""
    if isinstance(array, u.Quantity):
        unit = str(array[0].unit)
        if len(unit) > 0:
            unit = " [{}]".format(unit)
        if re.search(r"\d", unit):
            unit = re.sub(r"(\d)", r"^\1", unit)
            unit = unit.replace("[", r"$[").replace("]", r"]$")
        if "[" in title and "]" in title:
            _logger.warning(
                "Tried to add a unit from astropy.unit, "
                "but axis already has an explicit unit. Left axis title as is."
            )
            unit = ""

    return "{}{}".format(title, unit)


def setStyle(palette="default", bigPlot=False):
    """
    A function to set the plotting style as part of an effort to
    homogenize plot style across the framework.
    The function receives the colour palette name and whether it is
    a big plot or not. The latter sets the fonts and marker to be bigger in case it is a big plot.
    The available colour palettes are as follows:

    - classic (default): A classic colourful palette with strong colours and contrast.
    - modified classic: Similar to the classic, with slightly different colours.
    - autumn: A slightly darker autumn style colour palette.
    - purples: A pseudo sequential purple colour palette (not great for contrast).
    - greens: A pseudo sequential green colour palette (not great for contrast).

    To use the function, simply call it before plotting anything.
    The function is made public, so that it can be used outside the visualize module.
    However, it is highly recommended to create plots only through the visualize module.

    Parameters
    ----------
    palette: str
    bigPlot: bool

    Raises
    ------
    KeyError if provided palette does not exist.
    """

    if palette not in COLORS:
        raise KeyError("palette must be one of {}".format(", ".join(COLORS)))

    fontsize = {"default": 17, "bigPlot": 30}
    markersize = {"default": 8, "bigPlot": 18}
    plotSize = "default"
    if bigPlot:
        plotSize = "bigPlot"

    plt.rc("lines", linewidth=2, markersize=markersize[plotSize])
    plt.rc(
        "axes",
        prop_cycle=(
            cycler(color=COLORS[palette]) + cycler(linestyle=LINES) + cycler(marker=MARKERS)
        ),
    )
    plt.rc(
        "axes",
        titlesize=fontsize[plotSize],
        labelsize=fontsize[plotSize],
        labelpad=5,
        grid=True,
        axisbelow=True,
    )
    plt.rc("xtick", labelsize=fontsize[plotSize])
    plt.rc("ytick", labelsize=fontsize[plotSize])
    plt.rc("legend", loc="best", shadow=False, fontsize="medium")
    plt.rc("font", family="serif", size=fontsize[plotSize])

    return


def getColors(palette="default"):
    """
    Get the colour list of the palette requested.
    If no palette is provided, the default is returned.

    Parameters
    ----------
    palette: str

    Raises
    ------
    KeyError if provided palette does not exist.

    Returns
    -------
    list: colour list
    """

    if palette not in COLORS.keys():
        raise KeyError("palette must be one of {}".format(", ".join(COLORS)))

    return COLORS[palette]


def getMarkers():
    """
    Get the marker list used in this module.

    Returns
    -------
    list: marker list
    """

    return MARKERS


def getLines():
    """
    Get the line style list used in this module.

    Returns
    -------
    list: line style list
    """

    return LINES


def plot1D(data, **kwargs):
    """
    Produce a high contrast one dimensional plot from multiple data sets.
    A ratio plot can be added at the bottom to allow easy comparison.
    Additional options, such as plot title, plot legend, etc., are given in kwargs.
    Any option that can be changed after plotting (e.g., axes limits, log scale, etc.) should be
    done using the returned plt instance.

    Parameters
    ----------
    data: numpy structured array or a dictionary of structured arrays.
          Each structured array has two columns, the first is the x-axis and the second the y-axis.
          The titles of the columns are set as the axes titles.
          The labels of each dataset set are given in the dictionary keys
          and will be used in the legend.
    **kwargs:
        * palette: string
          Choose a colour palette (see setStyle for additional information).
        * title: string
          Set a plot title.
        * npLegend: bool
          Do not print a legend for the plot.
        * bigPlot: bool
          Increase marker and font sizes (like in a wide light curve).
        * noMarkers: bool
          Do not print markers.
        * emptyMarkers: bool
          Print empty (hollow) markers
        * plotRatio: bool
          Add a ratio plot at the bottom. The first entry in the data dictionary
          is used as the reference for the ratio.
          If data dictionary is not an OrderedDict, the reference will be random.
        * plotDifference: bool
          Add a difference plot at the bottom. The first entry in the data dictionary
          is used as the reference for the difference.
          If data dictionary is not an OrderedDict, the reference will be random.
        * Any additional kwargs for plt.plot

    Returns
    -------
    pyplot.figure
        A pyplot.figure instance in which the plot was produced
    """

    palette = kwargs.get("palette", "default")
    kwargs.pop("palette", None)
    bigPlot = kwargs.get("bigPlot", False)
    kwargs.pop("bigPlot", None)
    title = kwargs.get("title", "")
    kwargs.pop("title", None)
    noLegend = kwargs.get("noLegend", False)
    kwargs.pop("noLegend", None)
    noMarkers = kwargs.get("noMarkers", False)
    kwargs.pop("noMarkers", None)
    emptyMarkers = kwargs.get("emptyMarkers", False)
    kwargs.pop("emptyMarkers", None)

    if noMarkers:
        kwargs["marker"] = "None"
        kwargs["linewidth"] = 4
    if emptyMarkers:
        kwargs["markerfacecolor"] = "None"
        kwargs.pop("emptyMarkers", None)

    setStyle(palette, bigPlot)

    if not isinstance(data, dict):
        dataDict = dict()
        dataDict["_default"] = data
    else:
        dataDict = data

    plotRatio = kwargs.get("plotRatio", False)
    kwargs.pop("plotRatio", None)
    plotDifference = kwargs.get("plotDifference", False)
    kwargs.pop("plotDifference", None)
    if plotRatio or plotDifference:
        if len(dataDict) < 2:
            raise ValueError("Asked to plot a ratio or difference with just one set of data")

    if plotRatio or plotDifference:
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

    for label, dataNow in dataDict.items():
        assert len(dataNow.dtype.names) == 2, "Input array must have two columns with titles."
        xTitle, yTitle = dataNow.dtype.names[0], dataNow.dtype.names[1]
        xTitleUnit = _addUnit(xTitle, dataNow[xTitle])
        yTitleUnit = _addUnit(yTitle, dataNow[yTitle])
        plt.plot(dataNow[xTitle], dataNow[yTitle], label=label, **kwargs)

    if plotRatio or plotDifference:
        gs.update(hspace=0.02)
    else:
        plt.xlabel(xTitleUnit)
    plt.ylabel(yTitleUnit)

    if len(title) > 0:
        plt.title(title, y=1.02)
    if "_default" not in list(dataDict.keys()) and not noLegend:
        plt.legend()
    if not (plotRatio or plotDifference):
        plt.tight_layout()

    ##########################################################################################
    # Plot a ratio
    ##########################################################################################

    if plotRatio or plotDifference:
        plt.subplot(gs[1], sharex=ax1)
        # In order to advance the cycler one color/style,
        # so the colors stay consistent in the ratio, plot null data first.
        plt.plot([], [])

        # Use the first entry as the reference for the ratio.
        # If dataDict is not an OrderedDict, the reference will be random.
        dataRefName = next(iter(dataDict))
        for label, dataNow in dataDict.items():
            if label == dataRefName:
                continue
            else:
                xTitle, yTitle = dataNow.dtype.names[0], dataNow.dtype.names[1]
                xTitleUnit = _addUnit(xTitle, dataNow[xTitle])
                if plotRatio:
                    yValues = dataNow[yTitle] / dataDict[dataRefName][yTitle]
                else:
                    yValues = dataNow[yTitle] - dataDict[dataRefName][yTitle]
                plt.plot(dataNow[xTitle], yValues, **kwargs)

        plt.xlabel(xTitleUnit)
        yTitleRatio = "Ratio to {}".format(dataRefName)
        if len(yTitleRatio) > 20:
            yTitleRatio = "Ratio"
        if plotDifference:
            yTitleRatio = "Difference to {}".format(dataRefName)
        plt.ylabel(yTitleRatio)

        ylim = plt.gca().get_ylim()
        nbins = min(int((ylim[1] - ylim[0]) / 0.05 + 1), 6)
        plt.locator_params(axis="y", nbins=nbins)

    return fig


def plotTable(table, yTitle, **kwargs):
    """
    Produce a high contrast one dimensional plot from the data in an astropy.Table.
    A ratio plot can be added at the bottom to allow easy comparison.
    Additional options, such as plot title, plot legend, etc., are given in kwargs.
    Any option that can be changed after plotting (e.g., axes limits, log scale, etc.) should be
    done using the returned plt instance.

    Parameters
    ----------
    table: astropy.Table or astropy.QTable.
           The first column of the table is the x-axis and the second column is the y-axis.
           Any additional columns will be treated as additional data to plot.
           The column titles are used in the legend (except for the first column).
    yTitle: str
           The y-axis title.

    **kwargs:
        * palette: choose a colour palette (see setStyle for additional information).
        * title: set a plot title.
        * noLegend: do not print a legend for the plot.
        * bigPlot: increase marker and font sizes (like in a wide light curve).
        * noMarkers: do not print markers.
        * emptyMarkers: print empty (hollow) markers
        * plotRatio: bool
          Add a ratio plot at the bottom. The first entry in the data dictionary
          is used as the reference for the ratio.
          If data dictionary is not an OrderedDict, the reference will be random.
        * plotDifference: bool
          Add a difference plot at the bottom. The first entry in the data dictionary
          is used as the reference for the difference.
          If data dictionary is not an OrderedDict, the reference will be random.
        * Any additional kwargs for plt.plot

    Returns
    -------
    pyplot.fig

    """

    if len(table.keys()) < 2:
        raise ValueError("Table has to have at least two columns")

    xAxis = table.keys()[0]
    dataDict = OrderedDict()
    for column in table.keys()[1:]:
        dataDict[column] = QTable([table[xAxis], table[column]], names=[xAxis, yTitle])

    return plot1D(dataDict, **kwargs)


def plotHist2D(data, **kwargs):
    """
    Produce a two dimensional histogram plot.
    Any option that can be changed after plotting (e.g., axes limits, log scale, etc.) should be
    done using the returned plt instance.

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
        A pyplot.figure instance in which the plot was produced

    """

    cmap = plt.cm.gist_heat_r
    if "title" in kwargs:
        title = kwargs["title"]
        kwargs.pop("title", None)
    else:
        title = ""

    # Set default style since the usual options do not affect 2D plots (for now).
    setStyle()

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(figsize=(8, 6))

    ##########################################################################################
    # Plot the data
    ##########################################################################################

    plt.subplot(gs[0])
    assert len(data.dtype.names) == 2, "Input array must have two columns with titles."
    xTitle, yTitle = data.dtype.names[0], data.dtype.names[1]
    xTitleUnit = _addUnit(xTitle, data[xTitle])
    yTitleUnit = _addUnit(yTitle, data[yTitle])
    plt.hist2d(data[xTitle], data[yTitle], cmap=cmap, **kwargs)

    plt.xlabel(xTitleUnit)
    plt.ylabel(yTitleUnit)

    plt.gca().set_aspect("equal", adjustable="datalim")

    if len(title) > 0:
        plt.title(title, y=1.02)

    fig.tight_layout()

    return fig
