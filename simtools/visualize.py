#!/usr/bin/python3

import logging
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler
from collections import OrderedDict
from astropy import units
from astropy.table import QTable

__all__ = ['setStyle', 'plot1D', 'plotTable']

logger = logging.getLogger(__name__)


def _addUnit(title, array):
    '''
    A function to add a unit to "title" (presumably an axis title).
    The unit is extracted from the unit field of the array, in case array is an astropy quantity.
    If a unit is found, it is added to title in the form [unit].
    If a unit already is present in title (in the same form),
    a warning is printed and no unit is added.
    The function assumes array not to be empty and returns the modified title.
    '''

    unit = ''
    if isinstance(array, units.Quantity):
        unit = str(array[0].unit)
        if len(unit) > 0:
            unit = ' [{}]'.format(unit)
        if '[' in title and ']' in title:
            logger.warn('Tried to add a unit from astropy.unit, '
                        'but axis already has an explicit unit. Left axis title as is.')
            unit = ''

    return '{}{}'.format(title, unit)


def setStyle(palette='default', bigPlot=False):
    '''
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
    '''

    colors = dict()
    colors['classic'] = ['#ba2c54', '#5B90DC', '#FFAB44', '#0C9FB3', '#57271B', '#3B507D',
                         '#794D88', '#FD6989', '#8A978E', '#3B507D', '#D8153C', '#cc9214']
    colors['modified classic'] = ['#D6088F', '#424D9C', '#178084', '#AF99DA', '#F58D46', '#634B5B',
                                  '#0C9FB3', '#7C438A', '#328cd6', '#8D0F25', '#8A978E', '#ffcb3d']
    colors['autumn'] = ['#A9434D', '#4E615D', '#3C8DAB', '#A4657A', '#424D9C', '#DC575A',
                        '#1D2D38', '#634B5B', '#56276D', '#577580', '#134663', '#196096']
    colors['purples'] = ['#a57bb7', '#343D80', '#EA60BF', '#B7308E', '#E099C3', '#7C438A',
                         '#AF99DA', '#4D428E', '#56276D', '#CC4B93', '#DC4E76', '#5C4AE4']
    colors['greens'] = ['#268F92', '#abc14d', '#8A978E', '#0C9FB3', '#BDA962', '#B0CB9E',
                        '#769168', '#5E93A5', '#178084', '#B7BBAD', '#163317', '#76A63F']

    colors['default'] = colors['classic']

    if palette not in colors.keys():
        raise ValueError('palette must be one of {}'.format(', '.join(colors)))

    markers = ['o', 's', 'v', '^', '*', 'P', 'd', 'X', 'p', '<', '>', 'h']
    lines = [(0, ()),  # solid
             (0, (1, 1)),  # densely dotted
             (0, (3, 1, 1, 1)),  # densely dashdotted
             (0, (5, 5)),  # dashed
             (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
             (0, (5, 1)),  # desnely dashed
             (0, (1, 5)),  # dotted
             (0, (3, 5, 1, 5)),  # dashdotted
             (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
             (0, (5, 10)),  # loosely dashed
             (0, (1, 10)),  # loosely dotted
             (0, (3, 10, 1, 10)),  # loosely dashdotted
             ]
    fontsize = {'default': 15, 'bigPlot': 30}
    markersize = {'default': 8, 'bigPlot': 18}
    plotSize = 'default'
    if bigPlot:
        plotSize = 'bigPlot'

    plt.rc('lines', linewidth=2, markersize=markersize[plotSize])
    plt.rc('axes', prop_cycle=(cycler(color=colors[palette]) +
                               cycler(linestyle=lines) +
                               cycler(marker=markers)))
    plt.rc('axes', titlesize=fontsize[plotSize], labelsize=fontsize[plotSize],
           labelpad=5, grid=True, axisbelow=True)
    plt.rc('xtick', labelsize=fontsize[plotSize])
    plt.rc('ytick', labelsize=fontsize[plotSize])
    plt.rc('legend', loc='best', shadow=False, fontsize='x-large')

    return


def plot1D(data, **kwargs):
    '''
    Produce a one dimentional plot of the data in "data".
    "data" is assumed to be a structured array, or a dictionary of structured arrays.
    Each structured array has two columns, the first is the x-axis and the second the y-axis.
    The titles of the columns are the axes titles.
    The labels of each data set are given in the dictionary keys and will be put in a legend.
    The function returns a pyplot instance in plt.
    Additional options, such as plot title, plot legend, etc.
    are given in kwargs (list will be added to doc as function evolves).
    Any option that can be changed after plotting (e.g., axes limits, log scale, etc.) should be
    done using the returned plt instance.
    A growing list of options that have to be applied during plotting (e.g., markers, titles, etc.)
    are included here.

    Optional kwargs:
        * pallete - choose a colour pallete from
          "classic", "modified classic", "autumn", "purples" and "greens".
        * title - provide a plot title.
        * npLegend - do not print a legend for the plot.
        * bigPlot - increase marker and font sizes (like in a wide lightcurve).
        * noMarkers - do not print markers.
        * emptyMarkers - print empty (hollow) markers
        * plotRatio - add a ratio plot at the bottom. The first entry in the data dictionary
          is used as the reference for the ratio. If data dictionary is not an OrderedDict,
          the reference will be random.
    '''

    palette = 'default'
    bigPlot = False
    if 'palette' in kwargs:
        palette = kwargs['palette']
    if 'bigPlot' in kwargs:
        bigPlot = kwargs['bigPlot']
    setStyle(palette, bigPlot)

    if not isinstance(data, dict):
        dataDict = dict()
        dataDict['_default'] = data
    else:
        dataDict = data

    plotArgs = dict()
    if 'noMarkers' in kwargs:
        if kwargs['noMarkers']:
            plotArgs = {'marker': 'None', 'linewidth': 4}
    if 'emptyMarkers' in kwargs:
        if kwargs['emptyMarkers']:
            plotArgs = {'markerfacecolor': 'None'}

    plotRatio = False
    if 'plotRatio' in kwargs:
        if kwargs['plotRatio']:
            if len(dataDict) < 2:
                raise ValueError('Asked to plot ratio with just one set of data')
            else:
                plotRatio = True
    if plotRatio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        plt.figure(figsize=(8, 8))
    else:
        gs = gridspec.GridSpec(1, 1)
        plt.figure(figsize=(8, 6))

    ##########################################################################################
    # Plot the data
    ##########################################################################################

    plt.subplot(gs[0])

    for label, dataNow in dataDict.items():
        assert len(dataNow.dtype.names) == 2, 'Input array must have two columns with titles.'
        xTitle, yTitle = dataNow.dtype.names[0], dataNow.dtype.names[1]
        xTitleUnit = _addUnit(xTitle, dataNow[xTitle])
        yTitleUnit = _addUnit(yTitle, dataNow[yTitle])
        plt.plot(dataNow[xTitle], dataNow[yTitle], label=label, **plotArgs)

    if plotRatio:
        plt.gca().set_xticklabels([])
        gs.update(hspace=0.06)
    else:
        plt.xlabel(xTitleUnit)
    plt.ylabel(yTitleUnit)

    if 'title' in kwargs:
        title = kwargs['title']
    else:
        title = ''
    if len(title) > 0:
        plt.title(title, y=1.02)
    if '_default' not in list(dataDict.keys()) and 'noLegend' not in kwargs:
        plt.legend()
    if not plotRatio:
        plt.tight_layout()

    ##########################################################################################
    # Plot a ratio
    ##########################################################################################

    if plotRatio:
        plt.subplot(gs[1])
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
                plt.plot(dataNow[xTitle], dataNow[yTitle]/dataDict[dataRefName][yTitle], **plotArgs)

        plt.xlabel(xTitleUnit)
        yTitleRatio = 'Ratio to {}'.format(dataRefName)
        if len(yTitleRatio) > 20:
            yTitleRatio = 'Ratio'
        plt.ylabel(yTitleRatio)

        ylim = plt.gca().get_ylim()
        nbins = min(int((ylim[1] - ylim[0]) / 0.05 + 1), 6)
        plt.locator_params(axis='y', nbins=nbins)
        plt.gca().autoscale(enable=True, axis='x', tight=True)

    return plt


def plotTable(table, yTitle, **kwargs):
    '''
    Produce a one dimentional plot of the data in "table".
    "table" is assumed to be a astropy Table, where the first column is the x-axis and
    the second column is the y-axis. Any additional columns will be treated as additional
    data to plot. The column titles will be the labels of the data
    in the legend (except for the first column). The y-axis title needs to be provided
    explicitly to the function.
    The function returns a pyplot instance in plt.
    Additional options, such as plot title, plot legend, etc.
    are given in kwargs (see plot1D documentation).
    Any option that can be changed after plotting (e.g., axes limits, log scale, etc.) should be
    done using the returned plt instance.
    A growing list of options that have to be applied during plotting (e.g., markers, titles, etc.)
    are included here.
    For a list of options see the documentation of plot1D.
    '''
    if len(table.keys()) < 2:
        raise ValueError('Table has to have at least two columns')

    xAxis = table.keys()[0]
    dataDict = OrderedDict()
    for i_col, column in enumerate(table.keys()[1:]):
        dataDict[column] = QTable([table[xAxis], table[column]],
                                  names=[xAxis, yTitle])

    return plot1D(dataDict, **kwargs)
