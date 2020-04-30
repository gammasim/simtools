#!/usr/bin/python3

""" io_handler module """

import logging
import datetime
from pathlib import Path

from simtools.util import config as cfg

__all__ = ['getModelOutputDirectory', 'getRayTracingOutputDirectory', 'getCorsikaOutputDirectory']


def getOutputDirectory(filesLocation, label, mode):
    """ Return the path of the output directory.

    Args:
        filesLocation (Path)

        label (str)

    Returns:
        str: output Path

    """

    today = datetime.date.today()
    labelDir = label if label is not None else 'd-' + str(today)

    return Path(filesLocation).joinpath('simtools-files').joinpath(labelDir).joinpath(mode)


def getModelOutputDirectory(filesLocation, label):
    return getOutputDirectory(filesLocation, label, 'model')


def getRayTracingOutputDirectory(filesLocation, label):
    return getOutputDirectory(filesLocation, label, 'ray-tracing')


def getCorsikaOutputDirectory(filesLocation, label):
    return getOutputDirectory(filesLocation, label, 'corsika')


def getTestDataFile(fileName):
    return Path(cfg.get('testDataLocation')).joinpath('test-data').joinpath(fileName)


def getTestPlotFile(fileName):
    return Path(cfg.get('testDataLocation')).joinpath('test-plots').joinpath(fileName)
