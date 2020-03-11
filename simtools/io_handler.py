#!/usr/bin/python3

""" io_handler module """

import logging
from pathlib import Path
import datetime

__all__ = ['getModelOutputDirectory', 'getRayTracingOutputDirectory']


def getOutputDirectory(filesLocation, label, mode):
    """ Return the path of the output directory.

    Args:
        filesLocation (Path)

        label (str)

    Returns:
        str: output Path

    """

    today = datetime.date.today()
    labelDir = label + '-' + str(today) if label is not None else 'd-' + str(today)

    return Path(filesLocation).joinpath('simtools-files').joinpath(labelDir).joinpath(mode)


def getModelOutputDirectory(filesLocation, label):
    return getOutputDirectory(filesLocation, label, 'model')


def getRayTracingOutputDirectory(filesLocation, label):
    return getOutputDirectory(filesLocation, label, 'ray-tracing')
