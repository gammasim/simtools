""" Module to handle input and output conventions. """

import datetime
import logging
from pathlib import Path

import simtools.config as cfg

__all__ = ["getOutputDirectory", "getOutputFile", "getDataFile"]


def getOutputDirectory(filesLocation=None, label=None, dirType=None, test=False):
    """
    Get the output directory for the directory type dirType

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.
    dirType: str
        Name of the subdirectory (ray-tracing, model etc)
    test: bool
        If true, return test output location

    Returns
    -------
    Path
    """
    _logger = logging.getLogger(__name__)

    if test:
        outputDirectoryPrefix = Path(cfg.get("outputLocation")).joinpath("test-output")
    else:
        if filesLocation is None:
            filesLocation = cfg.get("outputLocation")
        outputDirectoryPrefix = Path(filesLocation).joinpath("simtools-output")

    today = datetime.date.today()
    labelDir = label if label is not None else "d-" + str(today)
    path = outputDirectoryPrefix.joinpath(labelDir)
    if dirType is not None:
        path = path.joinpath(dirType)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError:
        _logger.error("Error creating directory {}".format(str(path)))
        raise

    return path.absolute()


def getOutputFile(fileName, label=None, dirType=None, test=False):
    """
    Get path of an output file.

    Parameters
    ----------
    filesName: str
        File name.
    label: str
        Instance label.
    dirType: str
        Name of the subdirectory (ray-tracing, model etc)
    test: bool
        If true, return test output location

    Returns
    -------
    Path
    """
    return getOutputDirectory(label=label, dirType=dirType, test=test).joinpath(fileName).absolute()


def getDataFile(parentDir=None, fileName=None, test=False):
    """
    Get path of a data file, using the dataLocation taken from the config file.

    Parameters
    ----------
    parentDir: str
        Parent directory of the file.
    filesName: str
        File name.
    test: bool
        If true, return test resources location

    Returns
    -------
    Path
    """

    if test:
        filePrefix = Path("tests/resources/")
    else:
        filePrefix = Path(cfg.get("dataLocation")).joinpath(parentDir)
    return filePrefix.joinpath(fileName).absolute()
