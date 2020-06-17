''' Module to handle input and output conventions. '''

import logging
import datetime
from pathlib import Path

import simtools.config as cfg

__all__ = [
    'getModelOutputDirectory',
    'getRayTracingOutputDirectory',
    'getCorsikaOutputDirectory',
    'getTestDataFile',
    'getTestPlotFile'
]


def getOutputDirectory(filesLocation, label, mode):
    '''
    Get main output directory for a generic mode

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    '''
    today = datetime.date.today()
    labelDir = label if label is not None else 'd-' + str(today)
    path = Path(filesLocation).joinpath('simtools-files').joinpath(labelDir).joinpath(mode)
    return path.absolute()


def getModelOutputDirectory(filesLocation, label):
    '''
    Get output directory for model related files.

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    '''
    return getOutputDirectory(filesLocation, label, 'model')


def getRayTracingOutputDirectory(filesLocation, label):
    '''
    Get output directory for ray tracing related files.

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    '''
    return getOutputDirectory(filesLocation, label, 'ray-tracing')


def getCorsikaOutputDirectory(filesLocation, label):
    '''
    Get output directory for corsika related files.

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    '''
    return getOutputDirectory(filesLocation, label, 'corsika')


def getCameraEfficiencyOutputDirectory(filesLocation, label):
    '''
    Get output directory for camera efficiency related files.

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    '''
    return getOutputDirectory(filesLocation, label, 'camera-efficiency')


def getTestDataFile(fileName):
    '''
    Get path of a test file, using the  testDataLocation taken from the config file.

    Parameters
    ----------
    filesName: str
        File name

    Returns
    -------
    Path
    '''
    return Path(cfg.get('testDataLocation')).joinpath('test-data').joinpath(fileName)


def getTestPlotFile(fileName):
    '''
    Get path of a test plot file, using the  testDataLocation taken from the config file.

    Parameters
    ----------
    filesName: str
        File name

    Returns
    -------
    Path
    '''
    return Path(cfg.get('testDataLocation')).joinpath('test-plots').joinpath(fileName)
