""" Module to handle input and output conventions. """

import datetime
from pathlib import Path

import simtools.config as cfg

__all__ = [
    "getModelOutputDirectory",
    "getRayTracingOutputDirectory",
    "getCorsikaOutputDirectory",
    "getTestDataFile",
    "getTestDataDirectory",
    "getTestOutputFile",
    "getTestPlotFile",
]


def _getOutputDirectory(filesLocation, label, mode=None):
    """
    Get main output directory for a generic mode

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.
    mode: str
        Name of the subdirectory (ray-tracing, model etc)

    Returns
    -------
    Path
    """
    today = datetime.date.today()
    labelDir = label if label is not None else "d-" + str(today)
    path = Path(filesLocation).joinpath("simtools-output").joinpath(labelDir)
    if mode is not None:
        path = path.joinpath(mode)
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute()


def getModelOutputDirectory(filesLocation, label):
    """
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
    """
    return _getOutputDirectory(filesLocation, label, "model")


def getLayoutOutputDirectory(filesLocation, label):
    """
    Get output directory for layout related files.

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    """
    return _getOutputDirectory(filesLocation, label, "layout")


def getRayTracingOutputDirectory(filesLocation, label):
    """
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
    """
    return _getOutputDirectory(filesLocation, label, "ray-tracing")


def getCorsikaOutputDirectory(filesLocation, label):
    """
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
    """
    return _getOutputDirectory(filesLocation, label, "corsika")


def getCameraEfficiencyOutputDirectory(filesLocation, label):
    """
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
    """
    return _getOutputDirectory(filesLocation, label, "camera-efficiency")


def getApplicationOutputDirectory(filesLocation, label):
    """
    Get output directory for applications related files.

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    """
    return _getOutputDirectory(filesLocation, label, "application-plots")


def getArraySimulatorOutputDirectory(filesLocation, label):
    """
    Get output directory for array-simulator related files.

    Parameters
    ----------
    filesLocation: str, or Path
        Main location of the output files.
    label: str
        Instance label.

    Returns
    -------
    Path
    """
    return _getOutputDirectory(filesLocation, label, "array-simulator")


def getDataFile(parentDir, fileName):
    """
    Get path of a data file, using the dataLocation taken from the config file.

    Parameters
    ----------
    parentDir: str
        Parent directory of the file.
    filesName: str
        File name.

    Returns
    -------
    Path
    """
    return (
        Path(cfg.get("dataLocation")).joinpath(parentDir).joinpath(fileName).absolute()
    )


def getTestDataDirectory():
    """
    Get path of a test file directory, using the testDataLocation taken from the config file.

    Returns
    -------
    Path
    """
    return Path("tests/resources/")


def getTestDataFile(fileName):
    """
    Get path of a test file, using the testDataLocation taken from the config file.

    Parameters
    ----------
    filesName: str
        File name

    Returns
    -------
    Path
    """
    directory = getTestDataDirectory()
    return directory.joinpath(fileName)


def getTestOutputDirectory():
    """
    Get path of a test directory, using the outputLocation taken from the config file.
    Path is created, if it doesn't exist.

    Returns
    -------
    Path
    """
    path = Path(cfg.get("outputLocation")).joinpath("test-output")
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute()


def getTestOutputFile(fileName):
    """
    Get path of a test file, using the outputLocation taken from the config file.

    Parameters
    ----------
    filesName: str
        File name

    Returns
    -------
    Path
    """
    directory = getTestOutputDirectory()
    return directory.joinpath(fileName)


def getTestModelDirectory():
    """
    Get path of a test model directory, using the outputLocation taken from the config file.
    Path is created, if it doesn't exist.

    Returns
    -------
    Path
    """
    path = Path(cfg.get("outputLocation")).joinpath("model")
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute()


def getTestModelFile(fileName):
    """
    Get path of a model test file, using the outputLocation taken from the config file.

    Parameters
    ----------
    filesName: str
        File name

    Returns
    -------
    Path
    """
    directory = getTestModelDirectory()
    return directory.joinpath(fileName)


def getTestPlotFile(fileName):
    """
    Get path of a test plot file, using the testDataLocation taken from the config file.
    Path is created, if it doesn't exist.

    Parameters
    ----------
    filesName: str
        File name

    Returns
    -------
    Path
    """

    path = Path(cfg.get("outputLocation")).joinpath("test-plots")
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(fileName).absolute()
