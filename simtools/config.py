""" Module to deal with the interface with the global config information."""

import logging
import yaml
import copy
import os
from pathlib import Path

__all__ = ["setConfigFileName", "loadConfig", "get", "findFile", "change"]


class ConfigEnvironmentalVariableNotSet(Exception):
    pass


class ParameterNotFoundInConfigFile(Exception):
    pass


def setConfigFileName(fileName):
    """
    Redefines the config file name by resetting the a global variable.

    Parameters
    ----------
    fileName: str
        Config file name.
    """
    _logger = logging.getLogger(__name__)
    _logger.debug("Setting the config file name to {}".format(fileName))
    global CONFIG_FILE_NAME
    CONFIG_FILE_NAME = fileName


def loadConfig(fileName=None, useGlobals=True):
    """
    Load config file and return it as a dict.
    3 possible options for the config fileName:
    1st - fileName parameter is given (not None)
    2nd - CONFIG_FILE_NAME exists (set by setConfigFileName)
    3rd - ./config.yml

    Parameters
    ----------
    fileName: str, optional
        Config file name.
    ignoreGlobals: bool
        Use global config settings

    Returns
    -------
    dict
        A dictionary containing all the info from the global configuration setting.
    """
    if fileName is not None:
        thisFileName = fileName
    elif "CONFIG_FILE_NAME" in globals():
        thisFileName = CONFIG_FILE_NAME
    else:
        thisFileName = "config.yml"

    with open(thisFileName, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    # Running over the parameters set for change
    if useGlobals and "CONFIG_CHANGED_PARS" in globals():
        for par, value in CONFIG_CHANGED_PARS.items():
            config[par] = value

    return config


def get(par, useGlobals=True):
    """
    Get a single entry from the config settings.

    Parameters
    ----------
    par: str
        Name of the desired parameter.
    useGlobals: bool (default=True)
        Use global config settings

    Raises
    ------
    ParameterNotFoundInConfigFile
        In case the parameter is not in config.

    Returns
    -------
    Value of the entry from the config settings.
    """
    _logger = logging.getLogger(__name__)

    config = loadConfig(useGlobals=useGlobals)
    if par not in config.keys():
        msg = (
            "Configuration file does not contain an entry for the parameter "
            "{}".format(par)
        )
        _logger.error(msg)
        raise ParameterNotFoundInConfigFile(msg)
    else:
        # Enviroment variable
        if isinstance(config[par], str) and "$" in config[par]:
            return os.path.expandvars(config[par])
        else:
            return config[par]


def change(par, value):
    """
    Set to change a parameter to another value.

    Parameters
    ----------
    par: str
        Name of the parameter to change.
    value: any
        Value to be set to the parameter.
    """
    if "CONFIG_CHANGED_PARS" not in globals():
        global CONFIG_CHANGED_PARS
        CONFIG_CHANGED_PARS = dict()
    CONFIG_CHANGED_PARS[par] = value


def getConfigArg(name, value):
    """
    Get a config parameter if value is None. To be used to receive input arguments in classes.

    Parameters
    ----------
    name: str
        Name of the parameter
    value: str
        Input value.

    Returns
    -------
    Path
        Path of the desired parameter.
    """
    return value if value is not None else get(name)


def findFile(name, loc=None):
    """
    Search for model files inside of given directories, recursively, and return its full path.

    Parameters
    ----------
    name: str
        File name to be searched for.
    loc: Path, optional
        Location of where to search for the file. If not given, config information will be used.

    Returns
    -------
    Full path of the file to be found if existing. Otherwise, None

    Raises
    ------
    FileNotFoundError
        If the desired file is not found.
    """
    _logger = logging.getLogger(__name__)

    if loc is None:
        allLocations = get(par="modelFilesLocations")
    else:
        allLocations = copy.copy(loc)
    allLocations = (
        [allLocations] if not isinstance(allLocations, list) else allLocations
    )

    def _searchDirectory(directory, filename, rec=False):
        if not Path(directory).exists():
            msg = "Directory {} does not exist".format(directory)
            _logger.debug(msg)
            return None

        f = Path(directory).joinpath(filename)
        if f.exists():
            _logger.debug("File {} found in {}".format(filename, directory))
            return f
        if not rec:  # Not recursively
            return None

        for subdir in Path(directory).iterdir():
            if not subdir.is_dir():
                continue
            f = _searchDirectory(subdir, filename, True)
            if f is not None:
                return f
        return None

    # Searching file locally
    ff = _searchDirectory(".", name)
    if ff is not None:
        return ff
    # Searching file in given locations
    for ll in allLocations:
        ff = _searchDirectory(ll, name, True)
        if ff is not None:
            return ff
    msg = "File {} could not be found in {}".format(name, allLocations)
    _logger.error(msg)
    raise FileNotFoundError(msg)


def createDummyConfigFile(filename="config.yml", **kwargs):
    """
    Create a dummy config.yml file to be used in test enviroments only.

    Parameters
    ----------
    filename: str
        Name of the dummy config file (default=config.yml)
    **kwargs
        The default parameters can be overwritten using kwargs.
    """
    config = {
        "useMongoDB": False,
        "mongoDBConfigFile": None,
        "dataLocation": "./data/",
        "modelFilesLocations": ".",
        "simtelPath": ".",
        "outputLocation": ".",
        "extraCommands": []
    }

    # # Overwritting parameters with kwargs
    if len(kwargs) > 0:
        for key, value in kwargs.items():
            config[key] = value

    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile)


def createDummyDbDetails(filename="dbDetails.yml", **kwargs):
    """
    Create a dummy dbDetails.yml file to be used in test enviroments only.

    Parameters
    ----------
    filename: str
        Name of the dummy dbDetails file (default=dbDetails.yml)
    **kwargs
        The default parameters can be overwritten using kwargs.
    """
    pars = {
        "dbPort": None,
        "mongodbServer": None,
        "userDB": None,
        "passDB": None,
        "authenticationDatabase": "admin"
    }

    if len(kwargs) > 0:
        for key, value in kwargs.items():
            pars[key] = int(value) if key == "dbPort" else str(value)

    with open(filename, 'w') as outfile:
        yaml.dump(pars, outfile)
