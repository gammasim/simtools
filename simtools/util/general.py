import copy
import logging
import mmap
import os
import re
from collections import namedtuple
from pathlib import Path

import astropy.units as u
from astropy.io.misc import yaml

__all__ = [
    "validateConfigData",
    "collectDataFromYamlOrDict",
    "collectKwargs",
    "setDefaultKwargs",
    "sortArrays",
    "collectFinalLines",
    "getLogLevelFromUser",
    "separateArgsAndConfigData",
]


class UnableToIdentifyConfigEntry(Exception):
    pass


class MissingRequiredConfigEntry(Exception):
    pass


class InvalidConfigEntry(Exception):
    pass


class InvalidConfigData(Exception):
    pass


def fileHasText(file, text):
    """
    Check whether a file contain a certain piece of text.

    Parameters
    ----------
    file: str
        Path of the file.
    text: str
        Piece of text to be searched for.

    Returns
    -------
    bool
    """
    with open(file, "rb", 0) as stringFile, mmap.mmap(
        stringFile.fileno(), 0, access=mmap.ACCESS_READ
    ) as textFileInput:
        re_search_1 = re.compile(f"{text}".encode())
        searchResult_1 = re_search_1.search(textFileInput)
        if searchResult_1 is None:
            return False
        else:
            return True


def validateConfigData(configData, parameters):
    """
    Validate a generic configData dict by using the info
    given by the parameters dict. The entries will be validated
    in terms of length, units and names.

    See data/test-data/test_parameters.yml for an example of the structure
    of the parameters dict.

    Parameters
    ----------
    configData: dict
        Input config data.
    parameters: dict
        Parameter information necessary for validation.

    Raises
    ------
    UnableToIdentifyConfigEntry
        When an entry in configData cannot be identified among the parameters.
    MissingRequiredConfigEntry
        When a parameter without default value is not given in configData.
    InvalidConfigEntry
        When an entry in configData is invalid (wrong len, wrong unit, ...).

    Returns
    -------
    namedtuple:
        Containing the validated config data entries.
    """

    logger = logging.getLogger(__name__)

    # Dict to be filled and returned
    outData = dict()

    if configData is None:
        configData = dict()

    # Collecting all entries given as in configData.
    for keyData, valueData in configData.items():

        isIdentified = False
        # Searching for the key in the parameters.
        for parName, parInfo in parameters.items():
            names = parInfo.get("names", [])
            if keyData != parName and keyData.lower() not in [n.lower() for n in names]:
                continue
            # Matched parameter
            validatedValue = _validateAndConvertValue(parName, parInfo, valueData)
            outData[parName] = validatedValue
            isIdentified = True

        # Raising error for an unidentified input.
        if not isIdentified:
            msg = "Entry {} in configData cannot be identified.".format(keyData)
            logger.error(msg)
            raise UnableToIdentifyConfigEntry(msg)

    # Checking for parameters with default option.
    # If it is not given, filling it with the default value.
    for parName, parInfo in parameters.items():
        if parName in outData:
            continue
        elif "default" in parInfo.keys() and parInfo["default"] is not None:
            validatedValue = _validateAndConvertValue(parName, parInfo, parInfo["default"])
            outData[parName] = validatedValue
        elif "default" in parInfo.keys() and parInfo["default"] is None:
            outData[parName] = None
        else:
            msg = (
                "Required entry in configData {} ".format(parName)
                + "was not given (there may be more)."
            )
            logger.error(msg)
            raise MissingRequiredConfigEntry(msg)

    ConfigData = namedtuple("ConfigData", outData)
    return ConfigData(**outData)


def _validateAndConvertValue_without_units(value, valueKeys, parName, parInfo):
    """
    Validate input user parameter for input values without units.

    Parameters
    ----------
    value: list
       list of user input values
    valueKeys: list
       list of keys if user input was a dict; otherwise None
    parName: str
       name of parameter

    Returns
    -------
    list, dict
        validated and converted input data

    """
    logger = logging.getLogger(__name__)

    _, undefinedLength = _checkValueEntryLength(value, parName, parInfo)

    # Checking if values have unit and raising error, if so.
    if all([isinstance(v, str) for v in value]):
        # In case values are string, e.g. mirrorNumbers = 'all'
        # This is needed otherwise the elif condition will break
        pass
    elif any([u.Quantity(v).unit != u.dimensionless_unscaled for v in value]):
        msg = "Config entry {} should not have units".format(parName)
        logger.error(msg)
        raise InvalidConfigEntry(msg)

    if valueKeys:
        return {k: v for (k, v) in zip(valueKeys, value)}
    return value if len(value) > 1 or undefinedLength else value[0]


def _checkValueEntryLength(value, parName, parInfo):
    """
    Validate length of user input parmeters

    Parameters
    ----------
    value: list
        list of user input values
    parName: str
        name of parameter
    parInfo: dict
        dictionary with parameter info

    Returns
    -------
    valueLength: int
        length of input list
    undefinedLength: bool
        state of input list

    """
    logger = logging.getLogger(__name__)

    # Checking the entry length
    valueLength = len(value)
    logger.debug("Value len of {}: {}".format(parName, valueLength))
    undefinedLength = False
    try:
        if parInfo["len"] is None:
            undefinedLength = True
        elif valueLength != parInfo["len"]:
            msg = "Config entry with wrong len: {}".format(parName)
            logger.error(msg)
            raise InvalidConfigEntry(msg)
    except KeyError:
        logger.error("Missing len entry in parInfo")
        raise

    return valueLength, undefinedLength


def _validateAndConvertValue_with_units(value, valueKeys, parName, parInfo):
    """
    Validate input user parameter for input values with units.

    Parameters
    ----------
    value: list
       list of user input values
    valueKeys: list
       list of keys if user input was a dict; otherwise None
    parnName: str
       name of parameter

    Returns
    -------
    list, dict
        validated and converted input data

    """
    logger = logging.getLogger(__name__)

    valueLength, undefinedLength = _checkValueEntryLength(value, parName, parInfo)

    parUnit = copyAsList(parInfo["unit"])

    if undefinedLength and len(parUnit) != 1:
        msg = "Config entry with undefined length should have a single unit: {}".format(parName)
        logger.error(msg)
        raise InvalidConfigEntry(msg)
    elif len(parUnit) == 1:
        parUnit *= valueLength

    # Checking units and converting them, if needed.
    valueWithUnits = list()
    for arg, unit in zip(value, parUnit):
        # In case a entry is None, None should be returned.
        if unit is None or arg is None:
            valueWithUnits.append(arg)
            continue

        # Converting strings to Quantity
        if isinstance(arg, str):
            arg = u.quantity.Quantity(arg)

        if not isinstance(arg, u.quantity.Quantity):
            msg = "Config entry given without unit: {}".format(parName)
            logger.error(msg)
            raise InvalidConfigEntry(msg)
        elif not arg.unit.is_equivalent(unit):
            msg = "Config entry given with wrong unit: {}".format(parName)
            logger.error(msg)
            raise InvalidConfigEntry(msg)
        else:
            valueWithUnits.append(arg.to(unit).value)

    if valueKeys:
        return {k: v for (k, v) in zip(valueKeys, valueWithUnits)}

    return valueWithUnits if len(valueWithUnits) > 1 or undefinedLength else valueWithUnits[0]


def _validateAndConvertValue(parName, parInfo, valueIn):
    """
    Validate input user parameter and convert it to the right units, if needed.
    Returns the validated arguments in a list.
    """

    if isinstance(valueIn, dict):
        value = [d for (k, d) in valueIn.items()]
        valueKeys = [k for (k, d) in valueIn.items()]
    else:
        value = copyAsList(valueIn)
        valueKeys = None

    if "unit" not in parInfo.keys():
        return _validateAndConvertValue_without_units(value, valueKeys, parName, parInfo)

    return _validateAndConvertValue_with_units(value, valueKeys, parName, parInfo)


def collectDataFromYamlOrDict(inYaml, inDict, allowEmpty=False):
    """
    Collect input data that can be given either as a dict
    or as a yaml file.

    Parameters
    ----------
    inYaml: str
        Name of the Yaml file.
    inDict: dict
        Data as dict.
    allowEmpty: bool
        If True, an error won't be raised in case both yaml and dict are None.

    Returns
    -------
    data: dict
        Data as dict.
    """
    _logger = logging.getLogger(__name__)

    if inYaml is not None:
        if inDict is not None:
            _logger.warning("Both inDict inYaml were given - inYaml will be used")
        with open(inYaml) as file:
            data = yaml.load(file)
        return data
    elif inDict is not None:
        return dict(inDict)
    else:
        msg = "configData has not been provided (by yaml file neither by dict)"
        if allowEmpty:
            _logger.debug(msg)
            return None
        else:
            _logger.error(msg)
            raise InvalidConfigData(msg)


def collectKwargs(label, inKwargs):
    """
    Collect kwargs of the type label_* and return them as a dict.

    Parameters
    ----------
    label: str
    inKwargs: dict

    Returns
    -------
    Dict with the collected kwargs.
    """
    outKwargs = dict()
    for key, value in inKwargs.items():
        if label + "_" in key:
            outKwargs[key.replace(label + "_", "")] = value
    return outKwargs


def setDefaultKwargs(inKwargs, **kwargs):
    """
    Fill in a dict with a set of default kwargs and return it.

    Parameters
    ----------
    inKwargs: dict
        Input dict to be filled in with the default kwargs.
    **kwargs:
        Default kwargs to be set.

    Returns
    -------
    Dict containing the default kwargs.
    """
    for par, value in kwargs.items():
        if par not in inKwargs.keys():
            inKwargs[par] = value
    return inKwargs


def sortArrays(*args):
    orderArray = copy.copy(args[0])
    newArgs = list()
    for arg in args:
        _, a = zip(*sorted(zip(orderArray, arg)))
        newArgs.append(list(a))
    return newArgs


def collectFinalLines(file, nLines):
    """
    Parameters
    ----------
    file: str or Path
        File to collect the lines from.
    nLines: int
        Number of lines to be collected.

    Returns
    -------
    str: lines
    """
    fileInLines = list()
    with open(file, "r") as f:
        for line in f:
            fileInLines.append(line)
    collectedLines = fileInLines[-nLines:-1]
    out = ""
    for ll in collectedLines:
        out += ll
    return out


def getLogLevelFromUser(logLevel):
    """
    Map between logging level from the user to logging levels of the logging module.

    Parameters
    ----------
    logLevel: str
        Log level from the user

    Returns
    -------
    logging.LEVEL
        The requested logging level to be used as input to logging.setLevel()
    """

    possibleLevels = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logLevelLower = logLevel.lower()
    if logLevelLower not in possibleLevels:
        raise ValueError(
            '"{}" is not a logging level, only possible ones are {}'.format(
                logLevel, list(possibleLevels.keys())
            )
        )
    else:
        return possibleLevels[logLevelLower]


def copyAsList(value):
    """
    Copy value and, if it is not a list, turn it into a list with a single entry.

    Parameters
    ----------
    value: single variable of any type, or list

    Returns
    -------
    value: list
        Copy of value if it is a list of [value] otherwise.
    """
    if isinstance(value, str):
        return [value]
    else:
        try:
            return list(value)
        except Exception:
            return [value]


def separateArgsAndConfigData(expectedArgs, **kwargs):
    """
    Separate kwargs into the arguments expected for instancing a class and
    the dict to be given as configData.
    This function is specific for methods fromKwargs in classes which use the
    validateConfigData system.

    Parameters
    ----------
    expectedArgs: list of str
        List of arguments expected for the class.
    **kwargs:

    Returns
    -------
    dict, dict
        A dict with the args collected and another one with configData.
    """
    args = dict()
    configData = dict()
    for key, value in kwargs.items():
        if key in expectedArgs:
            args[key] = value
        else:
            configData[key] = value

    return args, configData


def program_is_executable(program):
    """
    Checks if program exists and is executable

    Follows https://stackoverflow.com/questions/377017/

    """

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def findFile(name, loc):
    """
    Search for files inside of given directories, recursively, and return its full path.

    Parameters
    ----------
    name: str
        File name to be searched for.
    loc: Path, optional
        Location of where to search for the file.

    Returns
    -------
    Full path of the file to be found if existing. Otherwise, None

    Raises
    ------
    FileNotFoundError
        If the desired file is not found.
    """
    _logger = logging.getLogger(__name__)

    allLocations = copy.copy(loc)
    allLocations = [allLocations] if not isinstance(allLocations, list) else allLocations

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
