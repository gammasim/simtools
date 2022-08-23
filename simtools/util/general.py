import copy
import logging
import mmap
import re
from collections import namedtuple

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
        if parName in outData.keys():
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


def _validateAndConvertValue(parName, parInfo, valueIn):
    """
    Validate input user parameter and convert it to the right units, if needed.
    Returns the validated arguments in a list.
    """

    logger = logging.getLogger(__name__)

    # Turning value into a list, if it is not.
    if isinstance(valueIn, dict):
        valueIsDict = True
        value = [d for (k, d) in valueIn.items()]
        valueKeys = [k for (k, d) in valueIn.items()]
    else:
        valueIsDict = False
        value = copyAsList(valueIn)

    # Checking the entry length
    valueLength = len(value)
    logger.debug("Value len of {}: {}".format(parName, valueLength))
    undefinedLength = False
    if parInfo["len"] is None:
        undefinedLength = True
    elif valueLength != parInfo["len"]:
        msg = "Config entry with wrong len: {}".format(parName)
        logger.error(msg)
        raise InvalidConfigEntry(msg)

    # Checking unit
    if "unit" not in parInfo.keys():

        # Checking if values have unit and raising error, if so.
        if all([isinstance(v, str) for v in value]):
            # In case values are string, e.g. mirrorNumbers = 'all'
            # This is needed otherwise the elif condition will break
            pass
        elif any([u.Quantity(v).unit != u.dimensionless_unscaled for v in value]):
            msg = "Config entry {} should not have units".format(parName)
            logger.error(msg)
            raise InvalidConfigEntry(msg)

        if valueIsDict:
            return {k: v for (k, v) in zip(valueKeys, value)}
        else:
            return value if len(value) > 1 or undefinedLength else value[0]

    else:
        # Turning parInfo['unit'] into a list, if it is not.
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

        if valueIsDict:
            return {k: v for (k, v) in zip(valueKeys, valueWithUnits)}
        else:
            return (
                valueWithUnits if len(valueWithUnits) > 1 or undefinedLength else valueWithUnits[0]
            )


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
