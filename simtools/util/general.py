import logging
import copy

import astropy.units as u
from astropy.io.misc import yaml

__all__ = [
    'collectArguments',
    'collectDataFromYamlOrDict',
    'collectKwargs',
    'setDefaultKwargs',
    'sortArrays',
    'collectFinalLines',
    'getLogLevelFromUser'
]


class ArgumentWithWrongUnit(Exception):
    pass


class ArgumentCannotBeCollected(Exception):
    pass


class MissingRequiredArgument(Exception):
    pass


def _unitsAreConvertible(quantity_1, quantity_2):
    '''
    Parameters
    ----------
    quantity_1: astropy.Quantity
    quantity_2: astropy.Quantity

    Returns
    -------
    Bool
    '''
    try:
        quantity_1.to(quantity_2)
        return True
    except Exception:
        return False


def _unitIsValid(quantity, unit):
    '''
    Parameters
    ----------
    quantity: astropy.Quantity
    unit: astropy.unit

    Returns
    -------
    Bool
    '''
    if unit is None and not isinstance(1 * quantity, u.quantity.Quantity):
        return True
    else:
        return _unitsAreConvertible(quantity, unit)


def _convertUnit(quantity, unit):
    '''
    Parameters
    ----------
    quantity: astropy.Quantity
    unit: astropy.unit

    Returns
    -------
    astropy.quantity
    '''
    return quantity if unit is None else quantity.to(unit).value


def collectArguments(obj, args, allInputs, **kwargs):
    '''
    Collect certain arguments and validate them.
    To be used during initialization of classes, where kwargs with
    physical meaning and units are expected.

    Note
    ----

    In ray_tracing class,

    .. code-block:: python

        collectArguments(
            self,
            args=['zenithAngle', 'offAxisAngle', 'sourceDistance'],
            allInputs=self.ALL_INPUTS,
            **kwargs
        )

    where,

    .. code-block:: python

        ALL_INPUTS = {
            'zenithAngle': {'default': 20, 'unit': u.deg},
            'offAxisAngle': {'default': [0, 1.5, 3.0], 'unit': u.deg, 'isList': True},
            'sourceDistance': {'default': 10, 'unit': u.km},
            'mirrorNumbers': {'default': [1], 'unit': None, 'isList': True}
        }


    Parameters
    ----------
    obj: class instance (self)
    args: list of str
        List of names of the parameters to be collected.
    allInputs: dict
        Dict with info about all the expected inputs. See example.
    **kwargs:
        kwargs from the input arguments.
    '''
    _logger = logging.getLogger(__name__)

    def processSingleArg(arg, inArgName, argG, argD):
        if _unitIsValid(argG, argD['unit']):
            obj.__dict__[inArgName] = _convertUnit(argG, argD['unit'])
        else:
            msg = 'Argument {} given with wrong unit'.format(arg)
            _logger.error(msg)
            raise ArgumentWithWrongUnit(msg)

    def processListArg(arg, inArgName, argG, argD):
        outArg = list()
        try:
            argG = list(argG)
        except Exception:
            argG = [argG]

        for aa in argG:
            if _unitIsValid(aa, argD['unit']):
                outArg.append(_convertUnit(aa, argD['unit']))
            else:
                msg = 'Argument {} given with wrong unit'.format(arg)
                _logger.error(msg)
                raise ArgumentWithWrongUnit(msg)
        obj.__dict__[inArgName] = outArg

    def processDictArg(arg, inArgName, argG, argD):
        outArg = dict()

        if not isinstance(argG, dict):
            msg = 'Argument is not a dict - aborting'
            _logger.error(msg)
            raise ArgumentCannotBeCollected(msg)

        for key, value in argG.items():
            if _unitIsValid(value, argD['unit']):
                outArg[key] = _convertUnit(value, argD['unit'])
            else:
                msg = 'Argument {} given with wrong unit'.format(arg)
                _logger.error(msg)
                raise ArgumentWithWrongUnit(msg)
        obj.__dict__[inArgName] = outArg

    for arg in args:
        inArgName = '_' + arg
        argData = allInputs[arg]

        if arg not in allInputs.keys():
            msg = 'Arg {} cannot be collected because it is not in allInputs'.format(arg)
            _logger.error(msg)
            raise ArgumentCannotBeCollected(msg)

        if arg in kwargs.keys():
            argGiven = kwargs[arg]
            if argGiven is None:
                obj.__dict__[inArgName] = None
            elif 'isDict' in argData and argData['isDict']:  # Dict
                processDictArg(arg, inArgName, argGiven, argData)
            elif 'isList' in argData and argData['isList']:  # List
                processListArg(arg, inArgName, argGiven, argData)
            else:  # Not a list or dict
                processSingleArg(arg, inArgName, argGiven, argData)

        elif 'default' in argData:
            obj.__dict__[inArgName] = argData['default']
        else:
            msg = 'Required argument (without default) {} was not given'.format(arg)
            _logger.warning(msg)
            raise MissingRequiredArgument(msg)

    return


def collectDataFromYamlOrDict(inYaml, inDict):
    '''
    Collect input data that can be given either as a dict
    or as a yaml file.

    Parameters
    ----------
    inYaml: str
        Name of the Yaml file.
    inDict: dict
        Data as dict.

    Returns
    -------
    data: dict
        Data as dict.
    '''
    _logger = logging.getLogger(__name__)

    if inYaml is not None:
        if inDict is not None:
            _logger.warning('Both inDict inYaml were given - inYaml will be used')
        with open(inYaml) as file:
            data = yaml.load(file)
        return data
    elif inDict is not None:
        return dict(inDict)
    else:
        _logger.error('No data was given - aborting')
        return None


def collectKwargs(label, inKwargs):
    '''
    Collect kwargs of the type label_* and return them as a dict.

    Parameters
    ----------
    label: str
    inKwargs: dict

    Returns
    -------
    Dict with the collected kwargs.
    '''
    outKwargs = dict()
    for key, value in inKwargs.items():
        if label + '_' in key:
            outKwargs[key.replace(label + '_', '')] = value
    return outKwargs


def setDefaultKwargs(inKwargs, **kwargs):
    '''
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
    '''
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
    '''
    Parameters
    ----------
    file: str or Path
        File to collect the lines from.
    nLines: int
        Number of lines to be collected.

    Returns
    -------
    str: lines
    '''
    fileInLines = list()
    with open(file, 'r') as f:
        for line in f:
            fileInLines.append(line)
    collectedLines = fileInLines[-nLines:-1]
    out = ''
    for ll in collectedLines:
        out += ll
    return out


def getLogLevelFromUser(logLevel):
    '''
    Map between logging level from the user to logging levels of the logging module.

    Parameters
    ----------
    logLevel: str
        Log level from the user

    Returns
    -------
    logging.LEVEL
        The requested logging level to be used as input to logging.setLevel()
    '''

    possibleLevels = {
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    logLevelLower = logLevel.lower()
    if logLevelLower not in possibleLevels:
        raise ValueError(
            '"{}" is not a logging level, only possible ones are {}'.format(
                logLevel,
                list(possibleLevels.keys())
            )
        )
    else:
        return possibleLevels[logLevelLower]

def copyAsList(value):
    '''
    Copy value and, if it is not a list, turn it into a list with a single entry.

    Parameters
    ----------
    value: single variable of any type, or list

    Returns
    -------
    value: list
        Copy of value if it is a list of [value] otherwise.
    '''
    return copy.copy(value) if isinstance(value, list) else [value]

