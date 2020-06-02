#!/usr/bin/python3

import logging
import math

from simtools.model.model_parameters import MODEL_PARS


__all__ = ['computeTelescopeTransmission', 'getTelescopeSize']

logger = logging.getLogger(__name__)


def computeTelescopeTransmission(pars, offAxis):
    '''
    Compute tel. transmission (0 < T < 1) for a given set of parameters
    as defined by the MC model and for a given off-axis angle.

    Parameters
    ----------
    pars: list of float
        Parameters of the telescope transmission. Len(pars) should be 4.
    offAxis: float
        Off-axis angle in deg.

    Returns
    -------
    float
        Telescope transmission.
    '''
    _degToRad = math.pi / 180.
    if pars[1] == 0:
        return pars[0]
    else:
        t = math.sin(offAxis*_degToRad) / (pars[3]*_degToRad)
        return pars[0] / (1. + pars[2] * t**pars[4])


def getTelescopeSize(telescopeType):
    '''
    Provide the telescope size (SST, MST or LST) for a given telescopeType.

    Parameters
    ----------
    telescopeType: str
        Ex. SST-2M-ASTRI, LST, ...

    Returns
    -------
    str
        'SST', 'MST' or 'LST'
    '''
    if 'SST' in telescopeType:
        return 'SST'
    elif 'MST' in telescopeType:
        return 'MST'
    elif 'LST' in telescopeType:
        return 'LST'
    else:
        logger.warning('Invalid telescopeType {}'.format(telescopeType))
        return None


def validateModelParameter(parNameIn, parValueIn):
    '''
    Validate model parameter based on the dict MODEL_PARS.

    Parameters
    ----------
    parNameIn: str
        Name of the parameter to be validated.
    parValueIn: str
        Value of the parameter to be validated.

    Returns
    -------
    (parName, parValue) after validated. parValueIn is converted to the proper type if that
    information is available in MODEL_PARS
    '''
    logger.debug('Validating parameter {}'.format(parNameIn))
    for parNameModel in MODEL_PARS.keys():
        if parNameIn == parNameModel or parNameIn in MODEL_PARS[parNameModel]['names']:
            parType = MODEL_PARS[parNameModel]['type']
            return parNameModel, parType(parValueIn)
    return parNameIn, parValueIn
