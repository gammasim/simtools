#!/usr/bin/python3

import logging
import math

from simtools.model.model_parameters import MODEL_PARS
from simtools.util import names


__all__ = [
    'computeTelescopeTransmission',
    'getTelescopeClass',
    'getCameraName',
    'isTwoMirrorTelescope'
]

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


def getCameraName(telescopeName):
    '''
    Get camera name from the telescope name.

    Parameters
    ----------
    telescopeName: str
        Telescope name (ex. South-LST-1)

    Returns
    -------
    str
        Camera name (validated by util.names)
    '''
    cameraName = ''
    telSite, telClass, telType = names.splitTelescopeName(telescopeName)
    if telClass == 'LST':
        cameraName = 'LST'
    elif telClass == 'MST':
        if 'FlashCam' in telType:
            cameraName = 'FlashCam'
        elif 'NectarCam' in telType:
            cameraName = 'NectarCam'
        else:
            logger.error('Camera not found for MST class telescope')
    elif telClass == 'SCT':
        cameraName = 'SCT'
    elif telClass == 'SST':
        if 'ASTRI' in telType:
            cameraName = 'ASTRI'
        elif 'GCT' in telType:
            cameraName = 'GCT'
        elif '1M' in telType:
            cameraName = '1M'
        else:
            cameraName = 'SST'
    else:
        logger.error('Invalid telescope name - please validate it first')

    cameraName = names.validateCameraName(cameraName)
    logger.debug('Camera name - {}'.format(cameraName))
    return cameraName


def getTelescopeClass(telescopeName):
    '''
    Get telescope class from telescope name.

    Parameters
    ----------
    telescopeName: str
        Telescope name (ex. South-LST-1)

    Returns
    -------
    str
        Telescope class (SST, MST, ...)
    '''
    telSite, telClass, telType = names.splitTelescopeName(telescopeName)
    return telClass


def isTwoMirrorTelescope(telescopeName):
    '''
    Check if the telescope is a two mirror design.

    Parameters
    ----------
    telescopeName: str
        Telescope name (ex. South-LST-1)

    Returns
    -------
    bool
        True if the telescope is a two mirror one.
    '''
    telSite, telClass, telType = names.splitTelescopeName(telescopeName)
    if telClass == 'SST':
        # Only 1M is False
        return False if '1M' in telType else True
    elif telClass == 'SCT':
        # SCT always two mirrors
        return True
    else:
        # All MSTs and LSTs
        return False
