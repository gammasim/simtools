''' Module to handle interaction with DB. '''

import logging
import datetime
import yaml
from pathlib import Path

import simtools.config as cfg
from simtools.util import names
from simtools.util.model import validateModelParameter, getTelescopeSize

__all__ = ['getArrayDB']


logger = logging.getLogger(__name__)


def getArrayDB(databaseLocation):
    '''
    Get array db info as a dict.

    Parameters
    ----------
    databaseLocation: str or Path

    Returns
    -------
    dict
    '''
    file = Path(databaseLocation).joinpath('arrays').joinpath('arrays.yml')
    out = dict()
    with open(file, 'r') as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)
    return out


def getModelParameters(telescopeType, version, onlyApplicable=False):
    '''
    Get parameters from DB for one specific type (telescopeType, site ...).

    Parameters
    ----------
    telescopeType: str
    version: str
        Version of the model.
    onlyApplicable: bool
        If True, only applicable parameters will be selected.

    Returns
    -------
    dict containing the parameters
    '''
    _telTypeValidated = names.validateName(telescopeType, names.allTelescopeTypeNames)
    _versionValidated = names.validateName(version, names.allModelVersionNames)

    if getTelescopeSize(_telTypeValidated) == 'MST':
        # MST-FlashCam or MST-NectarCam
        _whichTelLabels = [_telTypeValidated, 'MST-optics']
    elif _telTypeValidated == 'SST':
        # SST = SST-Camera + SST-Structure
        _whichTelLabels = ['SST-Camera', 'SST-Structure']
    else:
        _whichTelLabels = [_telTypeValidated]

    # Selecting version and applicable (if on)
    _pars = dict()
    for _tel in _whichTelLabels:
        _allPars = collectAllModelParameters(_tel, _versionValidated)

        # If tel is a struture, only applicable pars will be collected, always.
        # The default ones will be covered by the camera pars.
        _selectOnlyApplicable = onlyApplicable or _tel in ['MST-optics', 'SST-Structure']

        for parNameIn, parInfo in _allPars.items():

            if not parInfo['Applicable'] and _selectOnlyApplicable:
                continue

            parNameOut, parValueOut = validateModelParameter(
                parNameIn,
                parInfo[_versionValidated]
            )
            _pars[parNameOut] = parValueOut

    return _pars


def collectAllModelParameters(telescopeType, version):
    '''
    Collect all parameters from DB for one specific type (telescopeTYpe, site ...).
    No selection is applied.

    Parameters
    ----------
    telescopeType: str
    version: str
        Version of the model.

    Returns
    -------
    dict containing the parameters
    '''
    _fileNameDB = 'parValues-{}.yml'.format(telescopeType)
    _yamlFile = cfg.findFile(
        _fileNameDB,
        cfg.collectConfigArg('modelFilesLocations')
    )
    logger.debug('Reading DB file {}'.format(_yamlFile))
    with open(_yamlFile, 'r') as stream:
        _allPars = yaml.load(stream, Loader=yaml.FullLoader)
    return _allPars
