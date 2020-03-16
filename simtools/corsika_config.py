#!/usr/bin/python3

import logging
import yaml
from pathlib import Path

from simtools.util import names
from simtools import io_handler as io
from simtools.array_model import getArray


__all__ = ['CorsikaConfig']


ALL_PARAMETERS = {
    'RUNNR': {'len': 1, 'names': ['RUNNUMBER'], 'default': 1},
    'EVTNR': {'len': 1, 'names': ['EVENTNR', 'EVENTNUMBER'], 'default': 1},
    'NSHOW': {'len': 1, 'names': ['NSHOWERS']},
    'PRMPAR': {'len': 1, 'names': ['PRIMARY']},
    'ERANGE': {'len': 2, 'names': ['ENRANGE', 'ENERGYRANGE']},
    'ESLOPE': {'len': 1, 'names': ['ENSLOPE', 'ENERGYSLOPE']},
    'THETAP': {'len': 2, 'names': ['THETA', 'ZENITH']},
    'PHIP': {'len': 1, 'names': ['PHI', 'AZIMUTH']},
    'VIEWCONE': {'len': 2, 'names': ['CONE']},
    'CSCAT': {'len': 3, 'names': []}
}


PRIMARIES = {
    'GAMMA': {'number': 1, 'names': ['PHOTON']},
    'PROTON': {'number': 14, 'names': []}
}

SITES_PARS = {
    'Paranal': {
        'OBSLEV': [2150.e2],
        'ATMOSPHERE': [26, 'Y'],
        'MAGNET': [20.925, -9.119],
        'ARRANG':  [0.]
    },
    'LaPalma': {
        'OBSLEV': [2158.e2]
        'ATMOSPHERE': [36, 'Y']
        'MAGNET': [30.576, 23.571]
        'ARRANG': [-5.3195]
    }
}

INTERACTION_FLAGS = {
    'FIXHEI': [0., 0],
    'FIXCHI': [0.],
    'TSTART': ['T'],
    'ECUTS': [0.3, 0.1, 0.020, 0.020],
    'MUADDI': ['F'],
    'MUMULT': ['T'],
    'LONGI': ['T', 20., 'F', 'F'],
    'MAXPRT': [0],
    'ECTMAP': [1.e6],
    'STEPFC': [1.0]
}

CHERENKOV_EMISSION_PARS = {
    'CERSIZ': [5.],
    'CERFIL': ['F'],
    'CWAVLG': [240., 700.]
}

DEBUGGING_OUTPUT_PARS = {
    'DEBUG': ['F', 6, 'F', 1000000],
    'DATBAS': ['yes'],
    'DIRECT': [r'/dev/null'],
    'TELFIL': ['XFILEX']
}

IACT_TUNING_PARS = {
    'IACT': [
        ['SPLIT_AUTO', '15M'],
        ['IO_BUFFER', '800MB'],
        ['MAX_BUNCHES', '1000000']
    ]
}


def getTelescopeZ(size):
    if size == 'LST':
        return 16.
    elif size == 'MST':
        return 10.
    elif size == 'SST':
        return 5.
    else:
        logging.error('Wrong telescope size - {}'.format(size))
        raise Exception('Wrong telescope size - {}'.format(size))


def getSphereRadius(size):
    if size == 'LST':
        return 12.5
    elif size == 'MST':
        return 7.
    elif size == 'SST':
        return 3.5
    else:
        logging.error('Wrong sphere radius - {}'.format(size))
        raise Exception('Wrong sphere redius - {}'.format(size))


def writeTelescopes(file, array):
    mToCm = 1e2
    for n, tel in array.items():
        file.write('\nTELESCOPE {} {} {} {}'.format(
            tel['xPos'] * mToCm,
            tel['yPos'] * mToCm,
            getTelescopeZ(tel['size']) * mToCm,
            getSphereRadius(tel['size']) * mToCm
        ))

    file.write('\n')
    pass


class CorsikaConfig:
    def __init__(self, site, arrayName, databaseLocation, label=None, filesLocation=None, **kwargs):
        ''' Docs please '''
        logging.info('Init CorsikaConfig')

        self._label = label
        self._filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)
        self._site = names.validateName(site, names.allSiteNames)
        self._arrayName = names.validateName(arrayName, names.allArrayNames)
        self._array = getArray(self._arrayName, databaseLocation)

        self._loadArguments(**kwargs)

        print(self._parameters)

        print(self._array)

    def _loadArguments(self, **kwargs):
        self._parameters = dict()

        def validateAndFixArgs(parName, valueArgs):
            valueArgs = valueArgs if isinstance(valueArgs, list) else [valueArgs]
            if len(valueArgs) == 1 and parName == 'THETAP':  # fixing single value zenith angle
                valueArgs = valueArgs * 2
            if len(valueArgs) == 1 and parName == 'VIEWCONE':  # fixing single value viewcone
                valueArgs = [0, valueArgs[0]]

            if len(valueArgs) != parInfo['len']:
                logging.warning('Argument {} has wrong len'.format(keyArgs.upper()))

            return valueArgs

        indentifiedArgs = list()
        for keyArgs, valueArgs in kwargs.items():

            for parName, parInfo in ALL_PARAMETERS.items():

                if keyArgs.upper() == parName or keyArgs.upper() in parInfo['names']:
                    indentifiedArgs.append(keyArgs)
                    valueArgs = validateAndFixArgs(parName, valueArgs)
                    self._parameters[parName] = valueArgs

        unindentifiedArgs = [p for p in kwargs.keys() if p not in indentifiedArgs]
        if len(unindentifiedArgs) > 0:
            logging.warning(
                '{} argument were not properly identified: {} ...'.format(
                    len(unindentifiedArgs),
                    unindentifiedArgs[0]
                )
            )

    def exportFile(self):
        fileName = names.corsikaConfigFileName(
            arrayName=self._arrayName,
            site=self._site,
            zenith=self._parameters['THETAP'],
            viewCone=self._parameters['VIEWCONE'],
            label=self._label
        )
        fileDirectory = io.getCorsikaOutputDirectory(self._filesLocation, self._label)

        if not fileDirectory.exists():
            fileDirectory.mkdir(parents=True, exist_ok=True)
            logging.info('Creating directory {}'.format(fileDirectory))
        self._filePath = fileDirectory.joinpath(fileName)

        def writeParametersOneLine(file, pars):
            for par, values in pars.items():
                line = par + ' '
                for v in values:
                    line += str(v) + ' '
                line += '\n'
                file.write(line)

        def writeParametersManyLines(file, pars):
            for par, valueList in pars.items():
                for value in valueList:
                    newPars = {par: value}
                    writeParametersOneLine(file, newPars)

        with open(self._filePath, 'w') as file:
            writeParametersOneLine(file, self._parameters)
            file.write('\n# SITE PARAMETERS\n')
            writeParametersOneLine(file, SITES_PARS[self._site])
            file.write('\n# TELESCOPES\n')
            writeTelescopes(file, self._array)
            file.write('\n# INTERACTION FLAGS\n')
            writeParametersOneLine(file, INTERACTION_FLAGS)
            file.write('\n# CHERENKOV EMISSION PARAMETERS\n')
            writeParametersOneLine(file, CHERENKOV_EMISSION_PARS)
            file.write('\n# DEBUGGING OUTPUT PARAMETERS\n')
            writeParametersOneLine(file, DEBUGGING_OUTPUT_PARS)
            file.write('\n# IACT TUNING PARAMETERS\n')
            writeParametersManyLines(file, IACT_TUNING_PARS)
            file.write('\nEXIT')

    def addLine(self):
        pass
