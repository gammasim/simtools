#!/usr/bin/python3

import logging
import yaml
from pathlib import Path

from simtools.util import names
from simtools import io_handler as io

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
    'IACT': ['SPLIT_AUTO', '15M'],
    'IACT': ['IO_BUFFER', '800MB'],
    'IACT': ['MAX_BUNCHES', '1000000']
}


class CorsikaConfig:
    def __init__(self, site, array, label=None, filesLocation=None, **kwargs):
        ''' Docs please '''
        logging.info('Init CorsikaConfig')

        self._label = label
        self._filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)
        self._site = names.validateName(site, names.allSiteNames)
        # self._array = names.validateName(array, names.allArrayNames)
        self._loadArguments(**kwargs)

        print(self._parameters)

    def _loadArguments(self, **kwargs):
        self._parameters = dict()

        indentifiedArgs = list()
        for keyArgs, valueArgs in kwargs.items():

            for parName, parInfo in ALL_PARAMETERS.items():

                if keyArgs.upper() == parName or keyArgs.upper() in parInfo['names']:
                    indentifiedArgs.append(keyArgs)

                    valueArgs = valueArgs if isinstance(valueArgs, list) else [valueArgs]
                    if len(valueArgs) != parInfo['len']:
                        logging.warning('Argument {} has wrong len'.format(keyArgs.upper()))
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
        fileName = 'test-corsika-input.txt'
        fileDirectory = io.getCorsikaOutputDirectory(self._filesLocation, self._label)

        if not fileDirectory.exists():
            fileDirectory.mkdir(parents=True, exist_ok=True)
            logging.info('Creating directory {}'.format(fileDirectory))
        self._filePath = fileDirectory.joinpath(fileName)

        def writeParameters(file, pars):
            for par, value in pars.items():
                line = par + ' '
                for v in value:
                    line += str(v) + ' '
                line += '\n'
                file.write(line)

        with open(self._filePath, 'w') as file:
            writeParameters(file, self._parameters)
            file.write('\n# SITE PARAMETERS\n')
            writeParameters(file, SITES_PARS[self._site])
            file.write('\n# INTERACTION FLAGS\n')
            writeParameters(file, INTERACTION_FLAGS)
            file.write('\n# CHERENKOV EMISSION PARAMETERS\n')
            writeParameters(file, CHERENKOV_EMISSION_PARS)
            file.write('\n# DEBUGGING OUTPUT PARAMETERS\n')
            writeParameters(file, DEBUGGING_OUTPUT_PARS)
            file.write('\n# IACT TUNING PARAMETERS\n')
            writeParameters(file, IACT_TUNING_PARS)
            file.write('\nEXIT')

    def addLine(self):
        pass
