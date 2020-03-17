#!/usr/bin/python3

import logging
import yaml
from pathlib import Path
import random

from simtools.util import names
from simtools import io_handler as io
from simtools.array_model import getArray
from simtools import corsika_parameters as cors_pars


__all__ = ['CorsikaConfig']


class RequiredInputNotGiven(Exception):
    pass


class ArgumentsNotLoaded(Exception):
    pass


def _writeTelescopes(file, array):
    mToCm = 1e2
    for n, tel in array.items():
        file.write('\nTELESCOPE {} {} {} {} # {}'.format(
            tel['xPos'] * mToCm,
            tel['yPos'] * mToCm,
            cors_pars.TELESCOPE_Z[tel['size']] * mToCm,
            cors_pars.TELESCOPE_SPHERE_RADIUS[tel['size']] * mToCm,
            tel['size']
        ))
    file.write('\n')


def _writeSeeds(file, seeds):
    for s in seeds:
        file.write('SEED {} 0 0\n'.format(s))


def _convertPrimaryInput(value):
    for primName, primInfo in cors_pars.PRIMARIES.items():
        if value[0].upper() == primName or value[0].upper() in primInfo['names']:
            return [primInfo['number']]


class CorsikaConfig:
    def __init__(
        self,
        site,
        arrayName,
        databaseLocation,
        label=None,
        filesLocation=None,
        randomSeeds=False,
        **kwargs
    ):
        ''' Docs please '''
        logging.info('Init CorsikaConfig')

        self._label = label
        self._filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)
        self._site = names.validateName(site, names.allSiteNames)
        self._arrayName = names.validateName(arrayName, names.allArrayNames)
        self._array = getArray(self._arrayName, databaseLocation)

        self._loadArguments(**kwargs)
        self._loadSeeds(randomSeeds)
        print('Parameters')
        print(self._parameters)
        print('Seeds')
        print(self._seeds)
        print('Array')
        print(self._array)

    def _loadArguments(self, **kwargs):
        self._parameters = dict()

        def validateAndFixArgs(parName, valueArgs):
            valueArgs = valueArgs if isinstance(valueArgs, list) else [valueArgs]
            if len(valueArgs) == 1 and parName == 'THETAP':  # fixing single value zenith angle
                valueArgs = valueArgs * 2
            if len(valueArgs) == 1 and parName == 'VIEWCONE':  # fixing single value viewcone
                valueArgs = [0, valueArgs[0]]
            if parName == 'PRMPAR':
                valueArgs = _convertPrimaryInput(valueArgs)

            if len(valueArgs) != parInfo['len']:
                logging.warning('Argument {} has wrong len'.format(keyArgs.upper()))

            return valueArgs

        # Collecting all parameters given as arguments
        indentifiedArgs = list()
        for keyArgs, valueArgs in kwargs.items():

            for parName, parInfo in cors_pars.USER_PARAMETERS.items():

                if keyArgs.upper() == parName or keyArgs.upper() in parInfo['names']:
                    indentifiedArgs.append(keyArgs)
                    valueArgs = validateAndFixArgs(parName, valueArgs)
                    self._parameters[parName] = valueArgs

        # Checking for unindetified parameters
        unindentifiedArgs = [p for p in kwargs.keys() if p not in indentifiedArgs]
        if len(unindentifiedArgs) > 0:
            logging.warning(
                '{} argument were not properly identified: {} ...'.format(
                    len(unindentifiedArgs),
                    unindentifiedArgs[0]
                )
            )

        # Checking for parameters with default option
        # If it is not given. filling it with the default value
        requiredButNotGiven = list()
        for parName, parInfo in cors_pars.USER_PARAMETERS.items():
            if parName in self._parameters.keys():
                continue
            if 'default' in parInfo.keys():
                parValue = validateAndFixArgs(parName, parInfo['default'])
                self._parameters[parName] = parValue
            else:
                requiredButNotGiven.append(parName)
        if len(requiredButNotGiven) > 0:
            logging.error(
                'Required parameters not given ({} parameters: {} ...)'.format(
                    len(requiredButNotGiven),
                    requiredButNotGiven[0]
                )
            )
            raise RequiredInputNotGiven()

    def _loadSeeds(self, randomSeeds):
        if '_parameters' not in self.__dict__.keys():
            logging.error('_loadSeeds has be called after _loadArguments')
            raise ArgumentsNotLoaded()
        if randomSeeds:
            s = random.uniform(0, 1000)
        else:
            s = self._parameters['PRMPAR'][0] + self._parameters['RUNNR'][0]
        random.seed(s)
        self._seeds = [int(random.uniform(0, 1e7)) for i in range(4)]

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

        def _writeParametersSingleLine(file, pars):
            for par, values in pars.items():
                line = par + ' '
                for v in values:
                    line += str(v) + ' '
                line += '\n'
                file.write(line)

        def _writeParametersMultipleLines(file, pars):
            for par, valueList in pars.items():
                for value in valueList:
                    newPars = {par: value}
                    _writeParametersSingleLine(file, newPars)

        with open(self._filePath, 'w') as file:
            _writeParametersSingleLine(file, self._parameters)
            file.write('\n# SITE PARAMETERS\n')
            _writeParametersSingleLine(file, cors_pars.SITE_PARAMETERS[self._site])
            file.write('\n# SEEDS\n')
            _writeSeeds(file, self._seeds)
            file.write('\n# TELESCOPES\n')
            _writeTelescopes(file, self._array)
            file.write('\n# INTERACTION FLAGS\n')
            _writeParametersSingleLine(file, cors_pars.INTERACTION_FLAGS)
            file.write('\n# CHERENKOV EMISSION PARAMETERS\n')
            _writeParametersSingleLine(file, cors_pars.CHERENKOV_EMISSION_PARAMETERS)
            file.write('\n# DEBUGGING OUTPUT PARAMETERS\n')
            _writeParametersSingleLine(file, cors_pars.DEBUGGING_OUTPUT_PARAMETERS)
            file.write('\n# IACT TUNING PARAMETERS\n')
            _writeParametersMultipleLines(file, cors_pars.IACT_TUNING_PARAMETERS)
            file.write('\nEXIT')

    def addLine(self):
        pass
