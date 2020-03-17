#!/usr/bin/python3

import logging
import yaml
from pathlib import Path

from simtools.util import names
from simtools import io_handler as io
from simtools.array_model import getArray
from simtools import corsika_parameters as cors_pars


__all__ = ['CorsikaConfig']


def writeTelescopes(file, array):
    mToCm = 1e2
    for n, tel in array.items():
        file.write('\nTELESCOPE {} {} {} {}'.format(
            tel['xPos'] * mToCm,
            tel['yPos'] * mToCm,
            cors_pars.TELESCOPE_Z[tel['size']] * mToCm,
            cors_pars.TELESCOPE_SPHERE_RADIUS[tel['size']] * mToCm
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
        for parName, parInfo in cors_pars.USER_PARAMETERS.items():
            if 'default' not in parInfo.keys() or parName in self._parameters.keys():
                continue
            parValue = validateAndFixArgs(parName, parInfo['default'])
            self._parameters[parName] = parValue

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

        def writeParametersSingleLine(file, pars):
            for par, values in pars.items():
                line = par + ' '
                for v in values:
                    line += str(v) + ' '
                line += '\n'
                file.write(line)

        def writeParametersMultipleLines(file, pars):
            for par, valueList in pars.items():
                for value in valueList:
                    newPars = {par: value}
                    writeParametersSingleLine(file, newPars)

        with open(self._filePath, 'w') as file:
            writeParametersSingleLine(file, self._parameters)
            file.write('\n# SITE PARAMETERS\n')
            writeParametersSingleLine(file, cors_pars.SITE_PARAMETERS[self._site])
            file.write('\n# TELESCOPES\n')
            writeTelescopes(file, self._array)
            file.write('\n# INTERACTION FLAGS\n')
            writeParametersSingleLine(file, cors_pars.INTERACTION_FLAGS)
            file.write('\n# CHERENKOV EMISSION PARAMETERS\n')
            writeParametersSingleLine(file, cors_pars.CHERENKOV_EMISSION_PARAMETERS)
            file.write('\n# DEBUGGING OUTPUT PARAMETERS\n')
            writeParametersSingleLine(file, cors_pars.DEBUGGING_OUTPUT_PARAMETERS)
            file.write('\n# IACT TUNING PARAMETERS\n')
            writeParametersMultipleLines(file, cors_pars.IACT_TUNING_PARAMETERS)
            file.write('\nEXIT')

    def addLine(self):
        pass
