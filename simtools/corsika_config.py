#!/usr/bin/python3

import logging
import yaml
import random
from pathlib import Path

import astropy.units as u

import simtools.config as cfg
import simtools.io_handler as io
import simtools.corsika_parameters as cors_pars
from simtools.util import names
from simtools.model.array_model import getArray

__all__ = ['CorsikaConfig']


class RequiredInputNotGiven(Exception):
    pass


class ArgumentsNotLoaded(Exception):
    pass


class ArgumentWithWrongUnit(Exception):
    pass


class InvalidPrimary(Exception):
    pass


class CorsikaConfig:
    '''
    CorsikaConfig class.

    Methods
    -------
    setParameters(**kwargs)
    exportFile()
    getFile()
    '''
    def __init__(
        self,
        site,
        arrayName,
        label=None,
        databaseLocation=None,
        filesLocation=None,
        randomSeeds=False,
        logger=__name__,
        **kwargs
    ):
        '''
        CorsikaConfig init.

        Parameters
        ----------
        site: str
            Paranal or LaPalma
        arrayName: str
            Name of the array type. Ex 4LST, baseline ...
        label: str
            Instance label.
        databaseLocation: str
            Location of the db files.
        filesLocation: str or Path.
            Main location of the output file.
        randomSeeds: bool
            If True, seeds will be set randomly. If False, seeds will be defined based on the run
            number.
        logger: str
            Logger name to use in this instance
        **kwargs
            Set of parameters for the corsika config.
        '''

        self._logger = logging.getLogger(logger)
        self._logger.debug('Init CorsikaConfig')

        self._label = label
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)
        self._databaseLocation = cfg.getConfigArg('databaseLocation', databaseLocation)
        self._site = names.validateSiteName(site)
        self._arrayName = names.validateArrayName(arrayName)
        self._array = getArray(self._arrayName, self._databaseLocation)

        self.setParameters(**kwargs)
        self._loadSeeds(randomSeeds)
        self._isFileUpdated = False

    def setParameters(self, **kwargs):
        '''
        Set parameters for the corsika config.

        Parameters
        ----------
        **kwargs
        '''
        self._parameters = dict()

        def _validateAndFixArgs(parName, parInfo, valueArgs):
            valueArgs = valueArgs if isinstance(valueArgs, list) else [valueArgs]
            if len(valueArgs) == 1 and parName == 'THETAP':  # fixing single value zenith angle
                valueArgs = valueArgs * 2
            if len(valueArgs) == 1 and parName == 'VIEWCONE':  # fixing single value viewcone
                valueArgs = [0 * parInfo['unit'][0], valueArgs[0]]
            if parName == 'PRMPAR':
                valueArgs = self._convertPrimaryInput(valueArgs)

            if len(valueArgs) != parInfo['len']:
                self._logger.warning('Argument {} has wrong len'.format(keyArgs.upper()))

            if 'unit' in parInfo.keys():
                parUnit = (
                    [parInfo['unit']] if not isinstance(parInfo['unit'], list) else parInfo['unit']
                )

                newValueArgs = list()
                for (v, u) in zip(valueArgs, parUnit):
                    if u is None:
                        newValueArgs.append(v)
                        continue

                    try:
                        newValueArgs.append(v.to(u).value)
                    except u.core.UnitConversionError:
                        self._logger.error('Argument given with wrong unit: {}'.format(parName))
                        raise ArgumentWithWrongUnit()
                valueArgs = newValueArgs

            return valueArgs

        # Collecting all parameters given as arguments
        indentifiedArgs = list()
        for keyArgs, valueArgs in kwargs.items():
            for parName, parInfo in cors_pars.USER_PARAMETERS.items():
                if keyArgs.upper() == parName or keyArgs.upper() in parInfo['names']:
                    indentifiedArgs.append(keyArgs)
                    valueArgs = _validateAndFixArgs(parName, parInfo, valueArgs)
                    self._parameters[parName] = valueArgs

        # Checking for unindetified parameters
        unindentifiedArgs = [p for p in kwargs.keys() if p not in indentifiedArgs]
        if len(unindentifiedArgs) > 0:
            self._logger.warning(
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
                parValue = _validateAndFixArgs(parName, parInfo, parInfo['default'])
                self._parameters[parName] = parValue
            else:
                requiredButNotGiven.append(parName)
        if len(requiredButNotGiven) > 0:
            self._logger.error(
                'Required parameters not given ({} parameters: {} ...)'.format(
                    len(requiredButNotGiven),
                    requiredButNotGiven[0]
                )
            )
            raise RequiredInputNotGiven()

    def _writeTelescopes(self, file, array):
        '''
        Write telescope positions in the corsika input file.

        Parameters
        ----------
        file: file
            File where the telescope positions will be written.
        array: str
            Array type.
        '''
        self._logger.warning('Function is changing the inputs - Make it return the string instead')
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


    def _writeSeeds(self, file, seeds):
        '''
        Write seeds in the corsika input file.

        Parameters
        ----------
        file: file
            File where the telescope positions will be written.
        seeds: list of int
            List of seeds to be written.
        '''
        for s in seeds:
            file.write('SEED {} 0 0\n'.format(s))


    def _convertPrimaryInput(self, value):
        '''
        Convert a primary name into the right number.

        Parameters
        ----------
        value: str
            Input primary name.

        Raises
        ------
        InvalidPrimary
            If the input name is not found.
        '''
        for primName, primInfo in cors_pars.PRIMARIES.items():
            if value[0].upper() == primName or value[0].upper() in primInfo['names']:
                return [primInfo['number']]
        self._logger.error('Primary not valid')
        raise InvalidPrimary('Primary not valid')

    def _loadSeeds(self, randomSeeds):
        ''' Load seeds and store it in _seeds. '''
        if '_parameters' not in self.__dict__.keys():
            self._logger.error('_loadSeeds has be called after _loadArguments')
            raise ArgumentsNotLoaded()
        if randomSeeds:
            s = random.uniform(0, 1000)
        else:
            s = self._parameters['PRMPAR'][0] + self._parameters['RUNNR'][0]
        random.seed(s)
        self._seeds = [int(random.uniform(0, 1e7)) for i in range(4)]

    def exportFile(self):
        ''' Create and export corsika input file. '''
        configFileName = names.corsikaConfigFileName(
            arrayName=self._arrayName,
            site=self._site,
            zenith=self._parameters['THETAP'],
            viewCone=self._parameters['VIEWCONE'],
            label=self._label
        )
        outputFileName = names.corsikaOutputFileName(
            arrayName=self._arrayName,
            site=self._site,
            zenith=self._parameters['THETAP'],
            viewCone=self._parameters['VIEWCONE'],
            run=self._parameters['RUNNR'][0],
            label=self._label
        )
        fileDirectory = io.getCorsikaOutputDirectory(self._filesLocation, self._label)

        if not fileDirectory.exists():
            fileDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.info('Creating directory {}'.format(fileDirectory))
        self._configFilePath = fileDirectory.joinpath(configFileName)
        self._outputFilePath = fileDirectory.joinpath(outputFileName)

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

        with open(self._configFilePath, 'w') as file:
            _writeParametersSingleLine(file, self._parameters)
            file.write('\n# SITE PARAMETERS\n')
            _writeParametersSingleLine(file, cors_pars.SITE_PARAMETERS[self._site])
            file.write('\n# SEEDS\n')
            self._writeSeeds(file, self._seeds)
            file.write('\n# TELESCOPES\n')
            self._writeTelescopes(file, self._array)
            file.write('\n# INTERACTION FLAGS\n')
            _writeParametersSingleLine(file, cors_pars.INTERACTION_FLAGS)
            file.write('\n# CHERENKOV EMISSION PARAMETERS\n')
            _writeParametersSingleLine(file, cors_pars.CHERENKOV_EMISSION_PARAMETERS)
            file.write('\n# DEBUGGING OUTPUT PARAMETERS\n')
            _writeParametersSingleLine(file, cors_pars.DEBUGGING_OUTPUT_PARAMETERS)
            file.write('\n# OUTUPUT FILE\n')
            file.write('TELFIL {}'.format(self._outputFilePath))
            file.write('\n# IACT TUNING PARAMETERS\n')
            _writeParametersMultipleLines(file, cors_pars.IACT_TUNING_PARAMETERS)
            file.write('\nEXIT')

        self._isFileUpdated = True

    def getFile(self):
        '''
        Get the full path of the corsika input file.

        Returns
        -------
        Path of the input file.
        '''
        if not self._isFileUpdated:
            self.exportFile()
        return self._configFilePath

    def addLine(self):
        pass
