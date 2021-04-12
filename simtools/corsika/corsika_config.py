#!/usr/bin/python3

import logging
import random
from copy import copy

from astropy.io.misc import yaml

import simtools.config as cfg
import simtools.io_handler as io
from simtools.util import names
from simtools.layout.layout_array import LayoutArray
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ['CorsikaConfig']


class MissingRequiredInputInCorsikaConfigData(Exception):
    pass


class ArgumentsNotLoaded(Exception):
    pass


class InputWithWrongUnit(Exception):
    pass


class InvalidCorsikaParameterInfo(Exception):
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
        layoutName,
        label=None,
        filesLocation=None,
        randomSeeds=False,
        corsikaConfigData=None,
        corsikaConfigFile=None,
        corsikaParametersFile=None,
        **kwargs
    ):
        '''
        CorsikaConfig init.

        Parameters
        ----------
        site: str
            Paranal or LaPalma
        layoutName: str
            Name of the layout.
        layout: LayoutArray
            Instance of LayoutArray.
        label: str
            Instance label.
        filesLocation: str or Path.
            Main location of the output file.
        randomSeeds: bool
            If True, seeds will be set randomly. If False, seeds will be defined based on the run
            number.
        **kwargs
            Set of parameters for the corsika config.
        '''

        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init CorsikaConfig')

        self.label = label
        self.site = names.validateSiteName(site)

        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        # Grabbing layout name and building LayoutArray
        self.layoutName = names.validateLayoutArrayName(layoutName)
        self.layout = LayoutArray.fromLayoutArrayName(
            self.site + '-' + self.layoutName,
            label=self.label
        )

        # Load parameters
        self._loadCorsikaParametersFile(corsikaParametersFile)

        corsikaConfigData = collectDataFromYamlOrDict(corsikaConfigFile, corsikaConfigData)
        self.setUserParameters(corsikaConfigData)
        self._loadSeeds(randomSeeds)
        self._isFileUpdated = False

    def _loadCorsikaParametersFile(self, filename):
        if filename is not None:
            self._corsikaParametersFile = filename
        else:
            self._corsikaParametersFile = io.getDataFile('corsika', 'corsika_parameters.yml')
        self._logger.debug(
            'Loading CORSIKA parameters from file {}'.format(self._corsikaParametersFile)
        )
        with open(self._corsikaParametersFile, 'r') as f:
            self._corsikaParameters = yaml.load(f)

    def setUserParameters(self, corsikaConfigData):
        '''
        Set parameters for the corsika config.

        Parameters
        ----------
        **kwargs
        '''
        self._logger.debug('Setting user parameters from corsikaConfigData')
        self._userParameters = dict()

        userPars = self._corsikaParameters['USER_PARAMETERS']

        # Collecting all parameters given as arguments
        indentifiedArgs = list()
        for keyArgs, valueArgs in corsikaConfigData.items():
            # Looping over USER_PARAMETERS and searching for a match
            for parName, parInfo in userPars.items():
                if keyArgs.upper() != parName and keyArgs.upper() not in parInfo['names']:
                    continue
                # Matched parameter
                indentifiedArgs.append(keyArgs)
                validatedValueArgs = self._validateArgument(parName, parInfo, valueArgs)
                self._userParameters[parName] = validatedValueArgs

        # Checking for unindetified parameters
        unindentifiedArgs = [p for p in corsikaConfigData.keys() if p not in indentifiedArgs]
        if len(unindentifiedArgs) > 0:
            self._logger.warning(
                '{} arguments were not properly '.format(len(unindentifiedArgs))
                + 'identified: {} ...'.format(unindentifiedArgs[0])
            )

        # Checking for parameters with default option
        # If it is not given, filling it with the default value
        requiredButNotGiven = list()
        for parName, parInfo in userPars.items():
            if parName in self._userParameters.keys():
                continue
            elif 'default' in parInfo.keys():
                validatedValue = self._validateArgument(parName, parInfo, parInfo['default'])
                self._userParameters[parName] = validatedValue
            else:
                requiredButNotGiven.append(parName)

        if len(requiredButNotGiven) > 0:
            msg = (
                'Required parameters were not given ({} pars:'.format(len(requiredButNotGiven))
                + ' {} ...)'.format(requiredButNotGiven[0])
            )
            self._logger.error(msg)
            raise MissingRequiredInputInCorsikaConfigData(msg)
    # End of setParameters

    def _validateArgument(self, parName, parInfo, valueArgsIn):
        # Turning valueArgs into a list, if it is not.
        valueArgs = copy(valueArgsIn) if isinstance(valueArgsIn, list) else [valueArgsIn]

        if len(valueArgs) == 1 and parName in ['THETAP', 'PHIP']:
            # Fixing single value zenith or azimuth angle.
            # THETAP and PHIP should be written as a 2 values range in the CORSIKA input file
            valueArgs = valueArgs * 2
        elif len(valueArgs) == 1 and parName == 'VIEWCONE':
            # Fixing single value viewcone.
            # VIEWCONE should be written as a 2 values range in the CORSIKA input file
            valueArgs = [0 * parInfo['unit'][0], valueArgs[0]]
        elif parName == 'PRMPAR':
            valueArgs = self._convertPrimaryInput(valueArgs)

        if len(valueArgs) != parInfo['len']:
            self._logger.warning('Argument {} has wrong len'.format(parName))

        if 'unit' in parInfo.keys():
            # Turning parInfo['unit'] into a list, if it is not.
            parUnit = (
                copy(parInfo['unit']) if isinstance(parInfo['unit'], list) else [parInfo['unit']]
            )

            # Catching units with wrong len in the corsika_parameters file.
            if len(parUnit) != len(valueArgs):
                msg = (
                    'Parameter {} has units with wrong len: '.format(parName)
                    + '{} instead of {}'.format(len(parUnit), len(valueArgs))
                    + ' - please check your corsika_parameters file.'
                )
                self._logger.error(msg)
                raise InvalidCorsikaParameterInfo(msg)

            valueArgsWithUnits = list()
            for (v, u) in zip(valueArgs, parUnit):
                if u is None:
                    valueArgsWithUnits.append(v)
                    continue

                try:
                    valueArgsWithUnits.append(v.to(u).value)
                except u.core.UnitConversionError:
                    msg = 'Argument given with wrong unit: {}'.format(parName)
                    self._logger.error(msg)
                    raise InputWithWrongUnit(msg)
            valueArgs = valueArgsWithUnits

        return valueArgs
    # End of _validateArgument

    def printParameters(self):
        for par, value in self._userParameters.items():
            print('{} = {}'.format(par, value))

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
        for primName, primInfo in self._corsikaParameters['PRIMARIES'].items():
            if value[0].upper() == primName or value[0].upper() in primInfo['names']:
                return [primInfo['number']]
        self._logger.error('Primary not valid')
        raise InvalidPrimary('Primary not valid')

    def _loadSeeds(self, randomSeeds):
        ''' Load seeds and store it in _seeds. '''
        if '_userParameters' not in self.__dict__.keys():
            self._logger.error('_loadSeeds has be called after _loadArguments')
            raise ArgumentsNotLoaded()
        if randomSeeds:
            s = random.uniform(0, 1000)
        else:
            s = self._userParameters['PRMPAR'][0] + self._userParameters['RUNNR'][0]
        random.seed(s)
        self._seeds = [int(random.uniform(0, 1e7)) for i in range(4)]

    def exportInputFile(self):
        ''' Create and export corsika input file. '''
        self._setOutputFileAndDirectory()

        self._logger.debug('Exporting CORSIKA input file to {}'.format(self._configFilePath))

        def _getTextSingleLine(pars):
            text = ''
            for par, values in pars.items():
                line = par + ' '
                for v in values:
                    line += str(v) + ' '
                line += '\n'
                text += line
            return text

        def _getTextMultipleLines(pars):
            text = ''
            for par, valueList in pars.items():
                for value in valueList:
                    newPars = {par: value}
                    text += _getTextSingleLine(newPars)
            return text

        with open(self._configFilePath, 'w') as file:
            textParameters = _getTextSingleLine(self._userParameters)
            file.write(textParameters)

            file.write('\n* SITE PARAMETERS\n')
            textSiteParameters = _getTextSingleLine(
                self._corsikaParameters['SITE_PARAMETERS'][self.site]
            )
            file.write(textSiteParameters)

            file.write('\n* SEEDS\n')
            self._writeSeeds(file, self._seeds)

            file.write('\n* TELESCOPES\n')
            telescopeListText = self.layout.getCorsikaInputList()
            file.write(telescopeListText)

            file.write('\n* INTERACTION FLAGS\n')
            textInteractionFlags = _getTextSingleLine(self._corsikaParameters['INTERACTION_FLAGS'])
            file.write(textInteractionFlags)

            file.write('\n* CHERENKOV EMISSION PARAMETERS\n')
            textCherenkov = _getTextSingleLine(
                self._corsikaParameters['CHERENKOV_EMISSION_PARAMETERS']
            )
            file.write(textCherenkov)

            file.write('\n* DEBUGGING OUTPUT PARAMETERS\n')
            textDebugging = _getTextSingleLine(
                self._corsikaParameters['DEBUGGING_OUTPUT_PARAMETERS']
            )
            file.write(textDebugging)

            file.write('\n* OUTUPUT FILE\n')
            file.write('TELFIL {}\n'.format(self._outputFilePath.name))

            file.write('\n* IACT TUNING PARAMETERS\n')
            textIact = _getTextMultipleLines(self._corsikaParameters['IACT_TUNING_PARAMETERS'])
            file.write(textIact)

            file.write('\nEXIT')

        self._isFileUpdated = True
    # End of exportInputFile

    def _setOutputFileAndDirectory(self):
        configFileName = names.corsikaConfigFileName(
            arrayName=self.layoutName,
            site=self.site,
            zenith=self._userParameters['THETAP'],
            viewCone=self._userParameters['VIEWCONE'],
            label=self.label
        )
        outputFileName = names.corsikaOutputFileName(
            arrayName=self.layoutName,
            site=self.site,
            zenith=self._userParameters['THETAP'],
            viewCone=self._userParameters['VIEWCONE'],
            run=self._userParameters['RUNNR'][0],
            label=self.label
        )
        fileDirectory = io.getCorsikaOutputDirectory(self._filesLocation, self.label)

        if not fileDirectory.exists():
            fileDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.info('Creating directory {}'.format(fileDirectory))
        self._configFilePath = fileDirectory.joinpath(configFileName)
        self._outputFilePath = fileDirectory.joinpath(outputFileName)
    # End of setOutputFileAndDirectory

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
