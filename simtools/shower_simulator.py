
import logging
import os
import numpy as np
from pathlib import Path
from copy import copy

import simtools.config as cfg
import simtools.io_handler as io
from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ['ShowerSimulator']


class MissingRequiredEntryInShowerConfig(Exception):
    pass


class InvalidEntryInShowerConfig(Exception):
    pass


class InvalidRunsToSimulate(Exception):
    pass


class ShowerSimulator:
    '''
    Class for handling ray tracing simulations and analysis.

    Attributes
    ----------
    label: str
        Instance label.

    Methods
    -------
    simulate(test=False, force=False)
        Simulate RayTracing using SimtelRunner.
    analyse(export=True, force=False, useRX=False, noTelTransmission=False)
        Analyze RayTracing, meaning read simtel files, compute psfs and eff areas and store the
        results in _results.
    exportResults()
        Export results to a csv file.
    plot(key, **kwargs)
        Plot key vs off-axis angle.
    plotHistogram(key, **kwargs)
        Plot histogram of key (d80_cm, d80_deg, eff_area, eff_flen).
    getMean(key)
        Get mean value of key(d80_cm, d80_deg, eff_area, eff_flen).
    getStdDev(key)
        Get std dev of key(d80_cm, d80_deg, eff_area, eff_flen).
    images()
        Get list of PSFImages.
    '''

    def __init__(
        self,
        label=None,
        filesLocation=None,
        simtelSourcePath=None,
        showerConfigData=None,
        showerConfigFile=None
    ):
        '''
        RayTracing init.

        Parameters
        ----------
        telescopeModel: TelescopeModel
            Instance of the TelescopeModel class.
        label: str
            Instance label.
        simtelSourcePath: str (or Path), optional
            Location of sim_telarray installation. If not given, it will be taken from the
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        singleMirrorMode: bool
        useRandomFocalLength: bool
        **kwargs:
            Physical parameters with units (if applicable). Options: zenithAngle, offAxisAngle,
            sourceDistance, mirrorNumbers
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init CorsikaRunner')

        self.label = label

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)
        self._outputDirectory = io.getCorsikaOutputDirectory(self._filesLocation, self.label)
        self._outputDirectory.mkdir(parents=True, exist_ok=True)
        self._logger.debug(
            'Output directory {} - creating it, if needed.'.format(self._outputDirectory)
        )

        showerConfigData = collectDataFromYamlOrDict(showerConfigFile, showerConfigData)
        self._loadShowerConfigData(showerConfigData)
        self._setCorsikaRunner()

    # End of init

    def _loadShowerConfigData(self, showerConfigData):

        # Validating existence of site and layoutName entries.
        for par in ['site', 'layoutName']:
            if par not in showerConfigData:
                msg = '{} was not given in showerConfig'.format(par)
                self._logger.error(msg)
                raise MissingRequiredEntryInShowerConfig(msg)

        # Copying showerConfigData to corsikaConfigData
        self._corsikaConfigData = copy(showerConfigData)

        self.site = names.validateSiteName(showerConfigData['site'])
        self.layoutName = names.validateLayoutArrayName(showerConfigData['layoutName'])

        self._corsikaConfigData.pop('site')
        self._corsikaConfigData.pop('layoutName')

        # Grabbing runList and runRange
        runList = showerConfigData.get('runList', None)
        runRange = showerConfigData.get('runRange', None)
        # Validating and merging runList and runRange, if needed.
        self.runs = self._validateRunListAndRange(runList, runRange)

        # Removing runs key from corsikaConfigData
        self._corsikaConfigData.pop('runList', None)
        self._corsikaConfigData.pop('runRange', None)

        # Searching for corsikaParametersFile in showerConfig
        self._corsikaParametersFile = showerConfigData.get('corsikaParametersFile', None)
        self._corsikaConfigData.pop('corsikaParametersFile', None)

    def _setCorsikaRunner(self):
        self._corsikaRunner = CorsikaRunner(
            site=self.site,
            layoutName=self.layoutName,
            label=self.label,
            filesLocation=self._filesLocation,
            simtelSourcePath=self._simtelSourcePath,
            corsikaParametersFile=self._corsikaParametersFile,
            corsikaConfigData=self._corsikaConfigData
        )

    def run(self, runList=None, runRange=None):

        runsToSimulate = self._getRunsToSimulate(runList, runRange)
        self._logger.info('Running scripts for {} runs'.format(len(runsToSimulate)))

        self._logger.info('Starting running scripts')
        for run in runsToSimulate:
            runScript = self._corsikaRunner.getRunScriptFile(runNumber=run)

            self._logger.info('Run {} - Running script {}'.format(run, runScript))
            os.system(runScript)

    def submit(self, runList=None, runRange=None, submitCommand=None):

        subCmd = submitCommand if submitCommand is not None else cfg.get('submissionCommand')
        self._logger.info('Submission command: {}'.format(subCmd))

        runsToSimulate = self._getRunsToSimulate(runList, runRange)
        self._logger.info('Submitting run scripts for {} runs'.format(len(runsToSimulate)))

        self._logger.info('Starting submission')
        for run in runsToSimulate:
            runScript = self._corsikaRunner.getRunScriptFile(runNumber=run)
            self._logger.info('Run {} - Submitting script {}'.format(run, runScript))

            shellCommand = subCmd + ' ' + str(runScript)
            self._logger.debug(shellCommand)
            os.system(shellCommand)

    def _getRunsToSimulate(self, runList, runRange):

        if runList is None and runRange is None:
            if self.runs is None:
                msg = (
                    'Runs to simulate were not given as arguments nor '
                    + 'in showerConfigData - aborting'
                )
                self._logger.error(msg)
                raise InvalidRunsToSimulate(msg)
            else:
                return self.runs
        else:
            return self._validateRunListAndRange(runList, runRange)

    def _validateRunListAndRange(self, runList, runRange):

        if runList is None and runRange is None:
            self._logger.debug('Nothing to validate - runList and runRange not given.')
            return None

        validatedRuns = list()
        if runList is not None:
            if not all(isinstance(r, int) for r in runList):
                msg = 'runList must contain only integers.'
                self._logger.error(msg)
                raise InvalidRunsToSimulate(msg)
            else:
                self._logger.debug('runList: {}'.format(runList))
                validatedRuns = list(runList)

        if runRange is not None:
            if not all(isinstance(r, int) for r in runRange) or len(runRange) != 2:
                msg = 'runRange must contain two integers only.'
                self._logger.error(msg)
                raise InvalidRunsToSimulate(msg)
            else:
                runRange = np.arange(runRange[0], runRange[1])
                self._logger.debug('runRange: {}'.format(runRange))
                validatedRuns.extend(list(runRange))

        validatedRunsUnique = set(validatedRuns)
        return list(validatedRunsUnique)

    def getListOfOutputFiles(self, runList=None, runRange=None):
        self._logger.info('Getting list of output files')
        return self._getListOfFiles(runList=runList, runRange=runRange, which='output')

    def printListOfOutputFiles(self, runList=None, runRange=None):
        self._logger.info('Printing list of output files')
        self._printListOfFiles(runList=runList, runRange=runRange, which='output')

    def getListOfLogFiles(self, runList=None, runRange=None):
        self._logger.info('Getting list of log files')
        return self._getListOfFiles(runList=runList, runRange=runRange, which='log')

    def printListOfLogFiles(self, runList=None, runRange=None):
        self._logger.info('Printing list of log files')
        self._printListOfFiles(runList=runList, runRange=runRange, which='log')

    def _getListOfFiles(self, which, runList, runRange):
        runsToList = self._getRunsToSimulate(runList=runList, runRange=runRange)

        outputFiles = list()
        for run in runsToList:
            if which == 'output':
                file = self._corsikaRunner.getCorsikaOutputFile(runNumber=run)
            elif which == 'log':
                file = self._corsikaRunner.getCorsikaLogFile(runNumber=run)
            else:
                self._logger.error('Invalid type of files - log or output')
                return None
            outputFiles.append(str(file))

        return outputFiles

    def _printListOfFiles(self, which, runList, runRange):
        files = self._getListOfFiles(runList=runList, runRange=runRange, which=which)
        for f in files:
            print(f)

# End of ShowerSimulator
