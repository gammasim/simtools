
import logging
import os
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

        # Grabbing runs (and turn it into a list)
        self.runs = None if 'runs' not in showerConfigData.keys() else showerConfigData['runs']
        if not isinstance(self.runs, list):
            self._logger.debug('Turning runs into a list')
            self.runs = list(self.runs)

        # Validating runs - must be ints
        if not all(isinstance(r, int) for r in self.runs):
            msg = 'runs given in showerConfig must be all integers.'
            self._logger.error(msg)
            raise InvalidEntryInShowerConfig(msg)

        # Removing runs key from corsikaConfigData
        if 'runs' in self._corsikaConfigData.keys():
            self._corsikaConfigData.pop('runs')

        # Searching for corsikaParametersFile in showerConfig
        if 'corsikaParametersFile' in showerConfigData.keys():
            self._corsikaParametersFile = showerConfigData['corsikaParametersFile']
            self._corsikaConfigData.pop('corsikaParametersFile')
        else:
            # corsikaParametersFile not given - CorsikaConfig will use the default one
            self._corsikaParametersFile = None

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

    def run(self, runs=None):

        runsToSimulate = self._validateRunsToSimulate(runs)

        self._logger.info('Start running scripts')
        for run in runsToSimulate:
            runScript = self._corsikaRunner.getRunScriptFile(runNumber=run)

            self._logger.info('Run {} - Running script {}'.format(run, runScript))
            os.system(runScript)

    def submit(self, runs=None, submitCommand=None):

        subCmd = submitCommand if submitCommand is not None else cfg.get('submissionCommand')
        self._logger.info('Submission command: {}'.format(subCmd))

        runsToSimulate = self._validateRunsToSimulate(runs)

        for run in runsToSimulate:
            runScript = self._corsikaRunner.getRunScriptFile(runNumber=run)
            self._logger.info('Run {} - Submitting script {}'.format(run, runScript))

            shellCommand = subCmd + ' ' + str(runScript)
            self._logger.debug(shellCommand)
            os.system(shellCommand)

    def _validateRunsToSimulate(self, runs):

        if runs is not None:
            runsToSimulate = list(runs)
            if not all(isinstance(r, int) for r in runsToSimulate):
                msg = 'runs to simulate must be all integers - aborting simulation.'
                self._logger.error(msg)
                raise InvalidRunsToSimulate(msg)
        elif self.runs is not None:
            runsToSimulate = self.runs
        else:
            msg = (
                'runs to simulate were not given as arguments neither in showerConfigData'
                + ' - aborting simulation'
            )
            self._logger.error(msg)
            raise InvalidRunsToSimulate(msg)
        return runsToSimulate

    def getListOfOutputFiles(self, runs=None):
        self._logger.info('Getting list of output files')
        return self._getListOfFiles(runs=runs, which='output')

    def printListOfOutputFiles(self, runs=None):
        self._logger.info('Printing list of output files')
        self._printListOfFiles(runs=runs, which='output')

    def getListOfLogFiles(self, runs=None):
        self._logger.info('Getting list of log files')
        return self._getListOfFiles(runs=runs, which='log')

    def printListOfLogFiles(self, runs=None):
        self._logger.info('Printing list of log files')
        self._printListOfFiles(runs=runs, which='log')

    def _getListOfFiles(self, which, runs=None):
        runsToList = self._validateRunsToSimulate(runs)

        outputFiles = list()
        for run in runsToList:
            if which == 'output':
                file = self._corsikaRunner.getCorsikaOutputFile(runNumber=run)
            elif which == 'log':
                file = self._corsikaRunner.getCorsikaLogFile(runNumber=run)
            else:
                self._logger.error('Invalid type of files - log or output')
                return None
            outputFiles.append(file)

        return outputFiles

    def _printListOfFiles(self, which, runs=None):
        files = self._getListOfFiles(runs=runs, which=which)
        for f in files:
            print(f)

# End of ShowerSimulator
