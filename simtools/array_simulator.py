
import logging
import os
import numpy as np
from pathlib import Path
from copy import copy

import astropy.units as u

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen
from simtools.util import names
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.model.array_model import ArrayModel

__all__ = ['ArraySimulator']


class MissingRequiredEntryInArrayConfig(Exception):
    pass


class ArraySimulator:
    '''
    ShowerSimulator is responsible for managing simulation of showers. \
    It interfaces with simulation software-specific packages, like CORSIKA.

    The configuration is set as a dict showerConfigData or a yaml \
    file showerConfigFile. An example of showerConfigData can be found \
    below.

    .. code-block:: python

    self.showerConfigData = {
        'corsikaDataDirectory': './corsika-data',
        'site': 'South',
        'layoutName': 'Prod5',
        'runRange': [1, 100],
        'nshow': 10,
        'primary': 'gamma',
        'erange': [100 * u.GeV, 1 * u.TeV],
        'eslope': -2,
        'zenith': 20 * u.deg,
        'azimuth': 0 * u.deg,
        'viewcone': 0 * u.deg,
        'cscat': [10, 1500 * u.m, 0]
    }


    Attributes
    ----------
    site: str
        North or South.
    layoutName: str
        Name of the layout.
    label: str
        Instance label.

    Methods
    -------
    getRunScriptFile(runNumber)
        Get the full path of the run script file for a given run number.
    getRunLogFile(runNumber)
        Get the full path of the run log file.
    getCorsikaLogFile(runNumber)
        Get the full path of the CORSIKA log file.
    getCorsikaOutputFile(runNumber)
        Get the full path of the CORSIKA output file.
    '''

    def __init__(
        self,
        label=None,
        filesLocation=None,
        simtelSourcePath=None,
        configData=None,
        configFile=None
    ):
        '''
        ShowerSimulator init.

        Parameters
        ----------
        label: str
            Instance label.
        filesLocation: str or Path.
            Location of the output files. If not given, it will be set from \
            the config.yml file.
        simtelSourcePath: str or Path
            Location of source of the sim_telarray/CORSIKA package.
        showerConfigData: dict
            Dict with shower config data.
        showerConfigFile: str or Path
            Path to yaml file containing shower config data.
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init ArraySimulator')

        self.label = label

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        configData = gen.collectDataFromYamlOrDict(configFile, configData)
        self._loadArrayConfigData(configData)
        self._setSimtelRunner()

        self._results = dict()
        self._results['output'] = list()
        self._results['input'] = list()
        self._results['log'] = list()
    # End of init

    def _loadArrayConfigData(self, configData):
        ''' Validate showerConfigData and store the relevant data in variables.'''
        _arrayModelConfig, _restConfig = self._collectArrayModelParameters(configData)

        _parameterFile = io.getDataFile('parameters', 'array-simulator_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_restConfig, _parameters)

        self.arrayModel = ArrayModel(label=self.label, arrayConfigData=_arrayModelConfig)

    def _collectArrayModelParameters(self, configData):
        _arrayModelData = dict()
        _restData = copy(configData)

        try:
            _arrayModelData['site'] = _restData.pop('site')
            _arrayModelData['layoutName'] = _restData.pop('layoutName')
            _arrayModelData['modelVersion'] = _restData.pop('modelVersion')
            _arrayModelData['default'] = _restData.pop('default')
        except KeyError:
            msg = 'site, layoutName, modelVersion and/or default were not given in configData'
            self._logger.error(msg)
            raise MissingRequiredEntryInArrayConfig(msg)

        # Grabbing the telescope keys
        telKeys = [k for k in _restData.keys() if k[0:2] in ['L-', 'M-', 'S-']]
        for key in telKeys:
            _arrayModelData[key] = _restData.pop(key)

        return _arrayModelData, _restData

    def _setSimtelRunner(self):
        ''' Creating a CorsikaRunner and setting it to self._corsikaRunner. '''
        self._simtelRunner = SimtelRunnerArray(
            label=self.label,
            arrayModel=self.arrayModel,
            simtelSourcePath=self._simtelSourcePath,
            filesLocation=self._filesLocation,
            configData={
                'simtelDataDirectory': self.config.simtelDataDirectory,
                'primary': self.config.primary,
                'zenithAngle': self.config.zenithAngle * u.deg,
                'azimuthAngle': self.config.azimuthAngle * u.deg
            }
        )

    def run(self, inputFileList):
        '''
        Run simulation.

        Parameters
        ----------
        runList: list
            List of run numbers to be simulated.
        runRange: list
            List of len 2 with the limits ofthe range for runs to be simulated.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        '''

        # inputFile into list

        inputFileList = self._makeInputList(inputFileList)

        for file in inputFileList:
            run = self._guessRunFromFile(file)

            self._logger.info('Running scripts for run {}'.format(run))

            runScript = self._simtelRunner.getRunScript(run=run)
            self._logger.info('Run {} - Running script {}'.format(run, runScript))
            os.system(runScript)

            self._results['input'].append(file)
            self._results['output'].append(self._simtelRunner.getOutputFile(run))
            self._results['log'].append(self._simtelRunner.getLogFile(run))

    def _makeInputList(self, inputFileList):

        if not isinstance(inputFileList, list):
            return [inputFileList]
        else:
            return inputFileList

    def _guessRunFromFile(self, file):

        fileName = str(file)
        runStr = fileName[3:fileName.find('_')]

        try:
            runNumber = int(runStr)
            return runNumber
        except ValueError:
            msg = 'Run number could not be guessed from the input file name - using run = 1'
            self._logger.warning(msg)
            return 1

    def submit(self, inputFileList, submitCommand=None, extraCommands=None, test=False):
        '''
        Submit a run script as a job. The submit command can be given by \
        submitCommand or it will be taken from the config.yml file.

        Parameters
        ----------
        runList: list
            List of run numbers to be simulated.
        runRange: list
            List of len 2 with the limits ofthe range for runs to be simulated.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        '''

        subCmd = submitCommand if submitCommand is not None else cfg.get('submissionCommand')
        self._logger.info('Submission command: {}'.format(subCmd))

        inputFileList = self._makeInputList(inputFileList)

        self._logger.info('Starting submission')
        for file in inputFileList:
            run = self._guessRunFromFile(file)
            runScript = self._simtelRunner.getRunScript(
                run=run,
                extraCommands=extraCommands
            )
            self._logger.info('Run {} - Submitting script {}'.format(run, runScript))

            shellCommand = subCmd + ' ' + str(runScript)
            self._logger.debug(shellCommand)
            if not test:
                os.system(shellCommand)

            self._results['input'].append(str(file))
            self._results['output'].append(str(self._simtelRunner.getOutputFile(run)))
            self._results['log'].append(str(self._simtelRunner.getLogFile(run)))

    def getListOfOutputFiles(self, runList=None, runRange=None):
        '''
        Get list of output files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits ofthe range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.

        Returns
        -------
        list
            List with the full path of all the output files.
        '''
        self._logger.info('Getting list of output files')
        return self._results['output']

    def printListOfInputFiles(self):
        '''
        Get list of output files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        '''
        self._logger.info('Printing list of input files')
        self._printListOfFiles(which='input')

    def getListOfInputFiles(self):
        '''
        Get list of output files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits ofthe range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.

        Returns
        -------
        list
            List with the full path of all the output files.
        '''
        self._logger.info('Getting list of input files')
        return self._results['input']

    def printListOfOutputFiles(self):
        '''
        Get list of output files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        '''
        self._logger.info('Printing list of output files')
        self._printListOfFiles(which='output')

    def getListOfLogFiles(self):
        '''
        Get list of log files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.

        Returns
        -------
        list
            List with the full path of all the log files.
        '''
        self._logger.info('Getting list of log files')
        return self._results['log']

    def printListOfLogFiles(self):
        '''
        Print list of log files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        '''
        self._logger.info('Printing list of log files')
        self._printListOfFiles(which='log')

    def _printListOfFiles(self, which):
        for f in self._results[which]:
            print(f)

# End of ShowerSimulator
