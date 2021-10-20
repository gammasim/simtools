
import logging
import os
from copy import copy
from pathlib import Path

import astropy.units as u

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen
from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_histograms import SimtelHistograms
from simtools.simtel.simtel_runner_array import SimtelRunnerArray


__all__ = ['ArraySimulator']


class MissingRequiredEntryInArrayConfig(Exception):
    pass


class ArraySimulator:
    '''
    ArraySimulator is responsible for managing simulation of array of telescopes. \
    It interfaces with simulation software-specific packages, like sim_telarray.

    The configuration is set as a dict configData or a yaml \
    file configFile. An example of configData can be found \
    below.

    .. code-block:: python

    configData = {
        'dataDirectory': '(..)/data',
        'primary': 'gamma',
        'zenith': 20 * u.deg,
        'azimuth': 0 * u.deg,
        'viewcone': 0 * u.deg,
        # ArrayModel
        'site': 'North',
        'layoutName': '1LST',
        'modelVersion': 'Prod5',
        'default': {
            'LST': '1'
        },
        'M-01': 'FlashCam-D'
    }


    Attributes
    ----------
    label: str
        Instance label.
    config: NamedTuple
        Configurable parameters.
    arrayModel: ArrayModel
        Instance of ArrayModel.

    Methods
    -------
    run(inputFileList):
        Run simulation.
    submit(inputFileList, submitCommand=None, extraCommands=None, test=False):
        Submit a run script as a job. The submit command can be given by submitCommand \
        or it will be taken from the config.yml file.
    printHistograms():
        Print histograms and save a pdf file.
    getListOfOutputFiles():
        Get list of output files.
    getListOfInputFiles():
        Get list of input files.
    getListOfLogFiles():
        Get list of log files.
    printListOfOutputFiles():
        Print list of output files.
    printListOfInputFiles():
        Print list of output files.
    printListOfLogFiles():
        Print list of log files.
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
        ArraySimulator init.

        Parameters
        ----------
        label: str
            Instance label.
        filesLocation: str or Path.
            Location of the output files. If not given, it will be set from \
            the config.yml file.
        simtelSourcePath: str or Path
            Location of source of the sim_telarray/CORSIKA package.
        configData: dict
            Dict with configurable data.
        configFile: str or Path
            Path to yaml file containing configurable data.
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init ArraySimulator')

        self.label = label

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        # File location
        self._baseDirectory = io.getOutputDirectory(
            self._filesLocation,
            self.label
        )

        configData = gen.collectDataFromYamlOrDict(configFile, configData)
        self._loadArrayConfigData(configData)
        self._setSimtelRunner()

        # Storing list of files
        self._results = dict()
        self._results['output'] = list()
        self._results['hist'] = list()
        self._results['input'] = list()
        self._results['log'] = list()
    # End of init

    def _loadArrayConfigData(self, configData):
        ''' Load configData, create arrayModel and store reamnining parameters in config. '''
        _arrayModelConfig, _restConfig = self._collectArrayModelParameters(configData)

        _parameterFile = io.getDataFile('parameters', 'array-simulator_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_restConfig, _parameters)

        self.arrayModel = ArrayModel(label=self.label, arrayConfigData=_arrayModelConfig)

    def _collectArrayModelParameters(self, configData):
        '''
        Separate parameters from configData into parameters to create the arrayModel
        and reamnining parameters to be stored in config.
        '''
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
        ''' Creating a SimtelRunnerArray. '''
        self._simtelRunner = SimtelRunnerArray(
            label=self.label,
            arrayModel=self.arrayModel,
            simtelSourcePath=self._simtelSourcePath,
            filesLocation=self._filesLocation,
            configData={
                'simtelDataDirectory': self.config.dataDirectory,
                'primary': self.config.primary,
                'zenithAngle': self.config.zenithAngle * u.deg,
                'azimuthAngle': self.config.azimuthAngle * u.deg
            }
        )

    def _fillResultsWithoutRun(self, inputFileList):
        ''' Fill in the results dict wihtout calling run or submit. '''
        inputFileList = self._makeInputList(inputFileList)

        for file in inputFileList:
            run = self._guessRunFromFile(file)
            self._fillResults(file, run)

    def run(self, inputFileList):
        '''
        Run simulation.

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        '''
        inputFileList = self._makeInputList(inputFileList)

        for file in inputFileList:
            run = self._guessRunFromFile(file)

            self._logger.info('Running scripts for run {}'.format(run))

            runScript = self._simtelRunner.getRunScript(run=run)
            self._logger.info('Run {} - Running script {}'.format(run, runScript))
            os.system(runScript)

            self._fillResults(file, run)

    def submit(self, inputFileList, submitCommand=None, extraCommands=None, test=False):
        '''
        Submit a run script as a job. The submit command can be given by \
        submitCommand or it will be taken from the config.yml file.

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.
        submitCommand: str
            Command to be used before the script name.
        extraCommands: str or list of str
            Extra commands to be added to the run script before the run command,
        test: bool
            If True, job is not submitted.
        '''

        subCmd = submitCommand if submitCommand is not None else cfg.get('submissionCommand')
        self._logger.info('Submission command: {}'.format(subCmd))

        inputFileList = self._makeInputList(inputFileList)

        self._logger.info('Starting submission')
        for file in inputFileList:
            run = self._guessRunFromFile(file)

            runScript = self._simtelRunner.getRunScript(
                run=run,
                inputFile=file,
                extraCommands=extraCommands
            )
            self._logger.info('Run {} - Submitting script {}'.format(run, runScript))

            shellCommand = subCmd + ' ' + str(runScript)
            self._logger.debug(shellCommand)
            if not test:
                os.system(shellCommand)

            self._fillResults(file, run)

    def _makeInputList(self, inputFileList):
        ''' Enforce the input list to be a list. '''
        if not isinstance(inputFileList, list):
            return [inputFileList]
        else:
            return inputFileList

    def _guessRunFromFile(self, file):
        '''
        Finds the run number for a given input file name.
        Input file names must follow 'run1234_*' pattern.
        If not found, returns 1.
        '''
        fileName = str(Path(file).name)
        runStr = fileName[3:fileName.find('_')]

        try:
            runNumber = int(runStr)
            return runNumber
        except ValueError:
            msg = 'Run number could not be guessed from the input file name - using run = 1'
            self._logger.warning(msg)
            return 1

    def _fillResults(self, file, run):
        ''' Fill the results dict with input, output and log files. '''
        self._results['input'].append(str(file))
        self._results['output'].append(str(self._simtelRunner.getOutputFile(run)))
        self._results['hist'].append(str(self._simtelRunner.getHistogramFile(run)))
        self._results['log'].append(str(self._simtelRunner.getLogFile(run)))

    def printHistograms(self, inputFileList=None):
        '''
        Print histograms and save a pdf file.

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        path
            Path of the pdf file.
        '''

        if len(self._results['hist']) == 0 and inputFileList is not None:
            self._fillResultsWithoutRun(inputFileList)

        figName = self._baseDirectory.joinpath('histograms.pdf')
        histFileList = self.getListOfHistogramFiles()
        simtelHistograms = SimtelHistograms(histFileList)
        simtelHistograms.plotAndSaveFigures(figName)

        return figName

    def getListOfOutputFiles(self):
        '''
        Get list of output files.

        Returns
        -------
        list
            List with the full path of all the output files.
        '''
        self._logger.info('Getting list of output files')
        return self._results['output']

    def getListOfHistogramFiles(self):
        '''
        Get list of histogram files.

        Returns
        -------
        list
            List with the full path of all the histogram files.
        '''
        self._logger.info('Getting list of histogram files')
        return self._results['hist']

    def getListOfInputFiles(self):
        '''
        Get list of input files.

        Returns
        -------
        list
            List with the full path of all the intput files.
        '''
        self._logger.info('Getting list of input files')
        return self._results['input']

    def getListOfLogFiles(self):
        '''
        Get list of log files.

        Returns
        -------
        list
            List with the full path of all the log files.
        '''
        self._logger.info('Getting list of log files')
        return self._results['log']

    def printListOfOutputFiles(self):
        ''' Print list of output files. '''
        self._logger.info('Printing list of output files')
        self._printListOfFiles(which='output')

    def printListOfHistogramFiles(self):
        ''' Print list of histogram files. '''
        self._logger.info('Printing list of histogram files')
        self._printListOfFiles(which='hist')

    def printListOfInputFiles(self):
        ''' Print list of output files. '''
        self._logger.info('Printing list of input files')
        self._printListOfFiles(which='input')

    def printListOfLogFiles(self):
        ''' Print list of log files. '''
        self._logger.info('Printing list of log files')
        self._printListOfFiles(which='log')

    def _printListOfFiles(self, which):
        for f in self._results[which]:
            print(f)

# End of ShowerSimulator
