import logging
import os
from pathlib import Path

import simtools.io_handler as io
import simtools.util.general as gen
from simtools.util import names
from simtools.simtel.simtel_runner import SimtelRunner, InvalidOutputFile

__all__ = ['SimtelRunnerArray']


class SimtelRunnerArray(SimtelRunner):
    '''
    SimtelRunnerArray is the interface with sim_telarray to perform array simulations.

    Configurable parameters:
        simtelDataDirectory:
            len: 1
            default: null
            unit: null
        primary:
            len: 1
            unit: null
        zenithAngle:
            len: 1
            unit: deg
            default: 20 deg
        azimuthAngle:
            len: 1
            unit: deg
            default: 0 deg

    Attributes
    ----------
    label: str, optional
        Instance label.
    arrayModel: ArrayModel
        Instance of the ArrayModel class.
    config: namedtuple
        Contains the configurable parameters (zenithAngle).

    Methods
    -------
    getRunScript(self, test=False, inputFile=None, run=None)
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.
    run(test=False, force=False, inputFile=None, run=None)
        Run sim_telarray. test=True will make it faster and force=True will remove existing files
        and run again.
    '''

    def __init__(
        self,
        arrayModel,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
        configData=None,
        configFile=None
    ):
        '''
        SimtelRunnerArray.

        Parameters
        ----------
        arrayModel: str
            Instance of TelescopeModel class.
        label: str, optional
            Instance label. Important for output file naming.
        simtelSourcePath: str (or Path), optional
            Location of sim_telarray installation. If not given, it will be taken from the
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        configData: dict.
            Dict containing the configurable parameters.
        configFile: str or Path
            Path of the yaml file containing the configurable parameters.
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init SimtelRunnerArray')

        super().__init__(
            label=label,
            simtelSourcePath=simtelSourcePath,
            filesLocation=filesLocation
        )

        self.arrayModel = self._validateArrayModel(arrayModel)
        self.label = label if label is not None else self.arrayModel.label

        # File location
        self._baseDirectory = io.getOutputDirectory(
            self._filesLocation,
            self.label,
            'array'
        )

        # Loading configData
        _configDataIn = gen.collectDataFromYamlOrDict(configFile, configData)
        _parameterFile = io.getDataFile('parameters', 'simtel-runner-array_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_configDataIn, _parameters)

        self._loadSimtelDataDirectories()

    def _loadSimtelDataDirectories(self):
        '''
        Create sim_telarray output directories for data, log and input.

        If simtelDataDirectory is not given as a configurable parameter,
        the standard directory of simtools output (simtools-output) will
        be used. A sub directory simtel-data will be created and subdirectories for
        log and data will be created inside it.
        '''

        if self.config.simtelDataDirectory is None:
            # Default config value
            simtelBaseDir = self._baseDirectory
        else:
            simtelBaseDir = Path(self.config.simtelDataDirectory)

        simtelBaseDir = simtelBaseDir.joinpath('simtel-data')
        simtelBaseDir = simtelBaseDir.joinpath(self.arrayModel.site)
        simtelBaseDir = simtelBaseDir.joinpath(self.config.primary)
        simtelBaseDir = simtelBaseDir.absolute()

        self._simtelDataDir = simtelBaseDir.joinpath('data')
        self._simtelDataDir.mkdir(parents=True, exist_ok=True)
        self._simtelLogDir = simtelBaseDir.joinpath('log')
        self._simtelLogDir.mkdir(parents=True, exist_ok=True)

    def getLogFile(self, run):
        ''' Get full path of the simtel log file for a given run. '''
        fileName = names.simtelLogFileName(
            run=run,
            primary=self.config.primary,
            arrayName=self.arrayModel.layoutName,
            site=self.arrayModel.site,
            zenith=self.config.zenithAngle,
            azimuth=self.config.azimuthAngle,
            label=self.label
        )
        return self._simtelLogDir.joinpath(fileName)

    def getHistogramFile(self, run):
        ''' Get full path of the simtel histogram file for a given run. '''
        fileName = names.simtelHistogramFileName(
            run=run,
            primary=self.config.primary,
            arrayName=self.arrayModel.layoutName,
            site=self.arrayModel.site,
            zenith=self.config.zenithAngle,
            azimuth=self.config.azimuthAngle,
            label=self.label
        )
        return self._simtelDataDir.joinpath(fileName)

    def getOutputFile(self, run):
        ''' Get full path of the simtel output file for a given run. '''
        fileName = names.simtelOutputFileName(
            run=run,
            primary=self.config.primary,
            arrayName=self.arrayModel.layoutName,
            site=self.arrayModel.site,
            zenith=self.config.zenithAngle,
            azimuth=self.config.azimuthAngle,
            label=self.label
        )
        return self._simtelDataDir.joinpath(fileName)

    def _shallRun(self, run=None):
        ''' Tells if simulations should be run again based on the existence of output files. '''
        return not self.getOutputFile(run).exists()

    def _makeRunCommand(self, inputFile, run=1):
        ''' Builds and returns the command to run simtel_array. '''

        self._logFile = self.getLogFile(run)
        histogramFile = self.getHistogramFile(run)
        outputFile = self.getOutputFile(run)

        # Array
        command = str(self._simtelSourcePath.joinpath('sim_telarray/bin/sim_telarray'))
        command += ' -c {}'.format(self.arrayModel.getConfigFile())
        command += ' -I{}'.format(self.arrayModel.getConfigDirectory())
        command += super()._configOption('telescope_theta', self.config.zenithAngle)
        command += super()._configOption('telescope_phi', self.config.azimuthAngle)
        command += super()._configOption('power_law', '2.5')
        command += super()._configOption('histogram_file', histogramFile)
        command += super()._configOption('output_file', outputFile)
        command += super()._configOption('random_state', 'auto')
        command += super()._configOption('show', 'all')
        command += ' ' + str(inputFile)
        command += ' > ' + str(self._logFile) + ' 2>&1'

        return command
    # END of makeRunCommand

    def _checkRunResult(self, run):
        # Checking run
        if not self.getOutputFile(run).exists():
            msg = 'sim_telarray output file does not exist.'
            self._logger.error(msg)
            raise InvalidOutputFile(msg)
        else:
            self._logger.debug('Everything looks fine with the sim_telarray output file.')
