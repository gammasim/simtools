import logging
from pathlib import Path

import astropy.units as u

import simtools.io_handler as io
import simtools.util.general as gen
from simtools.util import names
from simtools.simtel.simtel_runner import SimtelRunner

__all__ = ['SimtelRunnerArray']


class SimtelRunnerArray(SimtelRunner):
    '''
    SimtelRunnerRayTracing is the interface with sim_telarray to perform ray tracing simulations.

    Configurable parameters:
        simtelDataDirectory:
            len: 1
            default: '.'
        primary:
            len: 1
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
    run(test=False, force=False)
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
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        # Loading configData
        _configDataIn = gen.collectDataFromYamlOrDict(configFile, configData)
        _parameterFile = io.getDataFile('parameters', 'simtel-runner-array_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_configDataIn, _parameters)

        self._loadSimtelDataDirectories()

    def _loadSimtelDataDirectories(self):
        ''' Create CORSIKA directories for data, log and input. '''
        simtelBaseDir = Path(self.config.simtelDataDirectory).joinpath('simtel-data')
        simtelBaseDir = simtelBaseDir.joinpath(self.arrayModel.site)
        simtelBaseDir = simtelBaseDir.joinpath(self.config.primary)
        simtelBaseDir = simtelBaseDir.absolute()

        self._simtelDataDir = simtelBaseDir.joinpath('data')
        self._simtelDataDir.mkdir(parents=True, exist_ok=True)
        self._simtelLogDir = simtelBaseDir.joinpath('log')
        self._simtelLogDir.mkdir(parents=True, exist_ok=True)

    def _getLogFile(self, run):
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

    def _getHistogramFile(self, run):
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

    def _getOutputFile(self, run):
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

    # def _getRunScript(self, test=False):
    #     self._logger.debug('Creating run bash script')
    #     self._scriptFile = self._baseDirectory.joinpath('run_script')
    #     self._logger.debug('Run bash script - {}'.format(self._scriptFile))

    #     command = self._makeRunCommand()
    #     with self._scriptFile.open('w') as file:
    #         # TODO: header
    #         file.write('#/usr/bin/bash\n\n')
    #         N = 1 if test else self.RUNS_PER_SET
    #         for _ in range(N):
    #             file.write('{}\n\n'.format(command))

    #     return self._scriptFile

    def _shallRun(self):
        ''' Tells if simulations should be run again based on the existence of output files. '''
        return True

    def _makeRunCommand(self, inputFile, run=1):
        ''' Return the command to run simtel_array. '''

        logFile = self._getLogFile(run)
        histogramFile = self._getHistogramFile(run)
        outputFile = self._getOutputFile(run)

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
        command += ' 2>&1 > ' + str(logFile)

        return command
    # END of makeRunCommand

    def _checkRunResult(self):
        # # Checking run
        # if not self._isPhotonListFileOK():
        #     self._logger.error('Photon list is empty.')
        # else:
        #     self._logger.debug('Everything looks fine with output file.')
        pass

    def _isPhotonListFileOK(self):
        # ''' Check if the photon list is valid,'''
        # with open(self._photonsFile, 'r') as ff:
        #     nLines = len(ff.readlines())

        # return nLines > 100
        return True
