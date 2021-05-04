import logging
import os
from pathlib import Path

import astropy.units as u

import simtools.io_handler as io
import simtools.config as cfg
import simtools.util.general as gen
from simtools.util import names
from simtools.model.telescope_model import TelescopeModel

__all__ = ['SimtelRunner']


class SimtelExecutionError(Exception):
    pass


class SimtelRunner:
    '''
    SimtelRunner is the interface with sim_telarray.

    Configurable parameters:
        zenithAngle:
            len: 1
            unit: deg
            default: 20 deg
        offAxisAngle:
            len: 1
            unit: deg
            default: 0 deg
        sourceDistance:
            len: 1
            unit: km
            default: 10 km
        useRandomFocalLength:
            len: 1
            default: False
        mirrorNumbers:
            len: 1
            default: 1

    Attributes
    ----------
    mode: str
        RayTracing, Trigger, etc.
    telescopeModel: TelescopeModel
        Instance of the TelescopeModel class.
    label: str, optional
        Instance label.
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
        mode,
        telescopeModel=None,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
        configData=None,
        configFile=None
    ):
        '''
        SimtelRunner.

        Parameters
        ----------
        mode: str
            RayTracing, Trigger, ...
        telescopeModel: str
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
        self._logger.debug('Init SimtelRunner')

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self.mode = names.validateSimtelModeName(mode)
        self.telescopeModel = self._validateTelescopeModel(telescopeModel)
        self.label = label if label is not None else self.telescopeModel.label

        # File location
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)
        self._baseDirectory = io.getOutputDirectory(
            self._filesLocation,
            self.label,
            self._getModeDirectory()
        )
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        # RayTracing - default parameters
        self._repNumber = 0
        self.RUNS_PER_SET = 1 if self._isSingleMirrorMode() else 20  # const
        self.PHOTONS_PER_RUN = 10000  # const

        # Loading configData
        _configDataIn = gen.collectDataFromYamlOrDict(configFile, configData)
        _parameterFile = io.getDataFile('parameters', 'simtel-runner_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_configDataIn, _parameters)

    # END of _init_

    def __repr__(self):
        return 'SimtelRunner(mode={}, label={})\n'.format(self.mode, self.label)

    def _isRayTracingMode(self):
        return 'RayTracing' in self.mode

    def _isSingleMirrorMode(self):
        return 'SingleMirror' in self.mode

    def _getModeDirectory(self):
        ''' Get mode sub-directory for output files directory. '''
        if self._isRayTracingMode():
            return 'ray-tracing'
        else:
            return 'generic'

    def _validateTelescopeModel(self, tel):
        ''' Validate TelescopeModel '''
        if isinstance(tel, TelescopeModel):
            self._logger.debug('TelescopeModel OK')
            return tel
        else:
            msg = 'Invalid TelescopeModel'
            self._logger.error(msg)
            raise ValueError(msg)

    def run(self, test=False, force=False):
        '''
        Run sim_telarray.

        Parameters
        ----------
        test: bool
            If True, make simulations faster.
        force: bool
            If True, remove possible existing output files and run again.
        '''
        self._logger.debug('Running at mode {}'.format(self.mode))
        # write all the important parameters

        if not self._shallRun() and not force:
            self._logger.debug('Skipping because file exists and force = False')
            return

        self._loadRequiredFiles()
        command = self._makeRunCommand()

        if test:
            self._logger.info('Running (test) with command:{}'.format(command))
            sysOutput = os.system(command)
        else:
            self._logger.info('Running ({}x) with command:{}'.format(self.RUNS_PER_SET, command))
            sysOutput = os.system(command)
            for _ in range(self.RUNS_PER_SET - 1):
                os.system(command)

        # Checking run
        if self._isRayTracingMode:
            if self._isPhotonListFileOK():
                self._logger.debug('Everything looks fine with simtel run')
            else:
                self._raiseSimtelError()
        elif self._simtelFailed():
            self._raiseSimtelError()
        else:
            self._logger.debug('Everything looks fine with simtel run')

    def _simtelFailed(self, sysOutput):
        return sysOutput != '0'

    def _isPhotonListFileOK(self):
        nLines = sum(1 for ll in open(self._photonsFileName, 'r'))
        return nLines > 100

    def _raiseSimtelError(self):
        msg = gen.collectFinalLines(self._logFileName, 10)
        self._logger.error(
            'Simtel Error - See below the relevant part of the simtel log file.\n'
            + '===== from simtel log file ======\n' + msg
            + '================================='
        )
        raise SimtelExecutionError()

    def _getRunBashScript(self, test=False):
        self._logger.debug('Creating run bash script')
        self._scriptFileName = self._baseDirectory.joinpath('run_script')
        self._logger.debug('Run bash script - {}'.format(self._scriptFileName))

        self._loadRequiredFiles()
        command = self._makeRunCommand()
        with self._scriptFileName.open('w') as file:
            # TODO: header
            file.write('#/usr/bin/bash\n\n')
            N = 1 if test else self.RUNS_PER_SET
            for _ in range(N):
                file.write('{}\n\n'.format(command))

        return self._scriptFileName

    def _shallRun(self):
        ''' Tells if simulations should be run again based on the existence of output files. '''
        if self._isRayTracingMode():
            photonsFileName = names.rayTracingFileName(
                self.telescopeModel.site,
                self.telescopeModel.name,
                self.config.sourceDistance,
                self.config.zenithAngle,
                self.config.offAxisAngle,
                self.config.mirrorNumber if self._isSingleMirrorMode() else None,
                self.label,
                'photons'
            )
            file = self._baseDirectory.joinpath(photonsFileName)
            return not file.exists()
        else:
            return False

    def _loadRequiredFiles(self):
        '''
        Which file are required for running depends on the mode.
        Here we define and write some information into these files. Log files are always requires.
        '''

        # RayTracing
        if self._isRayTracingMode():
            self._corsikaFileName = self._simtelSourcePath.joinpath('run9991.corsika.gz')

            # Loop to define and remove existing files.
            # Files will be named _baseFileName = self.__dict__['_' + base + 'FileName']
            for baseName in ['stars', 'photons', 'log']:
                fileName = names.rayTracingFileName(
                    self.telescopeModel.site,
                    self.telescopeModel.name,
                    self.config.sourceDistance,
                    self.config.zenithAngle,
                    self.config.offAxisAngle,
                    self.config.mirrorNumber if self._isSingleMirrorMode() else None,
                    self.label,
                    baseName
                )
                file = self._baseDirectory.joinpath(fileName)
                if file.exists():
                    file.unlink()
                # Defining the file name variable as an class atribute.
                self.__dict__['_' + baseName + 'FileName'] = file

            # Adding header to photon list file.
            with self._photonsFileName.open('w') as file:
                file.write('#{}\n'.format(50 * '='))
                file.write('# List of photons for RayTracing simulations\n')
                file.write('#{}\n'.format(50 * '='))
                file.write('# configFile = {}\n'.format(self.telescopeModel.getConfigFile()))
                file.write('# zenithAngle [deg] = {}\n'.format(self.config.zenithAngle))
                file.write('# offAxisAngle [deg] = {}\n'.format(self.config.offAxisAngle))
                file.write('# sourceDistance [km] = {}\n'.format(self.config.sourceDistance))
                if self._isSingleMirrorMode():
                    file.write('# mirrorNumber = {}\n\n'.format(self.config.mirrorNumber))

            # Filling in star file with a single star.
            with self._starsFileName.open('w') as file:
                file.write('0. {} 1.0 {}'.format(
                    90. - self.config.zenithAngle,
                    self.config.sourceDistance)
                )

        # Trigger
        # elif self._isTriggerMode()
        #     pass
    # END of _loadRequiredFiles

    def _makeRunCommand(self):
        ''' Return the command to run simtel_array. '''

        def _configOption(par, value=None):
            c = ' -C {}'.format(par)
            c += '={}'.format(value) if value is not None else ''
            return c

        if self._isSingleMirrorMode():
            _mirrorFocalLength = float(self.telescopeModel.getParameterValue('mirror_focal_length'))

        # RayTracing
        command = str(self._simtelSourcePath.joinpath('sim_telarray/bin/sim_telarray'))
        command += ' -c {}'.format(self.telescopeModel.getConfigFile())
        command += ' -I../cfg/CTA'
        command += _configOption('IMAGING_LIST', str(self._photonsFileName))
        command += _configOption('stars', str(self._starsFileName))
        command += _configOption('altitude', self.telescopeModel.getParameterValue('altitude'))
        command += _configOption(
            'telescope_theta',
            self.config.zenithAngle + self.config.offAxisAngle
        )
        command += _configOption('star_photons', str(self.PHOTONS_PER_RUN))
        command += _configOption('telescope_phi', '0')
        command += _configOption('camera_transmission', '1.0')
        command += _configOption('nightsky_background', 'all:0.')
        command += _configOption('trigger_current_limit', '1e10')
        command += _configOption('telescope_random_angle', '0')
        command += _configOption('telescope_random_error', '0')
        command += _configOption('convergent_depth', '0')
        command += _configOption('maximum_telescopes', '1')
        command += _configOption('show', 'all')
        command += _configOption('camera_filter', 'none')
        if self._isSingleMirrorMode():
            command += _configOption('focus_offset', 'all:0.')
            command += _configOption('camera_config_file', 'single_pixel_camera.dat')
            command += _configOption('camera_pixels', '1')
            command += _configOption('trigger_pixels', '1')
            command += _configOption('camera_body_diameter', '0')
            command += _configOption(
                'mirror_list',
                self.telescopeModel.getSingleMirrorListFile(
                    self.config.mirrorNumber,
                    self.config.useRandomFocalLength
                )
            )
            command += _configOption('focal_length', self.config.sourceDistance * u.km.to(u.cm))
            command += _configOption('dish_shape_length', _mirrorFocalLength)
            command += _configOption('mirror_focal_length', _mirrorFocalLength)
            command += _configOption('parabolic_dish', '0')
            # command += _configOption('random_focal_length', '0.')
            command += _configOption('mirror_align_random_distance', '0.')
            command += _configOption('mirror_align_random_vertical', '0.,28.,0.,0.')
        command += ' ' + str(self._corsikaFileName)
        command += ' 2>&1 > ' + str(self._logFileName) + ' 2>&1'

        return command
    # END of makeRunCommand
