import logging

import astropy.units as u

import simtools.io_handler as io
import simtools.util.general as gen
from simtools.util import names
from simtools.simtel.simtel_runner import SimtelRunner

__all__ = ['SimtelRunnerRayTracing']


class SimtelRunnerRayTracing(SimtelRunner):
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
        mirrorNumber:
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
        telescopeModel,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
        configData=None,
        configFile=None,
        singleMirrorMode=False
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
        self._logger.debug('Init SimtelRunnerRayTracing')

        super().__init__(
            label=label,
            simtelSourcePath=simtelSourcePath,
            filesLocation=filesLocation
        )

        self.telescopeModel = self._validateTelescopeModel(telescopeModel)
        self.label = label if label is not None else self.telescopeModel.label

        # File location
        self._baseDirectory = io.getOutputDirectory(
            self._filesLocation,
            self.label,
            'ray-tracing'
        )
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        self._singleMirrorMode = singleMirrorMode

        # RayTracing - default parameters
        self._repNumber = 0
        self.RUNS_PER_SET = 1 if self._singleMirrorMode else 20
        self.PHOTONS_PER_RUN = 10000

        # Loading configData
        _configDataIn = gen.collectDataFromYamlOrDict(configFile, configData)
        _parameterFile = io.getDataFile('parameters', 'simtel-runner-ray-tracing_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_configDataIn, _parameters)

        self._loadRequiredFiles()

    def _loadRequiredFiles(self):
        '''
        Which file are required for running depends on the mode.
        Here we define and write some information into these files. Log files are always requires.
        '''

        self._corsikaFile = self._simtelSourcePath.joinpath('run9991.corsika.gz')

        # Loop to define and remove existing files.
        # Files will be named _baseFile = self.__dict__['_' + base + 'File']
        for baseName in ['stars', 'photons', 'log']:
            fileName = names.rayTracingFileName(
                self.telescopeModel.site,
                self.telescopeModel.name,
                self.config.sourceDistance,
                self.config.zenithAngle,
                self.config.offAxisAngle,
                self.config.mirrorNumber if self._singleMirrorMode else None,
                self.label,
                baseName
            )
            file = self._baseDirectory.joinpath(fileName)
            if file.exists():
                file.unlink()
            # Defining the file name variable as an class atribute.
            self.__dict__['_' + baseName + 'File'] = file

        # Adding header to photon list file.
        with self._photonsFile.open('w') as file:
            file.write('#{}\n'.format(50 * '='))
            file.write('# List of photons for RayTracing simulations\n')
            file.write('#{}\n'.format(50 * '='))
            file.write('# configFile = {}\n'.format(self.telescopeModel.getConfigFile()))
            file.write('# zenithAngle [deg] = {}\n'.format(self.config.zenithAngle))
            file.write('# offAxisAngle [deg] = {}\n'.format(self.config.offAxisAngle))
            file.write('# sourceDistance [km] = {}\n'.format(self.config.sourceDistance))
            if self._singleMirrorMode:
                file.write('# mirrorNumber = {}\n\n'.format(self.config.mirrorNumber))

        # Filling in star file with a single star.
        with self._starsFile.open('w') as file:
            file.write('0. {} 1.0 {}'.format(
                90. - self.config.zenithAngle,
                self.config.sourceDistance)
            )

    def _getRunScript(self, test=False):
        self._logger.debug('Creating run bash script')
        self._scriptFile = self._baseDirectory.joinpath('run_script')
        self._logger.debug('Run bash script - {}'.format(self._scriptFile))

        command = self._makeRunCommand()
        with self._scriptFile.open('w') as file:
            # TODO: header
            file.write('#/usr/bin/bash\n\n')
            N = 1 if test else self.RUNS_PER_SET
            for _ in range(N):
                file.write('{}\n\n'.format(command))

        return self._scriptFile

    def _shallRun(self):
        ''' Tells if simulations should be run again based on the existence of output files. '''
        return not self._photonsFile.exists()

    def _makeRunCommand(self):
        ''' Return the command to run simtel_array. '''

        def _configOption(par, value=None):
            c = ' -C {}'.format(par)
            c += '={}'.format(value) if value is not None else ''
            return c

        if self._singleMirrorMode:
            _mirrorFocalLength = float(self.telescopeModel.getParameterValue('mirror_focal_length'))

        # RayTracing
        command = str(self._simtelSourcePath.joinpath('sim_telarray/bin/sim_telarray'))
        command += ' -c {}'.format(self.telescopeModel.getConfigFile())
        command += ' -I../cfg/CTA'
        command += _configOption('IMAGING_LIST', str(self._photonsFile))
        command += _configOption('stars', str(self._starsFile))
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
        if self._singleMirrorMode:
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
        command += ' ' + str(self._corsikaFile)
        command += ' 2>&1 > ' + str(self._logFile) + ' 2>&1'

        return command
    # END of makeRunCommand

    def _checkRunResult(self):
        # Checking run
        if not self._isPhotonListFileOK():
            self._logger.error('Photon list is empty.')
        else:
            self._logger.debug('Everything looks fine with output file.')

    def _isPhotonListFileOK(self):
        with open(self._photonsFile, 'r') as ff:
            nLines = len(ff.readlines())

        return nLines > 100
