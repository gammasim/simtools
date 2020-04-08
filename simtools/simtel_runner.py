#!/usr/bin/python3

import logging
from pathlib import Path
import os
from astropy import units

from simtools.util import names
from simtools.telescope_model import TelescopeModel
from simtools.util.general import collectArguments
from simtools import io_handler as io


class SimtelRunner:
    def __init__(
        self,
        simtelSourcePath,
        mode,
        telescopeModel=None,
        label=None,
        filesLocation=None,
        **kwargs
    ):
        ''' Comment '''
        self.log = logging.getLogger(__name__)
        self.log.info('Creating SimtelRunner')

        self.simtelSourcePath = Path(simtelSourcePath)

        self._mode = None
        self.mode = mode

        self.hasTelescopeModel = False
        self._telescopeModel = None
        self.telescopeModel = telescopeModel

        # RayTracing - default parameters
        self._repNumber = 0
        self.RUNS_PER_SET = 100 if 'SingleMirror' in self._mode else 100  # const
        self.PHOTONS_PER_RUN = 50000  # const

        # Label
        self._hasLabel = True
        if label is not None:
            self.label = label
        elif self.hasTelescopeModel:
            self.label = self._telescopeModel.label
        else:
            self._hasLabel = False

        # File location
        modeDir = 'ray-tracing' if 'RayTracing' in self.mode else 'generic'
        self._filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)
        self._baseDirectory = io.getOutputDirectory(self._filesLocation, self.label, modeDir)
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        collectArguments(
            self,
            ['zenithAngle', 'offAxisAngle', 'sourceDistance', 'repNumber'],
            **kwargs
        )

        if self.mode == 'RayTracingSingleMirror':
            self.telescopeModel.loadMirrorGeometryParameters()
    # end of _init_

    def __repr__(self):
        return 'SimtelRunner(mode={}, label={})\n'.format(self.mode, self.label)

    # Properties
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = names.validateName(value, names.allSimtelModeNames)

    @property
    def telescopeModel(self):
        return self._telescopeModel

    @telescopeModel.setter
    def telescopeModel(self, tel):
        if isinstance(tel, TelescopeModel):
            self._telescopeModel = tel
            self.hasTelescopeModel = True
        else:
            self._telescopeModel = None
            self.hasTelescopeModel = False
            if tel is not None:
                self.log.error('Invalid TelescopeModel')

    def run(self, test=False, force=False):
        self.log.info('Running at mode {}'.format(self.mode))
        # write all the important parameters

        if not self.shallRun() and not force:
            self.log.info('Skipping because file exists and force = False')
            return

        self.loadRequiredFiles()
        command = self.makeRunCommand()

        if test:
            self.log.info('Running (test) with command:{}'.format(command))
            os.system(command)
        else:
            numberRuns = self.RUNS_PER_SET
            self.log.info('Running ({}x) with command:{}'.format(numberRuns, command))
            for _ in range(numberRuns):
                os.system(command)

    def getRunBashScript(self, test=False):
        self.log.info('Creating run bash script')
        self._scriptFileName = self._baseDirectory.joinpath('run_script')
        self.log.debug('Run bash script - {}'.format(self._scriptFileName))

        self.loadRequiredFiles()
        command = self.makeRunCommand()
        with self._scriptFileName.open('w') as file:
            # TODO: header
            file.write('#/usr/bin/bash\n\n')
            N = 1 if test else self.RUNS_PER_SET
            for _ in range(N):
                file.write('{}\n\n'.format(command))

        return self._scriptFileName

    def shallRun(self):
        if 'RayTracing' in self.mode:
            photonsFileName = names.rayTracingFileName(
                self._telescopeModel.telescopeType,
                self._sourceDistance,
                self._zenithAngle,
                self._offAxisAngle,
                self._repNumber,
                self.label,
                'photons'
            )
            file = self._baseDirectory.joinpath(photonsFileName)
            return not file.exists()
        else:
            return False

    def loadRequiredFiles(self):
        ''' Which file are nrequired for running depend on the mode.
            Here we define and write some information into these files.
            Log files are always requires.
        '''
        ############
        # RayTracing
        if 'RayTracing' in self.mode:
            self._corsikaFileName = self.simtelSourcePath.joinpath('run9991.corsika.gz')

            # Loop to define and remove existing files
            # Files will be named _baseFileName = self.__dict__['_' + base + 'FileName']
            for base in ['stars', 'photons', 'log']:
                fileName = names.rayTracingFileName(
                    self._telescopeModel.telescopeType,
                    self._sourceDistance,
                    self._zenithAngle,
                    self._offAxisAngle,
                    self._repNumber,
                    self.label,
                    base
                )
                file = self._baseDirectory.joinpath(fileName)
                if file.exists():
                    file.unlink()
                self.__dict__['_' + base + 'FileName'] = file

            with self._photonsFileName.open('w') as file:
                file.write('#{}\n'.format(50*'='))
                file.write('# List of photons for RayTracing simulations\n')
                file.write('#{}\n'.format(50*'='))
                file.write('# configFile = {}\n'.format(self._telescopeModel.getConfigFile()))
                file.write('# zenithAngle [deg] = {}\n'.format(self._zenithAngle))
                file.write('# offAxisAngle [deg] = {}\n'.format(self._offAxisAngle))
                file.write('# sourceDistance [km] = {}\n'.format(self._sourceDistance))
                file.write('# repNumber = {}\n\n'.format(self._repNumber))

            with self._starsFileName.open('w') as file:
                file.write('0. {} 1.0 {}'.format(90. - self._zenithAngle, self._sourceDistance))
        ############
        # Trigger
        # elif self.mode == 'Trigger':
        #     pass

    def makeRunCommand(self):
        ''' Return the command to run simtel_array.
            It is only supposed to be called from inside the method run.
        '''
        def configOption(par, value=None):
            c = ' -C {}'.format(par)
            c += '={}'.format(value) if value is not None else ''
            return c

        # RayTracing
        command = str(self.simtelSourcePath.joinpath('sim_telarray/bin/sim_telarray'))
        command += ' -c {}'.format(self._telescopeModel.getConfigFile())
        command += ' -I../cfg/CTA'
        command += configOption('IMAGING_LIST', str(self._photonsFileName))
        command += configOption('stars', str(self._starsFileName))
        command += configOption('altitude', self._telescopeModel.getParameter('altitude'))
        command += configOption('telescope_theta', self._zenithAngle + self._offAxisAngle)
        command += configOption('star_photons', str(self.PHOTONS_PER_RUN))
        command += configOption('telescope_phi', '0')
        command += configOption('camera_transmission', '1.0')
        command += configOption('nightsky_background', 'all:0.')
        command += configOption('trigger_current_limit', '1e10')
        command += configOption('telescope_random_angle', '0')
        command += configOption('telescope_random_error', '0')
        command += configOption('convergent_depth', '0')
        command += configOption('maximum_telescopes', '1')
        command += configOption('show', 'all')
        command += configOption('camera_filter', 'none')
        if self.mode == 'RayTracingSingleMirror':
            command += configOption('focus_offset', 'all:0.')
            command += configOption('camera_config_file', 'single_pixel_camera.dat')
            command += configOption('camera_pixels', '1')
            command += configOption('trigger_pixels', '1')
            command += configOption('camera_body_diameter', '0')
            command += configOption('mirror_list', self.telescopeModel.getSingleMirrorListFile())
            command += configOption('focal_length', self._sourceDistance * units.km.to(units.cm))
            command += configOption('dish_shape_length', self.telescopeModel.mirrorFocalLength)
            command += configOption('mirror_focal_length', self.telescopeModel.mirrorFocalLength)
            command += configOption('parabolic_dish', '0')
            command += configOption('random_focal_length', '0.')
            command += configOption('mirror_align_random_distance', '0.')
            command += configOption('mirror_align_random_vertical', '0.,28.,0.,0.')
        command += ' ' + str(self._corsikaFileName)
        command += ' 2>&1 > ' + str(self._logFileName) + ' 2>&1'

        return command
    # end makeRunCommand
