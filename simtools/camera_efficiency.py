'''
Camera Efficiency simulations and analysis.

Author: Raul R Prado
'''

import logging
import matplotlib.pyplot as plt
from copy import copy
from pathlib import Path
from math import pi, tan

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools.util import names
from simtools.model.telescope_model import TelescopeModel
from simtools.model.model_parameters import RADIUS_CURV

__all__ = ['CameraEfficiency']

logger = logging.getLogger(__name__)


class CameraEfficiency:
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
        Plot histogram of key.
    getMean(key)
        Get mean value of key.
    getStdDev(key)
        Get std dev of key.
    images()
        Get list of PSFImages.
    '''

    def __init__(
        self,
        telescopeModel,
        label=None,
        simtelSourcePath=None,
        filesLocation=None
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
        self._simtelSourcePath = Path(cfg.collectConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.collectConfigArg('outputLocation', filesLocation)
        self._telescopeModel = self._validateTelescopeModel(telescopeModel)
        self.label = label if label is not None else self._telescopeModel.label

        self._baseDirectory = io.getCameraEfficiencyOutputDirectory(self._filesLocation, self.label)
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        self._hasResults = False

        # Results file
        fileNameResults = names.cameraEfficiencyResultsFileName(
                self._telescopeModel.telescopeType,
                self.label
        )
        self._fileResults = self._baseDirectory.joinpath(fileNameResults)
        # Log file
        fileNameLog = names.cameraEfficiencyLogFileName(
                self._telescopeModel.telescopeType,
                self.label
        )
        self._fileLog = self._baseDirectory.joinpath(fileNameLog)
    # END of init

    def __repr__(self):
        return 'CameraEfficiency(label={})\n'.format(self.label)

    def _validateTelescopeModel(self, tel):
        ''' Validate TelescopeModel '''
        if isinstance(tel, TelescopeModel):
            logger.debug('TelescopeModel OK')
            return tel
        else:
            msg = 'Invalid TelescopeModel'
            logger.error(msg)
            raise ValueError(msg)

    def simulate(self, force=False):
        '''
        Simulate RayTracing using SimtelRunner.

        Parameters
        ----------
        test: bool
            Test flag will make it faster by simulating much fewer photons.
        force: bool
            Force flag will remove existing files and simulate again.
        '''

        # Processing camera pixel features
        funnelShape = self._telescopeModel.camera.getFunnelShape()
        pixelShapeCmd = '-hpix' if funnelShape in [1, 3] else '-spix'
        pixelDiameter = self._telescopeModel.camera.getDiameter()

        # Processing focal length
        focalLength = self._telescopeModel.getParameter('effective_focal_length')
        if focalLength == 0.:
            logger.warning('Using focal_lenght because effective_focal_length is 0')
            focalLength = self._telescopeModel.getParameter('focal_length')

        # Processing mirror class
        mirrorClass = 1
        if self._telescopeModel.hasParameter('mirror_class'):
            mirrorClass = self._telescopeModel.getParameter('mirror_class')

        # Processing camera transmission
        cameraTransmission = 1
        if self._telescopeModel.hasParameter('camera_transmission'):
            cameraTransmission = self._telescopeModel.getParameter('camera_transmission')

        cmd = str(self._simtelSourcePath.joinpath('bin').joinpath('testeff'))
        cmd += ' -nm -nsb-extra'
        cmd += ' -alt {}'.format(self._telescopeModel.getParameter('altitude'))
        cmd += ' -fatm {}'.format(self._telescopeModel.getParameter('atmospheric_transmission'))
        cmd += ' -flen {}'.format(focalLength * 0.01)  # focal lenght in meters
        cmd += ' -fcur {}'.format(RADIUS_CURV[self._telescopeModel.telescopeType])
        cmd += ' {} {}'.format(pixelShapeCmd, pixelDiameter)
        cmd += ' -fmir {}'.format(self._telescopeModel.getParameter('mirror_list'))
        cmd += ' -fref {}'.format(self._telescopeModel.getParameter('mirror_reflectivity'))
        if mirrorClass == 2:
            cmd += ' -m2'
        cmd += ' -teltrans {}'.format(self._telescopeModel.getTelescopeTransmissionParameters()[0])
        cmd += ' -camtrans {}'.format(cameraTransmission)
        cmd += ' -fflt {}'.format(self._telescopeModel.getParameter('camera_filter'))
        cmd += ' -fang {}'.format(self._telescopeModel.camera.getFunnelEfficiencyFile())
        cmd += ' -fwl {}'.format(self._telescopeModel.camera.getFunnelWavelengthFile())
        cmd += ' -fqe {}'.format(self._telescopeModel.getParameter('quantum_efficiency'))
        cmd += ' {} {}'.format(200, 1000)  # lmin and lmax
        cmd += ' {} 1 20'.format(300)  # Xmax, ioatm, zenith angle
        cmd += ' 2>{}'.format(self._fileLog)
        cmd += ' >{}'.format(self._fileResults)

        print(cmd)

        # Moving to sim_telarray directory before running
        # cmd = 'cd {} && {}'.format(self.simTelArrayPath, cmd)

        # logStdout.info([['b', 'Running sim_telarray with cmd: {}'.format(cmd)]])
        # os.system(cmd)

    # END of simulate

    def analyze(self, export=True, force=False):
        '''
        Analyze RayTracing, meaning read simtel files, compute psfs and eff areas and store the
        results in _results.

        Parameters
        ----------
        export: bool
            If True, results will be exported to a file automatically. Alternativelly, exportResults
            function can be used.
        force: bool
            If True, existing results files will be removed and analysis will be done again.
        useRX: bool
            If True, calculations are done using the rx binary provided by sim_telarray. If False,
            calculations are done internally, by the module psf_analysis.
        noTelTransmission: bool
            If True, the telescope transmission is not applied.
        '''

    # END of analyze

    def exportResults(self):
        ''' Export results to a csv file. '''
        if not self._hasResults:
            logger.error('Cannot export results because it does not exist')
        else:
            logger.info('Exporting results to {}'.format(self._fileResults))
            table = Table(self._results)
            ascii.write(table, self._fileResults, format='basic', overwrite=True)

    def _readResults(self):
        ''' Read existing results file and store it in _results. '''
        table = ascii.read(self._fileResults, format='basic')
        self._results = dict(table)
        self._hasResults = True


# END of CameraEfficiency
