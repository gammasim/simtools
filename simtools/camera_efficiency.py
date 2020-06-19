'''
Camera Efficiency simulations and analysis.

Author: Raul R Prado
'''

import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path

import astropy.units as u
from astropy.io import ascii
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools.util import names
from simtools.util.general import collectArguments
from simtools.model.telescope_model import TelescopeModel
from simtools.model.model_parameters import RADIUS_CURV

__all__ = ['CameraEfficiency']


class CameraEfficiency:
    '''
    Class for handling camera efficiency simulations and analysis.

    Attributes
    ----------
    label: str
        Instance label.

    Methods
    -------
    simulate(force=False)
        Simulate camera efficiency using testeff from sim_telarray.
    analyse(export=True, force=False)
        Analyze output from testeff and store results in _results.
    exportResults()
        Export results to a csv file.
    plot(key, **kwargs)
        Plot key vs wavelength, where key may be cherenkov or nsb.
    '''
    ALL_INPUTS = {'zenithAngle': {'default': 20, 'unit': u.deg}}

    def __init__(
        self,
        telescopeModel,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
        logger=__name__,
        **kwargs
    ):
        '''
        CameraEfficiency init.

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
        **kwargs:
            Physical parameters with units (if applicable). Options: zenithAngle (default 20 deg).
        '''
        self._logger = logging.getLogger(logger)

        self._simtelSourcePath = Path(cfg.collectConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.collectConfigArg('outputLocation', filesLocation)
        self._telescopeModel = self._validateTelescopeModel(telescopeModel)
        self.label = label if label is not None else self._telescopeModel.label

        self._baseDirectory = io.getCameraEfficiencyOutputDirectory(self._filesLocation, self.label)
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        self._hasResults = False

        collectArguments(self, args=['zenithAngle'], allInputs=self.ALL_INPUTS, **kwargs)

        self._loadFiles()
    # END of init

    def __repr__(self):
        return 'CameraEfficiency(label={})\n'.format(self.label)

    def _validateTelescopeModel(self, tel):
        ''' Validate TelescopeModel '''
        if isinstance(tel, TelescopeModel):
            self._logger.debug('TelescopeModel OK')
            return tel
        else:
            msg = 'Invalid TelescopeModel'
            self._logger.error(msg)
            raise ValueError(msg)

    def _loadFiles(self):
        ''' Define the variables for the file names, including the results, simtel and log file. '''
        # Results file
        fileNameResults = names.cameraEfficiencyResultsFileName(
                self._telescopeModel.telescopeType,
                self._zenithAngle,
                self.label
        )
        self._fileResults = self._baseDirectory.joinpath(fileNameResults)
        # SimtelOutput file
        fileNameSimtel = names.cameraEfficiencySimtelFileName(
                self._telescopeModel.telescopeType,
                self._zenithAngle,
                self.label
        )
        self._fileSimtel = self._baseDirectory.joinpath(fileNameSimtel)
        # Log file
        fileNameLog = names.cameraEfficiencyLogFileName(
                self._telescopeModel.telescopeType,
                self._zenithAngle,
                self.label
        )
        self._fileLog = self._baseDirectory.joinpath(fileNameLog)

    def simulate(self, force=False):
        '''
        Simulate camera efficiency using testeff.

        Parameters
        ----------
        force: bool
            Force flag will remove existing files and simulate again.
        '''
        self._logger.info('Simulating CameraEfficiency')

        if self._fileSimtel.exists() and not force:
            self._logger.info('Simtel file exists and force=False - skipping simulation')
            return

        # Processing camera pixel features
        funnelShape = self._telescopeModel.camera.getFunnelShape()
        pixelShapeCmd = '-hpix' if funnelShape in [1, 3] else '-spix'
        pixelDiameter = self._telescopeModel.camera.getDiameter()

        # Processing focal length
        focalLength = self._telescopeModel.getParameter('effective_focal_length')
        if focalLength == 0.:
            self._logger.warning('Using focal_lenght because effective_focal_length is 0')
            focalLength = self._telescopeModel.getParameter('focal_length')

        # Processing mirror class
        mirrorClass = 1
        if self._telescopeModel.hasParameter('mirror_class'):
            mirrorClass = self._telescopeModel.getParameter('mirror_class')

        # Processing camera transmission
        cameraTransmission = 1
        if self._telescopeModel.hasParameter('camera_transmission'):
            cameraTransmission = self._telescopeModel.getParameter('camera_transmission')

        # cmd -> Command to be run at the shell
        cmd = str(self._simtelSourcePath.joinpath('sim_telarray/bin/testeff'))
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
        cmd += ' {} 1 {}'.format(300, self._zenithAngle)  # Xmax, ioatm, zenith angle
        cmd += ' 2>{}'.format(self._fileLog)
        cmd += ' >{}'.format(self._fileSimtel)

        # Moving to sim_telarray directory before running
        cmd = 'cd {} && {}'.format(self._simtelSourcePath.joinpath('sim_telarray'), cmd)

        self._logger.info('Running sim_telarray with cmd: {}'.format(cmd))
        os.system(cmd)
        return
    # END of simulate

    def analyze(self, export=True, force=False):
        '''
        Analyze camera efficiency output file and store the results in _results.

        Parameters
        ----------
        export: bool
            If True, results will be exported to a file automatically. Alternativelly, exportResults
            function can be used.
        force: bool
            If True, existing results files will be removed and analysis will be done again.
        '''
        self._logger.info('Analyzing CameraEfficiency')

        if self._fileResults.exists() and not force:
            self._logger.info('Results file exists and force=False - skipping analyze')
            self._readResults()
            return

        # List of parameters to be calculated and stored
        effPars = [
            'wl',
            'eff',
            'effAtm',
            'qe',
            'ref',
            'masts',
            'filt',
            'funnel',
            'atmTrans',
            'cher',
            'nsb',
            'atmCorr',
            'nsbSite',
            'nsbSiteEff',
            'nsbBe',
            'nsbBeEff',
            'C1',
            'C2',
            'C3',
            'C4',
            'C4x',
            'N1',
            'N2',
            'N3',
            'N4',
            'N4x'
        ]

        self._results = dict()
        for p in effPars:
            self._results[p] = list()

        with open(self._fileSimtel, 'r') as file:
            for line in file:
                words = line.split()
                if len(words) == 0 or '#' in words[0]:
                    continue
                try:
                    float(words[0])
                except:
                    continue

                if float(words[0]) < 200 or float(words[0]) > 1000:
                    continue
                numbers = [float(w) for w in words]
                for i in range(len(effPars) - 10):
                    self._results[effPars[i]].append(numbers[i])
                C1 = numbers[8] * (400 / numbers[0])**2
                C2 = C1 * numbers[4] * numbers[5]
                C3 = C2 * numbers[6] * numbers[7]
                C4 = C3 * numbers[3]
                C4x = C1 * numbers[3] * numbers[6] * numbers[7]
                self._results['C1'].append(C1)
                self._results['C2'].append(C2)
                self._results['C3'].append(C3)
                self._results['C4'].append(C4)
                self._results['C4x'].append(C4x)
                N1 = numbers[14]
                N2 = N1 * numbers[4] * numbers[5]
                N3 = N2 * numbers[6] * numbers[7]
                N4 = N3 * numbers[3]
                N4x = N1 * numbers[3] * numbers[6] * numbers[7]
                self._results['N1'].append(N1)
                self._results['N2'].append(N2)
                self._results['N3'].append(N3)
                self._results['N4'].append(N4)
                self._results['N4x'].append(N4x)

        self._hasResults = True
        if export:
            self.exportResults()
    # END of analyze

    def exportResults(self):
        ''' Export results to a csv file. '''
        if not self._hasResults:
            self._logger.error('Cannot export results because it does not exist')
        else:
            self._logger.info('Exporting results to {}'.format(self._fileResults))
            table = Table(self._results)
            ascii.write(table, self._fileResults, format='basic', overwrite=True)

    def _readResults(self):
        ''' Read existing results file and store it in _results. '''
        table = ascii.read(self._fileResults, format='basic')
        self._results = dict(table)
        self._hasResults = True

    def plot(self, key, **kwargs):
        '''
        Plot key vs wavelength.

        Parameters
        ----------
        key: str
            cherenkov or nsb
        **kwargs:
            kwargs for plt.plot

        Raises
        ------
        KeyError
            If key is not among the valid options.
        '''
        if key not in ['cherenkov', 'nsb']:
            msg = 'Invalid key to plot'
            self._logger.error(msg)
            raise KeyError(msg)

        ax = plt.gca()
        firstLetter = 'C' if key == 'cherenkov' else 'N'
        for par in ['1', '2', '3', '4', '4x']:
            ax.plot(
                self._results['wl'],
                self._results[firstLetter + par],
                label=firstLetter + par,
                **kwargs
            )

    def plotCherenkovEfficiency(self):
        '''
        Plot cherenkov efficiency vc wavelength.

        Returns
        -------
        plt
        '''
        self._logger.info('Plotting cherenkov efficiency vs wavelength')
        ax = plt.gca()

        ax.set_xlabel('wavelenght [nm]')
        ax.set_ylabel('cherenkov light efficiency')

        for par in ['C1', 'C2', 'C3', 'C4', 'C4x']:
            ax.plot(self._results['wl'], self._results[par], label=par)

        ax.legend(frameon=False)

        return plt

    def plotNSBEfficiency(self):
        '''
        Plot NSB efficiency vc wavelength.

        Returns
        -------
        plt
        '''
        self._logger.info('Plotting NSB efficiency vs wavelength')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xlabel('wavelenght [nm]')
        ax.set_ylabel('nightsky background light efficiency')

        for par in ['N1', 'N2', 'N3', 'N4', 'N4x']:
            ax.plot(self._results['wl'], self._results[par], label=par)

        ylim = ax.get_ylim()
        ax.set_ylim(1e-3, ylim[1])
        ax.legend(frameon=False)

        return plt
# END of CameraEfficiency
