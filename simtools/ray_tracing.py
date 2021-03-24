import logging
import subprocess
import matplotlib.pyplot as plt
from copy import copy
from pathlib import Path
from math import pi, tan

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import QTable

import simtools.config as cfg
import simtools.io_handler as io
from simtools.psf_analysis import PSFImage
from simtools.util import names
from simtools.util.model import computeTelescopeTransmission
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel_runner import SimtelRunner
from simtools.util.general import collectArguments
from simtools import visualize

__all__ = ['RayTracing']


class RayTracing:
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
    ALL_INPUTS = {
        'zenithAngle': {'default': 20, 'unit': u.deg},
        'offAxisAngle': {
            'default': [0.0],
            'unit': u.deg,
            'isList': True
        },
        'sourceDistance': {'default': 10, 'unit': u.km}
    }

    YLABEL = {
        'd80_cm': r'$D_{80}$',
        'd80_deg': r'$D_{80}$',
        'eff_area': 'Eff. mirror area',
        'eff_flen': 'Eff. focal length'
    }

    def __init__(
        self,
        telescopeModel,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
        singleMirrorMode=False,
        useRandomFocalLength=False,
        mirrorNumbers='all',
        **kwargs
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

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        self._telescopeModel = self._validateTelescopeModel(telescopeModel)

        self._singleMirrorMode = singleMirrorMode
        self._useRandomFocalLength = useRandomFocalLength

        # Default parameters
        if self._singleMirrorMode:
            collectArguments(
                self,
                args=['zenithAngle', 'offAxisAngle'],
                allInputs=self.ALL_INPUTS,
                **kwargs
            )
            mirFlen = self._telescopeModel.getParameter('mirror_focal_length')
            self._sourceDistance = 2 * float(mirFlen) * u.cm.to(u.km)  # km
            self._mirrorNumbers = mirrorNumbers
        else:
            collectArguments(
                self,
                args=['zenithAngle', 'offAxisAngle', 'sourceDistance'],
                allInputs=self.ALL_INPUTS,
                **kwargs
            )

        self.label = label if label is not None else self._telescopeModel.label

        self._outputDirectory = io.getRayTracingOutputDirectory(self._filesLocation, self.label)
        self._outputDirectory.mkdir(parents=True, exist_ok=True)

        if self._singleMirrorMode:
            if self._mirrorNumbers == 'all':
                self._mirrorNumbers = list(range(0, self._telescopeModel.mirrors.numberOfMirrors))
            if not isinstance(self._mirrorNumbers, list):
                self._mirrorNumbers = [self._mirrorNumbers]

        self._hasResults = False

        # Results file
        fileNameResults = names.rayTracingResultsFileName(
            self._telescopeModel.telescopeName,
            self._sourceDistance,
            self._zenithAngle,
            self.label
        )
        self._outputDirectory.joinpath('results').mkdir(parents=True, exist_ok=True)
        self._fileResults = self._outputDirectory.joinpath('results').joinpath(fileNameResults)
    # END of init

    def __repr__(self):
        return 'RayTracing(label={})\n'.format(self.label)

    def _validateTelescopeModel(self, tel):
        ''' Validate TelescopeModel '''
        if isinstance(tel, TelescopeModel):
            self._logger.debug('TelescopeModel OK')
            return tel
        else:
            msg = 'Invalid TelescopeModel'
            self._logger.error(msg)
            raise ValueError(msg)

    def simulate(self, test=False, force=False):
        '''
        Simulate RayTracing using SimtelRunner.

        Parameters
        ----------
        test: bool
            Test flag will make it faster by simulating much fewer photons.
        force: bool
            Force flag will remove existing files and simulate again.
        '''
        allMirrors = self._mirrorNumbers if self._singleMirrorMode else [0]
        for thisOffAxis in self._offAxisAngle:
            for thisMirror in allMirrors:
                self._logger.info('Simulating RayTracing for offAxis={}, mirror={}'.format(
                    thisOffAxis,
                    thisMirror
                ))
                simtel = SimtelRunner(
                    simtelSourcePath=self._simtelSourcePath,
                    filesLocation=self._filesLocation,
                    mode='ray-tracing' if not self._singleMirrorMode else 'raytracing-singlemirror',
                    telescopeModel=self._telescopeModel,
                    zenithAngle=self._zenithAngle * u.deg,
                    sourceDistance=self._sourceDistance * u.km,
                    offAxisAngle=thisOffAxis * u.deg,
                    mirrorNumber=thisMirror,
                    useRandomFocalLength=self._useRandomFocalLength,
                    logger=self._logger.name
                )
                simtel.run(test=test, force=force)
    # END of simulate

    def analyze(self, export=True, force=False, useRX=False, noTelTransmission=False):
        '''
        Analyze RayTracing, meaning read simtel files, compute psfs and eff areas and store the
        results in _results.

        Parameters
        ----------
        export: bool
            If True, results will be exported to a file automatically. Alternativelly,
            exportResults function can be used.
        force: bool
            If True, existing results files will be removed and analysis will be done again.
        useRX: bool
            If True, calculations are done using the rx binary provided by sim_telarray. If False,
            calculations are done internally, by the module psf_analysis.
        noTelTransmission: bool
            If True, the telescope transmission is not applied.
        '''

        doAnalyze = (not self._fileResults.exists() or force)

        focalLength = float(self._telescopeModel.getParameter('focal_length'))
        telTransmissionPars = (
            self._telescopeModel.getTelescopeTransmissionParameters()
            if not noTelTransmission else [1, 0, 0, 0]
        )

        cmToDeg = 180. / pi / focalLength

        self._psfImages = dict()
        if doAnalyze:
            _rows = list()
        else:
            self._readResults()

        allMirrors = self._mirrorNumbers if self._singleMirrorMode else [0]
        for thisOffAxis in self._offAxisAngle:
            for thisMirror in allMirrors:
                self._logger.debug('Analyzing RayTracing for offAxis={}'.format(thisOffAxis))
                if self._singleMirrorMode:
                    self._logger.debug('mirrorNumber={}'.format(thisMirror))

                photonsFileName = names.rayTracingFileName(
                    self._telescopeModel.telescopeName,
                    self._sourceDistance,
                    self._zenithAngle,
                    thisOffAxis,
                    thisMirror if self._singleMirrorMode else None,
                    self.label,
                    'photons'
                )

                photonsFile = self._outputDirectory.joinpath(photonsFileName)
                telTransmission = computeTelescopeTransmission(telTransmissionPars, thisOffAxis)
                image = PSFImage(focalLength, None, self._logger.name)
                image.readSimtelFile(photonsFile)
                self._psfImages[thisOffAxis] = copy(image)

                if not doAnalyze:
                    continue

                if useRX:
                    d80_cm, centroidX, centroidY, effArea = self._processRX(photonsFile)
                    d80_deg = d80_cm * cmToDeg
                    image.setPSF(d80_cm, fraction=0.8, unit='cm')
                    image.centroidX = centroidX
                    image.centroidY = centroidY
                    image.setEffectiveArea(effArea * telTransmission)
                else:
                    d80_cm = image.getPSF(0.8, 'cm')
                    d80_deg = image.getPSF(0.8, 'deg')
                    centroidX = image.centroidX
                    centroidY = image.centroidY
                    effArea = image.getEffectiveArea() * telTransmission

                effFlen = (
                    np.nan if thisOffAxis == 0 else centroidX / tan(thisOffAxis * pi / 180.)
                )
                _currentResults = (
                    thisOffAxis * u.deg,
                    d80_cm * u.cm,
                    d80_deg * u.deg,
                    effArea * u.m * u.m,
                    effFlen * u.cm
                )
                if self._singleMirrorMode:
                    _currentResults += (thisMirror,)
                _rows.append(_currentResults)
        # END for offAxis, mirrorNumber

        if doAnalyze:
            _columns = ['Off-axis angle']
            _columns.extend(list(self.YLABEL.keys()))
            if self._singleMirrorMode:
                _columns.append('mirror_number')
            self._results = QTable(rows=_rows, names=_columns)

        self._hasResults = True
        # Exporting
        if export:
            self.exportResults()
    # END of analyze

    def _processRX(self, file):
        '''
        Process sim_telarray photon list with rx binary and return the results (d80, centroids and
        eff area).

        Parameters
        ----------
        file: str or Path
            Photon list file.

        Returns
        -------
        (d80_cm, xMean, yMean, effArea)
        '''
        # Use -n to disable the cog optimization
        rxOutput = subprocess.check_output(
            '{}/sim_telarray/bin/rx -f 0.8 -v < {}'.format(self._simtelSourcePath, file),
            shell=True
        )
        rxOutput = rxOutput.split()
        d80_cm = 2 * float(rxOutput[0])
        xMean = float(rxOutput[1])
        yMean = float(rxOutput[2])
        effArea = float(rxOutput[5])
        return d80_cm, xMean, yMean, effArea

    def exportResults(self):
        ''' Export results to a csv file. '''
        if not self._hasResults:
            self._logger.error('Cannot export results because it does not exist')
        else:
            self._logger.info('Exporting results to {}'.format(self._fileResults))
            ascii.write(self._results, self._fileResults, format='ecsv', overwrite=True)

    def _readResults(self):
        ''' Read existing results file and store it in _results. '''
        self._results = ascii.read(self._fileResults, format='ecsv')
        self._hasResults = True

    def plot(self, key, save=False, **kwargs):
        '''
        Plot key vs off-axis angle and save the figure in pdf.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen
        save: bool
            If True, figure will be saved.
        **kwargs:
            kwargs for plt.plot

        Raises
        ------
        KeyError
            If key is not among the valid options.
        '''
        if key not in self.YLABEL.keys():
            msg = 'Invalid key to plot'
            self._logger.error(msg)
            raise KeyError(msg)

        self._logger.info('Plotting {} vs off-axis angle'.format(key))

        plt = visualize.plotTable(
            self._results['Off-axis angle', key],
            self.YLABEL[key],
            noLegend=True,
            **kwargs
        )

        if save:
            plotFileName = names.rayTracingPlotFileName(
                key,
                self._telescopeModel.telescopeName,
                self._sourceDistance,
                self._zenithAngle,
                self.label
            )
            self._outputDirectory.joinpath('figures').mkdir(exist_ok=True)
            plotFile = self._outputDirectory.joinpath('figures').joinpath(plotFileName)
            self._logger.info('Ssaving fig in {}'.format(plotFile))
            plt.savefig(plotFile)

    def plotHistogram(self, key, **kwargs):
        '''
        Plot histogram of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen
        **kwargs:
            kwargs for plt.hist

        Raises
        ------
        KeyError
            If key is not among the valid options.
        '''
        if key not in self.YLABEL.keys():
            msg = 'Invalid key to plot'
            self._logger.error(msg)
            raise KeyError(msg)

        ax = plt.gca()
        ax.hist([r.value for r in self._results[key]], **kwargs)

    def getMean(self, key):
        '''
        Get mean value of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen

        Returns
        -------
        float
            Mean value of key.

        Raises
        ------
        KeyError
            If key is not among the valid options.
        '''
        if key not in self.YLABEL.keys():
            msg = 'Invalid key to plot'
            self._logger.error(msg)
            raise KeyError(msg)
        return np.mean(self._results[key])

    def getStdDev(self, key):
        '''
        Get std dev of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen

        Returns
        -------
        float
            Srd deviation of key.

        Raises
        ------
        KeyError
            If key is not among the valid options.
        '''
        if key not in self.YLABEL.keys():
            msg = 'Invalid key to plot'
            self._logger.error(msg)
            raise KeyError(msg)
        return np.std(self._results[key])

    def images(self):
        '''
        Get list of PSFImages.

        Returns
        -------
        List of PSFImage's
        '''
        images = list()
        for thisOffAxis in self._offAxisAngle:
            if thisOffAxis in self._psfImages.keys():
                images.append(self._psfImages[thisOffAxis])
        if len(images) == 0:
            self._logger.error('No image found')
            return None
        return images

# END of RayTracing
