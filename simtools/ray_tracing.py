''' Ray Tracing simulations and analysis. '''

import logging
import os
import subprocess
import math
import matplotlib.pyplot as plt
from copy import copy
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools.psf_analysis import PSFImage
from simtools.util import names
from simtools.util.model import computeTelescopeTransmission
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel_runner import SimtelRunner
from simtools.util.general import collectArguments, collectKwargs, setDefaultKwargs

__all__ = ['RayTracing']


class RayTracing:
    ALL_INPUTS = {
        'zenithAngle': {'default': 20, 'unit': u.deg},
        'offAxisAngle': {
            'default': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            'unit': u.deg,
            'isList': True
        },
        'sourceDistance': {'default': 10, 'unit': u.km},
        'mirrorNumbers': {'default': [1], 'unit': None, 'isList': True}
    }

    def __init__(
        self,
        telescopeModel,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
        singleMirrorMode=False,
        useRandomFocalLength=False,
        **kwargs
    ):
        """
        RayTracing init

        Parameters
        ---------
        name
            A string to assign to the `name` instance attribute.
        """
        self._simtelSourcePath = Path(cfg.collectConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.collectConfigArg('outputLocation', filesLocation)

        self.hasTelescopeModel = False
        self._telescopeModel = None
        self.telescopeModel = telescopeModel

        self._singleMirrorMode = singleMirrorMode
        self._useRandomFocalLength = useRandomFocalLength

        # Default parameters
        if self._singleMirrorMode:
            collectArguments(
                self,
                args=['zenithAngle', 'offAxisAngle', 'mirrorNumbers'],
                allInputs=self.ALL_INPUTS,
                **kwargs
            )
            mirFlen = self.telescopeModel.getParameter('mirror_focal_length')
            self._sourceDistance = 2 * float(mirFlen) * u.cm.to(u.km)  # km
        else:
            collectArguments(
                self,
                args=['zenithAngle', 'offAxisAngle', 'sourceDistance'],
                allInputs=self.ALL_INPUTS,
                **kwargs
            )

        # Label
        self._hasLabel = True
        if label is not None:
            self.label = label
        elif self.hasTelescopeModel:
            self.label = self._telescopeModel.label
        else:
            self._hasLabel = False
            self.label = None

        self._baseDirectory = io.getRayTracingOutputDirectory(self._filesLocation, self.label)
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        if self._singleMirrorMode:
            if self._mirrorNumbers == 'all':
                self._mirrorNumbers = list(range(1, self._telescopeModel.numberOfMirrors + 1))
            if not isinstance(self._mirrorNumbers, list):
                self._mirrorNumbers = [self._mirrorNumbers]

        self._hasResults = False

        # Results file
        fileNameResults = names.rayTracingResultsFileName(
                self._telescopeModel.telescopeType,
                self._sourceDistance,
                self._zenithAngle,
                self.label
        )
        self._fileResults = self._baseDirectory.joinpath(fileNameResults)

        # end of init

    def __repr__(self):
        return 'RayTracing(label={})\n'.format(self.label)

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
                logging.error('Invalid TelescopeModel')

    def simulate(self, test=False, force=False):
        """Simulate RayTracing."""
        allMirrors = self._mirrorNumbers if self._singleMirrorMode else [0]
        for thisOffAxis in self._offAxisAngle:
            for thisMirror in allMirrors:
                logging.info('Simulating RayTracing for offAxis={}, mirror={}'.format(
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
                    useRandomFocalLength=self._useRandomFocalLength
                )
                simtel.run(test=test, force=force)

    def analyze(self, export=True, force=False, useRX=False, noTelTransmission=False):
        """Analyze RayTracing."""

        if self._fileResults.exists() and not force:
            logging.info('Skipping analyze because file exists and force = False')
            self.readResults()
            focalLength = float(self._telescopeModel.getParameter('focal_length'))
            self._psfImages = dict()
            allMirrors = self._mirrorNumbers if self._singleMirrorMode else [0]
            for thisOffAxis in self._offAxisAngle:
                for thisMirror in allMirrors:
                    logging.debug('Reading images for offAxis={}'.format(thisOffAxis))
                    if self._singleMirrorMode:
                        logging.debug('mirrorNumber={}'.format(thisMirror))

                    photonsFileName = names.rayTracingFileName(
                        self._telescopeModel.telescopeType,
                        self._sourceDistance,
                        self._zenithAngle,
                        thisOffAxis,
                        thisMirror if self._singleMirrorMode else None,
                        self.label,
                        'photons'
                    )
                    image = PSFImage(focalLength)
                    image.readSimtelFile(self._baseDirectory.joinpath(photonsFileName))
                    self._psfImages[thisOffAxis] = copy(image)
            return
            # end for offAxis            return

        focalLength = float(self._telescopeModel.getParameter('focal_length'))
        # FUTURE: telTransmission processing (from str to list of floats)
        # should be done by TelescopeModel class, not here
        telTransmissionPars = list()
        for p in self._telescopeModel.getParameter('telescope_transmission').split():
            telTransmissionPars.append(float(p))
        cmToDeg = 180. / math.pi / focalLength

        self._results = dict()
        self._results['off_axis'] = list()
        self._results['d80_cm'] = list()
        self._results['d80_deg'] = list()
        self._results['eff_area'] = list()
        self._results['eff_flen'] = list()
        if self._singleMirrorMode:
            self._results['mirror_no'] = list()
        self._psfImages = dict()

        allMirrors = self._mirrorNumbers if self._singleMirrorMode else [0]
        for thisOffAxis in self._offAxisAngle:
            for thisMirror in allMirrors:
                logging.info('Analyzing RayTracing for offAxis={}'.format(thisOffAxis))
                if self._singleMirrorMode:
                    logging.info('mirrorNumber={}'.format(thisMirror))

                photonsFileName = names.rayTracingFileName(
                    self._telescopeModel.telescopeType,
                    self._sourceDistance,
                    self._zenithAngle,
                    thisOffAxis,
                    thisMirror if self._singleMirrorMode else None,
                    self.label,
                    'photons'
                )
                file = self._baseDirectory.joinpath(photonsFileName)
                telTransmission = 1 if noTelTransmission else computeTelescopeTransmission(
                    telTransmissionPars,
                    thisOffAxis
                )
                image = PSFImage(focalLength)
                image.readSimtelFile(file)

                if useRX:
                    d80_cm, xPosMean, yPosMean, effArea = self.processRX(file)
                    d80_deg = d80_cm * cmToDeg
                    image.setPSF(d80_cm, fraction=0.8, unit='cm')
                    image.centroidX = xPosMean
                    image.centroidY = yPosMean
                    image.setEffectiveArea(effArea * telTransmission)
                else:
                    d80_cm = image.getPSF(0.8, 'cm')
                    d80_deg = image.getPSF(0.8, 'deg')
                    xPosMean = image.centroidX
                    yPosMean = image.centroidY
                    effArea = image.getEffectiveArea() * telTransmission

                self._psfImages[thisOffAxis] = image
                effFlen = (
                    'nan' if thisOffAxis == 0 else xPosMean / math.tan(thisOffAxis * math.pi / 180.)
                )
                self._results['off_axis'].append(thisOffAxis)
                self._results['d80_cm'].append(d80_cm)
                self._results['d80_deg'].append(d80_deg)
                self._results['eff_area'].append(effArea)
                self._results['eff_flen'].append(effFlen)
                if self._singleMirrorMode:
                    self._results['mirror_no'].append(thisMirror)
        # end for offAxis

        self._hasResults = True

        # Exporting
        if export:
            self.exportResults()

    def processRX(self, file):
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
        logging.info('Exporting results')
        if not self._hasResults:
            logging.error('Cannot export results because it does not exist')
        else:
            table = Table(self._results)
            ascii.write(table, self._fileResults, format='basic', overwrite=True)

    def readResults(self):
        table = ascii.read(self._fileResults, format='basic')
        self._results = dict(table)
        self._hasResults = True

    def _validateWhich(self, which):
        if which not in ['d80_cm', 'd80_deg', 'eff_area', 'eff_flen']:
            logging.error('Invalid option for plotting RayTracing')
            return
        return

    def plot(self, which='d80_cm', **kwargs):
        self._validateWhich(which=which)

        ax = plt.gca()
        ax.plot(self._results['off_axis'], self._results[which], **kwargs)

    def plotHistogram(self, which='d80_cm', **kwargs):
        self._validateWhich(which=which)

        ax = plt.gca()
        ax.hist(self._results[which], **kwargs)

    def getMean(self, which='d80_cm'):
        self._validateWhich(which=which)
        return np.mean(self._results[which])

    def getStdDev(self, which='d80_cm'):
        self._validateWhich(which=which)
        return np.std(self._results[which])

    def images(self):
        images = list()
        for thisOffAxis in self._offAxisAngle:
            if thisOffAxis in self._psfImages.keys():
                images.append(self._psfImages[thisOffAxis])
        if len(images) == 0:
            loggin.warning('No image found')
            return None
        return images

# end of RayTracing
