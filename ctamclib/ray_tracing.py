#!/usr/bin/python3

import logging
from pathlib import Path
import numpy as np
import os
import subprocess
from astropy.io import ascii
from astropy.table import Table
import math
import matplotlib.pyplot as plt

from ctamclib.util import names
from ctamclib.telescope_model import TelescopeModel
from ctamclib.simtel_runner import SimtelRunner
from ctamclib.util.general import collectArguments


class RayTracing:
    def __init__(
        self,
        simtelSourcePath,
        telescopeModel,
        label=None,
        filesLocation=None,
        **kwargs
    ):
        ''' Comment '''

        self._simtelSourcePath = simtelSourcePath
        self._filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)
        self._baseDirectory = self._filesLocation.joinpath('CTAMCFiles')

        self.hasTelescopeModel = False
        self._telescopeModel = None
        self.telescopeModel = telescopeModel

        # Default parameters
        self._zenithAngle = 20                          # deg
        self._offAxisAngle = np.linspace(0.0, 3.0, 7)   # deg
        self._sourceDistance = 10                       # km

        # Label
        self._hasLabel = True
        if label is not None:
            self.label = label
        elif self.hasTelescopeModel:
            self.label = self._telescopeModel.label
        else:
            self._hasLabel = False
            self.label = None

        collectArguments(self, ['zenithAngle', 'offAxisAngle', 'sourceDistance'], **kwargs)
        self._hasResults = False

        # Results file
        fileNameResults = names.rayTracingResultsFileName(
                self._telescopeModel.telescopeType,
                self._sourceDistance,
                self._zenithAngle,
                self.label
        )
        dirResults = self._baseDirectory.joinpath('Results')
        dirResults.mkdir(parents=True, exist_ok=True)
        self._fileResults = dirResults.joinpath(fileNameResults)

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
        """ Simulating RayTracing"""
        for thisOffAxis in self._offAxisAngle:
            logging.info('Simulating RayTracing for offAxis={}'.format(thisOffAxis))
            simtel = SimtelRunner(
                simtelSourcePath=self._simtelSourcePath,
                filesLocation=self._filesLocation,
                mode='ray-tracing',
                telescopeModel=self._telescopeModel,
                zenithAngle=self._zenithAngle,
                sourceDistance=self._sourceDistance,
                offAxisAngle=thisOffAxis
            )
            simtel.run(test=test, force=force)

    def analyze(self, export=True, force=False):
        """ Analyzing RayTracing"""

        if self._fileResults.exists() and not force:
            logging.info('Skipping analyze because file exists and force = False')
            self.readResults()
            return

        focalLength = float(self._telescopeModel.getParameter('focal_length'))
        cmToDeg = 180. / math.pi / focalLength

        self._results = dict()
        self._results['off_axis'] = list()
        self._results['d80_cm'] = list()
        self._results['d80_deg'] = list()
        self._results['eff_area'] = list()
        self._results['eff_flen'] = list()

        for thisOffAxis in self._offAxisAngle:
            logging.info('Analyzing RayTracing for offAxis={}'.format(thisOffAxis))
            photonsFileName = names.rayTracingFileName(
                self._telescopeModel.telescopeType,
                self._sourceDistance,
                self._zenithAngle,
                thisOffAxis,
                self.label,
                'photons'
            )
            file = self._baseDirectory.joinpath('RayTracing').joinpath(photonsFileName)
            rxOutput = subprocess.check_output(
                '{}/sim_telarray/bin/rx -f 0.8 -v < {}'.format(self._simtelSourcePath, file),
                shell=True
            )
            rxOutput = rxOutput.split()
            d80 = float(rxOutput[0])
            xMean = float(rxOutput[1])
            yMean = float(rxOutput[2])
            effArea = float(rxOutput[5])
            effFlen = 'nan' if thisOffAxis == 0 else xMean / math.tan(thisOffAxis * math.pi / 180.)
            self._results['off_axis'].append(thisOffAxis)
            self._results['d80_cm'].append(d80)
            self._results['d80_deg'].append(d80 * cmToDeg)
            self._results['eff_area'].append(effArea)
            self._results['eff_flen'].append(effFlen)

        self._hasResults = True

        # Exporting
        if export:
            self.exportResults()

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

    def plot(self, which='d80_cm', **kwargs):
        if which not in ['d80_cm', 'd80_deg', 'eff_area', 'eff_flen']:
            logging.error('Invalid option for plotting RayTracing')
            return

        ax = plt.gca()
        ax.plot(self._results['off_axis'], self._results[which], **kwargs)
