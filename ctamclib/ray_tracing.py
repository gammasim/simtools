#!/usr/bin/python3

import logging
from pathlib import Path
import numpy as np
import os
import subprocess

from ctamclib import names
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
        self._baseDirectory = self._filesLocation.joinpath('CTAMCFiles/RayTracing')

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

    def analyze(self, export=True):
        """ Analyzing RayTracing"""

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
            file = self._baseDirectory.joinpath(photonsFileName)
            # os.system('{}/sim_telarray/bin/rx -f 0.8 -v < {}'.format(self._simtelSourcePath, file))
            rxOutput = subprocess.check_output(
                '{}/sim_telarray/bin/rx -f 0.8 -v < {}'.format(self._simtelSourcePath, file),
                shell=True
            )
            rxOutput = rxOutput.split()
            d80 = float(rxOutput[0])
            self._results['off_axis'].append(thisOffAxis)
            self._results['d80_cm'].append(d80)
            self._results['eff_area'].append(float(rxOutput[5]))

            xMean = float(rxOutput[1])
            yMean = float(rxOutput[2])
            nPhotons = int(rxOutput[3])
            effArea = float(rxOutput[5])

            print(d80, xMean, yMean, nPhotons, effArea)
