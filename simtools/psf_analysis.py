#!/usr/bin/python3

"""
psf_analysis.py
====================================
Testing docstrings
"""

import logging
import numpy as np
import os
from astropy.io import ascii
from astropy.table import Table
import math
import matplotlib.pyplot as plt
from astropy import units


from simtools.util.general import (
    collectArguments,
    collectKwargs,
    setDefaultKwargs
)

__all__ = ['PSFImage']


class PSFImage:
    def __init__(self, focalLength):
        """ PSFImage only knows the list of photon position in cm
            No information about the telescope (e.g focal length) should be used in here.
        """
        self.log = logging.getLogger(__name__)

        self.xPos = list()
        self.yPos = list()
        self.xPosMean = None
        self.yPosMean = None
        self.effArea = None
        self.totalPhotons = None
        self.totalArea = None
        self.d80_cm = None
        self.d80_deg = None
        self._storedPSF = dict()
        self._cmToDeg = 180. / math.pi / focalLength

    def __repr__(self):
        return 'PSFImage ({}/{} photons)'.format(self.detectedPhotons, self.totalPhotons)

    def readSimtelFile(self, file):
        self.log.info('Reading SimtelFile {}'.format(file))
        self.totalPhotons = 0
        self.totalArea = None
        with open(file, 'r') as f:
            for line in f:
                self.processSimtelLine(line)

        if len(self.xPos) == 0 or len(self.yPos) == 0 or len(self.xPos) != len(self.yPos):
            self.log.error('Problems reading Simtel file - invalid data')

        self.xPosMean = np.mean(self.xPos)
        self.yPosMean = np.mean(self.yPos)
        self.detectedPhotons = len(self.xPos)
        self.effArea = self.detectedPhotons * self.totalArea / self.totalPhotons

    def loadPSF(self):
        self.d80_cm = self.getPSF(0.8)
        self.d80_deg = self.d80_cm * self._cmToDeg

    def processSimtelLine(self, line):
        words = line.split()
        if 'falling on an area of' in line:
            self.totalPhotons += int(words[4])
            area = float(words[14])
            if self.totalArea is None:
                self.totalArea = area
            elif area != self.totalArea:
                self.log.error(
                    'Conflicting value of the total area found'
                    ' {} != {}'.format(self.totalArea, area)
                )
        elif '#' in line or len(words) == 0:
            pass
        else:
            self.xPos.append(float(words[2]))
            self.yPos.append(float(words[3]))

    def getPSF(self, fraction):
        if fraction not in self._storedPSF.keys():
            self.computePSF(fraction)
        return self._storedPSF[fraction]

    def computePSF(self, fraction):
        self._storedPSF[fraction] = self.findPSF(fraction)

    def findPSF(self, fraction):
        self.log.debug('Finding PSF for fraction = {}'.format(fraction))

        xPos2 = [i**2 for i in self.xPos]
        yPos2 = [i**2 for i in self.yPos]
        xSig = math.sqrt(np.mean(xPos2) - self.xPosMean**2)
        ySig = math.sqrt(np.mean(yPos2) - self.yPosMean**2)
        rSig = math.sqrt(xSig**2 + ySig**2)

        numberTarget = fraction * self.detectedPhotons
        rad = 1.5 * rSig
        number0 = self.sumPhotonsInRadius(rad)
        A = 0.5 * math.sqrt(rad * rad / number0)
        delta = number0 - numberTarget
        nIter = 0
        foundRadius = False
        while not foundRadius and nIter < 100:
            nIter += 1
            dr = -delta * A / math.sqrt(numberTarget)
            while rad + dr < 0:
                dr *= 0.5
            rad += dr
            number = self.sumPhotonsInRadius(rad)
            delta = number - numberTarget
            foundRadius = math.fabs(delta) < self.detectedPhotons / 1000.

        if foundRadius:
            return 2 * rad
        else:
            self.log.warning('Could not find PSF efficiently')
            psf = self.findPSFByScanning(numberTarget, rSig)
            return psf

    def findPSFByScanning(self, numberTarget, rSig):
        self.log.debug('Finding PSF by scanning')

        def scan(dr, radMin, radMax):
            r0, r1 = radMin, radMin + dr
            s0, s1 = 0, 0
            foundRadius = False
            while not foundRadius:
                s0, s1 = self.sumPhotonsInRadius(r0), self.sumPhotonsInRadius(r1)
                if s0 < numberTarget and s1 > numberTarget:
                    foundRadius = True
                    break
                if r1 > radMax:
                    break
                r0 += dr
                r1 += dr
            if foundRadius:
                return (r0 + r1) / 2, r0, r1
            else:
                self.log.error('Could not find PSF by scanning')
                return 0, radMin, radMax

        # Step 0
        rad, radMin, radMax = scan(0.1 * rSig, 0, 4 * rSig)
        # Step 1
        rad, radMin, radMax = scan(0.005 * rSig, radMin, radMax)
        return rad

    def sumPhotonsInRadius(self, radius):
        n = 0
        for x, y in zip(self.xPos, self.yPos):
            d2 = (x - self.xPosMean)**2 + (y - self.yPosMean)**2
            n += 1 if d2 < radius**2 else 0
        return n

    def plot(self, **kwargs):
        ''' kwargs for histogram: image_*
            kwargs for PSF circle: psf_*
        '''
        ax = plt.gca()
        fac = 1
        xToPlot = fac * np.array(self.xPos) - fac * self.xPosMean
        yToPlot = fac * np.array(self.yPos) - fac * self.yPosMean

        kwargs = setDefaultKwargs(
            kwargs,
            image_bins=80,
            image_cmap=plt.cm.gist_heat_r,
            psf_color='k',
            psf_fill=False,
            psf_lw=2,
            psf_ls='--'
        )
        kwargsForImage = collectKwargs('image', kwargs)
        kwargsForPSF = collectKwargs('psf', kwargs)

        ax.hist2d(xToPlot, yToPlot, **kwargsForImage)

        circle = plt.Circle((0, 0), self.d80_cm / 2, **kwargsForPSF)
        ax.add_artist(circle)

# end of PSFImage
