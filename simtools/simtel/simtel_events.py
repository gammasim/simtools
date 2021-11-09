import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from eventio.simtel import SimTelFile


__all__ = ['SimtelEvents']


class InconsistentInputFile(Exception):
    pass


class SimtelEvents:
    '''
    This class handle sim_telarray histograms.
    Histogram files are handled by using eventio library.

    Methods
    -------
    plotAndSaveFigures(figName)
        Plot all histograms and save a single pdf file.
    '''

    def __init__(
        self,
        inputFiles=None
    ):
        '''
        SimtelHistograms

        Parameters
        ----------
        histogramFiles: list
            List of sim_telarray histogram files (str of Path).

        '''
        self._logger = logging.getLogger(__name__)
        self.loadInputFiles(inputFiles)
        self.loadHeader()

    def loadInputFiles(self, files):
        if not hasattr(self, 'inputFiles'):
            self.inputFiles = list()

        if files is None:
            msg = 'No input file was given'
            self._logger.debug(msg)
            return

        if not isinstance(files, list):
            files = [files]

        for file in files:
            self.inputFiles.append(file)
        return

    def loadHeader(self):

        self._numberOfFiles = len(self.inputFiles)
        keysToGrab = ['obsheight', 'n_showers', 'n_use', 'core_range', 'diffuse', 'viewcone',
                      'E_range', 'spectral_index', 'B_total']
        # Dict has to have its keys defined and filled beforehands
        self._mcHeader = {k: 0 for k in keysToGrab}

        self.summaryEvents = {
            'energy': np.array([]),
            'r_core': np.array([])
        }

        def _areHeadersConsistent(header0, header1):
            comparison = dict()
            for k in keysToGrab:
                value = (header0[k] == header1[k])
                comparison[k] = value if isinstance(value, bool) else all(value)

            return all(comparison)

        isFirstFile = True
        numberOfTriggeredEvents = 0
        for file in self.inputFiles:
            with SimTelFile(file) as f:

                for event in f:
                    en = event['mc_shower']['energy']
                    rc = math.sqrt(
                        math.pow(event['mc_event']['xcore'], 2)
                        + math.pow(event['mc_event']['ycore'], 2)
                    )

                    self.summaryEvents['energy'] = np.append(self.summaryEvents['energy'], en)
                    self.summaryEvents['r_core'] = np.append(self.summaryEvents['r_core'], rc)
                    numberOfTriggeredEvents += 1

                if isFirstFile:
                    # First file - grabbing parameters
                    self._mcHeader.update({k: copy(f.mc_run_headers[0][k]) for k in keysToGrab})
                else:
                    # Remaining files - Checking whether the parameters are consistent
                    if not _areHeadersConsistent(self._mcHeader, f.mc_run_headers[0]):
                        msg = 'MC header pamameters from different files are inconsistent'
                        self._logger.error(msg)
                        raise InconsistentInputFile(msg)

                isFirstFile = False

        # Calculating number of events
        self._mcHeader['n_events'] = (
            self._mcHeader['n_use'] * self._mcHeader['n_showers'] * self._numberOfFiles
        )
        self._mcHeader['n_triggered'] = numberOfTriggeredEvents
        return

    def countTriggeredEvents(self, energyRange=None, coreMax=None, coneMax=None):
        if energyRange is None:
            energyRange = self._mcHeader['E_range']

        if coreMax is None:
            coreMax = self._mcHeader['core_range'][1]

        isInEnergyRange = list(map(
            lambda e: e > energyRange[0] and e < energyRange[1],
            self.summaryEvents['energy']
        ))
        isInCoreRange = list(map(
            lambda r: r < coreMax,
            self.summaryEvents['r_core']
        ))
        return np.sum(np.array(isInEnergyRange) * np.array(isInCoreRange))

    def selectEvents(self, energyRange=None, coreMax=None, coneMax=None):
        if energyRange is None:
            energyRange = self._mcHeader['E_range']

        if coreMax is None:
            coreMax = self._mcHeader['core_range'][1]

        selectedEvents = list()
        for file in self.inputFiles:
            with SimTelFile(file) as f:

                for event in f:
                    energy = event['mc_shower']['energy']
                    x_core = event['mc_event']['xcore']
                    y_core = event['mc_event']['ycore']
                    r_core = math.sqrt(math.pow(x_core, 2) + math.pow(y_core, 2))

                    if energy < energyRange[0] or energy > energyRange[1]:
                        continue

                    if r_core > coreMax:
                        continue

                    selectedEvents.append(event)
        return selectedEvents

    def countSimulatedEvents(self, energyRange=None, coreMax=None, coneMax=None):
        if energyRange is None:
            energyRange = self._mcHeader['E_range']

        if coreMax is None:
            coreMax = self._mcHeader['core_range'][1]

        # energy factor
        def integral(erange):
            power = self._mcHeader['spectral_index'] + 1
            return math.pow(erange[0], power) - math.pow(erange[1], power)

        energy_factor = integral(energyRange) / integral(self._mcHeader['E_range'])

        # core factor
        core_factor = math.pow(coreMax, 2) / math.pow(self._mcHeader['core_range'][1], 2)

        # cone factor
        # TODO

        return self._mcHeader['n_events'] * energy_factor * core_factor
