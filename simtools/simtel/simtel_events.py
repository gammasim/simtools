import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    def loadHeader(self):

        self._numberOfFiles = len(self.inputFiles)
        self._mcHeader = dict()
        keysToGrab = ['obsheight', 'n_showers', 'n_use', 'core_range', 'diffuse', 'viewcone',
                      'E_range', 'spectral_index', 'B_total']

        def _areHeadersConsistent(header0, header1):
            comparison = dict()
            for k in keysToGrab:
                value = (header0[k] == header1[k])
                comparison[k] = value if isinstance(value, bool) else all(value)

            return all(comparison)

        for file in self.inputFiles:
            with SimTelFile(file) as f:

                if len(self._mcHeader) == 0:
                    # First file - grabbing parameters
                    self._mcHeader = {key: f.mc_run_headers[0][key] for key in keysToGrab}
                else:
                    # Remaining files - Checking whether the parameters are consistent
                    if not _areHeadersConsistent(self._mcHeader, f.mc_run_headers[0]):
                        msg = 'MC header pamameters from different files are inconsistent'
                        self._logger.error(msg)
                        raise InconsistentInputFile(msg)

        # Calculating number of events
        self._mcHeader['n_events'] = (
            self._mcHeader['n_use'] * self._mcHeader['n_showers'] * self._numberOfFiles
        )

        return
