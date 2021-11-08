import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eventio.simtel import SimTelFile


__all__ = ['SimtelEvents']


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
