import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type

__all__ = ['SimtelHistograms']


class SimtelHistograms:
    '''
    '''

    def __init__(
        self,
        histogramFiles
    ):
        '''
        '''
        self._logger = logging.getLogger(__name__)

        self.histogramFiles = histogramFiles

    def plotAndSaveFigures(self, figName):
        allHists = self._combineHistogramFiles()

        print(allHists)

    def _combineHistogramFiles(self):

        # Processing and combining histograms from multiple files
        combinedHists = list()

        n_files = 0
        for file in self.histogramFiles:

            count_file = 1
            with EventIOFile(file) as f:

                for i, o in enumerate(yield_toplevel_of_type(f, Histograms)):
                    try:
                        hists = o.parse()
                    except Exception:
                        print('Problematic file {}'.format(file))
                        count_file = 0
                        continue

                    if len(combinedHists) == 0:
                        # First file
                        combinedHists = copy.copy(hists)

                    else:
                        # Remaning files
                        for hist, thisCombinedHist in zip(hists, combinedHists):

                            # Checking consistency of histograms
                            for key_to_test in ['lower_x', 'upper_x', 'n_bins_x', 'title']:
                                if hist[key_to_test] != thisCombinedHist[key_to_test]:
                                    print('WARNING: {} is not consistency'.format(key_to_test))

                            thisCombinedHist['data'] = np.add(
                                thisCombinedHist['data'],
                                hist['data']
                            )

                    n_files += count_file

        return combinedHists
