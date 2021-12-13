import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type

__all__ = ['SimtelHistograms']


class BadHistogramFormat(Exception):
    pass


class SimtelHistograms:
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
        histogramFiles
    ):
        '''
        SimtelHistograms

        Parameters
        ----------
        histogramFiles: list
            List of sim_telarray histogram files (str of Path).

        '''
        self._logger = logging.getLogger(__name__)
        self._histogramFiles = histogramFiles

    def plotAndSaveFigures(self, figName):
        '''
        Plot all histograms and save a single pdf file.

        Parameters
        ----------
        figName: str
            Name of the output figure file.
        '''
        self._combineHistogramFiles()
        self._plotCombinedHistograms(figName)

    @property
    def numberOfHistograms(self):
        if not isattr(self, 'combinedHists'):
            self._combineHistogramFiles()
        return len(self.combinedHists)

    def _combineHistogramFiles(self):
        ''' Combine histograms from all files into one single list of histograms. '''
        # Processing and combining histograms from multiple files
        self.combinedHists = list()

        nFiles = 0
        for file in self._histogramFiles:

            countFile = True
            with EventIOFile(file) as f:

                for i, o in enumerate(yield_toplevel_of_type(f, Histograms)):
                    try:
                        hists = o.parse()
                    except Exception:
                        self._logger.warning('Problematic file {}'.format(file))
                        countFile = False
                        continue

                    if len(self.combinedHists) == 0:
                        # First file
                        self.combinedHists = copy.copy(hists)

                    else:
                        # Remaning files
                        for hist, thisCombinedHist in zip(hists, self.combinedHists):

                            # Checking consistency of histograms
                            for key_to_test in ['lower_x', 'upper_x', 'n_bins_x', 'title']:
                                if hist[key_to_test] != thisCombinedHist[key_to_test]:
                                    msg = 'Trying to add histograms with inconsistent dimensions'
                                    self._logger.error(msg)
                                    raise BadHistogramFormat(msg)

                            thisCombinedHist['data'] = np.add(
                                thisCombinedHist['data'],
                                hist['data']
                            )

                    nFiles += int(countFile)

        self._logger.debug('End of reading {} files'.format(nFiles))

        return

    def _plotCombinedHistograms(self, figName):
        '''
        Plot all histograms into pdf pages and save the figure as a pdf file.

        Parameters
        ----------
        figName: str
            Name of the output figure file.
        '''

        pdfPages = PdfPages(figName)
        for iHist in range(len(self.combinedHists)):

            self._logger.debug('Processing: {}'.format(self.combinedHists[iHist]['title']))

            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()

            self.plotOneHistogram(iHist, ax)

            plt.tight_layout()
            pdfPages.savefig(fig)
            plt.close()

        pdfPages.close()


    def plotOneHistogram(self, iHist, ax):

        hist = self.combinedHists[iHist]
        ax.set_title(hist['title'])

        def _get_bins(hist, axis=0):
            ax_str = 'x' if axis == 0 else 'y'
            return np.linspace(
                hist['lower_' + ax_str],
                hist['upper_' + ax_str],
                hist['n_bins_' + ax_str] + 1
            )

        def _get_ax_lim(hist, axis=0):
            if np.sum(hist['data']) == 0:
                return 0, 1

            bins = _get_bins(hist, axis=axis)

            if hist['data'].ndim == 1:
                non_zero = np.where(hist['data'] != 0)
            else:
                marginal = np.sum(hist['data'], axis=axis)
                non_zero = np.where(marginal != 0)

            return bins[non_zero[0][0]], bins[non_zero[0][-1] + 1]


        if hist['n_bins_y'] > 0:
            # 2D histogram

            xlim = _get_ax_lim(hist, axis=0)
            ylim = _get_ax_lim(hist, axis=1)

            if np.sum(hist['data']) == 0:
                ax.text(
                    0.5,
                    0.5,
                    'EMPTY',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes
                )
                return

            x_bins = _get_bins(hist, axis=0)
            y_bins = _get_bins(hist, axis=1)

            ax.pcolormesh(x_bins, y_bins, hist['data'])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        else:
            # 1D histogram

            xlim = _get_ax_lim(hist, axis=0)

            if np.sum(hist['data']) == 0:
                ax.text(
                    0.5,
                    0.5,
                    'EMPTY',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes
                )
                return

            x_bins = _get_bins(hist, axis=0)
            centers = 0.5 * (x_bins[:-1] + x_bins[1:])
            ax.hist(centers, bins=x_bins, weights=hist['data'])
            ax.set_xlim(xlim)
        return
    # End of plotOneHistogram
