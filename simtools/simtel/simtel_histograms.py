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
        combinedHists = self._combineHistogramFiles()
        self._plotCombinedHistograms(combinedHists, figName)

    def _combineHistogramFiles(self):

        # Processing and combining histograms from multiple files
        combinedHists = list()

        nFiles = 0
        for file in self.histogramFiles:

            countFile = 1
            with EventIOFile(file) as f:

                for i, o in enumerate(yield_toplevel_of_type(f, Histograms)):
                    try:
                        hists = o.parse()
                    except Exception:
                        self._logger.warning('Problematic file {}'.format(file))
                        countFile = 0
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
                                    self._logger.warning(
                                        'WARNING: {} is not consistency'.format(key_to_test)
                                    )

                            thisCombinedHist['data'] = np.add(
                                thisCombinedHist['data'],
                                hist['data']
                            )

                    nFiles += countFile

        self._logger.debug('End of reading {} files'.format(nFiles))

        return combinedHists

    def _plotCombinedHistograms(self, combinedHists, figName):

        def _get_bins(hist, axis=0):
            ax_str = 'x' if axis == 0 else 'y'
            return np.linspace(
                hist['lower_' + ax_str],
                hist['upper_' + ax_str],
                hist['n_bins_' + ax_str] + 1
            )

        def _get_ax_lim(hist, axis=0):
            if np.sum(hist['data']) < 1:
                return 0, 1

            bins = _get_bins(hist, axis=axis)

            if hist['data'].ndim == 1:
                non_zero = np.where(hist['data'] != 0)
            else:
                marginal = np.sum(hist['data'], axis=axis)
                non_zero = np.where(marginal != 0)

            return bins[non_zero[0][0]], bins[non_zero[0][-1] + 1]

        pdfPages = PdfPages(figName)
        for hist in combinedHists:

            self._logger.debug('Processing: {}'.format(hist['title']))

            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()
            ax.set_title(hist['title'])

            if hist['n_bins_y'] > 0:
                # 2D histogram

                xlim = _get_ax_lim(hist, axis=0)
                ylim = _get_ax_lim(hist, axis=1)

                if np.sum(hist['data']) < 1:
                    ax.text(
                        0.5,
                        0.5,
                        'EMPTY',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes
                    )
                    continue

                x_bins = _get_bins(hist, axis=0)
                y_bins = _get_bins(hist, axis=1)

                ax.pcolormesh(x_bins, y_bins, hist['data'])
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            else:
                # 1D histogram

                xlim = _get_ax_lim(hist, axis=0)

                if np.sum(hist['data']) < 1:
                    ax.text(
                        0.5,
                        0.5,
                        'EMPTY',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes
                    )
                    continue

                x_bins = _get_bins(hist, axis=0)
                centers = 0.5 * (x_bins[:-1] + x_bins[1:])
                ax.hist(centers, bins=x_bins, weights=hist['data'])
                ax.set_xlim(xlim)

            plt.tight_layout()
            pdfPages.savefig(fig)
            plt.close()

        pdfPages.close()
    # End
