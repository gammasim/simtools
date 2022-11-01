import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type
from matplotlib.backends.backend_pdf import PdfPages

__all__ = ["SimtelHistograms"]


class BadHistogramFormat(Exception):
    pass


class SimtelHistograms:
    """
    This class handle sim_telarray histograms.
    Histogram files are handled by using eventio library.

    Methods
    -------
    plot_and_save_figures(figName)
        Plot all histograms and save a single pdf file.
    plot_one_histogram(iHist, ax)
        Plot a single histogram referent to the index iHist.

    Attributes
    ----------
    number_of_histograms
        Number of histograms
    combinedHistograms
        List of histogram data.
    """

    def __init__(self, histogramFiles, test=False):
        """
        SimtelHistograms

        Parameters
        ----------
        histogramFiles: list
            List of sim_telarray histogram files (str of Path).
        test: bool
            If True, only a fraction of the histograms will be processed, leading to \
        a much shorter runtime.
        """
        self._logger = logging.getLogger(__name__)
        self._histogramFiles = histogramFiles
        self._isTest = test

    def plot_and_save_figures(self, figName):
        """
        Plot all histograms and save a single pdf file.

        Parameters
        ----------
        figName: str
            Name of the output figure file.
        """
        self._combine_histogram_files()
        self._plot_combined_histograms(figName)

    @property
    def number_of_histograms(self):
        """Returns number of histograms."""
        if not hasattr(self, "combinedHists"):
            self._combine_histogram_files()
        return len(self.combinedHists)

    def get_histogram_title(self, iHist):
        """
        Returns the title of the histogram with index iHist.

        Parameters
        ----------
        iHist: int
            Histogram index.

        Returns
        -------
        str: histogram title
        """
        if not hasattr(self, "combinedHists"):
            self._combine_histogram_files()
        return self.combinedHists[iHist]["title"]

    def _combine_histogram_files(self):
        """Combine histograms from all files into one single list of histograms."""
        # Processing and combining histograms from multiple files
        self.combinedHists = list()

        nFiles = 0
        for file in self._histogramFiles:

            countFile = True
            with EventIOFile(file) as f:

                for o in yield_toplevel_of_type(f, Histograms):
                    try:
                        hists = o.parse()
                    except Exception:
                        self._logger.warning("Problematic file {}".format(file))
                        countFile = False
                        continue

                    if len(self.combinedHists) == 0:
                        # First file
                        self.combinedHists = copy.copy(hists)

                    else:
                        # Remaining files
                        for hist, thisCombinedHist in zip(hists, self.combinedHists):

                            # Checking consistency of histograms
                            for key_to_test in [
                                "lower_x",
                                "upper_x",
                                "n_bins_x",
                                "title",
                            ]:
                                if hist[key_to_test] != thisCombinedHist[key_to_test]:
                                    msg = "Trying to add histograms with inconsistent dimensions"
                                    self._logger.error(msg)
                                    raise BadHistogramFormat(msg)

                            thisCombinedHist["data"] = np.add(
                                thisCombinedHist["data"], hist["data"]
                            )

                    nFiles += int(countFile)

        self._logger.debug("End of reading {} files".format(nFiles))

        return

    def _plot_combined_histograms(self, figName):
        """
        Plot all histograms into pdf pages and save the figure as a pdf file.

        Parameters
        ----------
        figName: str
            Name of the output figure file.
        """

        pdfPages = PdfPages(figName)
        for iHist, histo in enumerate(self.combinedHists):

            # Test case: processing only 1/10 of the histograms
            if self._isTest and iHist % 10 != 0:
                self._logger.debug("Skipping (test=True): {}".format(histo["title"]))
                continue

            self._logger.debug("Processing: {}".format(histo["title"]))

            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()

            self.plot_one_histogram(iHist, ax)

            plt.tight_layout()
            pdfPages.savefig(fig)
            plt.close()

        pdfPages.close()

    def plot_one_histogram(self, iHist, ax):
        """
        Plot a single histogram referent to the index iHist.

        Parameters
        ----------
        iHist: int
            Index of the histogram to be plotted.
        ax: matplotlib.axes.Axes
            Axes in which to plot the histogram.
        """

        hist = self.combinedHists[iHist]
        ax.set_title(hist["title"])

        def _get_bins(hist, axis=0):
            ax_str = "x" if axis == 0 else "y"
            return np.linspace(
                hist["lower_" + ax_str],
                hist["upper_" + ax_str],
                hist["n_bins_" + ax_str] + 1,
            )

        def _get_ax_lim(hist, axis=0):
            if np.sum(hist["data"]) == 0:
                return 0, 1

            bins = _get_bins(hist, axis=axis)

            if hist["data"].ndim == 1:
                non_zero = np.where(hist["data"] != 0)
            else:
                marginal = np.sum(hist["data"], axis=axis)
                non_zero = np.where(marginal != 0)

            return bins[non_zero[0][0]], bins[non_zero[0][-1] + 1]

        if hist["n_bins_y"] > 0:
            # 2D histogram

            xlim = _get_ax_lim(hist, axis=0)
            ylim = _get_ax_lim(hist, axis=1)

            if np.sum(hist["data"]) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "EMPTY",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                return

            x_bins = _get_bins(hist, axis=0)
            y_bins = _get_bins(hist, axis=1)

            ax.pcolormesh(x_bins, y_bins, hist["data"])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        else:
            # 1D histogram

            xlim = _get_ax_lim(hist, axis=0)

            if np.sum(hist["data"]) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "EMPTY",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                return

            x_bins = _get_bins(hist, axis=0)
            centers = 0.5 * (x_bins[:-1] + x_bins[1:])
            ax.hist(centers, bins=x_bins, weights=hist["data"])
            ax.set_xlim(xlim)
        return
