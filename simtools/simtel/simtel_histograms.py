import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type
from matplotlib.backends.backend_pdf import PdfPages

__all__ = ["BadHistogramFormat", "SimtelHistograms"]


class BadHistogramFormat(Exception):
    """Exception for bad histogram format."""

    pass


class SimtelHistograms:
    """
    This class handle sim_telarray histograms. Histogram files are handled by using eventio library.

    Parameters
    ----------
    histogram_files: list
        List of sim_telarray histogram files (str of Path).
    test: bool
        If True, only a fraction of the histograms will be processed, leading to a much shorter\
         runtime.
    """

    def __init__(self, histogram_files, test=False):
        """
        Initialize SimtelHistograms
        """
        self._logger = logging.getLogger(__name__)
        self._histogram_files = histogram_files
        self._is_test = test

    def plot_and_save_figures(self, fig_name):
        """
        Plot all histograms and save a single pdf file.

        Parameters
        ----------
        fig_name: str
            Name of the output figure file.
        """
        self._combine_histogram_files()
        self._plot_combined_histograms(fig_name)

    @property
    def number_of_histograms(self):
        """Returns number of histograms."""
        if not hasattr(self, "combined_hists"):
            self._combine_histogram_files()
        return len(self.combined_hists)

    def get_histogram_title(self, i_hist):
        """
        Returns the title of the histogram with index i_hist.

        Parameters
        ----------
        i_hist: int
            Histogram index.

        Returns
        -------
        str
            Histogram title.
        """
        if not hasattr(self, "combined_hists"):
            self._combine_histogram_files()
        return self.combined_hists[i_hist]["title"]

    def _combine_histogram_files(self):
        """Combine histograms from all files into one single list of histograms."""
        # Processing and combining histograms from multiple files
        self.combined_hists = list()

        n_files = 0
        for file in self._histogram_files:

            count_file = True
            with EventIOFile(file) as f:

                for o in yield_toplevel_of_type(f, Histograms):
                    try:
                        hists = o.parse()
                    except Exception:
                        self._logger.warning("Problematic file {}".format(file))
                        count_file = False
                        continue

                    if len(self.combined_hists) == 0:
                        # First file
                        self.combined_hists = copy.copy(hists)

                    else:
                        # Remaining files
                        for hist, this_combined_hist in zip(hists, self.combined_hists):

                            # Checking consistency of histograms
                            for key_to_test in [
                                "lower_x",
                                "upper_x",
                                "n_bins_x",
                                "title",
                            ]:
                                if hist[key_to_test] != this_combined_hist[key_to_test]:
                                    msg = "Trying to add histograms with inconsistent dimensions"
                                    self._logger.error(msg)
                                    raise BadHistogramFormat(msg)

                            this_combined_hist["data"] = np.add(
                                this_combined_hist["data"], hist["data"]
                            )

                    n_files += int(count_file)

        self._logger.debug("End of reading {} files".format(n_files))

        return

    def _plot_combined_histograms(self, fig_name):
        """
        Plot all histograms into pdf pages and save the figure as a pdf file.

        Parameters
        ----------
        fig_name: str
            Name of the output figure file.
        """

        pdf_pages = PdfPages(fig_name)
        for i_hist, histo in enumerate(self.combined_hists):

            # Test case: processing only 1/10 of the histograms
            if self._is_test and i_hist % 10 != 0:
                self._logger.debug("Skipping (test=True): {}".format(histo["title"]))
                continue

            self._logger.debug("Processing: {}".format(histo["title"]))

            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()

            self.plot_one_histogram(i_hist, ax)

            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close()

        pdf_pages.close()

    def plot_one_histogram(self, i_hist, ax):
        """
        Plot a single histogram referent to the index i_hist.

        Parameters
        ----------
        i_hist: int
            Index of the histogram to be plotted.
        ax: matplotlib.axes.Axes
            Instance of matplotlib.axes.Axes in which to plot the histogram.
        """

        hist = self.combined_hists[i_hist]
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
