import copy
import logging

import numpy as np
from ctapipe.io import write_table
from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type

from simtools import version
from simtools.io_operations.hdf5_handler import fill_hdf5_table
from simtools.utils.names import sanitize_name

__all__ = ["BadHistogramFormat", "SimtelHistograms"]


class BadHistogramFormat(Exception):
    """Exception for bad histogram format."""


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
        if not isinstance(histogram_files, list):
            histogram_files = [histogram_files]
        self._histogram_files = histogram_files
        self._is_test = test
        self.combined_hists = None
        self.__meta_dict = None

    @property
    def number_of_histograms(self):
        """Returns number of histograms."""
        if self.combined_hists is None:
            self.combine_histogram_files()
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
        if self.combined_hists is None:
            self.combine_histogram_files()
        return self.combined_hists[i_hist]["title"]

    def get_list_of_histograms(self):
        """Returns a list with the histograms for each file.

        Returns
        -------
        list:
            List of all histograms read from the files.

        """

        list_of_histograms = []
        for file in self._histogram_files:
            with EventIOFile(file) as f:
                for o in yield_toplevel_of_type(f, Histograms):
                    try:
                        hists = o.parse()
                    except Exception:  # pylint: disable=broad-except
                        self._logger.warning(f"Problematic file {file}")
                        continue
                    list_of_histograms.append(hists)

        return list_of_histograms

    def _check_consistency(self, first_hist_file, second_hist_file):
        """
        Checks whether two histograms have the same format.
        Raises an error in case they are not consistent.

        Parameters
        ----------
        first_hist_file: dict
            One histogram from a single file.
        second_hist_file: dict
            One histogram from a single file.

        Raises
        ------
        BadHistogramFormat:
            if the format of the histograms have inconsistent dimensions.
        """
        for key_to_test in [
            "lower_x",
            "upper_x",
            "n_bins_x",
            "title",
        ]:
            if first_hist_file[key_to_test] != second_hist_file[key_to_test]:
                msg = "Trying to add histograms with inconsistent dimensions"
                self._logger.error(msg)
                raise BadHistogramFormat(msg)

    def combine_histogram_files(self):
        """Add the values of the same type of histogram from the various lists into a single
        histogram list."""
        # Processing and combining histograms from multiple files
        self.combined_hists = []
        list_of_histograms = self.get_list_of_histograms()
        n_files = 0
        for hists_one_file in list_of_histograms:
            count_file = True

            if len(self.combined_hists) == 0:
                # First file
                self.combined_hists = copy.copy(hists_one_file)

            else:
                for hist, this_combined_hist in zip(hists_one_file, self.combined_hists):
                    self._check_consistency(hist, this_combined_hist)

                    this_combined_hist["data"] = np.add(this_combined_hist["data"], hist["data"])

            n_files += int(count_file)

        self._logger.debug(f"End of reading {n_files} files")

    def derive_trigger_rate_histogram(self):
        """
        Calculates the trigger rate histograms.
        The estimate is based on the existing histograms defined by the id=1 and id=2 in
        `pyeventio`.
        """

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

    @property
    def _meta_dict(self):
        """
        Define the meta dictionary for exporting the histograms.

        Returns
        -------
        dict
            Meta dictionary for the hdf5 files with the histograms.
        """

        if self.__meta_dict is None:
            self.__meta_dict = {
                "simtools_version": version.__version__,
                "note": "Only lower bin edges are given.",
            }
        return self.__meta_dict

    def export_histograms(self, hdf5_file_name, overwrite=False):
        """
        Export the histograms to hdf5 files.

        Parameters
        ----------
        hdf5_file_name: str
            Name of the file to be saved with the hdf5 tables.
        overwrite: bool
            If True overwrites the histograms already saved in the hdf5 file.
        """
        for histogram in self.combined_hists:
            x_bin_edges_list = np.linspace(
                histogram["lower_x"],
                histogram["upper_x"],
                num=histogram["n_bins_x"] + 1,
                endpoint=True,
            )
            if histogram["n_bins_y"] > 0:
                y_bin_edges_list = np.linspace(
                    histogram["lower_y"],
                    histogram["upper_y"],
                    num=histogram["n_bins_y"] + 1,
                    endpoint=True,
                )
            else:
                y_bin_edges_list = None

            self._meta_dict["Title"] = sanitize_name(histogram["title"])

            table = fill_hdf5_table(
                hist=histogram["data"],
                x_bin_edges=x_bin_edges_list,
                y_bin_edges=y_bin_edges_list,
                x_label=None,
                y_label=None,
                meta_data=self._meta_dict,
            )

            self._logger.debug(
                f"Writing histogram with name {self._meta_dict['Title']} to " f"{hdf5_file_name}."
            )
            # overwrite takes precedence over append
            if overwrite is True:
                append = False
            else:
                append = True
            write_table(
                table,
                hdf5_file_name,
                f"/{self._meta_dict['Title']}",
                append=append,
                overwrite=overwrite,
            )
