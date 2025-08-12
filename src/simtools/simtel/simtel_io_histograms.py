"""Reads the content of multiples files from sim_telarray."""

import copy
import logging

import numpy as np
from ctapipe.io import write_table
from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type

from simtools import version
from simtools.io.hdf5_handler import fill_hdf5_table
from simtools.simtel.simtel_io_histogram import (
    HistogramIdNotFoundError,
    InconsistentHistogramFormatError,
    SimtelIOHistogram,
)
from simtools.utils.names import sanitize_name

__all__ = [
    "SimtelIOHistograms",
]


class SimtelIOHistograms:
    """
    Read the content of either multiple histogram (.hdata, or .hdata.zst) or sim_telarray files.

    Allow both the .hdata.zst histogram and the .simtel.zst output file type.
    It uses the SimtelIOHistogram class to deal with individual files.
    Histogram files are ultimately handled by using eventio library.

    Parameters
    ----------
    histogram_files: list
        List of sim_telarray histogram files (str of Path).
    test: bool
        If True, only a fraction of the histograms will be processed, leading to a much shorter\
         runtime.
    area_from_distribution: bool
        If true, the area thrown (the area in which the simulated events are distributed)
        in the trigger rate calculation is estimated based on the event distribution.
        The expected shape of the distribution of events as function of the core distance is
        triangular up to the maximum distance. The weighted mean radius of the triangular
        distribution is 2/3 times the upper edge. Therefore, when using the
        ``area_from_distribution`` flag, the mean distance times 3/2, returns just the position of
        the upper edge in the triangle distribution with little impact of the binning and little
        dependence on the scatter area defined in the simulation. This is special useful when
        calculating trigger rate for individual telescopes.
        If false, the area thrown is estimated based on the maximum distance as given in
        the simulation configuration.
    energy_range: list
        The energy range used in the simulation. It must be passed as a list of floats and the
        energy must be in TeV (as in the CORSIKA configuration).
        This argument is only needed and used if histogram_file is a .hdata file, in which case the
        energy range cannot be retrieved directly from the file.
    view_cone: list
        The view cone used in the simulation. It must be passed as a list of floats and the
        view cone must be in deg (as in the CORSIKA configuration).
        This argument is only needed and used if histogram_file is a .hdata file, in which case the
        view cone cannot be retrieved directly from the file.
    """

    def __init__(
        self,
        histogram_files,
        test=False,
        area_from_distribution=False,
        energy_range=None,
        view_cone=None,
    ):
        """Initialize SimtelIOHistograms."""
        self._logger = logging.getLogger(__name__)
        if not isinstance(histogram_files, list):
            histogram_files = [histogram_files]
        self.histogram_files = histogram_files
        self.view_cone = view_cone
        self.energy_range = energy_range
        self._is_test = test
        self._combined_hists = None
        self._list_of_histograms = None
        self.__meta_dict = None
        self.area_from_distribution = area_from_distribution

    def calculate_trigger_rates(self, print_info=False, stack_files=False):
        """
        Calculate the triggered and simulated event rate considering the histograms in each file.

        It returns also a list with the tables where the energy dependent trigger rate for each
        file can be found.

        Parameters
        ----------
        print_info: bool
            if True, prints out the information about the histograms such as energy range, area,
            etc.
        stack_files: bool
            if True, stack the histograms from the different files into single histograms.
            Useful to increase event statistics when calculating the trigger rate.

        Returns
        -------
        sim_event_rates: list of astropy.Quantity[1/time]
            The simulated event rates.
        triggered_event_rates: list of astropy.Quantity[1/time]
            The triggered event rates.
        triggered_event_rate_uncertainties: list of astropy.Quantity[1/time]
            The uncertainties in the triggered event rates.
        trigger_rate_in_tables: list of astropy.QTable
            The energy dependent trigger rates.
            Only filled if stack_files is False.
        """
        if stack_files:
            (
                sim_event_rates,
                triggered_event_rates,
                triggered_event_rate_uncertainties,
            ) = self._rates_for_stacked_files()
            trigger_rate_in_tables = []
        else:
            (
                sim_event_rates,
                triggered_event_rates,
                triggered_event_rate_uncertainties,
                trigger_rate_in_tables,
            ) = self._rates_for_each_file(print_info)

        return (
            sim_event_rates,
            triggered_event_rates,
            triggered_event_rate_uncertainties,
            trigger_rate_in_tables,
        )

    def _fill_stacked_events(self):
        """
        Retrieve the simulated and triggered event histograms from the stacked histograms instead.

        Returns
        -------
        first_hist_file: dict
            The simulated 2D event histogram.
        second_hist_file: dict
            The triggered 2D event histogram.

        Raises
        ------
        HistogramIdNotFoundError:
            if histogram ids not found. Problem with the file.
        """
        sim_hist = None
        trig_hist = None
        for _, one_hist in enumerate(self.combined_hists):
            if one_hist["id"] == 1:
                sim_hist = one_hist
            elif one_hist["id"] == 2:
                trig_hist = one_hist

        if sim_hist is None or trig_hist is None:
            msg = (
                "Simulated and triggered histograms were not found in the stacked histograms."
                " Please check sim_telarray files!"
            )
            self._logger.error(msg)
            raise HistogramIdNotFoundError

        return sim_hist, trig_hist

    def get_stacked_num_events(self):
        """
        Return stacked number of simulated events and triggered events.

        Returns
        -------
        int:
            total number of simulated events for the stacked dataset.
        int:
            total number of triggered events for the stacked dataset.
        """
        stacked_num_simulated_events = 0
        stacked_num_triggered_events = 0
        for _, file in enumerate(self.histogram_files):
            simtel_hist_instance = SimtelIOHistogram(
                file,
                area_from_distribution=self.area_from_distribution,
                energy_range=self.energy_range,
                view_cone=self.view_cone,
            )
            _simulated, _triggered = simtel_hist_instance.total_number_of_events
            stacked_num_simulated_events += _simulated
            stacked_num_triggered_events += _triggered
        return stacked_num_simulated_events, stacked_num_triggered_events

    def _rates_for_stacked_files(self):
        """
        Calculate trigger rate for the stacked case.

        Returns
        -------
        sim_event_rates: list of astropy.Quantity[1/time]
            The simulated event rates.
        triggered_event_rates: list of astropy.Quantity[1/time]
            The triggered event rates.
        triggered_event_rate_uncertainties: list of astropy.Quantity[1/time]
            The uncertainties in the triggered event rates.
        trigger_rate_in_tables: list of astropy.QTable
            The energy dependent trigger rates.
            Only filled if stack_files is False.
        """
        logging.info("Estimates for the stacked histograms:")
        sim_hist, trig_hist = self._fill_stacked_events()
        # Using a dummy instance of SimtelIOHistogram to calculate the trigger rate for the
        # stacked files
        simtel_hist_instance = SimtelIOHistogram(
            self.histogram_files[0],
            area_from_distribution=self.area_from_distribution,
            energy_range=self.energy_range,
            view_cone=self.view_cone,
        )

        stacked_num_simulated_events, stacked_num_triggered_events = self.get_stacked_num_events()
        logging.info(f"Total number of simulated events: {stacked_num_simulated_events} events")
        logging.info(f"Total number of triggered events: {stacked_num_triggered_events} events")
        obs_time = simtel_hist_instance.estimate_observation_time(stacked_num_simulated_events)
        logging.info(
            "Estimated equivalent observation time corresponding to the number of"
            f"events simulated: {obs_time.value} s"
        )
        sim_event_rate = stacked_num_simulated_events / obs_time
        logging.info(f"Simulated event rate: {sim_event_rate.value:.4e} Hz")
        (
            triggered_event_rate,
            _,
        ) = simtel_hist_instance.compute_system_trigger_rate(
            events_histogram=sim_hist, triggered_events_histogram=trig_hist
        )
        triggered_event_rate_uncertainty = simtel_hist_instance.estimate_trigger_rate_uncertainty(
            triggered_event_rate, stacked_num_simulated_events, stacked_num_triggered_events
        )
        logging.info(
            f"System trigger event rate for stacked files: "
            f"{triggered_event_rate.value:.4e} \u00b1 "
            f"{triggered_event_rate_uncertainty.value:.4e} Hz"
        )
        return (
            [sim_event_rate],
            [triggered_event_rate],
            [triggered_event_rate_uncertainty],
        )

    def _rates_for_each_file(self, print_info=False):
        """
        Calculate trigger rate for each file.

        Returns
        -------
        sim_event_rates: list of astropy.Quantity[1/time]
            The simulated event rates.
        triggered_event_rates: list of astropy.Quantity[1/time]
            The triggered event rates.
        triggered_event_rate_uncertainties: list of astropy.Quantity[1/time]
            The uncertainties in the triggered event rates.
        """
        triggered_event_rates = []
        sim_event_rates = []
        trigger_rate_in_tables = []
        triggered_event_rate_uncertainties = []
        for i_file, file in enumerate(self.histogram_files):
            simtel_hist_instance = SimtelIOHistogram(
                file,
                area_from_distribution=self.area_from_distribution,
                energy_range=self.energy_range,
                view_cone=self.view_cone,
            )
            if print_info:
                simtel_hist_instance.print_info()

            _simulated_events, _triggered_events = simtel_hist_instance.total_number_of_events
            logging.info(f"Histogram {i_file + 1}:")
            logging.info(f"Total number of simulated events: {_simulated_events} events")
            logging.info(f"Total number of triggered events: {_triggered_events} events")

            obs_time = simtel_hist_instance.estimate_observation_time(_simulated_events)
            logging.info(
                f"Estimated equivalent observation time corresponding to the number of "
                f"events simulated: {obs_time.value} s"
            )
            if obs_time != 0:
                sim_event_rate = _simulated_events / obs_time
            else:
                sim_event_rate = 0.0 * obs_time.unit
                logging.warning("Observation time is zero, cannot calculate event rate.")
            sim_event_rates.append(sim_event_rate)
            logging.info(f"Simulated event rate: {sim_event_rate.value:.4e} Hz")

            (
                triggered_event_rate,
                triggered_event_rate_uncertainty,
            ) = simtel_hist_instance.compute_system_trigger_rate()
            logging.info(
                f"System trigger event rate: "
                # pylint: disable=E1101
                f"{triggered_event_rate.value:.4e} \u00b1 "
                # pylint: disable=E1101
                f"{triggered_event_rate_uncertainty.value:.4e} Hz"
            )
            triggered_event_rates.append(triggered_event_rate)
            triggered_event_rate_uncertainties.append(triggered_event_rate_uncertainty)
            trigger_rate_in_tables.append(simtel_hist_instance.trigger_info_in_table())
        return (
            sim_event_rates,
            triggered_event_rates,
            triggered_event_rate_uncertainties,
            trigger_rate_in_tables,
        )

    @property
    def number_of_files(self):
        """Returns number of histograms."""
        return len(self.histogram_files)

    def _check_consistency(self, first_hist_file, second_hist_file):
        """
        Check whether two histograms have the same format.

        Raises an error in case they are not consistent.

        Parameters
        ----------
        first_hist_file: dict
            One histogram from a single file.
        second_hist_file: dict
            One histogram from a single file.

        Raises
        ------
        InconsistentHistogramFormatError:
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
                raise InconsistentHistogramFormatError

    @property
    def list_of_histograms(self):
        """
        Returns a list with the histograms for each file.

        Returns
        -------
        list:
            List of histograms.
        """
        if self._list_of_histograms is None:
            self._list_of_histograms = []
            for file in self.histogram_files:
                with EventIOFile(file) as f:
                    for o in yield_toplevel_of_type(f, Histograms):
                        hists = o.parse()
                        self._list_of_histograms.append(hists)
        return self._list_of_histograms

    @property
    def combined_hists(self):
        """
        Combine histograms of same type of histogram.

        Histograms are read from various lists into a single histogram list.
        """
        # Processing and combining histograms from multiple files
        if self._combined_hists is None:
            self._combined_hists = []
            for histogram_index, hists_one_file in enumerate(self.list_of_histograms):
                if histogram_index == 0:
                    # First file
                    self._combined_hists = copy.copy(hists_one_file)

                else:
                    for hist, this_combined_hist in zip(hists_one_file, self._combined_hists):
                        self._check_consistency(hist, this_combined_hist)

                        this_combined_hist["data"] = np.add(
                            this_combined_hist["data"], hist["data"]
                        )

            self._logger.debug(f"End of reading {len(self.histogram_files)} files")
        return self._combined_hists

    @combined_hists.setter
    def combined_hists(self, new_combined_hists):
        """
        Setter for combined_hists.

        Parameters
        ----------
        new_combined_hists:
            Combined histograms.
        """
        self._combined_hists = new_combined_hists

    def plot_one_histogram(self, histogram_index, ax):
        """
        Plot a single histogram referent to the index histogram_index.

        Parameters
        ----------
        histogram_index: int
            Index of the histogram to be plotted.
        ax: matplotlib.axes.Axes
            Instance of matplotlib.axes.Axes in which to plot the histogram.
        """
        hist = self.combined_hists[histogram_index]
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
        Export sim_telarray histograms to hdf5 files.

        Parameters
        ----------
        hdf5_file_name: str
            Name of the file to be saved with the hdf5 tables.
        overwrite: bool
            If True overwrites histograms already saved in the hdf5 file.
        """
        self._logger.info(f"Exporting histograms to {hdf5_file_name}.")
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
                f"Writing histogram with name {self._meta_dict['Title']} to {hdf5_file_name}."
            )
            # overwrite takes precedence over append
            write_table(
                table,
                hdf5_file_name,
                f"/{self._meta_dict['Title']}",
                append=not overwrite,
                overwrite=overwrite,
            )
