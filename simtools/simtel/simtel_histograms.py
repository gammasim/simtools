import copy
import logging

import numpy as np
from astropy import units as u
from ctao_cosmic_ray_spectra.spectral import PowerLaw, irfdoc_proton_spectrum
from ctapipe.io import write_table
from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type
from eventio.simtel import MCRunHeader

from simtools import version
from simtools.io_operations.hdf5_handler import fill_hdf5_table
from simtools.utils.names import sanitize_name

__all__ = ["InconsistentHistogramFormat", "SimtelHistograms"]


class InconsistentHistogramFormat(Exception):
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
        self._combined_hists = None
        self.__meta_dict = None
        self._config = None
        self._initialize_lists()

    @property
    def number_of_histograms(self):
        """Returns number of histograms."""
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
        return self.combined_hists[i_hist]["title"]

    def _initialize_lists(self):
        """
        Initializes lists of histograms and files.

        Returns
        -------
        list:
            List of histograms.
        """
        self.list_of_histograms = []
        self.list_of_files = []
        for file in self._histogram_files:
            self.list_of_files.append(EventIOFile(file))
            with EventIOFile(file) as f:
                for obj in yield_toplevel_of_type(f, Histograms):
                    hists = obj.parse()
                    self.list_of_histograms.append(hists)

    @property
    def config(self):
        """
        Returns information about the input parameters for the simulation.

        Returns
        -------
        dict:
            dictionary with information about the simulation (pyeventio MCRunHeader object).
        """
        if self._config is None:
            for readfile in self.list_of_files:
                with readfile as f:
                    for obj in f:
                        if isinstance(obj, MCRunHeader):
                            self._config = obj.parse()
        return self._config

    @property
    def total_num_simulated_events(self):
        """
        Returns the total number of simulated events.
        It already accounts for the NSHOW and NUSE found used in the configuration to produce
        the histograms.

        Returns
        -------
        int:
            total number of simulated events.
        """
        print(self.config["n_showers"], self.config["n_use"], self.config["n_showers"] * self.config["n_use"])
        return self.config["n_showers"] * self.config["n_use"]

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
        InconsistentHistogramFormat:
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
                raise InconsistentHistogramFormat(msg)

    @property
    def combined_hists(self):
        """Add the values of the same type of histogram from the various lists into a single
        histogram list."""
        # Processing and combining histograms from multiple files
        if self._combined_hists is None:
            self._combined_hists = []
            for i_hist, hists_one_file in enumerate(self.list_of_histograms):
                if i_hist == 0:
                    # First file
                    self._combined_hists = copy.copy(hists_one_file)

                else:
                    for hist, this_combined_hist in zip(hists_one_file, self._combined_hists):
                        self._check_consistency(hist, this_combined_hist)

                        this_combined_hist["data"] = np.add(
                            this_combined_hist["data"], hist["data"]
                        )

            self._logger.debug(f"End of reading {len(self.list_of_histograms)} files")
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

    def _derive_trigger_rate_histograms(self, livetime):
        """
        Calculates the trigger rate histograms, i.e., the ratio in which the events
        are triggered in each bin of impact distance and log energy for each histogram file for
        the livetime defined by `livetime`.
        The livetime gives the amount of time used in a small production to produce the histograms
        used. It is assumed that the livetime is the same for all the histogram files used and that
        the radius (x-axis in the histograms) is given in meters.

        Parameters
        ----------
        livetime: astropy.Quantity
            Time used in the simulation that produced the histograms. E.g., 1*u.h.

        Returns
        -------
        list:
            List with the trigger rate histograms for each file.
        """
        if isinstance(livetime, u.Quantity):
            livetime = livetime.to(u.s)
        else:
            livetime = livetime * u.s
        events_histogram = {}
        trigged_events_histogram = {}
        # Save the appropriate histograms to a dictionary
        for i_file, hists_one_file in enumerate(self.list_of_histograms):
            for hist in hists_one_file:
                if hist["id"] == 1:
                    events_histogram[i_file] = hist

                elif hist["id"] == 2:
                    trigged_events_histogram[i_file] = hist

        list_of_trigger_rate_hists = []

        # Calculate the event rate histograms
        for i_file, hists_one_file in enumerate(self.list_of_histograms):

            view_cone = self.config["viewcone"] * u.deg
            logging.info(f"View cone: {view_cone.value} deg")

            energy_range = [self.config["E_range"][0] * u.TeV, self.config["E_range"][1] * u.TeV]
            logging.info(f"Energy range: {energy_range}")

            total_area = np.pi * (((self.config["core_range"][1] - self.config["core_range"][0])
                                   * u.m).to(u.cm)) ** 2
            logging.debug(f"Min. core range: {self.config['core_range'][0]} m")
            logging.debug(f"Max. core range: {self.config['core_range'][1]} m")
            logging.info(f"Total area: {(total_area.to(u.m**2)).value} m2")

            obs_time = self.estimate_observation_time(view_cone, energy_range, total_area)
            logging.info(f"Estimated observation time: {obs_time.value} s")

            radius_axis = np.linspace(
                events_histogram[i_file]["lower_x"],
                events_histogram[i_file]["upper_x"],
                events_histogram[i_file]["n_bins_x"] + 1,
                endpoint=True,
            )
            energy_axis = np.logspace(
                events_histogram[i_file]["lower_y"],
                events_histogram[i_file]["upper_y"],
                events_histogram[i_file]["n_bins_y"] + 1,
                endpoint=True,
            )

            event_ratio_histogram = copy.copy(events_histogram[i_file])

            event_ratio_histogram["data"] = np.zeros_like(trigged_events_histogram[i_file]["data"])
            bins_with_events = trigged_events_histogram[i_file]["data"] != 0

            # Radial distribution of triggered events per E divided by the radial distribution
            # of simulated events per E (gives a radial distribution of trigger probability per E
            event_ratio_histogram["data"][bins_with_events] = (
                trigged_events_histogram[i_file]["data"][bins_with_events]
                / events_histogram[i_file]["data"][bins_with_events]
            )

            # TODO: apply any correction factor here (energy and area distribution)

            # Radial distribution of trigger probability per E integrated in area at each radius
            # (gives a trigger probability per E)
            integrated_event_ratio_per_energy = np.zeros_like(energy_axis[:-1])
            areas = np.pi * np.diff(radius_axis**2)

            for i_energy, _ in enumerate(energy_axis[:-1]):
                integrated_event_ratio_per_energy[i_energy] = np.sum(
                    event_ratio_histogram["data"][bins_with_events][i_energy] * areas
                )

            # Trigger probability per E integrated in E
            # (gives a trigger probability, i.e. a normalization)
            hist_normalization = np.sum(integrated_event_ratio_per_energy * np.diff(energy_axis))

            if self.config["diffuse"] == 1:
                norm_unit = 1 / (u.m**2 * u.s * u.sr * u.TeV)
            else:
                norm_unit = 1 / (u.m**2 * u.s * u.TeV)


            non_norm_simulated_power_law_function = PowerLaw(
                normalization=1 * norm_unit, index=self.config["spectral_index"], e_ref=1 * u.TeV
            )
            non_norm_simulated_events_rate = (
                non_norm_simulated_power_law_function.derive_events_rate(
                    inner=view_cone[0],
                    outer=view_cone[1],
                    area=total_area,
                    energy_min=energy_range[0],
                    energy_max=energy_range[1],
                )
            )

            factor = self.total_num_simulated_events / non_norm_simulated_events_rate.value
            norm_simulated_power_law_function = PowerLaw(
                normalization=factor * norm_unit,
                index=self.config["spectral_index"],
                e_ref=1 * u.TeV,
            )
            print(norm_simulated_power_law_function)
            print(self.total_num_simulated_events, norm_simulated_power_law_function.derive_events_rate(
                    inner=view_cone[0],
                    outer=view_cone[1],
                    area=total_area,
                    energy_min=energy_range[0],
                    energy_max=energy_range[1],
                ))

            # Keeping only the necessary information for proceeding with integration
            keys_to_keep = [
                "data",
                "lower_x",
                "lower_y",
                "upper_x",
                "upper_y",
                "entries",
                "n_bins_x",
                "n_bins_y",
            ]
            event_ratio_histogram = {
                key: event_ratio_histogram[key]
                for key in keys_to_keep
                if key in event_ratio_histogram
            }

            list_of_trigger_rate_hists.append(event_ratio_histogram)
        return list_of_trigger_rate_hists

    def estimate_observation_time(self, view_cone, energy_range, total_area):
        """
        Estimates the observation time comprised by the number of events in the simulation.
        It uses the CTAO reference cosmic-ray spectra, the total number of particles simulated,
        and other information from the simulation configuration `self.config`.

        Parameters
        ----------
        view_cone: list of astropy.Quantity[deg]
            The view cone used in the simulation.
        energy_range: list of astropy.Quantity[energy]
            The energy range [Emin, Emax] used in the simulation.
        total_area: astropy.Quantity[area]
            Total ground area used in the simulation (CSCAT).

        Returns
        -------
        float: astropy.Quantity[time]
            Estimated observation time based on the total number of particles simulated.
        """
        first_estimate = irfdoc_proton_spectrum.derive_number_events(
            view_cone[0], view_cone[1], 1*u.s, total_area, energy_range[0],energy_range[1]
        )
        return self.total_num_simulated_events/first_estimate



    def trigger_rate_per_histogram(self, livetime):
        """
        Estimates the trigger rate for each histogram passed.

        Parameters
        ----------
        livetime: astropy.Quantity[time]
            Time used in the simulation that produced the histograms.
        """
        list_of_trigger_rate_hists = self._derive_trigger_rate_histograms(livetime=livetime)
        return list_of_trigger_rate_hists

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
