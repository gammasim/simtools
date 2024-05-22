import copy
import logging
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import QTable
from ctao_cr_spectra.definitions import IRFDOC_PROTON_SPECTRUM
from ctao_cr_spectra.spectral import cone_solid_angle
from ctapipe.io import write_table
from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type
from eventio.simtel import MCRunHeader

from simtools import version
from simtools.io_operations.hdf5_handler import fill_hdf5_table
from simtools.utils.names import sanitize_name

__all__ = [
    "InconsistentHistogramFormat",
    "HistogramIdNotFound",
    "SimtelHistogram",
    "SimtelHistograms",
]


class InconsistentHistogramFormat(Exception):
    """Exception for bad histogram format."""


class HistogramIdNotFound(Exception):
    """Exception for histogram ID not found."""


class SimtelHistogram:
    """
    This class handles a single histogram (or simtel_array output) file.

    Parameters
    ----------
    histogram_file: str
        The histogram (.hdata.zst) or simtel_array (.simtel.zst) file.
    rht: bool
        If true, the area thrown in the trigger rate calculation is estimated exactly as in the
        hessio rht.cc tool. If false, it is estimated based on the maximum distance as given in
        the simulation configuration.
        Note: The expected shape of the distribution of events as function of the core distance is
        triangular up to the maximum distance.The weighted mean radius of the triangular
        distribution is 2/3 times the upper edge. Thus when using the ``rht`` flag, the mean
        distance times 3/2, returns just the position of the upper edge in the triangle
        distribution with little impact of the binning.

    """

    trigger_rate: None

    def __init__(self, histogram_file, rht=False):
        """
        Initialize SimtelHistogram class.

        """
        self._logger = logging.getLogger(__name__)
        self.histogram_file = histogram_file
        if not Path(histogram_file).exists():
            msg = f"File {histogram_file} does not exist."
            self._logger.error(msg)
            raise FileNotFoundError
        self._config = None
        self._view_cone = None
        self._total_area = None
        self._solid_angle = None
        self._energy_range = None
        self._total_num_simulated_events = None
        self._total_num_triggered_events = None
        self._histogram = None
        self._histogram_file = None
        self._initialize_histogram()
        self.trigger_rate = None
        self.trigger_rate_uncertainty = None
        self.trigger_rate_per_energy_bin = None
        self.energy_axis = None
        self.radius_axis = None
        self.rht = rht

    def _initialize_histogram(self):
        """
        Initializes lists of histograms and files.

        Returns
        -------
        list:
            List of histograms.
        """
        with EventIOFile(self.histogram_file) as f:
            for obj in yield_toplevel_of_type(f, Histograms):
                self.histogram = obj.parse()

    @property
    def number_of_histogram_types(self):
        """Returns number of histograms."""
        return len(self.histogram)

    def get_histogram_type_title(self, histogram_index):
        """
        Returns the title of the histogram with index histogram_index.

        Parameters
        ----------
        histogram_index: int
            Histogram index.

        Returns
        -------
        str
            Histogram title.
        """
        return self.histogram[histogram_index]["title"]

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
            with EventIOFile(self.histogram_file) as f:
                for obj in f:
                    if isinstance(obj, MCRunHeader):
                        self._config = obj.parse()
        return self._config

    @property
    def total_num_simulated_events(self):
        """
        Returns the total number of simulated events.
        the histograms.

        Returns
        -------
        int:
            total number of simulated events.
        """
        if self._total_num_simulated_events is None:
            self._total_num_simulated_events = []
            logging.debug(
                f"Number of simulated showers (CORSIKA NSHOW): {self.config['n_showers']}"
            )
            logging.debug(
                "Number of times each simulated shower is used: " f"{self.config['n_use']}"
            )
            self._total_num_simulated_events = self.config["n_showers"] * self.config["n_use"]
            logging.debug(f"Number of total simulated showers: {self._total_num_simulated_events}")
        return self._total_num_simulated_events

    @property
    def total_num_triggered_events(self):
        """
        Returns the total number of triggered events.
        Please note that this value is not supposed to match the trigger rate x estimated
        observation time, as the simulation is optimized for computational time and the energy
        distribution assumed is not necessarily the reference cosmic-ray spectra.

        Returns
        -------
        int:
            total number of simulated events.
        """

        if self._total_num_triggered_events is None:
            _, triggered_hist = self.fill_event_histogram_dicts()
            self._total_num_triggered_events = np.round(np.sum(triggered_hist["data"]))
            logging.debug(f"Number of triggered events: {self._total_num_triggered_events}")
        return self._total_num_triggered_events

    def fill_event_histogram_dicts(self):
        """
        Get data from the total simulated event and the triggered event histograms.

        Returns
        -------
        dict:
            Information about the histograms with simulated events.
        dict:
            Information about the histograms with triggered events.

        Raises
        ------
        HistogramIdNotFound:
            if histogram ids not found. Problem with the file.
        """
        # Save the appropriate histograms to variables
        found_simulated_events_hist = False
        found_triggered_events_hist = False
        events_histogram = None
        triggered_events_histogram = None
        for hist in self.histogram:
            if hist["id"] == 1:
                events_histogram = hist
                found_simulated_events_hist = True
            elif hist["id"] == 2:
                triggered_events_histogram = hist
                found_triggered_events_hist = True
            if found_simulated_events_hist * found_triggered_events_hist:
                if "triggered_events_histogram" in locals():
                    return events_histogram, triggered_events_histogram
        msg = "Histograms ids not found. Please check your files."

        self._logger.error(msg)
        raise HistogramIdNotFound

    @property
    def view_cone(self):
        """
        View cone used in the simulation.

        Returns
        -------
        list:
            view cone used in the simulation [min, max]
        """
        if self._view_cone is None:
            self._view_cone = self.config["viewcone"] * u.deg
        return self._view_cone

    @property
    def solid_angle(self):
        """
        Solid angle corresponding to the view cone.

        Returns
        -------
        astropy.Quantity[u.sr]:
            Solid angle corresponding to the view cone.
        """
        if self._solid_angle is None:
            self._solid_angle = cone_solid_angle(self.view_cone[1]) - cone_solid_angle(
                self.view_cone[0]
            )
        return self._solid_angle

    @property
    def total_area(self):
        """
        Total area covered by the simulated events (original CORSIKA CSCAT), i.e., area thrown.

        Returns
        -------
        astropy.Quantity[area]:
            Total area covered on the ground covered by the simulation.
        """
        if self._total_area is None:

            if self.rht is True:
                events_histogram, _ = self.fill_event_histogram_dicts()
                self._initialize_histogram_axes(events_histogram)
                rht_max_radius = 1.5 * np.average(
                    self.radius_axis[:-1], weights=np.sum(events_histogram["data"], axis=0)
                )
                self._total_area = (np.pi * (rht_max_radius * u.m) ** 2).to(u.cm**2)
            else:
                self._total_area = (
                    np.pi
                    * (
                        ((self.config["core_range"][1] - self.config["core_range"][0]) * u.m).to(
                            u.cm
                        )
                    )
                    ** 2
                )
        return self._total_area

    @property
    def energy_range(self):
        """
        Energy range used in the simulation.

        Returns
        -------
        list:
            Energy range used in the simulation [min, max]
        """
        if self._energy_range is None:
            self._energy_range = [
                self.config["E_range"][0] * u.TeV,
                self.config["E_range"][1] * u.TeV,
            ]
        return self._energy_range

    @staticmethod
    def _produce_triggered_to_sim_fraction_hist(events_histogram, triggered_events_histogram):
        """
        Produce a new histogram with the fraction of triggered events over the simulated events.
        The dimension of the histogram is reduced, as the rates are summed for all the bins in
        impact distance.

        Parameters
        ----------
        events_histogram:
            A dictionary with "data" corresponding to a 2D histogram (impact distance x energy)
            for the simulated events.
        triggered_events_histogram:
            A dictionary with "data" corresponding to a 2D histogram (impact distance x energy)
            for the triggered events.

        Returns
        -------
        event_ratio_histogram:
            The new histogram with the fraction of triggered over simulated events.
        """

        simulated_events_per_energy_bin = np.sum(events_histogram["data"], axis=1)

        triggered_events_per_energy_bin = np.sum(triggered_events_histogram["data"], axis=1)
        ratio_per_energy_bin = np.zeros_like(triggered_events_per_energy_bin, dtype=float)

        non_zero_indices = np.nonzero(simulated_events_per_energy_bin)[0]
        ratio_per_energy_bin[non_zero_indices] = (
            triggered_events_per_energy_bin[non_zero_indices]
            / simulated_events_per_energy_bin[non_zero_indices]
        )
        return ratio_per_energy_bin

    def compute_system_trigger_rate(self):
        """
        Compute the system trigger rate and its uncertainty, which are saved as class attributes.
        """

        if self.trigger_rate is None:
            # Get the simulated and triggered 2D histograms from the simtel_array output file
            events_histogram, triggered_events_histogram = self.fill_event_histogram_dicts()

            # Calculate triggered/simulated event 1D histogram (energy dependent)
            triggered_to_sim_fraction_hist = self._produce_triggered_to_sim_fraction_hist(
                events_histogram, triggered_events_histogram
            )
            self._initialize_histogram_axes(triggered_events_histogram)

            # Getting the particle distribution function according to the reference
            particle_distribution_function = self.get_particle_distribution_function(
                label="reference"
            )

            # Integrating the flux between the consecutive energy bins. The result given in
            # cm-2s-1sr-1
            flux_per_energy_bin = self._integrate_in_energy_bin(
                particle_distribution_function, self.energy_axis
            )

            # Derive the trigger rate per energy bin
            self.trigger_rate_per_energy_bin = (
                triggered_to_sim_fraction_hist
                * flux_per_energy_bin
                * self.total_area
                * self.solid_angle
            ).decompose()

            # Derive the system trigger rate
            self.trigger_rate = np.sum(self.trigger_rate_per_energy_bin)

            # Derive the uncertainty in the system trigger rate estimate
            self.trigger_rate_uncertainty = self._estimate_trigger_rate_uncertainty()

        return self.trigger_rate, self.trigger_rate_uncertainty

    def trigger_info_in_table(self):
        """
        Provide the trigger rate per energy bin in tabulated form.

        Returns
        -------
        astropy.QTable:
            The QTable instance with the trigger rate per energy bin.
        """
        meta = self.produce_trigger_meta_data()
        trigger_rate_per_energy_bin_table = QTable(
            [self.energy_axis[:-1] * u.TeV, (self.trigger_rate_per_energy_bin.to(u.Hz))],
            names=("Energy (TeV)", "Trigger rate (Hz)"),
            meta=meta,
        )
        return trigger_rate_per_energy_bin_table

    def produce_trigger_meta_data(self):
        """
        Produce the meta data to include in the tabulated form of the trigger rate per energy bin.
        It shows some information from the input file (simtel_array file) and the final estimate
        system trigger rate.

        Returns
        -------
        dict:
            dictionary with the metadata.
        """
        return {
            "simtel_array_file": self.histogram_file,
            "simulation_input": self.print_info(mode="silent"),
            # pylint: disable=E1101
            "system_trigger_rate (Hz)": self.trigger_rate.value,
        }

    def _integrate_in_energy_bin(self, particle_distribution_function, energy_axis):
        """
        Helper function to integrate the particle distribution function between the consecutive
        energy bins given by the energy_axis array.

        Parameters
        ----------
        particle_distribution_function: ctao_cr_spectra.spectral.PowerLaw
            The function describing the spectral distribution.
        energy_axis: numpy.array
            The array with the simulated particle energies.

        Returns
        -------
        astropy.Quantity:
            astropy.Quantity of a numpy array with the energy integrated flux.
        """
        unit = None
        flux_per_energy_bin = []
        for i_energy, _ in enumerate(energy_axis[:-1]):
            integrated_flux = particle_distribution_function.integrate_energy(
                energy_axis[i_energy] * u.TeV, energy_axis[i_energy + 1] * u.TeV
            ).decompose(bases=[u.s, u.cm, u.sr])
            if unit is None:
                unit = integrated_flux.unit

            flux_per_energy_bin.append(integrated_flux.value)

        return np.array(flux_per_energy_bin) * unit

    def _initialize_histogram_axes(self, events_histogram):
        """
        Initialize the two axes of a histogram: the array with the edges of the bins in core
        distance and the edges of the array with the energy bins.

        Parameters
        ----------
        events_histogram:
            A single histogram from where to extract axis information.
        """
        self.radius_axis = np.linspace(
            events_histogram["lower_x"],
            events_histogram["upper_x"],
            events_histogram["n_bins_x"] + 1,
            endpoint=True,
        )

        self.energy_axis = np.logspace(
            events_histogram["lower_y"],
            events_histogram["upper_y"],
            events_histogram["n_bins_y"] + 1,
            endpoint=True,
        )

    def get_particle_distribution_function(self, label="reference"):
        """
        Get the particle distribution function, depending on whether one wants the reference CR
        distribution or the distribution used in the simulation.This is controlled by label.
        By using label="reference", one gets the distribution function according to a pre-defined CR
        distribution, while by using label="simulation", the spectral index of the distribution
        function from the simulation is used.

        Parameters
        ----------
        label: str
            label defining which distribution function. Possible values are: "reference", or
            "simulation".

        Returns
        -------
        ctao_cr_spectra.spectral.PowerLaw
            The function describing the spectral distribution.
        """

        if label == "reference":
            particle_distribution_function = copy.copy(IRFDOC_PROTON_SPECTRUM)
        elif label == "simulation":
            particle_distribution_function = self._get_simulation_spectral_distribution_function()
        else:
            msg = f"label {label} is not valid. Please use either 'reference' or 'simulation'."
            self._logger.error(msg)
            raise ValueError
        return particle_distribution_function

    def _get_simulation_spectral_distribution_function(self):
        """
        Get the simulation particle energy distribution according to its configuration.

        Returns
        -------
        ctao_cr_spectra.spectral.PowerLaw
            The function describing the spectral distribution.
        """
        spectral_distribution = copy.copy(IRFDOC_PROTON_SPECTRUM)
        spectral_distribution.index = self.config["spectral_index"]
        return spectral_distribution

    def estimate_observation_time(self):
        """
        Estimates the observation time corresponding to the simulated number of events.
        It uses the CTAO reference cosmic-ray spectra, the total number of particles simulated,
        and other information from the simulation configuration self.config.

        Returns
        -------
        astropy.Quantity[time]
            Estimated observation time based on the total number of particles simulated.
        """
        first_estimate = IRFDOC_PROTON_SPECTRUM.compute_number_events(
            self.view_cone[0],
            self.view_cone[1],
            1 * u.s,
            self.total_area,
            self.energy_range[0],
            self.energy_range[1],
        )
        obs_time = (self.total_num_simulated_events / first_estimate) * u.s
        return obs_time

    def _estimate_trigger_rate_uncertainty(self):
        """
        Estimate the trigger rate uncertainty, based on the number of simulated events.
        Poisson statistics are assumed.

        Returns
        -------
        astropy.Quantity[1/time]
            Uncertainty in the trigger rate estimate.
        """
        return (
            np.sqrt(self.total_num_triggered_events / self.estimate_observation_time().value) * u.s
        )

    def print_info(self, mode=None):
        """
        Print information on the geometry and input parameters.

        Returns
        -------
        dict:
            Dictionary with the information, e.g., view angle, energy range, etc.
        """
        info_dict = {
            "view_cone": self.view_cone,
            "solid_angle": self.solid_angle,
            "total_area": self.total_area,
            "energy_range": self.energy_range,
            "total_num_simulated_events": self.total_num_simulated_events,
            "total_num_triggered_events": self.total_num_triggered_events,
        }
        if mode != "silent":
            print(info_dict)
        return info_dict


class SimtelHistograms:
    """
    This class handles sim_telarray histograms. Histogram files are handled by using eventio
    library.
    Input files may either be histogram (.hdata.zst) or simtel_array output (.simtel) files.

    Parameters
    ----------
    histogram_files: list
        List of sim_telarray histogram files (str of Path).
    test: bool
        If True, only a fraction of the histograms will be processed, leading to a much shorter\
         runtime.
    rht: bool
        If true, the area thrown in the trigger rate calculation is estimated exactly as in the
        hessio rht.cc tool. If false, it is estimated based on the maximum distance as given in
        the simulation configuration.
    """

    def __init__(self, histogram_files, test=False, rht=False):
        """
        Initialize SimtelHistograms
        """
        self._logger = logging.getLogger(__name__)
        if not isinstance(histogram_files, list):
            histogram_files = [histogram_files]
        self.histogram_files = histogram_files
        self._is_test = test
        self._combined_hists = None
        self._list_of_histograms = None
        self.__meta_dict = None
        self.rht = rht

    def calculate_trigger_rates(self, print_info=False):
        """
        Calculate the triggered and simulated event rate considering the histograms in each file.
        It returns also a list with the tables where the energy dependent trigger rate for each
        file can be found.

        Parameters
        ----------
        print_info: bool
            if True, prints out the information about the histograms such as energy range, area,
            etc.

        Returns
        -------
        sim_event_rates: list of astropy.Quantity[1/time]
            The simulated event rates.
        triggered_event_rates: list of astropy.Quantity[1/time]
            The triggered event rates.
        trigger_rate_in_tables: list of astropy.QTable
            The energy dependent trigger rates.
        """
        triggered_event_rates = []
        sim_event_rates = []
        trigger_rate_in_tables = []
        triggered_event_rate_uncertainties = []
        for i_file, file in enumerate(self.histogram_files):
            simtel_hist_instance = SimtelHistogram(file, rht=self.rht)
            if print_info:
                simtel_hist_instance.print_info()

            logging.info(f"Histogram {i_file + 1}:")
            logging.info(
                "Total number of simulated events: "
                f"{simtel_hist_instance.total_num_simulated_events} events"
            )
            logging.info(
                "Total number of triggered events: "
                f"{simtel_hist_instance.total_num_triggered_events} events"
            )

            obs_time = simtel_hist_instance.estimate_observation_time()
            logging.info(
                f"Estimated equivalent observation time corresponding to the number of "
                f"events simulated: {obs_time.value} s"
            )
            sim_event_rate = simtel_hist_instance.total_num_simulated_events / obs_time
            sim_event_rates.append(sim_event_rate)
            logging.info(f"Simulated event rate: {sim_event_rate.value:.4e} Hz")

            (
                triggered_event_rate,
                triggered_event_rate_uncertainty,
            ) = simtel_hist_instance.compute_system_trigger_rate()
            logging.info(
                f"System trigger event rate: "
                # pylint: disable=E1101
                f"{triggered_event_rate.value:.4e} \u00B1 "
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
                raise InconsistentHistogramFormat

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
        """Add the values of the same type of histogram from the various lists into a single
        histogram list."""
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
