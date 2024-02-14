import copy
import logging

import numpy as np
from astropy import units as u
from ctao_cr_spectra.spectral import PowerLaw, irfdoc_proton_spectrum
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
    This class handles a single histogram file.
    """

    def __init__(self, histogram_file):
        self.histogram_file = histogram_file
        self._config = None
        self._view_cone = None
        self._total_area = None
        self._energy_range = None
        self._total_num_simulated_events = None
        self._total_num_triggered_events = None
        self._histogram = None
        self._histogram_file = None
        self._initialize_histogram()

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

    def get_histogram_type_title(self, i_hist):
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
        return self.histogram[i_hist]["title"]

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
        It already accounts for the NSHOW and NUSE found used in the configuration to produce
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
                "Number of times each simulated shower is used (CORSIKA NUSE): "
                f"{self.config['n_use']}"
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
        distribution assumed is not necessarily the CTAO reference cosmic-ray spectra.

        Returns
        -------
        int:
            total number of simulated events.
        """

        if self._total_num_triggered_events is None:
            _, triggered_hist = self.fill_event_histogram_dicts()
            self._total_num_triggered_events = np.round(np.sum(triggered_hist["data"]))
            logging.debug(f"Number of triggered showers: {self._total_num_triggered_events}")
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
        # Save the appropriate histograms to dictionaries
        found_one = False
        found_two = False
        for hist in self.histogram:
            if hist["id"] == 1:
                events_histogram = hist
                found_one = True
            elif hist["id"] == 2:
                triggered_events_histogram = hist
                found_two = True
            if found_one * found_two:
                break

        if "triggered_events_histogram" in locals():
            return events_histogram, triggered_events_histogram
        msg = "Histograms ids not found. Please check your files."
        logging.error(msg)
        raise HistogramIdNotFound

    @property
    @u.quantity_input(energy=u.deg)
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
    @u.quantity_input(energy=u.cm**2)
    def total_area(self):
        """
        Total area covered by the simulated events (original CORSIKA CSCAT).

        Returns
        -------
        astropy.Quantity[area]:
            Total area covered on the ground covered by the simulation.
        """
        if self._total_area is None:
            self._total_area = (
                np.pi
                * (((self.config["core_range"][1] - self.config["core_range"][0]) * u.m).to(u.cm))
                ** 2
            )
        return self._total_area

    @property
    @u.quantity_input(energy=u.TeV)
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

        Parameters
        ----------
        events_histogram:
            A histogram for the simulated events.
        triggered_events_histogram:
            A histogram for the triggered events.

        Returns
        -------
        event_ratio_histogram:
            The new histogram with the fraction of triggered over simulated events.
        """
        event_ratio_histogram = copy.copy(events_histogram)

        event_ratio_histogram["data"] = np.zeros_like(triggered_events_histogram["data"])

        # Radial distribution of triggered events per E divided by the radial distribution
        # of simulated events per E (gives a radial distribution of trigger probability per E
        non_zero_indices = events_histogram["data"] != 0
        event_ratio_histogram["data"][non_zero_indices] = (
            triggered_events_histogram["data"][non_zero_indices]
            / events_histogram["data"][non_zero_indices]
        )
        return event_ratio_histogram

    def _initialize_histogram_axes(self, events_histogram):
        """
        Initialize the two axes of a histogram.

        Parameters
        ----------
        events_histogram:
            A single histogram from where to extract axis information.

        Returns
        -------
        radius_axis: numpy.array
            The array with the impact distance.
        energy_axis: numpy.array
            The array with the simulated particle energies.
        """
        radius_axis = np.linspace(
            events_histogram["lower_x"],
            events_histogram["upper_x"],
            events_histogram["n_bins_x"] + 1,
            endpoint=True,
        )
        energy_axis = np.logspace(
            events_histogram["lower_y"],
            events_histogram["upper_y"],
            events_histogram["n_bins_y"] + 1,
            endpoint=True,
        )
        return radius_axis, energy_axis

    @staticmethod
    def _integrate_hist_in_area(radius_axis, events_histogram):
        """
        Integrate the histogram in area and keep the energy dependence.
        In a sequence of integrations, one has to integrate in area first, and then in energy.

        Parameters
        ----------
        radius_axis: numpy.array
            The array with the impact distance.
        events_histogram:
            A histogram for the simulated events.

        Returns
        -------
        integrated_event_ratio_per_energy: numpy.array (1-D)
            The area-integrated, energy-dependent array.

        """
        energy_axis_size = np.size(events_histogram["data"], axis=0)
        integrated_event_ratio_per_energy = np.zeros(energy_axis_size)
        areas = radius_axis[:-1] * np.diff(radius_axis)

        for i_energy in range(energy_axis_size - 1):
            integrated_event_ratio_per_energy[i_energy] = np.sum(
                events_histogram["data"][i_energy] * areas
            )
        return integrated_event_ratio_per_energy

    def _integrate_array_in_energy(self, energy_axis, events_array, re_weight=True):
        """
        Integrate the event ratio distribution in energy.

        Parameters
        ----------
        energy_axis: numpy.array
            The array with the simulated particle energies.
        events_array: numpy.array
            Array with the area-integrated trigger to simulated event ratio distribution as
            function of energy.
        re_weight: bool
            if True, re-weights the particle spectral distribution to correct to the expected one.

        Returns
        -------
        float:
            energy-integrated event ratio distribution, i.e., the system trigger probability.
        """
        # Get the particle distribution for the specified energy axis and re_weight flag
        particle_distribution = self.get_particle_distribution(energy_axis, re_weight=re_weight)

        normalized_pdf = particle_distribution / np.sum(particle_distribution)

        # Trigger probability per E integrated in E according to the given energy distribution
        # (gives a trigger probability, i.e. a normalization)
        trigger_probability = np.sum(events_array * normalized_pdf[:-1] * np.diff(energy_axis))
        logging.debug(f"System trigger probability: {trigger_probability}.")
        return trigger_probability

    def get_particle_distribution(self, energy_axis, re_weight=True):
        """
        Get the particle distribution for the specified energy_axis.

        Parameters
        ----------
        energy_axis: numpy.array
            The array with the simulated particle energies.
        re_weight: bool
            if True, re-weights the particle spectral distribution to correct to the expected one.

        Returns
        -------
        particle_distribution: numpy.array
            The array with the particle distribution as function of energy.
        """

        particle_distribution_function = self.get_particle_distribution_function(
            re_weight=re_weight
        )
        return particle_distribution_function(energy_axis * u.TeV)

    def get_particle_distribution_function(self, re_weight=True):
        """
        Get the particle distribution function, depending on whether one wants to re-weight to the
        expected cosmic-ray spectral distribution or not.

        Parameters
        ----------
        re_weight: bool
            if True, re-weights the particle spectral distribution to correct to the expected one.

        Returns
        -------
        ctao_cr_spectra.spectral.PowerLaw
            The function describing the spectral distribution.
        """
        # Define the particle distribution
        particle_distribution_function = copy.copy(irfdoc_proton_spectrum)
        if re_weight:
            correction_factor = self.get_correction_factor()
        else:
            correction_factor = 1
        particle_distribution_function.normalization /= correction_factor

        return particle_distribution_function

    def get_correction_factor(self):
        """
        Get the correction factor for the energy distribution to account for differences in the
        expected cosmic-ray spectral distribution and the cosmic-ray distribution assumed for
        the simulation.

        Returns
        -------
        float
            The correction factor.
        """
        logging.debug("Getting particle distribution.")

        simulation_energy_distribution = self.get_simulation_spectral_distribution()
        # Expected integrated CR flux
        cr_energy_integrated = irfdoc_proton_spectrum.integrate_energy(
            self.energy_range[0], self.energy_range[1]
        )
        # Simulated integrated flux (differs from above due to optimization of computational time)
        simulation_energy_integrated = simulation_energy_distribution.integrate_energy(
            self.energy_range[0], self.energy_range[1]
        )
        # Estimate a normalization factor, which also means the fraction of computational time
        # spared by using a different distribution. `time_economy_factor` expected to be > 1.
        time_economy_factor = cr_energy_integrated / simulation_energy_integrated
        return time_economy_factor

    def get_simulation_spectral_distribution(self):
        """
        Get the simulation particle energy distribution according to its configuration.

        Returns
        -------
        numpy.array
            The differential flux of the energy distribution.
        """
        if self.config["diffuse"] == 1:
            norm_unit = 1 / (u.cm**2 * u.s * u.sr * u.TeV)
        else:
            norm_unit = 1 / (u.cm**2 * u.s * u.TeV)

        non_norm_simulated_power_law_function = PowerLaw(
            normalization=1 * norm_unit, index=self.config["spectral_index"], e_ref=1 * u.TeV
        )
        non_norm_simulated_events_rate = non_norm_simulated_power_law_function.derive_events_rate(
            inner=self.view_cone[0],
            outer=self.view_cone[1],
            area=self.total_area,
            energy_min=self.energy_range[0],
            energy_max=self.energy_range[1],
        )

        factor = self.total_num_simulated_events / non_norm_simulated_events_rate.value
        norm_simulated_power_law_function = PowerLaw(
            normalization=factor * norm_unit,
            index=self.config["spectral_index"],
            e_ref=1 * u.TeV,
        )
        return norm_simulated_power_law_function

    def _calculate_system_trigger_rate(self, trigger_probability, particle_distribution_function):
        """
        Calculate the system trigger rate based on the system trigger probability and the particle
        distribution function.

        Parameters
        ----------
        trigger_probability: float
            The system trigger probability.
        particle_distribution_function:
            The function from ctao_cr_spectra that describes the particle energy
            distribution.

        Returns
        -------
        float:
            The system trigger ratio.
        """
        system_trigger_rate = (
            trigger_probability
            * particle_distribution_function.derive_events_rate(
                inner=self.view_cone[0],
                outer=self.view_cone[1],
                area=self.total_area,
                energy_min=self.energy_range[0],
                energy_max=self.energy_range[1],
            )
        )
        logging.debug(f"{system_trigger_rate.to(1 / u.s).value} Hz")
        return system_trigger_rate

    def estimate_observation_time(self):
        """
        Estimates the observation time comprised by the number of events in the simulation.
        It uses the CTAO reference cosmic-ray spectra, the total number of particles simulated,
        and other information from the simulation configuration `self.config`.

        Returns
        -------
        float: astropy.Quantity[time]
            Estimated observation time based on the total number of particles simulated.
        """
        first_estimate = irfdoc_proton_spectrum.derive_number_events(
            self.view_cone[0],
            self.view_cone[1],
            1 * u.s,
            self.total_area,
            self.energy_range[0],
            self.energy_range[1],
        )
        obs_time = self.total_num_simulated_events / first_estimate * u.s
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
        return np.sqrt(self.total_num_triggered_events) / self.estimate_observation_time()

    def trigger_rate_per_histogram(self, re_weight=True):
        """
        Calculate the trigger rate histograms, i.e., the number of triggered events per second.

        Parameters
        ----------
        re_weight: bool
            if True, re-weights the particle spectral distribution to correct to the expected one.

        Returns
        -------
        list:
            List with the trigger rate histograms for each file.
        """

        events_histogram, triggered_events_histogram = self.fill_event_histogram_dicts()

        # Produce triggered/simulated event histogram
        triggered_to_sim_fraction_hist = self._produce_triggered_to_sim_fraction_hist(
            events_histogram, triggered_events_histogram
        )

        radius_axis, energy_axis = self._initialize_histogram_axes(events_histogram)
        # Radial distribution of trigger probability per E integrated in area at each radius
        # (gives a trigger probability per E)
        triggered_to_sim_fraction_per_energy = self._integrate_hist_in_area(
            radius_axis, triggered_to_sim_fraction_hist
        )

        trigger_probability = self._integrate_array_in_energy(
            energy_axis, triggered_to_sim_fraction_per_energy, re_weight=re_weight
        )
        system_trigger_rate = self._calculate_system_trigger_rate(
            trigger_probability,
            self.get_particle_distribution_function(re_weight=re_weight),
        )

        system_trigger_rate_uncertainty = self._estimate_trigger_rate_uncertainty()

        return system_trigger_rate, system_trigger_rate_uncertainty


class SimtelHistograms:
    """
    This class handles sim_telarray histograms. Histogram files are handled by using eventio
    library.

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
        self.histogram_files = histogram_files
        self._is_test = test
        self._combined_hists = None
        self._list_of_histograms = None
        self.__meta_dict = None

    def calculate_event_rates(self):
        """
        Calculate the triggered and simulated event rate for the histograms in each file.

        Returns
        -------
        sim_event_rates: list of astropy.Quantity[1/time]
            The simulated event rates.
        triggered_event_rates: list of astropy.Quantity[1/time]
            The triggered event rates.
        """
        triggered_event_rates = []
        sim_event_rates = []
        for i_file, file in enumerate(self.histogram_files):
            simtel_hist_instance = SimtelHistogram(file)

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
            logging.info(f"Estimated equivalent observation time: {obs_time.value} s")

            sim_event_rate = simtel_hist_instance.total_num_simulated_events / obs_time
            sim_event_rates.append(sim_event_rate)
            logging.info(f"Simulated event rate: {sim_event_rate.value:.4e} Hz")

            (
                triggered_event_rate,
                triggered_event_rate_uncertainty,
            ) = simtel_hist_instance.trigger_rate_per_histogram()
            triggered_event_rates.append(triggered_event_rate)
            logging.info(
                f"System trigger event rate: {triggered_event_rate.value:.4e} \u00B1 "
                f"{triggered_event_rate_uncertainty.value:.4e} Hz"
            )
        return sim_event_rates, triggered_event_rates

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
                raise InconsistentHistogramFormat(msg)

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
