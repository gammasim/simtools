import copy
import logging
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import QTable
from ctao_cr_spectra.definitions import IRFDOC_PROTON_SPECTRUM
from ctao_cr_spectra.spectral import cone_solid_angle
from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type
from eventio.simtel import MCRunHeader

__all__ = [
    "InconsistentHistogramFormat",
    "HistogramIdNotFound",
    "SimtelHistogram",
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

    """

    def __init__(self, histogram_file, area_from_distribution=False):
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
        self.area_from_distribution = area_from_distribution

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

            if self.area_from_distribution is True:
                events_histogram, _ = self.fill_event_histogram_dicts()
                self._initialize_histogram_axes(events_histogram)
                area_from_distribution_max_radius = 1.5 * np.average(
                    self.radius_axis[:-1], weights=np.sum(events_histogram["data"], axis=0)
                )
                self._total_area = (np.pi * (area_from_distribution_max_radius * u.m) ** 2).to(
                    u.cm**2
                )
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

    def compute_system_trigger_rate(self, events_histogram=None, triggered_events_histogram=None):
        """
        Compute the system trigger rate and its uncertainty, which are saved as class attributes.
        If events_histogram and triggered_events_histogram are passed, they are used to calculate
        the trigger rate and trigger rate uncertainty, instead of the histograms from the file.
        This is specially useful when calculating the trigger rate for stacked files, in which case
        one can pass the histograms resulted from stacking the files to this function.
        Default is filling from the file.

        Parameters
        ----------
        events_histogram:
            A dictionary with "data" corresponding to a 2D histogram (core distance x energy)
            for the simulated events.
        triggered_events_histogram:
            A dictionary with "data" corresponding to a 2D histogram (core distance x energy)
            for the triggered events.
        """

        if self.trigger_rate is None:
            # Get the simulated and triggered 2D histograms from the simtel_array output file
            if events_histogram is None and triggered_events_histogram is None:
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
            self.trigger_rate_uncertainty = self.estimate_trigger_rate_uncertainty(
                self.trigger_rate,
                np.sum(events_histogram["data"]),
                np.sum(triggered_events_histogram["data"]),
            )

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

    def estimate_observation_time(self, stacked_num_simulated_events=None):
        """
        Estimates the observation time corresponding to the simulated number of events.
        It uses the CTAO reference cosmic-ray spectra, the total number of particles simulated,
        and other information from the simulation configuration self.config.
        If stacked_num_simulated_events is given, the observation time is estimated from it instead
        of from the simulation configuration (useful for the stacked trigger rate estimate).

        Parameters
        ----------
        stacked_num_simulated_events: int
            total number of simulated events for the stacked dataset.

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
        if stacked_num_simulated_events is None:
            return (self.total_num_simulated_events / first_estimate) * u.s
        return (stacked_num_simulated_events / first_estimate) * u.s

    def estimate_trigger_rate_uncertainty(
        self, trigger_rate_estimate, num_simulated_events, num_triggered_events
    ):
        """
        Estimate the trigger rate uncertainty, based on the number of simulated and triggered
        events. Poisson Statistics are assumed. The uncertainty is calculated based on propagation
        of the individual uncertainties.
        If stacked_num_simulated_events is passed, the uncertainty is estimated based on it instead
        of based on the total number of trigger events from the simulation configuration
        (useful for the stacked trigger rate estimate).

        Parameters
        ----------
        trigger_rate_estimate: astropy.Quantity[1/time]
            The already estimated the trigger rate.
        num_simulated_events: int
            Total number of simulated events.
        num_triggered_events: int
            Total number of triggered events.

        Returns
        -------
        astropy.Quantity[1/time]
            Uncertainty in the trigger rate estimate.
        """
        # pylint: disable=E1101
        return (
            trigger_rate_estimate.value
            * np.sqrt(1 / num_triggered_events + 1 / num_simulated_events)
        ) * trigger_rate_estimate.unit

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
