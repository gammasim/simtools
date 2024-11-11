"""
Provides functionality to evaluate statistical errors from dl2_mc_events_file FITS files.

Classes
-------
StatisticalErrorEvaluator
    Handles error calculation for given dl2_mc_events_file FITS files and specified metrics.


"""

import logging

import numpy as np
from astropy import units as u
from astropy.io import fits

_logger = logging.getLogger(__name__)


class StatisticalErrorEvaluator:
    """
    Evaluates statistical errors from a dl2_mc_events_file FITS file.

    Parameters
    ----------
    file_path : str
        Path to the dl2_mc_events_file FITS file.
    file_type : str
        Type of the file, either 'On-source' or 'Offset'.
    metrics : dict, optional
        Dictionary of metrics to evaluate. Default is None.
    grid_point : tuple, optional
        Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).
    """

    def __init__(
        self,
        file_path: str,
        file_type: str,
        metrics: dict[str, float] | None = None,
        grid_point: tuple[float, float, float, float, float] | None = None,
    ):
        """
        Init the evaluator with a dl2_mc_events_file FITS file, its type, and metrics to calculate.

        Parameters
        ----------
        file_path : str
            The path to the dl2_mc_events_file FITS file.
        file_type : str
            The type of the file ('On-source' or 'Offset').
        metrics : dict, optional
            Dictionary specifying which metrics to compute and their reference values.
        grid_point : tuple, optional
            Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).
        """
        self.file_path = file_path
        self.file_type = file_type
        self.metrics = metrics or {}
        self.grid_point = grid_point

        self.data = self.load_data_from_file()

        self.error_eff_area = None
        self.error_sig_eff_gh = None
        self.error_energy_estimate_bdt_reg_tree = None
        self.sigma_energy = None
        self.delta_energy = None
        self.error_gamma_ray_psf = None
        self.error_image_template_methods = None

        self.metric_results = None
        self.scaled_events = None
        self.scaled_events_gridpoint = None
        self.energy_threshold = None

    def load_data_from_file(self):
        """
        Load data from the dl2_mc_events_file FITS file and return dictionaries with units.

        Returns
        -------
        dict
            Dictionary containing data from the dl2_mc_events_file FITS file with units.
        """
        data = {}
        try:
            with fits.open(self.file_path) as hdul:
                events_data = hdul["EVENTS"].data  # pylint: disable=E1101
                sim_events_data = hdul["SIMULATED EVENTS"].data  # pylint: disable=E1101

                event_units = {}
                for idx, col_name in enumerate(events_data.columns.names, start=1):
                    unit_key = f"TUNIT{idx}"
                    if unit_key in hdul["EVENTS"].header:  # pylint: disable=E1101
                        event_units[col_name] = u.Unit(
                            hdul["EVENTS"].header[unit_key]  # pylint: disable=E1101
                        )
                    else:
                        event_units[col_name] = None

                sim_units = {}
                for idx, col_name in enumerate(sim_events_data.columns.names, start=1):
                    unit_key = f"TUNIT{idx}"
                    if unit_key in hdul["SIMULATED EVENTS"].header:  # pylint: disable=E1101
                        sim_units[col_name] = u.Unit(
                            hdul["SIMULATED EVENTS"].header[unit_key]  # pylint: disable=E1101
                        )
                    else:
                        sim_units[col_name] = None

                # Check and apply units to each column, raising an error if the unit is missing
                if not event_units["ENERGY"]:
                    raise ValueError("Unit for ENERGY in EVENTS data is missing.")
                event_energies_reco = events_data["ENERGY"] * event_units["ENERGY"]

                if not event_units["MC_ENERGY"]:
                    raise ValueError("Unit for MC_ENERGY in EVENTS data is missing.")
                event_energies_mc = events_data["MC_ENERGY"] * event_units["MC_ENERGY"]

                if not sim_units["MC_ENERG_LO"]:
                    raise ValueError("Unit for MC_ENERG_LO in SIMULATED EVENTS data is missing.")
                bin_edges_low = sim_events_data["MC_ENERG_LO"] * sim_units["MC_ENERG_LO"]

                if not sim_units["MC_ENERG_HI"]:
                    raise ValueError("Unit for MC_ENERG_HI in SIMULATED EVENTS data is missing.")
                bin_edges_high = sim_events_data["MC_ENERG_HI"] * sim_units["MC_ENERG_HI"]

                simulated_event_histogram = sim_events_data["EVENTS"] * u.count

                viewcone = hdul[3].data["viewcone"][0][1]  # pylint: disable=E1101
                core_range = hdul[3].data["core_range"][0][1]  # pylint: disable=E1101

                data = {
                    "event_energies_reco": event_energies_reco,
                    "event_energies_mc": event_energies_mc,
                    "bin_edges_low": bin_edges_low,
                    "bin_edges_high": bin_edges_high,
                    "simulated_event_histogram": simulated_event_histogram,
                    "viewcone": viewcone,
                    "core_range": core_range,
                }
                unique_azimuths = np.unique(events_data["PNT_AZ"]) * u.deg
                unique_zeniths = np.unique(events_data["PNT_ALT"]) * u.deg
                if self.grid_point is None:
                    _logger.info(f"Unique azimuths: {unique_azimuths}")
                    _logger.info(f"Unique zeniths: {unique_zeniths}")

                    if len(unique_azimuths) == 1 and len(unique_zeniths) == 1:
                        _logger.info(
                            f"Setting initial grid point with azimuth: {unique_azimuths[0]}"
                            f" zenith: {unique_zeniths[0]}"
                        )
                        self.grid_point = (
                            1 * u.TeV,
                            unique_azimuths[0],
                            unique_zeniths[0],
                            0,
                            0 * u.deg,
                        )  # Initialize grid point with azimuth and zenith
                    else:
                        _logger.warning(
                            "Multiple unique values found for azimuth or zenith. "
                            "The grid point will be set based on the first unique values."
                        )
                else:
                    _logger.warning(
                        f"Grid point already set to: {self.grid_point}. "
                        "Overwriting with new values from file."
                    )

                    self.grid_point = (
                        1 * u.TeV,
                        unique_azimuths[0],
                        unique_zeniths[0],
                        0,
                        0 * u.deg,
                    )
                    _logger.info(f"New grid point values: {self.grid_point}")

        except (FileNotFoundError, KeyError) as e:
            _logger.error(f"Error loading file {self.file_path}: {e}")
            raise FileNotFoundError from e
        return data

    def create_bin_edges(self):
        """
        Create unique energy bin edges.

        Returns
        -------
        bin_edges : array
            Array of unique energy bin edges.
        """
        bin_edges_low = self.data["bin_edges_low"]
        bin_edges_high = self.data["bin_edges_high"]
        bin_edges = np.concatenate([bin_edges_low, [bin_edges_high[-1]]])
        return np.unique(bin_edges)

    def compute_histogram(self, event_energies_reco, bin_edges):
        """
        Compute histogram for triggered events.

        Parameters
        ----------
        event_energies_reco : array
            Array of energies of the observed events.
        bin_edges : array
            Array of energy bin edges.

        Returns
        -------
        triggered_event_histogram : array
            Histogram of triggered events.
        """
        event_energies_reco = event_energies_reco.to(bin_edges.unit)

        triggered_event_histogram, _ = np.histogram(event_energies_reco.value, bins=bin_edges.value)
        return triggered_event_histogram * u.count

    def compute_efficiency_and_errors(self, triggered_event_counts, simulated_event_counts):
        """
        Compute energy efficiency and its statistical error using the binomial distribution.

        Parameters
        ----------
        triggered_event_counts : array with units
            Histogram counts of the triggered events.
        simulated_event_counts : array with units
            Histogram counts of the simulated events.

        Returns
        -------
        efficiencies : array
            Array of calculated efficiencies.
        uncertainties : array
            Array of calculated uncertainties.
        relative_errors : array
            Array of relative uncertainties.
        """
        # Ensure the inputs have compatible units
        triggered_event_counts = triggered_event_counts.to(u.ct)
        simulated_event_counts = simulated_event_counts.to(u.ct)

        # Compute efficiencies, ensuring the output is dimensionless
        efficiencies = np.divide(
            triggered_event_counts,
            simulated_event_counts,
            out=np.zeros_like(triggered_event_counts),
            where=simulated_event_counts > 0,
        ).to(u.dimensionless_unscaled)

        # Set up a mask for valid data with a unit-consistent threshold
        valid = (
            (simulated_event_counts > 0 * u.ct)
            & (triggered_event_counts <= simulated_event_counts)
            & (triggered_event_counts > 5 * u.ct)
        )

        uncertainties = np.zeros_like(triggered_event_counts) * u.ct**-0.5

        if np.any(valid):
            uncertainties[valid] = np.sqrt(
                np.maximum(
                    triggered_event_counts[valid]
                    * (1 - triggered_event_counts[valid] / simulated_event_counts[valid]),
                    0,
                )
            )

        # Compute relative errors
        relative_errors = np.divide(
            uncertainties,
            simulated_event_counts,
            out=np.zeros_like(uncertainties, dtype=float),
            where=uncertainties > 0,
        )

        return efficiencies, uncertainties, relative_errors

    def calculate_energy_threshold(self):
        """
        Calculate the energy threshold where the effective area exceeds 10% of its maximum value.

        Returns
        -------
        float
            Energy threshold value.
        """
        bin_edges = self.create_bin_edges()
        triggered_event_histogram = self.compute_histogram(
            self.data["event_energies_mc"], bin_edges
        )
        simulated_event_histogram = self.data["simulated_event_histogram"]

        efficiencies, _, _ = self.compute_efficiency_and_errors(
            triggered_event_histogram, simulated_event_histogram
        )

        # Determine the effective area threshold (10% of max effective area)
        max_efficiency = np.max(efficiencies)
        threshold_efficiency = 0.1 * max_efficiency

        threshold_index = np.argmax(efficiencies >= threshold_efficiency)
        if threshold_index == 0 and efficiencies[0] < threshold_efficiency:
            return

        self.energy_threshold = bin_edges[threshold_index]

    def calculate_error_eff_area(self):
        """
        Calculate the uncertainties on the effective collection area.

        Returns
        -------
        errors : dict
            Dictionary with uncertainties for the file.
        """
        bin_edges = self.create_bin_edges()
        triggered_event_histogram = self.compute_histogram(
            self.data["event_energies_mc"], bin_edges
        )
        simulated_event_histogram = self.data["simulated_event_histogram"]
        _, _, relative_errors = self.compute_efficiency_and_errors(
            triggered_event_histogram, simulated_event_histogram
        )
        return {"relative_errors": relative_errors}

    def calculate_error_sig_eff_gh(self):
        """
        Calculate the uncertainties on signal efficiency in gamma-hadron separation.

        Returns
        -------
        float
            The calculated uncertainty for signal efficiency.
        """
        # implement
        return 0.02  # Placeholder value

    def calculate_error_energy_estimate_bdt_reg_tree(self):
        """
        Calculate the uncertainties in energy estimation.

        Returns
        -------
        float
            The calculated uncertainty for energy estimation.
        """
        logging.info("Calculating Energy Resolution Error")

        event_energies_reco = self.data["event_energies_reco"]
        event_energies_mc = self.data["event_energies_mc"]

        if len(event_energies_reco) != len(event_energies_mc):
            raise ValueError(f"Mismatch in the number of energies for file {self.file_path}")

        energy_deviation = (event_energies_reco - event_energies_mc) / event_energies_mc

        bin_edges = self.create_bin_edges()
        bin_indices = np.digitize(event_energies_reco, bin_edges) - 1

        energy_deviation_by_bin = [
            energy_deviation[bin_indices == i] for i in range(len(bin_edges) - 1)
        ]

        # Calculate sigma for each bin
        sigma_energy = [np.std(d) if len(d) > 0 else np.nan for d in energy_deviation_by_bin]

        # Calculate delta_energy as the mean deviation for each bin
        delta_energy = [np.mean(d) if len(d) > 0 else np.nan for d in energy_deviation_by_bin]

        # Combine sigma into a single measure
        overall_uncertainty = np.nanmean(sigma_energy)

        return overall_uncertainty, sigma_energy, delta_energy

    def calculate_error_gamma_ray_psf(self):
        """
        Calculate the uncertainties on gamma-ray PSF (68, 80, and 95% containment radius).

        Returns
        -------
        float
            The calculated uncertainty for gamma-ray PSF.
        """
        # implement
        return 0.01  # Placeholder value

    def calculate_error_image_template_methods(self):
        """
        Calculate the uncertainties relevant for image template methods.

        Returns
        -------
        float
            The calculated uncertainty for image template methods.
        """
        # implement
        return 0.05  # Placeholder value

    def calculate_metrics(self):
        """Calculate all defined metrics as specified in self.metrics and store results."""
        if "error_eff_area" in self.metrics:
            self.error_eff_area = self.calculate_error_eff_area()
            ref_value = self.metrics.get("error_eff_area")
            if self.error_eff_area:
                avg_error = np.mean(self.error_eff_area["relative_errors"])
                _logger.info(
                    f"Effective Area Error (avg): {avg_error:.3f}, Reference: {ref_value:.3f}"
                )

        if "error_sig_eff_gh" in self.metrics:
            self.error_sig_eff_gh = self.calculate_error_sig_eff_gh()
            ref_value = self.metrics.get("error_sig_eff_gh")
            _logger.info(
                f"Signal Efficiency Error: {self.error_sig_eff_gh:.3f}, Reference: {ref_value:.3f}"
            )

        if "error_energy_estimate_bdt_reg_tree" in self.metrics:
            self.error_energy_estimate_bdt_reg_tree, self.sigma_energy, self.delta_energy = (
                self.calculate_error_energy_estimate_bdt_reg_tree()
            )
            ref_value = self.metrics.get("error_energy_estimate_bdt_reg_tree")
            _logger.info(
                f"Energy Estimate Error: {self.error_energy_estimate_bdt_reg_tree:.3f}, "
                f"Reference: {ref_value:.3f}"
            )

        if "error_gamma_ray_psf" in self.metrics:
            self.error_gamma_ray_psf = self.calculate_error_gamma_ray_psf()
            ref_value = self.metrics.get("error_gamma_ray_psf")
            _logger.info(
                f"Gamma-Ray PSF Error: {self.error_gamma_ray_psf:.3f}, Reference: {ref_value:.3f}"
            )

        if "error_image_template_methods" in self.metrics:
            self.error_image_template_methods = self.calculate_error_image_template_methods()
            ref_value = self.metrics.get("error_image_template_methods")
            _logger.info(
                f"Image Template Methods Error: {self.error_image_template_methods:.3f}, "
                f"Reference: {ref_value:.3f}"
            )

        self.metric_results = {
            "error_eff_area": self.error_eff_area,
            "error_sig_eff_gh": self.error_sig_eff_gh,
            "error_energy_estimate_bdt_reg_tree": self.error_energy_estimate_bdt_reg_tree,
            "error_gamma_ray_psf": self.error_gamma_ray_psf,
            "error_image_template_methods": self.error_image_template_methods,
        }

    def calculate_max_error_for_effective_area(self):
        """
        Calculate the maximum relative error for effective area.

        Returns
        -------
        max_error : float
            Maximum relative error.
        """
        if "relative_errors" in self.metric_results["error_eff_area"]:
            return np.max(self.metric_results["error_eff_area"]["relative_errors"])
        if self.error_eff_area:
            return np.max(self.error_eff_area["relative_errors"])
        return None

    def calculate_scaled_events(self):
        """
        Calculate the scaled number of events for a specific grid point.

        Parameters
        ----------
        grid_point : tuple, optional
            Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).

        Returns
        -------
        float
            Scaled number of events for the specified grid point.
        """
        if not self.grid_point:
            raise Warning("Grid point data is not available for this evaluator.")

        bin_edges = self.create_bin_edges()
        simulated_event_histogram = self.data["simulated_event_histogram"]
        if not simulated_event_histogram.size:
            raise ValueError("Simulated event histogram is empty.")

        # Add here the implementation that uses a combination of the required metrics
        # Currently we only use the rel error on the eff area for scaling
        energy = self.grid_point[0]
        bin_idx = np.digitize(energy, bin_edges) - 1
        if bin_idx < 0 or bin_idx >= len(simulated_event_histogram):
            raise ValueError("Grid point is outside the range of the current file's data.")
        self.scaled_events = (
            self.data["simulated_event_histogram"]
            * self.error_eff_area["relative_errors"]
            / self.metrics["error_eff_area"]
        )
        self.scaled_events_gridpoint = self.scaled_events[bin_idx]
        return self.scaled_events_gridpoint

    def calculate_overall_metric(self, metric="average"):
        """
        Calculate an overall metric for the statistical errors.

        Parameters
        ----------
        metric : str
            The metric to calculate ('average', 'maximum').

        Returns
        -------
        dict
            Dictionary with overall maximum errors for each metric.
        """
        # Decide how to combine the metrics
        if self.metric_results is None:
            raise ValueError("Metrics have not been computed yet.")

        overall_max_errors = {}

        for metric_name, result in self.metric_results.items():
            if metric_name == "error_eff_area":
                max_errors = self.calculate_max_error_for_effective_area()
                overall_max_errors[metric_name] = max_errors if max_errors else 0
            elif metric_name in [
                "error_sig_eff_gh",
                "error_energy_estimate_bdt_reg_tree",
                "error_gamma_ray_psf",
                "error_image_template_methods",
            ]:
                overall_max_errors[metric_name] = result
            else:
                raise ValueError(f"Unsupported result type for {metric_name}: {type(result)}")
        _logger.info(f"overall_max_errors {overall_max_errors}")
        all_max_errors = list(overall_max_errors.values())
        if metric == "average":
            overall_metric = np.mean(all_max_errors)
        elif metric == "maximum":
            overall_metric = np.max(all_max_errors)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return overall_metric
