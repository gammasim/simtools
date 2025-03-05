"""Evaluate statistical uncertainties from DL2 MC event files."""

import logging

import numpy as np
from astropy import units as u
from astropy.io import fits

__all__ = ["StatisticalErrorEvaluator"]


class StatisticalErrorEvaluator:
    """
    Evaluates statistical uncertainties from a DL2 MC event file.

    Parameters
    ----------
    file_path : str
        Path to the DL2 MC event file.
    file_type : str
        Type of the file, either 'point-like' or 'cone'.
    metrics : dict, optional
        Dictionary of metrics to evaluate. Default is None.
    grid_point : tuple, optional
        Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).
    """

    def __init__(
        self,
        file_path: str,
        file_type: str,
        metrics: dict[str, float],
        grid_point: tuple[float, float, float, float, float] | None = None,
    ):
        """Init the evaluator with a DL2 MC event file, its type, and metrics to calculate."""
        self._logger = logging.getLogger(__name__)
        self.file_type = file_type
        self.metrics = metrics
        self.grid_point = grid_point

        self.data = self.load_data_from_file(file_path)

        self.uncertainty_effective_area = None
        self.energy_estimate = None
        self.sigma_energy = None
        self.delta_energy = None

        self.metric_results = None
        self.energy_threshold = None

    def _load_event_data(self, hdul, data_type):
        """
        Load data and units for the event and simulated data data.

        Parameters
        ----------
        hdul : HDUList
            The HDUList object.
        data_type: str
            The type of data to load ('EVENTS' or 'SIMULATED EVENTS').

        Returns
        -------
        dict
            Dictionary containing units for the event data.
        """
        _data = hdul[data_type].data  # pylint: disable=E1101
        _header = hdul[data_type].header  # pylint: disable=E1101
        _units = {}
        for idx, col_name in enumerate(_data.columns.names, start=1):
            unit_key = f"TUNIT{idx}"
            if unit_key in _header:
                _units[col_name] = u.Unit(_header[unit_key])
            else:
                _units[col_name] = None
        return _data, _units

    def _set_grid_point(self, events_data):
        """Set azimuth/zenith angle of grid point."""
        unique_azimuths = np.unique(events_data["PNT_AZ"]) * u.deg
        unique_zeniths = 90 * u.deg - np.unique(events_data["PNT_ALT"]) * u.deg
        if len(unique_azimuths) > 1 or len(unique_zeniths) > 1:
            msg = (
                f"Multiple values found for azimuth ({unique_azimuths}) zenith ({unique_zeniths})."
            )
            self._logger.error(msg)
            raise ValueError(msg)
        if self.grid_point is not None:
            self._logger.warning(
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
        self._logger.info(f"Grid point values: {self.grid_point}")

    def load_data_from_file(self, file_path):
        """
        Load data from the DL2 MC event file and return dictionaries with units.

        Returns
        -------
        dict
            Dictionary containing data from the DL2 MC event file with units.
        """
        data = {}
        try:
            with fits.open(file_path) as hdul:
                events_data, event_units = self._load_event_data(hdul, "EVENTS")
                sim_events_data, sim_units = self._load_event_data(hdul, "SIMULATED EVENTS")

                data = {
                    "event_energies_reco": events_data["ENERGY"] * event_units["ENERGY"],
                    "event_energies_mc": events_data["MC_ENERGY"] * event_units["MC_ENERGY"],
                    "bin_edges_low": sim_events_data["MC_ENERG_LO"] * sim_units["MC_ENERG_LO"],
                    "bin_edges_high": sim_events_data["MC_ENERG_HI"] * sim_units["MC_ENERG_HI"],
                    "simulated_event_histogram": sim_events_data["EVENTS"] * u.count,
                    "viewcone": hdul[3].data["viewcone"][0][1],  # pylint: disable=E1101
                    "core_range": hdul[3].data["core_range"][0][1],  # pylint: disable=E1101
                }
                self._set_grid_point(events_data)

        except FileNotFoundError as e:
            error_message = f"Error loading file {file_path}: {e}"
            self._logger.error(error_message)
            raise FileNotFoundError(error_message) from e
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

    def compute_reconstructed_event_histogram(self, event_energies_reco, bin_edges):
        """
        Compute histogram of events as function of reconstructed energy.

        Parameters
        ----------
        event_energies_reco : array
            Array of reconstructed energy per event.
        bin_edges : array
            Array of energy bin edges.

        Returns
        -------
        reconstructed_event_histogram : array
            Histogram of reconstructed events.
        """
        event_energies_reco = event_energies_reco.to(bin_edges.unit)

        reconstructed_event_histogram, _ = np.histogram(
            event_energies_reco.value, bins=bin_edges.value
        )
        return reconstructed_event_histogram * u.count

    def compute_efficiency_and_errors(self, reconstructed_event_counts, simulated_event_counts):
        """
        Compute reconstructed event efficiency and its uncertainty assuming binomial distribution.

        Parameters
        ----------
        reconstructed_event_counts : array with units
            Histogram counts of reconstructed events.
        simulated_event_counts : array with units
            Histogram counts of simulated events.

        Returns
        -------
        efficiencies : array
            Array of calculated efficiencies.
        relative_errors : array
            Array of relative uncertainties.
        """
        # Ensure the inputs have compatible units
        reconstructed_event_counts = (
            reconstructed_event_counts.to(u.ct)
            if isinstance(reconstructed_event_counts, u.Quantity)
            else reconstructed_event_counts * u.ct
        )
        simulated_event_counts = (
            simulated_event_counts.to(u.ct)
            if isinstance(simulated_event_counts, u.Quantity)
            else simulated_event_counts * u.ct
        )

        if np.any(reconstructed_event_counts > simulated_event_counts):
            raise ValueError("Reconstructed event counts exceed simulated event counts.")

        # Compute efficiencies, ensuring the output is dimensionless
        efficiencies = np.divide(
            reconstructed_event_counts,
            simulated_event_counts,
            out=np.zeros_like(reconstructed_event_counts),
            where=simulated_event_counts > 0,
        ).to(u.dimensionless_unscaled)

        # Set up a mask for valid data with a unit-consistent threshold
        valid = (simulated_event_counts > 0) & (reconstructed_event_counts > 0)

        uncertainties = np.zeros_like(reconstructed_event_counts.value) * u.dimensionless_unscaled

        if np.any(valid):
            uncertainties[valid] = np.sqrt(
                np.maximum(
                    simulated_event_counts[valid]
                    / reconstructed_event_counts[valid]
                    * (1 - reconstructed_event_counts[valid] / simulated_event_counts[valid]),
                    0,
                )
            )

        # Compute relative errors
        relative_errors = np.divide(
            uncertainties,
            np.sqrt(simulated_event_counts.value),
            out=np.zeros_like(uncertainties, dtype=float),
            where=uncertainties > 0,
        )

        return efficiencies, relative_errors

    def calculate_energy_threshold(self, requested_eff_area_fraction=0.1):
        """
        Calculate the energy threshold where the effective area exceeds 10% of its maximum value.

        Returns
        -------
        float
            Energy threshold value.
        """
        bin_edges = self.create_bin_edges()
        reconstructed_event_histogram = self.compute_reconstructed_event_histogram(
            self.data["event_energies_mc"], bin_edges
        )
        simulated_event_histogram = self.data["simulated_event_histogram"]

        efficiencies, _ = self.compute_efficiency_and_errors(
            reconstructed_event_histogram, simulated_event_histogram
        )

        # Determine the effective area threshold (10% of max effective area)
        max_efficiency = np.max(efficiencies)
        threshold_efficiency = requested_eff_area_fraction * max_efficiency

        threshold_index = np.argmax(efficiencies >= threshold_efficiency)
        if threshold_index == 0 and efficiencies[0] < threshold_efficiency:
            return

        self.energy_threshold = bin_edges[threshold_index]

    def calculate_uncertainty_effective_area(self):
        """
        Calculate the uncertainties on the effective collection area.

        Returns
        -------
        errors : dict
            Dictionary with uncertainties for the file.
        """
        bin_edges = self.create_bin_edges()
        reconstructed_event_histogram = self.compute_reconstructed_event_histogram(
            self.data["event_energies_mc"], bin_edges
        )
        simulated_event_histogram = self.data["simulated_event_histogram"]
        _, relative_errors = self.compute_efficiency_and_errors(
            reconstructed_event_histogram, simulated_event_histogram
        )
        return {"relative_errors": relative_errors}

    def calculate_energy_estimate(self):
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
            raise ValueError(
                f"Mismatch in the number of energies: {len(event_energies_reco)} vs "
                f"{len(event_energies_mc)}"
            )

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

    def calculate_metrics(self):
        """Calculate all defined metrics as specified in self.metrics and store results."""
        if "uncertainty_effective_area" in self.metrics:
            self.uncertainty_effective_area = self.calculate_uncertainty_effective_area()
            if self.uncertainty_effective_area:
                energy_range = self.metrics.get("uncertainty_effective_area", {}).get(
                    "energy_range"
                )
                min_energy, max_energy = (
                    energy_range["value"][0] * u.Unit(energy_range["unit"]),
                    energy_range["value"][1] * u.Unit(energy_range["unit"]),
                )

                valid_errors = [
                    error
                    for energy, error in zip(
                        self.data["bin_edges_low"],
                        self.uncertainty_effective_area["relative_errors"],
                    )
                    if min_energy <= energy <= max_energy
                ]
                self.uncertainty_effective_area["max_error"] = (
                    max(valid_errors) if valid_errors else 0.0
                )
                ref_value = self.metrics.get("uncertainty_effective_area", {}).get("target_error")[
                    "value"
                ]
                self._logger.info(
                    f"Effective Area Error (max in validity range): "
                    f"{self.uncertainty_effective_area['max_error'].value:.6f}, "
                    f"Reference: {ref_value:.3f}"
                )

        if "energy_estimate" in self.metrics:
            self.energy_estimate, self.sigma_energy, self.delta_energy = (
                self.calculate_energy_estimate()
            )
            ref_value = self.metrics.get("energy_estimate", {}).get("target_error")["value"]
            self._logger.info(
                f"Energy Estimate Error: {self.energy_estimate:.3f}, Reference: {ref_value:.3f}"
            )
        else:
            raise ValueError("Invalid metric specified.")
        self.metric_results = {
            "uncertainty_effective_area": self.uncertainty_effective_area,
            "energy_estimate": self.energy_estimate,
        }
        return self.metric_results

    def calculate_max_error_for_effective_area(self):
        """
        Calculate the maximum relative error for effective area.

        Returns
        -------
        max_error : float
            Maximum relative error.
        """
        if "relative_errors" in self.metric_results["uncertainty_effective_area"]:
            return np.max(self.metric_results["uncertainty_effective_area"]["relative_errors"])
        if self.uncertainty_effective_area:
            return np.max(self.uncertainty_effective_area["relative_errors"])
        return None

    def calculate_overall_metric(self, metric="average"):
        """
        Calculate an overall metric for the statistical uncertainties.

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
            if metric_name == "uncertainty_effective_area":
                max_errors = self.calculate_max_error_for_effective_area()
                overall_max_errors[metric_name] = max_errors if max_errors else 0
            elif metric_name in [
                "error_gamma_ray_psf",
            ]:
                overall_max_errors[metric_name] = result
            else:
                raise ValueError(f"Unsupported result type for {metric_name}: {type(result)}")
        self._logger.info(f"overall_max_errors {overall_max_errors}")
        all_max_errors = list(overall_max_errors.values())
        if metric == "average":
            overall_metric = np.mean(all_max_errors)
        elif metric == "maximum":
            overall_metric = np.max(all_max_errors)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return overall_metric
