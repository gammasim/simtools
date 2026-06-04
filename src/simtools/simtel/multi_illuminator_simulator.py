"""Multi-illuminator simulation orchestration with parallel execution."""

import logging

from simtools.job_execution.process_pool import determine_max_workers, process_pool_map_ordered
from simtools.model.illuminator_visibility import IlluminatorTelescopeVisibility
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.utils import general

_logger = logging.getLogger(__name__)

_NO_RESULTS_MSG = "No simulations have been run yet. Call simulate() first."


def _simulate_illuminator_telescope_pair(job_spec):
    """
    Simulate a single illuminator-telescope pair.

    This function is executed in a separate process.

    Parameters
    ----------
    job_spec : dict
        Job specification containing:
        - illuminator: str - illuminator name
        - telescope: str - telescope name
        - site: str - site name (North/South)
        - label: str - base label for the simulation
        - config: dict - light emission configuration

    Returns
    -------
    dict
        Result dictionary with simulation outcome:
        - illuminator: str
        - telescope: str
        - success: bool
        - error: str or None
    """
    illuminator = job_spec["illuminator"]
    telescope = job_spec["telescope"]
    config = job_spec["config"].copy()

    # Update configuration for this specific pair
    config["telescope"] = telescope
    config["light_source"] = illuminator

    label = job_spec["label"]

    try:
        _logger.info(f"Starting simulation for {illuminator} -> {telescope}")

        simulator = SimulatorLightEmission(
            light_emission_config=config,
            telescope=telescope,
            label=label,
        )

        simulator.simulate()
        simulator.validate_simulations()

        _logger.info(f"Completed simulation for {illuminator} -> {telescope}")

        return {
            "illuminator": illuminator,
            "telescope": telescope,
            "wavelength": job_spec.get("wavelength"),
            "success": True,
            "error": None,
        }

    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning(f"Failed simulation for {illuminator} -> {telescope}: {exc}")
        return {
            "illuminator": illuminator,
            "telescope": telescope,
            "wavelength": job_spec.get("wavelength"),
            "success": False,
            "error": str(exc),
        }


class MultiIlluminatorSimulator:
    """
    Orchestrate parallel simulations of multiple illuminator-telescope pairs.

    This class manages simulations for all valid illuminator-telescope combinations
    defined in a visibility table, running them in parallel across multiple CPU cores.

    Parameters
    ----------
    config : dict
        Base configuration for light emission simulations. Will be updated for
        each illuminator-telescope pair. Should contain:
        - site: str (North/South)
        - model_version: str
        - number_of_events: int
        - other SimulatorLightEmission parameters
    visibility_data : dict, optional
        Dictionary with "columns" and "rows" keys containing the visibility table.
        The expected structure is:
        - columns: ["illuminator_id", "telescope_id", "visible"]
        - rows: list of [illuminator_id, telescope_id, visible] lists
        If not provided, the visibility table is retrieved from the site model
        using the site and model_version from config.
    label : str, optional
        Base label for all simulations. Pair-specific labels will be appended.
    max_workers : int or None, optional
        Maximum number of parallel worker processes. If None, uses 60% of CPU cores.
        If <= 0, uses all available cores.
    """

    def __init__(self, config, visibility_data=None, label=None, max_workers=None):
        """Initialize the multi-illuminator simulator."""
        self._logger = logging.getLogger(__name__)

        # Load visibility table from provided data or from the site model
        if visibility_data is None:
            visibility_data = self._load_visibility_from_site_model(config)

        self.visibility = IlluminatorTelescopeVisibility(visibility_data)

        self.base_config = config
        self.label = label or "multi_illuminator"
        self.max_workers = determine_max_workers(max_workers)
        self.results = None

        self._logger.info(f"Will use {self.max_workers} parallel workers")

    @staticmethod
    def _load_visibility_from_site_model(config):
        """
        Load visibility data from the site model database.

        Parameters
        ----------
        config : dict
            Configuration containing "site" and "model_version" keys.

        Returns
        -------
        dict
            Visibility table dictionary with "columns" and "rows" keys.
        """
        # Import here to avoid loading SiteModel when visibility_data is provided directly
        from simtools.model.site_model import SiteModel  # pylint: disable=import-outside-toplevel

        site_model = SiteModel(
            site=config["site"],
            model_version=config["model_version"],
        )
        return site_model.get_parameter_value("illuminator_telescope_visibility")

    def simulate(self, wavelengths=None, illuminators=None, telescopes=None):
        """
        Run simulations for all valid illuminator-telescope pairs in parallel.

        Optionally simulate across multiple wavelengths. If wavelengths are not
        specified and not in base_config, all model wavelengths will be used.

        Parameters
        ----------
        wavelengths : list of astropy.units.Quantity, optional
            List of wavelengths to simulate. If None, uses wavelength from base_config
            if present, otherwise all model wavelengths.
        illuminators : list of str, optional
            Restrict simulations to specific illuminators. If None, simulate all.
        telescopes : list of str, optional
            Restrict simulations to specific telescopes. If None, simulate all.

        Returns
        -------
        list of dict
            List of simulation results. Each dict contains:
            - illuminator: str
            - telescope: str
            - wavelength: astropy.units.Quantity (if wavelength simulation)
            - success: bool
            - error: str or None
        """
        # Get all valid pairs
        all_pairs = self.visibility.get_valid_pairs()

        # Apply filters first (before using a pair to fetch wavelengths)
        if illuminators is not None:
            all_pairs = [(ill, tel) for ill, tel in all_pairs if ill in illuminators]

        if telescopes is not None:
            all_pairs = [(ill, tel) for ill, tel in all_pairs if tel in telescopes]

        if not all_pairs:
            self._logger.warning("No valid pairs to simulate after filtering")
            self.results = []
            return self.results

        # Determine wavelengths: explicit param > config > model
        if wavelengths is not None:
            wavelengths = general.ensure_list(wavelengths)
        elif "wavelength" in self.base_config and self.base_config["wavelength"] is not None:
            wavelengths = general.ensure_list(self.base_config["wavelength"])
        else:
            # Fetch from model using a sample pair
            self._logger.info("No wavelength specified, fetching from model")

            sample_illuminator, sample_telescope = all_pairs[0]
            config_for_query = self.base_config.copy()
            config_for_query["telescope"] = sample_telescope
            config_for_query["light_source"] = sample_illuminator

            wavelengths = SimulatorLightEmission.get_available_wavelengths_from_config(
                config_for_query
            )
            self._logger.info(
                f"Using {len(wavelengths)} wavelengths from model: "
                f"{[w.value for w in wavelengths]} nm"
            )

        # Build job specs for all wavelength-pair combinations
        job_specs = []
        for wavelength in wavelengths:
            for illuminator, telescope in all_pairs:
                # Create config with wavelength
                config_with_wl = self.base_config.copy()
                config_with_wl["wavelength"] = wavelength

                job_spec = {
                    "illuminator": illuminator,
                    "telescope": telescope,
                    "wavelength": wavelength,
                    "site": self.base_config.get("site"),
                    "label": self.label,
                    "config": config_with_wl,
                }
                job_specs.append(job_spec)

        self._logger.info(
            f"Simulating {len(all_pairs)} pair(s) x {len(wavelengths)} wavelength(s) "
            f"= {len(job_specs)} jobs with {self.max_workers} workers"
        )

        # Execute in parallel
        self.results = process_pool_map_ordered(
            _simulate_illuminator_telescope_pair,
            job_specs,
            max_workers=self.max_workers,
        )

        # Log summary
        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful
        self._logger.info(
            f"Simulation complete: {successful} successful, {failed} failed "
            f"across {len(wavelengths)} wavelength(s)"
        )

        return self.results

    def get_summary(self):
        """
        Get a summary of simulation results.

        Returns
        -------
        dict
            Summary containing:
            - total: int - total number of pairs simulated
            - successful: int - number of successful simulations
            - failed: int - number of failed simulations
            - success_rate: float - fraction of successful simulations

        Raises
        ------
        RuntimeError
            If simulations have not been run yet.
        """
        if self.results is None:
            raise RuntimeError(_NO_RESULTS_MSG)

        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
        }

    def get_failed_pairs(self):
        """
        Get list of illuminator-telescope pairs that failed simulation.

        Returns
        -------
        list of tuple
            List of (illuminator, telescope) tuples for failed simulations.

        Raises
        ------
        RuntimeError
            If simulations have not been run yet.
        """
        if self.results is None:
            raise RuntimeError(_NO_RESULTS_MSG)

        return [(r["illuminator"], r["telescope"]) for r in self.results if not r["success"]]

    def get_failed_results(self):
        """
        Get detailed results for failed simulations.

        Returns
        -------
        list of dict
            List of result dictionaries for failed simulations only.

        Raises
        ------
        RuntimeError
            If simulations have not been run yet.
        """
        if self.results is None:
            raise RuntimeError(_NO_RESULTS_MSG)

        return [r for r in self.results if not r["success"]]
