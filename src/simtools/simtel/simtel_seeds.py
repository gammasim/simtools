"""Seeds for sim_telarray simulations."""

import logging
from pathlib import Path

from simtools import settings
from simtools.constants import SIMTEL_MAX_SEED
from simtools.io import ascii_handler
from simtools.utils import names, random
from simtools.version import semver_to_int


class SimtelSeeds:
    """Manage seeds for sim_telarray simulations."""

    def __init__(
        self, output_path=None, site=None, model_version=None, zenith_angle=None, azimuth_angle=None
    ):
        """
        Initialize seeds for sim_telarray simulations.

        Two seeds are set for sim_telarray simulations:

            - Instrument seed: used to randomize the instrument setup
            - Simulation seed: used for the shower simulation

        Parameters
        ----------
        output_path : str or Path or None
            Output path for the seed file.
        site : str or None
            Site name.
        model_version : str or None
            Model version.
        zenith_angle : float or None
            Zenith angle.
        azimuth_angle : float or None
            Azimuth angle.
        """
        self._logger = logging.getLogger(__name__)

        self.instrument_seed = settings.config.args.get("sim_telarray_instrument_seed", None)
        self.instruments = settings.config.args.get(
            "sim_telarray_random_instrument_instances", None
        )
        self.simulation_seed = settings.config.args.get("sim_telarray_seed", None)
        self.seed_file = settings.config.args.get("sim_telarray_seed_file", None)
        if output_path is not None:
            self.seed_file = Path(output_path) / self.seed_file

        self.seed_string = self.initialize_seeds(site, model_version, zenith_angle, azimuth_angle)

    def initialize_seeds(self, site, model_version, zenith_angle, azimuth_angle):
        """Initialize seeds based on provided parameters."""
        if isinstance(self.simulation_seed, list):
            return self._set_fixed_seeds()

        if not self.simulation_seed:
            self.simulation_seed = random.seeds(max_seed=SIMTEL_MAX_SEED)

        if not self.instruments or self.instruments <= 1:
            return self._generate_seed_pair()

        return self._generate_seeds_with_file(site, model_version, zenith_angle, azimuth_angle)

    def _set_fixed_seeds(self):
        """
        Set fixed seeds to be using for testing purposes only.

        Fixes both instrument and simulation seeds.
        """
        try:
            seed_string = f"{self.simulation_seed[0]},{self.simulation_seed[1]}"
        except IndexError as exc:
            raise IndexError(
                "Two seeds must be provided for testing purposes: "
                "first for instrument, second for shower simulation."
            ) from exc
        self._logger.warning(f"Using fixed test seeds: {seed_string}")
        return seed_string

    def _generate_seed_pair(self):
        """Generate seed string."""
        if not self.instrument_seed:
            self.instrument_seed = random.seeds(max_seed=SIMTEL_MAX_SEED)

        self._logger.info(
            f"Generated sim_telarray seeds - Instrument: {self.instrument_seed}, "
            f"Shower simulation: {self.simulation_seed}"
        )
        return f"{self.instrument_seed},{self.simulation_seed}"

    def _generate_seeds_with_file(self, site, model_version, zenith_angle, azimuth_angle):
        """Generate a seed file for the instrument seeds and return the seed string."""
        self.instrument_seed = self._get_instrument_seed(
            site, model_version, zenith_angle, azimuth_angle
        )

        self._logger.info(
            f"Writing random instrument seed file {self.seed_file}"
            f" (instrument seed {self.instrument_seed})"
        )
        if self.instruments > 1024:
            raise ValueError("Number of random instances of instrument must be less than 1024")
        random_integers = random.seeds(
            n_seeds=self.instruments,
            max_seed=SIMTEL_MAX_SEED,
            fixed_seed=self.instrument_seed,
        )
        with open(self.seed_file, "w", encoding="utf-8") as file:
            file.write(
                "# Random seeds for instrument configuration generated with seed "
                f"{self.instrument_seed} (model version {model_version}, site {site})\n"
                f"# Zenith angle: {zenith_angle}, Azimuth angle: {azimuth_angle}\n"
            )
            for number in random_integers:
                file.write(f"{number}\n")

        return f"file-by-run:{self.seed_file},{self.simulation_seed}"

    def _get_instrument_seed(self, site, model_version, zenith_angle, azimuth_angle):
        """
        Get configuration dependent instrument seed.

        Three different scenarios are possible:

            - instrument seed provided through configuration: use it
            - site, model_version, zenith_angle, azimuth_angle provided:
              generate a seed based on these parameters
            - none of the above: generate a random seed

        Parameters
        ----------
        site : str or None
            Site name.
        model_version : str or None
            Model version.
        zenith_angle : float or None
            Zenith angle.
        azimuth_angle : float or None
            Azimuth angle.

        Returns
        -------
        int
            Instrument seed.
        """
        # Use the instrument seed from the configuration if provided
        if self.instrument_seed:
            return self.instrument_seed

        # Generate a seed based on site, model_version, zenith_angle, and azimuth_angle
        if model_version and zenith_angle is not None and azimuth_angle is not None:
            try:
                key_index = next(
                    i + 1
                    for i, (_, values) in enumerate(names.site_names().items())
                    if site in values
                )
            except StopIteration as exc:
                raise ValueError(f"Unknown site: {site!r}") from exc

            seed = semver_to_int(model_version) * 10000000
            seed = seed + key_index * 1000000
            seed = seed + int(zenith_angle) * 1000
            return seed + int(azimuth_angle)

        # Generate a random instrument seed
        return random.seeds(max_seed=SIMTEL_MAX_SEED)

    def save_seeds(self, path):
        """
        Save the seeds to a file.

        Parameters
        ----------
        path : str or Path
            Path to the seed file.
        """
        seed_dict = {
            "instrument_seed": self.instrument_seed,
            "simulation_seed": self.simulation_seed,
        }
        ascii_handler.write_data_to_file(path, seed_dict)
