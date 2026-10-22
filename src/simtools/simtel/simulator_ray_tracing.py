"""Simulation runner for ray tracing simulations."""

import logging
import math
from collections import namedtuple

from simtools import settings
from simtools.io import io_handler
from simtools.ray_tracing import parameter_changes
from simtools.runners.simtel_runner import SimtelRunner
from simtools.utils import names

# pylint: disable=no-member
# The line above is needed because there are members which are created
# by adding them to the __dict__ of the class rather than directly.


def telescope_pointing(zenith_angle, off_axis_x, off_axis_y, azimuth_angle=0.0):
    """Telescope zenith and azimuth angles [deg] for a star offset in the camera.

    The star is at azimuth ``azimuth_angle`` and zenith ``zenith_angle``; the
    telescope points ``off_axis_x`` upwards (camera x points down) and
    ``off_axis_y`` sideways (camera y points right).
    """
    phi = azimuth_angle
    if zenith_angle > 1.0e-6:
        phi += off_axis_y / math.sin(math.radians(zenith_angle))
    return zenith_angle - off_axis_x, phi


class SimulatorRayTracing(SimtelRunner):
    """
    Perform ray tracing simulations with sim_telarray.

    Parameters
    ----------
    telescope_model: TelescopeModel
        telescope model
    site_model: SiteModel
        site model
    label: str
        label used for output file naming.
    config_data: namedtuple
        namedtuple containing the configurable parameters as values (expected units in
        brackets): zenith_angle (deg), off_axis_x (deg), off_axis_y (deg),
        source_distance (km), single_mirror_mode, mirror_numbers, photons_per_run,
        parameter_changes (model-parameter values replacing those of the model) and
        seeds (instrument and simulation seed of each run).
    force_simulate: bool
        Remove existing files and force re-running of the ray-tracing simulation.
    test: bool
        If activated, application will be faster by simulating fewer photons.
    """

    def __init__(
        self,
        telescope_model,
        site_model,
        label=None,
        config_data=None,
        force_simulate=False,
        test=False,
    ):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)
        super().__init__(label=label)

        self.telescope_model = telescope_model
        self.site_model = site_model
        self.label = label if label is not None else self.telescope_model.label

        self.io_handler = io_handler.IOHandler()
        self._base_directory = self.io_handler.get_output_directory()

        self.config = (
            self._config_to_namedtuple(config_data)
            if isinstance(config_data, dict)
            else config_data
        )
        self._rep_number = 0
        # (Instrument seed, simulation seed) of each run; None lets sim_telarray draw them.
        self.seeds = list(
            (config_data.get("seeds") if isinstance(config_data, dict) else None) or [None]
        )
        self.runs_per_set = len(self.seeds)
        self.photons_per_run = self.config.photons_per_run or (5000 if test else 100000)
        self._single_pixel_camera_file = None
        self._funnel_file = None
        self._write_config = True

        self._load_required_files(force_simulate)

    def _load_required_files(self, force_simulate):
        """
        Load required files for the simulation. Depends on the running mode.

        Initialize files for the simulation.

        Parameters
        ----------
        force_simulate: bool
            Remove existing files and force re-running of the ray-tracing simulation.
        """
        # Loop to define and remove existing files.
        # Files will be named _base_file = self.__dict__['_' + base + 'File']
        for base_name in ["stars", "photons", "log"]:
            file_name = names.generate_file_name(
                file_type=f"ray_tracing_{base_name}",
                suffix=".log" if base_name == "log" else ".lis",
                site=self.telescope_model.site,
                telescope_model_name=self.telescope_model.name,
                source_distance=(
                    None if self.config.single_mirror_mode else self.config.source_distance
                ),
                zenith_angle=self.config.zenith_angle,
                off_axis_x=self.config.off_axis_x,
                off_axis_y=self.config.off_axis_y,
                mirror_number=(
                    self.config.mirror_numbers if self.config.single_mirror_mode else None
                ),
                label=self.label,
            )
            file = self._base_directory.joinpath(file_name)
            if file.exists() and force_simulate:
                file.unlink()
            # Defining the file name variable as an class attribute.
            self.__dict__["_" + base_name + "_file"] = file

        if not file.exists() or force_simulate:
            config_file_path = self.telescope_model.config_file_path
            # Adding header to photon list file.
            with self._photons_file.open("w", encoding="utf-8") as file:
                file.write(f"#{50 * '='}\n")
                file.write("# List of photons for RayTracing simulations\n")
                file.write(f"#{50 * '='}\n")
                file.write(f"# config_file = {config_file_path}\n")
                file.write(f"# zenith_angle [deg] = {self.config.zenith_angle}\n")
                file.write(f"# off_axis_x [deg] = {self.config.off_axis_x}\n")
                file.write(f"# off_axis_y [deg] = {self.config.off_axis_y}\n")
                file.write(f"# source_distance [km] = {self.config.source_distance}\n")
                if self.config.single_mirror_mode:
                    file.write(f"# mirror_number = {self.config.mirror_numbers}\n\n")

            # Filling a star file with a single light source defined by
            # - azimuth
            # - elevation
            # - flux
            # - distance of light source
            with self._stars_file.open("w", encoding="utf-8") as file:
                file.write(
                    f"0. {90.0 - self.config.zenith_angle} 1.0 {self.config.source_distance}\n"
                )

        if self.config.single_mirror_mode:
            self._logger.debug("For single mirror mode, need to prepare the single pixel camera.")
            self._write_out_single_pixel_camera_files()

    def make_run_command(self, run_number=None, input_file=None):  # pylint: disable=unused-argument
        """
        Generate sim_telarray run command. Export sim_telarray configuration file(s).

        The run_number and input_file parameters are not relevant for the ray tracing simulation.
        """
        if self._write_config:
            self.telescope_model.write_sim_telarray_config_file(
                additional_models=self.site_model,
                label=self.label,
            )

        config_file_path = self.telescope_model.config_file_path

        telescope_theta, telescope_phi = telescope_pointing(
            self.config.zenith_angle,
            self.config.off_axis_x,
            self.config.off_axis_y,
            self.config.azimuth_angle,
        )

        options = {
            "random_state": "none",
            "IMAGING_LIST": str(self._photons_file),
            "stars": str(self._stars_file),
            "altitude": self.site_model.get_parameter_value("corsika_observation_level"),
            "telescope_theta": telescope_theta,
            "telescope_phi": telescope_phi,
            "star_photons": str(self.photons_per_run),
            "camera_transmission": "1.0",
            "nightsky_background": "all:0.",
            "trigger_current_limit": "1e10",
            "telescope_random_angle": "0",
            "telescope_random_error": "0",
            "convergent_depth": "0",
            "maximum_telescopes": "1",
            "show": "all",
            "camera_filter": "none",
        }

        if self.config.single_mirror_mode:
            options.update(
                {
                    "camera_config_file": str(self._single_pixel_camera_file),
                    "camera_pixels": "1",
                    "trigger_pixels": "1",
                }
            )
        options.update(parameter_changes.sim_telarray_options(self.config.parameter_changes))

        cmd = [
            str(settings.config.sim_telarray_exe),
            "-c",
            str(config_file_path),
            f"-I{self.telescope_model.config_file_directory}",
        ]
        for key, value in options.items():
            cmd.extend(["-C", f"{key}={value}"])
        cmd.append(str(settings.config.corsika_dummy_file))

        return cmd, self._log_file, self._log_file.with_suffix(".err")

    def _command_for_run(self, command, run_index):
        """Run with the instrument and simulation seed of its random instrument instance."""
        seeds = self.seeds[run_index]
        if seeds is None:
            return command
        return [*command[:-1], "-C", f"random_seed={seeds[0]},{seeds[1]}", command[-1]]

    def _check_run_result(self, run_number=None):  # pylint: disable=unused-argument
        """
        Check run results.

        Photon list files should have at least 100 lines.

        Returns
        -------
        bool
            True if photon list is not empty.

        Raises
        ------
        RuntimeError
            if Photon list is empty.
        """
        with open(self._photons_file, "rb") as ff:
            n_lines = sum(1 for _ in ff)
        if n_lines < 100:
            raise RuntimeError("Photon list is empty.")
        return True

    def _write_out_single_pixel_camera_files(self):
        """Write out per-label single pixel camera + funnel files.

        These files are referenced by sim_telarray and must not be shared across
        parallel worker processes (otherwise they can be truncated mid-read).
        """
        self._single_pixel_camera_file = (
            self._base_directory / f"single_pixel_camera_{self.label}.dat"
        )
        self._funnel_file = self._base_directory / f"funnel_perfect_{self.label}.dat"

        # Funnel file (referenced by absolute path from the camera file).
        with self._funnel_file.open("w", encoding="utf-8") as file:
            file.write(
                "# Perfect light collection where the angular efficiency of funnels is needed\n"
            )
            file.write("0    1.0\n")
            file.write("30   1.0\n")
            file.write("60   1.0\n")
            file.write("90   1.0\n")

        # Camera config.
        with self._single_pixel_camera_file.open("w", encoding="utf-8") as file:
            file.write("# Single pixel camera\n")
            file.write(f'PixType 1   0  0 300   1 300 0.00   "{self._funnel_file}"\n')
            file.write("Pixel 0 1 0. 0.  0  0  0 0x00 1\n")
            file.write("Trigger 1 of 0\n")

    def _config_to_namedtuple(self, data_dict):
        """Convert dict to namedtuple for configuration."""
        config_data = namedtuple(
            "Config",
            [
                "zenith_angle",
                "azimuth_angle",
                "off_axis_x",
                "off_axis_y",
                "source_distance",
                "single_mirror_mode",
                "mirror_numbers",
                "photons_per_run",
                "parameter_changes",
            ],
        )
        return config_data(
            zenith_angle=data_dict["zenith_angle"],
            azimuth_angle=data_dict.get("azimuth_angle", 0.0),
            off_axis_x=data_dict["off_axis_x"],
            off_axis_y=data_dict["off_axis_y"],
            source_distance=data_dict["source_distance"],
            single_mirror_mode=data_dict["single_mirror_mode"],
            mirror_numbers=data_dict.get("mirror_numbers"),
            photons_per_run=data_dict.get("photons_per_run"),
            parameter_changes=data_dict.get("parameter_changes") or {},
        )
