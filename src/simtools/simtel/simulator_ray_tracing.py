"""Simulation runner for ray tracing simulations."""

import logging
from collections import namedtuple

import astropy.units as u

from simtools import settings
from simtools.io import io_handler
from simtools.runners.simtel_runner import SimtelRunner
from simtools.utils import names
from simtools.utils.general import clear_default_sim_telarray_cfg_directories

# pylint: disable=no-member
# The line above is needed because there are members which are created
# by adding them to the __dict__ of the class rather than directly.


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
        brackets): zenith_angle (deg), off_axis_angle (deg), source_distance (km),
        single_mirror_mode, use_random_focal_length,
        mirror_numbers.
    force_simulate: bool
        Remove existing files and force re-running of the ray-tracing simulation.
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
        self._logger.debug("Init SimulatorRayTracing")

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
        self.runs_per_set = 1 if self.config.single_mirror_mode else 20
        self.photons_per_run = 100000 if not test else 5000

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
                off_axis_angle=self.config.off_axis_angle,
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
            # Adding header to photon list file.
            with self._photons_file.open("w", encoding="utf-8") as file:
                file.write(f"#{50 * '='}\n")
                file.write("# List of photons for RayTracing simulations\n")
                file.write(f"#{50 * '='}\n")
                file.write(f"# config_file = {self.telescope_model.config_file_path}\n")
                file.write(f"# zenith_angle [deg] = {self.config.zenith_angle}\n")
                file.write(f"# off_axis_angle [deg] = {self.config.off_axis_angle}\n")
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

    def _make_run_command(self, run_number=None, input_file=None):  # pylint: disable=unused-argument
        """
        Generate sim_telarray run command. Export sim_telarray configuration file(s).

        The run_number and input_file parameters are not relevant for the ray tracing simulation.
        """
        self.telescope_model.write_sim_telarray_config_file(additional_models=self.site_model)

        if self.config.single_mirror_mode:
            # Note: no mirror length defined for dual-mirror telescopes
            _mirror_focal_length = float(
                self.telescope_model.get_parameter_value("mirror_focal_length")
            )

        # RayTracing
        command = str(settings.config.sim_telarray_exe)
        command += f" -c {self.telescope_model.config_file_path}"
        command += f" -I{self.telescope_model.config_file_directory}"
        command += super().get_config_option("random_state", "none")
        command += super().get_config_option("IMAGING_LIST", str(self._photons_file))
        command += super().get_config_option("stars", str(self._stars_file))
        command += super().get_config_option(
            "altitude", self.site_model.get_parameter_value("corsika_observation_level")
        )
        command += super().get_config_option(
            "telescope_theta",
            self.config.zenith_angle + self.config.off_axis_angle,
        )
        command += super().get_config_option("star_photons", str(self.photons_per_run))
        command += super().get_config_option("telescope_phi", "0")
        command += super().get_config_option("camera_transmission", "1.0")
        command += super().get_config_option("nightsky_background", "all:0.")
        command += super().get_config_option("trigger_current_limit", "1e10")
        command += super().get_config_option("telescope_random_angle", "0")
        command += super().get_config_option("telescope_random_error", "0")
        command += super().get_config_option("convergent_depth", "0")
        command += super().get_config_option("maximum_telescopes", "1")
        command += super().get_config_option("show", "all")
        command += super().get_config_option("camera_filter", "none")
        if self.config.single_mirror_mode:
            command += super().get_config_option("focus_offset", "all:0.")
            # Use per-label camera config to avoid file clobbering in parallel runs.
            command += super().get_config_option(
                "camera_config_file", str(self._single_pixel_camera_file)
            )
            command += super().get_config_option("camera_pixels", "1")
            command += super().get_config_option("trigger_pixels", "1")
            command += super().get_config_option("camera_body_diameter", "0")
            command += super().get_config_option(
                "mirror_list",
                self.telescope_model.get_single_mirror_list_file(
                    self.config.mirror_numbers, self.config.use_random_focal_length
                ),
            )
            command += super().get_config_option(
                "focal_length", self.config.source_distance * u.km.to(u.cm)
            )
            command += super().get_config_option("dish_shape_length", _mirror_focal_length)
            command += super().get_config_option("mirror_focal_length", _mirror_focal_length)
            command += super().get_config_option("parabolic_dish", "0")
            command += super().get_config_option("mirror_align_random_distance", "0.")
            command += super().get_config_option("mirror_align_random_vertical", "0.,28.,0.,0.")
            # Pass mirror_reflection_random_angle on command line to override .cfg file
            rnda = self.telescope_model.get_parameter_value("mirror_reflection_random_angle")
            if isinstance(rnda, list):
                rnda_str = ",".join(str(v) for v in rnda)
            else:
                rnda_str = str(rnda)
            command += super().get_config_option("mirror_reflection_random_angle", rnda_str)
        command += " " + str(settings.config.corsika_dummy_file)

        return clear_default_sim_telarray_cfg_directories(command), self._log_file, self._log_file

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
        self._single_pixel_camera_file = self._base_directory / f"single_pixel_camera_{self.label}.dat"
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
            file.write(
                f'PixType 1   0  0 300   1 300 0.00   "{self._funnel_file}"\n'
            )
            file.write("Pixel 0 1 0. 0.  0  0  0 0x00 1\n")
            file.write("Trigger 1 of 0\n")

    def _config_to_namedtuple(self, data_dict):
        """Convert dict to namedtuple for configuration."""
        config_data = namedtuple(
            "Config",
            [
                "zenith_angle",
                "off_axis_angle",
                "source_distance",
                "single_mirror_mode",
                "use_random_focal_length",
                "mirror_numbers",
            ],
        )
        return config_data(
            zenith_angle=data_dict["zenith_angle"],
            off_axis_angle=data_dict["off_axis_angle"],
            source_distance=data_dict["source_distance"],
            single_mirror_mode=data_dict["single_mirror_mode"],
            use_random_focal_length=data_dict["use_random_focal_length"],
            mirror_numbers=data_dict["mirror_numbers"],
        )
