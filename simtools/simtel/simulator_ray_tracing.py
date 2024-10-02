"""Simulation runner for ray tracing simulations."""

import logging

import astropy.units as u

from simtools.io_operations import io_handler
from simtools.runners.simtel_runner import SimtelRunner
from simtools.utils import names

__all__ = ["SimulatorRayTracing"]

# pylint: disable=no-member
# The line above is needed because there are members which are created
# by adding them to the __dict__ of the class rather than directly.


class SimulatorRayTracing(SimtelRunner):
    """
    SimulatorRayTracing is the interface with sim_telarray to perform ray tracing simulations.

    Parameters
    ----------
    telescope_model: str
        Instance of TelescopeModel class.
    label: str
        Instance label. Important for output file naming.
    simtel_path: str or Path
        Location of sim_telarray installation.
    config_data: namedtuple
        namedtuple containing the configurable parameters: zenith_angle,
        off_axis_angle, source_distance, single_mirror_mode, use_random_focal_length,
        mirror_numbers.
    config_file: str or Path
        Path of the yaml file containing the configurable parameters.
    force_simulate: bool
        Remove existing files and force re-running of the ray-tracing simulation.
    """

    def __init__(
        self,
        telescope_model,
        label=None,
        simtel_path=None,
        config_data=None,
        force_simulate=False,
        test=False,
    ):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimulatorRayTracing")

        super().__init__(label=label, simtel_path=simtel_path)

        self.telescope_model = telescope_model
        self.label = label if label is not None else self.telescope_model.label

        self.io_handler = io_handler.IOHandler()
        self._base_directory = self.io_handler.get_output_directory(self.label, "ray-tracing")

        self.config = config_data
        self._rep_number = 0
        self.runs_per_set = 1 if self.config.single_mirror_mode else 20
        self.photons_per_run = 100000 if not test else 5000

        self._load_required_files(force_simulate)

    def _load_required_files(self, force_simulate):
        """
        Which file are required for running depends on the mode.

        Here we define and write some information into these files. Log files are always required.

        Parameters
        ----------
        force_simulate: bool
            Remove existing files and force re-running of the ray-tracing simulation.
        """
        # This file is not actually needed and does not exist in simtools.
        # However, we need to provide the name of a CORSIKA input file to sim_telarray
        # so it is set up here.
        self._corsika_file = self._simtel_path.joinpath("run9991.corsika.gz")

        # Loop to define and remove existing files.
        # Files will be named _base_file = self.__dict__['_' + base + 'File']
        for base_name in ["stars", "photons", "log"]:
            file_name = names.generate_file_name(
                file_type=base_name,
                suffix=".log" if base_name == "log" else ".lis",
                site=self.telescope_model.site,
                telescope_model_name=self.telescope_model.name,
                source_distance=self.config.source_distance.to("km").value,
                zenith_angle=self.config.zenith_angle.to("deg").value,
                off_axis_angle=self.config.off_axis_angle.to("deg").value,
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
                file.write(f"# config_file = {self.telescope_model.get_config_file()}\n")
                file.write(f"# zenith_angle [deg] = {self.config.zenith_angle.to('deg').value}\n")
                file.write(
                    f"# off_axis_angle [deg] = {self.config.off_axis_angle.to('deg').value}\n"
                )
                file.write(
                    f"# source_distance [km] = {self.config.source_distance.to('km').value}\n"
                )
                if self.config.single_mirror_mode:
                    file.write(f"# mirror_number = {self.config.mirror_numbers}\n\n")

            # Filling in star file with a single light source.
            # Parameters defining light source:
            # - azimuth
            # - elevation
            # - flux
            # - distance of light source
            with self._stars_file.open("w", encoding="utf-8") as file:
                file.write(
                    f"0. {90.0 - self.config.zenith_angle.to('deg').value} "
                    f"1.0 {self.config.source_distance.to('km').value}\n"
                )

        if self.config.single_mirror_mode:
            self._logger.debug("For single mirror mode, need to prepare the single pixel camera.")
            self._write_out_single_pixel_camera_file()

    def _make_run_command(
        self, run_number=None, input_file=None
    ):  # pylint: disable=unused-argument
        """Return the command to run simtel_array."""
        if self.config.single_mirror_mode:
            # TODO SSTs without mirror_focal_length
            _mirror_focal_length = float(
                self.telescope_model.get_parameter_value("mirror_focal_length")
            )

        # RayTracing
        command = str(self._simtel_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {self.telescope_model.get_config_file()}"
        command += f" -I{self.telescope_model.config_file_directory}"
        command += super().get_config_option("random_state", "none")
        command += super().get_config_option("IMAGING_LIST", str(self._photons_file))
        command += super().get_config_option("stars", str(self._stars_file))
        command += super().get_config_option(
            "altitude", self.telescope_model.get_parameter_value("corsika_observation_level")
        )
        command += super().get_config_option(
            "telescope_theta",
            self.config.zenith_angle.to("deg").value + self.config.off_axis_angle.to("deg").value,
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
        # TODO this is a hack
        if self.config.single_mirror_mode:
            command += super().get_config_option("focus_offset", "all:0.")
            command += super().get_config_option("camera_config_file", "single_pixel_camera.dat")
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
                "focal_length", self.config.source_distance.value * u.km.to(u.cm)
            )
            command += super().get_config_option("dish_shape_length", _mirror_focal_length)
            command += super().get_config_option("mirror_focal_length", _mirror_focal_length)
            command += super().get_config_option("parabolic_dish", "0")
            command += super().get_config_option("mirror_align_random_distance", "0.")
            command += super().get_config_option("mirror_align_random_vertical", "0.,28.,0.,0.")
        command += " " + str(self._corsika_file)
        command += " 2>&1 > " + str(self._log_file) + " 2>&1"

        return command

    def _check_run_result(self, run_number=None):  # pylint: disable=unused-argument
        """Check run results.

        Raises
        ------
        RuntimeError
            if Photon list is empty.
        """
        # Checking run
        if not self._is_photon_list_file_ok():
            msg = "Photon list is empty."
            self._logger.error(msg)
            raise RuntimeError(msg)

    def _is_photon_list_file_ok(self):
        """Check if the photon list is valid."""
        n_lines = 0
        with open(self._photons_file, "rb") as ff:
            for _ in ff:
                n_lines += 1
                if n_lines > 100:
                    break

        return n_lines > 100

    def _write_out_single_pixel_camera_file(self):
        """Write out the single pixel camera file."""
        with self.telescope_model.config_file_directory.joinpath("single_pixel_camera.dat").open(
            "w"
        ) as file:
            file.write("# Single pixel camera\n")
            file.write('PixType 1   0  0 300   1 300 0.00   "funnel_perfect.dat"\n')
            file.write("Pixel 0 1 0. 0.  0  0  0 0x00 1\n")
            file.write("Trigger 1 of 0\n")

        # need to also write out the funnel_perfect.dat file
        with self.telescope_model.config_file_directory.joinpath("funnel_perfect.dat").open(
            "w"
        ) as file:
            file.write(
                "# Perfect light collection where the angular efficiency of funnels is needed\n"
            )
            file.write("0    1.0\n")
            file.write("30   1.0\n")
            file.write("60   1.0\n")
            file.write("90   1.0\n")
