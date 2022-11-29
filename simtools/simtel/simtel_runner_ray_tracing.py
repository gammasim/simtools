import logging

import astropy.units as u

import simtools.util.general as gen
from simtools import io_handler
from simtools.simtel.simtel_runner import SimtelRunner
from simtools.util import names

__all__ = ["SimtelRunnerRayTracing"]

# pylint: disable=no-member
# The line above is needed because there are members which are created
# by adding them to the __dict__ of the class rather than directly.


class SimtelRunnerRayTracing(SimtelRunner):
    """
    SimtelRunnerRayTracing is the interface with sim_telarray to perform ray tracing simulations.

    Configurable parameters:
        zenith_angle:
            len: 1
            unit: deg
            default: 20 deg
        off_axis_angle:
            len: 1
            unit: deg
            default: 0 deg
        source_distance:
            len: 1
            unit: km
            default: 10 km
        use_random_focal_length:
            len: 1
            default: False
        mirror_number:
            len: 1
            default: 1

    Parameters
    ----------
    telescope_model: str
        Instance of TelescopeModel class.
    label: str
        Instance label. Important for output file naming.
    simtel_source_path: str or Path
        Location of sim_telarray installation.
    config_data: dict
        Dict containing the configurable parameters.
    config_file: str or Path
        Path of the yaml file containing the configurable parameters.
    single_mirror_mode: bool
        True for single mirror simulations.
    force_simulate: bool
        Remove existing files and force re-running of the ray-tracing simulation.
    """

    def __init__(
        self,
        telescope_model,
        label=None,
        simtel_source_path=None,
        config_data=None,
        config_file=None,
        single_mirror_mode=False,
        force_simulate=False,
    ):
        """
        Initialize SimtelRunner.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerRayTracing")

        super().__init__(label=label, simtel_source_path=simtel_source_path)

        self.telescope_model = self._validate_telescope_model(telescope_model)
        self.label = label if label is not None else self.telescope_model.label

        self.io_handler = io_handler.IOHandler()
        self._base_directory = self.io_handler.get_output_directory(self.label, "ray-tracing")

        self._single_mirror_mode = single_mirror_mode

        # RayTracing - default parameters
        self._rep_number = 0
        self.RUNS_PER_SET = 1 if self._single_mirror_mode else 20
        self.PHOTONS_PER_RUN = 100000

        # Loading config_data
        _config_data_in = gen.collect_data_from_yaml_or_dict(config_file, config_data)
        _parameter_file = self.io_handler.get_input_data_file(
            "parameters", "simtel-runner-ray-tracing_parameters.yml"
        )
        _parameters = gen.collect_data_from_yaml_or_dict(_parameter_file, None)
        self.config = gen.validate_config_data(_config_data_in, _parameters)

        self._load_required_files(force_simulate)

    def _load_required_files(self, force_simulate):
        """
        Which file are required for running depends on the mode.
        Here we define and write some information into these files. Log files are always required.
        """

        # This file is not actually needed and does not exist in gammasim-tools.
        # However, we need to provide the name of a CORSIKA input file to sim_telarray
        # so it is set up here.
        self._corsika_file = self._simtel_source_path.joinpath("run9991.corsika.gz")

        # Loop to define and remove existing files.
        # Files will be named _base_file = self.__dict__['_' + base + 'File']
        for base_name in ["stars", "photons", "log"]:
            file_name = names.ray_tracing_file_name(
                self.telescope_model.site,
                self.telescope_model.name,
                self.config.source_distance,
                self.config.zenith_angle,
                self.config.off_axis_angle,
                self.config.mirror_number if self._single_mirror_mode else None,
                self.label,
                base_name,
            )
            file = self._base_directory.joinpath(file_name)
            if file.exists() and force_simulate:
                file.unlink()
            # Defining the file name variable as an class attribute.
            self.__dict__["_" + base_name + "_file"] = file

        if not file.exists() or force_simulate:
            # Adding header to photon list file.
            with self._photons_file.open("w") as file:
                file.write("#{}\n".format(50 * "="))
                file.write("# List of photons for RayTracing simulations\n")
                file.write("#{}\n".format(50 * "="))
                file.write("# config_file = {}\n".format(self.telescope_model.get_config_file()))
                file.write("# zenith_angle [deg] = {}\n".format(self.config.zenith_angle))
                file.write("# off_axis_angle [deg] = {}\n".format(self.config.off_axis_angle))
                file.write("# source_distance [km] = {}\n".format(self.config.source_distance))
                if self._single_mirror_mode:
                    file.write("# mirror_number = {}\n\n".format(self.config.mirror_number))

            # Filling in star file with a single light source.
            # Parameters defining light source:
            # - azimuth
            # - elevation
            # - flux
            # - distance of light source
            with self._stars_file.open("w") as file:
                file.write(
                    "0. {} 1.0 {}\n".format(
                        90.0 - self.config.zenith_angle, self.config.source_distance
                    )
                )

    def _shall_run(self, **kwargs):  # pylint: disable=unused-argument; applies only to this line
        """Tells if simulations should be run again based on the existence of output files."""
        return not self._is_photon_list_file_ok()

    def _make_run_command(self, **kwargs):  # pylint: disable=unused-argument
        """Return the command to run simtel_array."""

        if self._single_mirror_mode:
            _mirror_focal_length = float(
                self.telescope_model.get_parameter_value("mirror_focal_length")
            )

        # RayTracing
        command = str(self._simtel_source_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += " -c {}".format(self.telescope_model.get_config_file())
        command += " -I../cfg/CTA"
        command += " -I{}".format(self.telescope_model.get_config_directory())
        command += super()._config_option("IMAGING_LIST", str(self._photons_file))
        command += super()._config_option("stars", str(self._stars_file))
        command += super()._config_option(
            "altitude", self.telescope_model.get_parameter_value("altitude")
        )
        command += super()._config_option(
            "telescope_theta", self.config.zenith_angle + self.config.off_axis_angle
        )
        command += super()._config_option("star_photons", str(self.PHOTONS_PER_RUN))
        command += super()._config_option("telescope_phi", "0")
        command += super()._config_option("camera_transmission", "1.0")
        command += super()._config_option("nightsky_background", "all:0.")
        command += super()._config_option("trigger_current_limit", "1e10")
        command += super()._config_option("telescope_random_angle", "0")
        command += super()._config_option("telescope_random_error", "0")
        command += super()._config_option("convergent_depth", "0")
        command += super()._config_option("maximum_telescopes", "1")
        command += super()._config_option("show", "all")
        command += super()._config_option("camera_filter", "none")
        if self._single_mirror_mode:
            command += super()._config_option("focus_offset", "all:0.")
            command += super()._config_option("camera_config_file", "single_pixel_camera.dat")
            command += super()._config_option("camera_pixels", "1")
            command += super()._config_option("trigger_pixels", "1")
            command += super()._config_option("camera_body_diameter", "0")
            command += super()._config_option(
                "mirror_list",
                self.telescope_model.get_single_mirror_list_file(
                    self.config.mirror_number, self.config.use_random_focal_length
                ),
            )
            command += super()._config_option(
                "focal_length", self.config.source_distance * u.km.to(u.cm)
            )
            command += super()._config_option("dish_shape_length", _mirror_focal_length)
            command += super()._config_option("mirror_focal_length", _mirror_focal_length)
            command += super()._config_option("parabolic_dish", "0")
            # command += super()._config_option('random_focal_length', '0.')
            command += super()._config_option("mirror_align_random_distance", "0.")
            command += super()._config_option("mirror_align_random_vertical", "0.,28.,0.,0.")
        command += " " + str(self._corsika_file)
        command += " 2>&1 > " + str(self._log_file) + " 2>&1"

        return command

    def _check_run_result(self, **kwargs):  # pylint: disable=unused-argument
        """Checking run results.

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

        self._logger.debug("Everything looks fine with output file.")

    def _is_photon_list_file_ok(self):
        """Check if the photon list is valid,"""
        n_lines = 0
        with open(self._photons_file, "r") as ff:
            for _ in ff:
                n_lines += 1
                if n_lines > 100:
                    break

        return n_lines > 100
