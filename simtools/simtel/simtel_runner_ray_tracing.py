import logging

import astropy.units as u

import simtools.utils.general as gen
from simtools.io_operations import io_handler
from simtools.simtel.simtel_runner import SimtelRunner
from simtools.utils import names

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
        single_mirror_mode:
            len: 1
            default: False
        use_random_focal_length:
            len: 1
            default: False
        mirror_numbers:
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
        force_simulate=False,
        test=False,
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

        # Loading config_data
        self.config = gen.validate_config_data(
            gen.collect_data_from_file_or_dict(config_file, config_data),
            self.ray_tracing_default_configuration(True),
        )

        # RayTracing - default parameters
        self._rep_number = 0
        self.runs_per_set = 1 if self.config.single_mirror_mode else 20
        self.photons_per_run = 100000 if not test else 10000

        self._load_required_files(force_simulate)

    def _load_required_files(self, force_simulate):
        """
        Which file are required for running depends on the mode.
        Here we define and write some information into these files. Log files are always required.
        """

        # This file is not actually needed and does not exist in simtools.
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
                self.config.mirror_numbers if self.config.single_mirror_mode else None,
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
            with self._photons_file.open("w", encoding="utf-8") as file:
                file.write(f"#{50 * '='}\n")
                file.write("# List of photons for RayTracing simulations\n")
                file.write(f"#{50 * '='}\n")
                file.write(f"# config_file = {self.telescope_model.get_config_file()}\n")
                file.write(f"# zenith_angle [deg] = {self.config.zenith_angle}\n")
                file.write(f"# off_axis_angle [deg] = {self.config.off_axis_angle}\n")
                file.write(f"# source_distance [km] = {self.config.source_distance}\n")
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
                    f"0. {90.0 - self.config.zenith_angle} 1.0 {self.config.source_distance}\n"
                )

    def _shall_run(self, **kwargs):  # pylint: disable=unused-argument; applies only to this line
        """Tells if simulations should be run again based on the existence of output files."""
        return not self._is_photon_list_file_ok()

    def _make_run_command(self, **kwargs):  # pylint: disable=unused-argument
        """Return the command to run simtel_array."""

        if self.config.single_mirror_mode:
            _mirror_focal_length = float(
                self.telescope_model.get_parameter_value("mirror_focal_length")
            )

        # RayTracing
        command = str(self._simtel_source_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {self.telescope_model.get_config_file()}"
        command += " -I../cfg/CTA"
        command += f" -I{self.telescope_model.get_config_directory()}"
        command += super()._config_option("random_state", "none")
        command += super()._config_option("IMAGING_LIST", str(self._photons_file))
        command += super()._config_option("stars", str(self._stars_file))
        command += super()._config_option(
            "altitude", self.telescope_model.get_parameter_value("altitude")
        )
        command += super()._config_option(
            "telescope_theta", self.config.zenith_angle + self.config.off_axis_angle
        )
        command += super()._config_option("star_photons", str(self.photons_per_run))
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
        if self.config.single_mirror_mode:
            command += super()._config_option("focus_offset", "all:0.")
            command += super()._config_option("camera_config_file", "single_pixel_camera.dat")
            command += super()._config_option("camera_pixels", "1")
            command += super()._config_option("trigger_pixels", "1")
            command += super()._config_option("camera_body_diameter", "0")
            command += super()._config_option(
                "mirror_list",
                self.telescope_model.get_single_mirror_list_file(
                    self.config.mirror_numbers, self.config.use_random_focal_length
                ),
            )
            command += super()._config_option(
                "focal_length", self.config.source_distance * u.km.to(u.cm)
            )
            command += super()._config_option("dish_shape_length", _mirror_focal_length)
            command += super()._config_option("mirror_focal_length", _mirror_focal_length)
            command += super()._config_option("parabolic_dish", "0")
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
        with open(self._photons_file, "rb") as ff:
            for _ in ff:
                n_lines += 1
                if n_lines > 100:
                    break

        return n_lines > 100

    @staticmethod
    def ray_tracing_default_configuration(config_runner=False):
        """
        Get default ray tracing configuration.

        Returns
        -------
        dict
            Default configuration for ray tracing.

        """

        return {
            "zenith_angle": {
                "len": 1,
                "unit": u.Unit("deg"),
                "default": 20.0 * u.deg,
                "names": ["zenith", "theta"],
            },
            "off_axis_angle": {
                "len": 1 if config_runner else None,
                "unit": u.Unit("deg"),
                "default": 0.0 * u.deg,
                "names": ["offaxis", "offset"],
            },
            "source_distance": {
                "len": 1,
                "unit": u.Unit("km"),
                "default": 10.0 * u.km,
                "names": ["sourcedist", "srcdist"],
            },
            "single_mirror_mode": {"len": 1, "default": False},
            "use_random_focal_length": {"len": 1, "default": False},
            "mirror_numbers": {
                "len": 1 if config_runner else None,
                "default": 1 if config_runner else "all",
            },
        }
