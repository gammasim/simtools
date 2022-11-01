import logging

import astropy.units as u

import simtools.util.general as gen
from simtools import io_handler
from simtools.simtel.simtel_runner import SimtelRunner
from simtools.util import names

__all__ = ["SimtelRunnerRayTracing"]


class SimtelRunnerRayTracing(SimtelRunner):
    """
    SimtelRunnerRayTracing is the interface with sim_telarray to perform ray tracing simulations.

    Configurable parameters:
        zenithAngle:
            len: 1
            unit: deg
            default: 20 deg
        offAxisAngle:
            len: 1
            unit: deg
            default: 0 deg
        sourceDistance:
            len: 1
            unit: km
            default: 10 km
        useRandomFocalLength:
            len: 1
            default: False
        mirrorNumber:
            len: 1
            default: 1

    Attributes
    ----------
    label: str, optional
        Instance label.
    telescopeModel: TelescopeModel
        Instance of the TelescopeModel class.
    config: namedtuple
        Contains the configurable parameters (zenithAngle).

    Methods
    -------
    get_run_script(self, test=False, inputFile=None, runNumber=None)
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.
    run(test=False, force=False)
        Run sim_telarray. test=True will make it faster and force=True will remove existing files
        and run again.
    """

    def __init__(
        self,
        telescopeModel,
        label=None,
        simtelSourcePath=None,
        configData=None,
        configFile=None,
        singleMirrorMode=False,
        forceSimulate=False,
    ):
        """
        SimtelRunner.

        Parameters
        ----------
        telescopeModel: str
            Instance of TelescopeModel class.
        label: str, optional
            Instance label. Important for output file naming.
        simtelSourcePath: str (or Path)
            Location of sim_telarray installation.
        configData: dict.
            Dict containing the configurable parameters.
        configFile: str or Path
            Path of the yaml file containing the configurable parameters.
        singleMirrorMode: bool
            True for single mirror simulations.
        forceSimulate: bool
            Remove existing files and force re-running of the ray-tracing simulation
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerRayTracing")

        super().__init__(label=label, simtelSourcePath=simtelSourcePath)

        self.telescopeModel = self._validate_telescope_model(telescopeModel)
        self.label = label if label is not None else self.telescopeModel.label

        self.io_handler = io_handler.IOHandler()
        self._baseDirectory = self.io_handler.get_output_directory(self.label, "ray-tracing")

        self._singleMirrorMode = singleMirrorMode

        # RayTracing - default parameters
        self._repNumber = 0
        self.RUNS_PER_SET = 1 if self._singleMirrorMode else 20
        self.PHOTONS_PER_RUN = 100000

        # Loading configData
        _configDataIn = gen.collect_data_from_yaml_or_dict(configFile, configData)
        _parameterFile = self.io_handler.get_input_data_file(
            "parameters", "simtel-runner-ray-tracing_parameters.yml"
        )
        _parameters = gen.collect_data_from_yaml_or_dict(_parameterFile, None)
        self.config = gen.validate_config_data(_configDataIn, _parameters)

        self._load_required_files(forceSimulate)

    def _load_required_files(self, forceSimulate):
        """
        Which file are required for running depends on the mode.
        Here we define and write some information into these files. Log files are always required.
        """

        self._corsikaFile = self._simtelSourcePath.joinpath("run9991.corsika.gz")

        # Loop to define and remove existing files.
        # Files will be named _baseFile = self.__dict__['_' + base + 'File']
        for baseName in ["stars", "photons", "log"]:
            fileName = names.ray_tracing_file_name(
                self.telescopeModel.site,
                self.telescopeModel.name,
                self.config.sourceDistance,
                self.config.zenithAngle,
                self.config.offAxisAngle,
                self.config.mirrorNumber if self._singleMirrorMode else None,
                self.label,
                baseName,
            )
            file = self._baseDirectory.joinpath(fileName)
            if file.exists() and forceSimulate:
                file.unlink()
            # Defining the file name variable as an class attribute.
            self.__dict__["_" + baseName + "File"] = file

        if not file.exists() or forceSimulate:
            # Adding header to photon list file.
            with self._photonsFile.open("w") as file:
                file.write("#{}\n".format(50 * "="))
                file.write("# List of photons for RayTracing simulations\n")
                file.write("#{}\n".format(50 * "="))
                file.write("# configFile = {}\n".format(self.telescopeModel.get_config_file()))
                file.write("# zenithAngle [deg] = {}\n".format(self.config.zenithAngle))
                file.write("# offAxisAngle [deg] = {}\n".format(self.config.offAxisAngle))
                file.write("# sourceDistance [km] = {}\n".format(self.config.sourceDistance))
                if self._singleMirrorMode:
                    file.write("# mirrorNumber = {}\n\n".format(self.config.mirrorNumber))

            # Filling in star file with a single light source.
            # Parameters defining light source:
            # - azimuth
            # - elevation
            # - flux
            # - distance of light source
            with self._starsFile.open("w") as file:
                file.write(
                    "0. {} 1.0 {}\n".format(
                        90.0 - self.config.zenithAngle, self.config.sourceDistance
                    )
                )

    def _shall_run(self, runNumber=None):
        """Tells if simulations should be run again based on the existence of output files."""
        return not self._is_photon_list_file_ok()

    def _make_run_command(self, inputFile, runNumber=None):
        """Return the command to run simtel_array."""

        if self._singleMirrorMode:
            _mirrorFocalLength = float(
                self.telescopeModel.get_parameter_value("mirror_focal_length")
            )

        # RayTracing
        command = str(self._simtelSourcePath.joinpath("sim_telarray/bin/sim_telarray"))
        command += " -c {}".format(self.telescopeModel.get_config_file())
        command += " -I../cfg/CTA"
        command += " -I{}".format(self.telescopeModel.get_config_directory())
        command += super()._config_option("IMAGING_LIST", str(self._photonsFile))
        command += super()._config_option("stars", str(self._starsFile))
        command += super()._config_option(
            "altitude", self.telescopeModel.get_parameter_value("altitude")
        )
        command += super()._config_option(
            "telescope_theta", self.config.zenithAngle + self.config.offAxisAngle
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
        if self._singleMirrorMode:
            command += super()._config_option("focus_offset", "all:0.")
            command += super()._config_option("camera_config_file", "single_pixel_camera.dat")
            command += super()._config_option("camera_pixels", "1")
            command += super()._config_option("trigger_pixels", "1")
            command += super()._config_option("camera_body_diameter", "0")
            command += super()._config_option(
                "mirror_list",
                self.telescopeModel.get_single_mirror_list_file(
                    self.config.mirrorNumber, self.config.useRandomFocalLength
                ),
            )
            command += super()._config_option(
                "focal_length", self.config.sourceDistance * u.km.to(u.cm)
            )
            command += super()._config_option("dish_shape_length", _mirrorFocalLength)
            command += super()._config_option("mirror_focal_length", _mirrorFocalLength)
            command += super()._config_option("parabolic_dish", "0")
            # command += super()._config_option('random_focal_length', '0.')
            command += super()._config_option("mirror_align_random_distance", "0.")
            command += super()._config_option("mirror_align_random_vertical", "0.,28.,0.,0.")
        command += " " + str(self._corsikaFile)
        command += " 2>&1 > " + str(self._logFile) + " 2>&1"

        return command

    def _check_run_result(self, runNumber=None):
        # Checking run
        if not self._is_photon_list_file_ok():
            self._logger.error("Photon list is empty.")
        else:
            self._logger.debug("Everything looks fine with output file.")

    def _is_photon_list_file_ok(self):
        """Check if the photon list is valid,"""
        nLines = 0
        with open(self._photonsFile, "r") as ff:
            for _ in ff:
                nLines += 1
                if nLines > 100:
                    break

        return nLines > 100
