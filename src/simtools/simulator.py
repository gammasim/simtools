"""Simulator class for managing simulations of showers and array of telescopes."""

import logging
import shutil
from pathlib import Path

import numpy as np
from astropy import units as u

from simtools import settings
from simtools.corsika.corsika_config import CorsikaConfig
from simtools.io import io_handler, table_handler
from simtools.job_execution import job_manager
from simtools.model.array_model import ArrayModel
from simtools.runners import corsika_runner, corsika_simtel_runner, runner_services, simtel_runner
from simtools.sim_events import file_info, writer
from simtools.simtel.simulator_array import SimulatorArray
from simtools.testing.sim_telarray_metadata import assert_sim_telarray_metadata
from simtools.utils import general, names
from simtools.version import semver_to_int


class Simulator:
    """
    Simulation of showers and of the array of telescopes.

    Interface with the simulation software packages (e.g., CORSIKA or sim_telarray).
    A single run is simulated per instance, possibly for multiple model versions.

    Parameters
    ----------
    label: str
        Instance label.
    extra_commands: str or list of str
        Extra commands to be added to the run script before the run command.
    """

    def __init__(self, label=None, extra_commands=None):
        """Initialize Simulator class."""
        self.logger = logging.getLogger(__name__)
        self.label = label

        self.site = settings.config.args.get("site", None)
        self.model_version = settings.config.args.get("model_version", None)

        self.simulation_software = settings.config.args.get(
            "simulation_software", "corsika_sim_telarray"
        )
        self.run_mode = settings.config.args.get("run_mode", None)

        self.io_handler = io_handler.IOHandler()

        self._extra_commands = extra_commands
        self.sim_telarray_seeds = None
        self.run_number = self._initialize_from_tool_configuration()

        self.array_models, self.corsika_configurations = self._initialize_array_models()
        self._simulation_runner = self._initialize_simulation_runner()
        self.runner_service = runner_services.RunnerServices(
            self._get_first_corsika_config(),
            "sub",
            label,
        )
        self.file_list = self.runner_service.load_files(self.run_number)

    @property
    def simulation_software(self):
        """The attribute simulation_software."""
        return self._simulation_software

    @simulation_software.setter
    def simulation_software(self, simulation_software):
        """
        Set and test simulation_software type.

        Parameters
        ----------
        simulation_software: choices: [sim_telarray, corsika, corsika_sim_telarray]
            implemented are sim_telarray and CORSIKA or corsika_sim_telarray
            (running CORSIKA and piping it directly to sim_telarray)

        Raises
        ------
        ValueError

        """
        if simulation_software not in ["sim_telarray", "corsika", "corsika_sim_telarray"]:
            raise ValueError(f"Invalid simulation software: {simulation_software}")
        self._simulation_software = simulation_software.lower()

    def _initialize_from_tool_configuration(self):
        """Initialize simulator from tool configuration."""
        self.sim_telarray_seeds = {
            "seed": settings.config.args.get("sim_telarray_instrument_seeds"),
            "random_instrument_instances": settings.config.args.get(
                "sim_telarray_random_instrument_instances"
            ),
            "seed_file_name": "sim_telarray_instrument_seeds.txt",  # name only; no directory
        }

        if settings.config.args.get("corsika_file"):
            run_number = file_info.get_corsika_run_number(settings.config.args["corsika_file"])
        else:
            run_number = settings.config.args.get(
                "run_number_offset", 0
            ) + settings.config.args.get("run_number", 1)
        return runner_services.validate_corsika_run_number(run_number)

    def _initialize_array_models(self):
        """
        Initialize array simulation models and CORSIKA config (one per model version).

        Returns
        -------
        list, list
            List of ArrayModel and CorsikaConfig objects.
        """
        versions = general.ensure_iterable(self.model_version)

        array_model = []
        corsika_configurations = []

        for version in versions:
            array_model.append(
                ArrayModel(
                    label=self.label,
                    site=self.site,
                    layout_name=settings.config.args.get("array_layout_name"),
                    model_version=version,
                    calibration_device_types=self._get_calibration_device_types(self.run_mode),
                    overwrite_model_parameters=settings.config.args.get(
                        "overwrite_model_parameters"
                    ),
                )
            )
            corsika_configurations.append(
                CorsikaConfig(
                    array_model=array_model[-1],
                    label=self.label,
                    run_number=self.run_number,
                )
            )
            array_model[-1].sim_telarray_seeds = {
                "seed": self._get_seed_for_random_instrument_instances(
                    self.sim_telarray_seeds["seed"],
                    version,
                    corsika_configurations[-1].zenith_angle,
                    corsika_configurations[-1].azimuth_angle,
                ),
                "random_instrument_instances": self.sim_telarray_seeds[
                    "random_instrument_instances"
                ],
                "seed_file_name": self.sim_telarray_seeds["seed_file_name"],
            }

        #  'corsika_sim_telarray' allows for multiple model versions (multipipe option)
        corsika_configurations = (
            corsika_configurations
            if self.simulation_software == "corsika_sim_telarray"
            else corsika_configurations[0]
        )

        return array_model, corsika_configurations

    def _get_seed_for_random_instrument_instances(
        self, seed, model_version, zenith_angle, azimuth_angle
    ):
        """
        Generate seed for random instances of the instrument.

        Parameters
        ----------
        seed : str
            Seed string given through configuration.
        model_version: str
            Model version.
        zenith_angle: float
            Zenith angle of the observation (in degrees).
        azimuth_angle: float
            Azimuth angle of the observation (in degrees).

        Returns
        -------
        int
            Seed for random instances of the instrument.
        """
        if seed:
            return int(seed.split(",")[0].strip())

        def key_index(key):
            try:
                return list(names.site_names()).index(key) + 1
            except ValueError:
                return 1

        seed = semver_to_int(model_version) * 10000000
        seed = seed + key_index(self.site) * 1000000
        seed = seed + (int)(zenith_angle) * 1000
        return seed + (int)(azimuth_angle)

    def _initialize_simulation_runner(self):
        """
        Initialize corsika configuration and simulation runners.

        Returns
        -------
        CorsikaRunner or SimulatorArray or CorsikaSimtelRunner
            Simulation runner object.
        """
        runner_class = {
            "corsika": corsika_runner.CorsikaRunner,
            "sim_telarray": SimulatorArray,
            "corsika_sim_telarray": corsika_simtel_runner.CorsikaSimtelRunner,
        }.get(self.simulation_software)

        runner_args = {
            "label": self.label,
            "corsika_config": self.corsika_configurations,
        }

        if runner_class is not SimulatorArray:
            runner_args["corsika_seeds"] = settings.config.args.get("corsika_seeds", False)
            runner_args["curved_atmosphere_min_zenith_angle"] = settings.config.args.get(
                "curved_atmosphere_min_zenith_angle", 65 * u.deg
            )
        if runner_class is not corsika_runner.CorsikaRunner:
            runner_args["sim_telarray_seeds"] = self.sim_telarray_seeds
        if runner_class is corsika_simtel_runner.CorsikaSimtelRunner:
            runner_args["sequential"] = settings.config.args.get("sequential", False)

        return runner_class(**runner_args)

    def simulate(self):
        """
        Prepare and submit a run script as a job.

        Writes submission scripts using the simulation runners and submits the
        run script to the job manager. Collects generated files.
        """
        self._simulation_runner.prepare_run(
            run_number=self.run_number,
            corsika_file=self._get_corsika_file(),
            sub_script=self.runner_service.get_file_name("sub_script", self.run_number),
            extra_commands=self._extra_commands,
        )
        self.update_file_lists()

        job_manager.submit(
            command=self.runner_service.get_file_name("sub_script", self.run_number),
            out_file=self.runner_service.get_file_name("sub_out", self.run_number),
            err_file=self.runner_service.get_file_name("sub_err", self.run_number),
            env=simtel_runner.SIM_TELARRAY_ENV,
        )

    def _get_corsika_file(self):
        """
        Get the CORSIKA input file if applicable (for sim_telarray simulations).

        Returns
        -------
        Path, None
            Path to the CORSIKA input file.
        """
        if self.simulation_software == "sim_telarray":
            return settings.config.args.get("corsika_file", None)
        return None

    def get_files(self, file_type):
        """
        Get file(s) generated by simulations.

        Not all file types are available for all simulation types.
        Returns an empty list for an unknown file type.

        Parameters
        ----------
        file_type : str
            File type to be listed.

        Returns
        -------
        Path or list[Path]:
            File or list with the full path of output files of a certain file type.

        """
        return self.file_list[file_type]

    def _make_resources_report(self):
        """
        Prepare a simple report on computing wall clock time used in the simulations.

        Returns
        -------
        str
           string reporting on computing resources

        """
        runtime = []
        _resources = self._simulation_runner.get_resources(self.get_files(file_type="sub_out"))
        if _resources.get("runtime"):
            runtime.append(_resources["runtime"])

        mean_runtime = np.mean(runtime)

        resource_summary = f"Mean wall time/run [sec]: {mean_runtime}"
        if "n_events" in _resources and _resources["n_events"] > 0:
            resource_summary += f", #events/run: {_resources['n_events']}"

        return resource_summary

    def report(self):
        """
        Report on simulations and computing resources used.

        Includes run time per run only at this point.
        """
        _corsika_config = self._get_first_corsika_config()
        self.logger.info(
            f"Production run complete for primary {_corsika_config.primary_particle} showers "
            f"from {_corsika_config.azimuth_angle} azimuth and "
            f"{_corsika_config.zenith_angle} zenith "
            f"at {self.site} site, using {self.model_version} model."
        )
        self.logger.info(
            f"Computing for {self.simulation_software} Simulations: {self._make_resources_report()}"
        )

    def save_file_lists(self):
        """Save file lists for output and log files."""
        outdir = self.io_handler.get_output_directory()

        for file_type, files in self.file_list.items():
            if not files or any(f is None for f in files):
                self.logger.debug(f"No files to save for {file_type} files.")
                continue

            path = outdir / f"{file_type}_files.txt"
            self.logger.info(f"Saving list of {file_type} files to {path}")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(map(str, files)) + "\n")

    def save_reduced_event_lists(self):
        """
        Save reduced event lists with event data on simulated and triggered events.

        The files are saved with the same name as the sim_telarray output file
        but with a 'hdf5' extension.
        """
        if "sim_telarray" not in self.simulation_software:
            self.logger.warning(
                "Reduced event lists can only be saved for sim_telarray simulations."
            )
            return

        input_files = self.get_files(file_type="sim_telarray_output")
        output_files = self.get_files(file_type="sim_telarray_event_data")
        for input_file, output_file in zip(input_files, output_files):
            generator = writer.EventDataWriter([input_file])
            table_handler.write_tables(
                tables=generator.process_files(),
                output_file=Path(output_file),
                overwrite_existing=True,
            )

    def pack_for_register(self, directory_for_grid_upload=None):
        """
        Pack simulation output files for registering on the grid.

        Creates separate tarballs for each model version's log files.

        Parameters
        ----------
        directory_for_grid_upload: str
            Directory for the tarball with output files.

        """
        self.logger.info(
            f"Packing output files for registering on the grid ({directory_for_grid_upload})"
        )
        output_files = self.get_files(file_type="sim_telarray_output")
        log_files = self.get_files(file_type="sim_telarray_log")
        histogram_files = self.get_files(file_type="sim_telarray_histogram")
        reduced_event_files = (
            self.get_files(file_type="sim_telarray_event_data")
            if settings.config.args.get("save_reduced_event_lists")
            else []
        )

        directory_for_grid_upload = (
            Path(directory_for_grid_upload)
            if directory_for_grid_upload
            else self.io_handler.get_output_directory().joinpath("directory_for_grid_upload")
        )
        directory_for_grid_upload.mkdir(parents=True, exist_ok=True)

        # Group files by model version
        for model in self.array_models:
            model_version = model.model_version
            model_logs = [f for f in log_files if model_version in str(f)]

            if not model_logs:
                continue

            tar_name = Path(model_logs[0]).name.replace("simtel.log.gz", "log_hist.tar.gz")
            tar_path = directory_for_grid_upload / tar_name

            files_to_tar = (
                model_logs
                + [f for f in histogram_files if model_version in str(f)]
                + [str(self.get_files(file_type="corsika_log"))]
                + list(general.ensure_iterable(model.pack_model_files()))
            )
            general.pack_tar_file(tar_path, files_to_tar)

        for file_to_move in output_files + reduced_event_files:
            destination_file = directory_for_grid_upload / Path(file_to_move).name
            if destination_file.exists():
                self.logger.warning(f"Overwriting existing file: {destination_file}")
            shutil.move(file_to_move, destination_file)

        self.logger.info(f"Grid output files grid placed in {directory_for_grid_upload!s}")

    def validate_metadata(self):
        """Validate metadata in the sim_telarray output files."""
        if "sim_telarray" not in self.simulation_software:
            self.logger.info("No sim_telarray files to validate.")
            return

        for model in self.array_models:
            files = general.ensure_iterable(self.get_files(file_type="sim_telarray_output"))

            output_file = next((f for f in files if model.model_version in str(f)), None)
            if output_file:
                self.logger.info(f"Validating metadata for {output_file}")
                assert_sim_telarray_metadata(output_file, model)
                self.logger.info(f"Metadata for sim_telarray file {output_file} is valid.")
            else:
                self.logger.warning(
                    f"No sim_telarray file found for model version {model.model_version}: {files}"
                )

    @staticmethod
    def _get_calibration_device_types(run_mode):
        """
        Get the list of calibration device types based on the run mode.

        Parameters
        ----------
        run_mode: str
            Run mode of the simulation.

        Returns
        -------
        list
            List of calibration device types.
        """
        if run_mode == "direct_injection":
            return ["flat_fielding"]
        return []

    def _get_first_corsika_config(self):
        """
        Return first instance from list of CORSIKA configurations.

        Most values stored in the CORSIKA configurations are identical,
        with the exception of the simulation model version dependent parameters.

        Returns
        -------
        CorsikaConfig
            First CORSIKA configuration instance.
        """
        try:
            return (
                self.corsika_configurations[0]
                if isinstance(self.corsika_configurations, list)
                else self.corsika_configurations
            )
        except (IndexError, TypeError) as exc:
            raise ValueError("CORSIKA configuration not found for verification.") from exc

    def verify_simulations(self):
        """
        Verify simulations.

        This includes checking the number of simulated events.

        """
        self.logger.info("Verifying simulations.")

        _corsika_config = self._get_first_corsika_config()
        expected_shower_events = _corsika_config.shower_events
        expected_mc_events = _corsika_config.mc_events

        self.logger.info(
            f"Expected number of shower events: {expected_shower_events}, "
            f"expected number of MC events: {expected_mc_events}"
        )
        if self.simulation_software in ["corsika_sim_telarray", "sim_telarray"]:
            self._verify_simulated_events_in_sim_telarray(
                expected_shower_events, expected_mc_events
            )
        if self.simulation_software == "corsika":
            self._verify_simulated_events_corsika(expected_mc_events)
        if settings.config.args.get("save_reduced_event_lists"):
            self._verify_simulated_events_in_reduced_event_lists(expected_mc_events)

    def _verify_simulated_events_corsika(self, expected_mc_events, tolerance=1.0e-3):
        """
        Verify the number of simulated events in CORSIKA output files.

        Allow for a small mismatch in the number of requested events.

        Parameters
        ----------
        expected_mc_events: int
            Expected number of simulated MC events.

        Raises
        ------
        ValueError
            If the number of simulated events does not match the expected number.
        """

        def consistent(a, b, tol):
            return abs(a - b) / max(a, b) <= tol

        event_errors = []

        file = self.get_files(file_type="corsika_output")
        shower_events, _ = file_info.get_simulated_events(file)

        if shower_events != expected_mc_events:
            if consistent(shower_events, expected_mc_events, tol=tolerance):
                self.logger.warning(
                    f"Small mismatch in number of events in: {file}: "
                    f"shower events: {shower_events} (expected: {expected_mc_events})"
                )
            else:
                event_errors.append(
                    f"Number of simulated MC events ({shower_events}) does not match "
                    f"the expected number ({expected_mc_events}) in CORSIKA {file}."
                )
        else:
            self.logger.info(
                f"Consistent number of events in: {file}: shower events: {shower_events}"
            )

        if event_errors:
            self.logger.error("Inconsistent event counts found in CORSIKA output:")
            for error in event_errors:
                self.logger.error(f" - {error}")
            error_message = "Inconsistent event counts found in CORSIKA output:\n" + "\n".join(
                f" - {error}" for error in event_errors
            )
            raise ValueError(error_message)

    def _verify_simulated_events_in_sim_telarray(self, expected_shower_events, expected_mc_events):
        """
        Verify the number of simulated events.

        Parameters
        ----------
        expected_shower_events: int
            Expected number of simulated shower events.
        expected_mc_events: int
            Expected number of simulated MC events.

        Raises
        ------
        ValueError
            If the number of simulated events does not match the expected number.
        """
        event_errors = []
        for file in general.ensure_iterable(self.get_files(file_type="sim_telarray_output")):
            shower_events, mc_events = file_info.get_simulated_events(file)

            if (shower_events, mc_events) != (expected_shower_events, expected_mc_events):
                event_errors.append(
                    f"Event mismatch: shower/MC events in {file}: {shower_events}/{mc_events}"
                    f" (expected: {expected_shower_events}/{expected_mc_events})"
                )
            else:
                self.logger.info(
                    f"Consistent number of events in: {file}: "
                    f"shower events: {shower_events}, "
                    f"MC events: {mc_events}"
                )

        if event_errors:
            self.logger.error("Inconsistent event counts found:")
            for error in event_errors:
                self.logger.error(f" - {error}")
            error_message = "Inconsistent event counts found:\n" + "\n".join(
                f" - {error}" for error in event_errors
            )
            raise ValueError(error_message)

    def _verify_simulated_events_in_reduced_event_lists(self, expected_mc_events):
        """
        Verify the number of simulated events in reduced event lists.

        Parameters
        ----------
        expected_mc_events: int
            Expected number of simulated MC events.

        Raises
        ------
        ValueError
            If the number of simulated events does not match the expected number.
        """
        event_errors = []
        for file in self.get_files(file_type="sim_telarray_event_data"):
            tables = table_handler.read_tables(file, ["SHOWERS"])
            try:
                mc_events = len(tables["SHOWERS"])
            except KeyError as exc:
                raise ValueError(f"SHOWERS table not found in reduced event list {file}.") from exc

            if mc_events != expected_mc_events:
                event_errors.append(
                    f"Number of simulated MC events ({mc_events}) does not match "
                    f"the expected number ({expected_mc_events}) in reduced event list {file}."
                )
            else:
                self.logger.info(
                    f"Consistent number of events in reduced event list: {file}: MC events:"
                    f" {mc_events}"
                )

        if event_errors:
            self.logger.error("Inconsistent event counts found in reduced event lists:")
            for error in event_errors:
                self.logger.error(f" - {error}")
            error_message = "Inconsistent event counts found in reduced event lists:\n" + "\n".join(
                f" - {error}" for error in event_errors
            )
            raise ValueError(error_message)

    def update_file_lists(self):
        """
        Update file lists with all data, log, histogram and submission files.

        Some of these files are generated by the simulation runners.
        """
        if self.file_list is None:
            self.file_list = self._simulation_runner.file_list
        else:
            self.file_list.update(self._simulation_runner.file_list)
