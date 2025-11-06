"""Simulator class for managing simulations of showers and array of telescopes."""

import gzip
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy import units as u

import simtools.utils.general as gen
from simtools.corsika.corsika_config import CorsikaConfig
from simtools.io import io_handler, table_handler
from simtools.job_execution.job_manager import JobManager
from simtools.model.array_model import ArrayModel
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.simtel.simtel_io_event_writer import SimtelIOEventDataWriter
from simtools.simtel.simtel_io_file_info import get_simulated_events
from simtools.simtel.simulator_array import SimulatorArray
from simtools.testing.sim_telarray_metadata import assert_sim_telarray_metadata
from simtools.version import semver_to_int


class InvalidRunsToSimulateError(Exception):
    """Exception for invalid runs to simulate."""


class Simulator:
    """
    Simulator is managing the simulation of showers and of the array of telescopes.

    It interfaces with simulation software packages (e.g., CORSIKA or sim_telarray).
    A single run is simulated per instance, possibly for multiple model versions.

    The configuration is set as a dict corresponding to the command line configuration groups
    (especially simulation_software, simulation_model, simulation_parameters).

    Parameters
    ----------
    args_dict : dict
        Configuration dictionary
        (includes simulation_software, simulation_model, simulation_parameters groups).
    label: str
        Instance label.
    extra_commands: str or list of str
        Extra commands to be added to the run script before the run command.
    db_config: dict
        Database configuration.
    """

    def __init__(
        self,
        args_dict,
        label=None,
        extra_commands=None,
        db_config=None,
    ):
        """Initialize Simulator class."""
        self.logger = logging.getLogger(__name__)
        self.label = label

        self.args_dict = args_dict
        self.db_config = db_config

        self.simulation_software = self.args_dict.get("simulation_software", "corsika_sim_telarray")
        self.logger.debug(f"Init Simulator {self.simulation_software}")
        self.run_mode = args_dict.get("run_mode", None)

        self.io_handler = io_handler.IOHandler()

        self.run_number = self.args_dict.get("run_number_offset", 0) + self.args_dict.get(
            "run_number", 1
        )
        self._results = defaultdict(list)
        self._test = self.args_dict.get("test", False)
        self._extra_commands = extra_commands

        self.sim_telarray_seeds = {
            "seed": self.args_dict.get("sim_telarray_instrument_seeds"),
            "random_instrument_instances": self.args_dict.get(
                "sim_telarray_random_instrument_instances"
            ),
            "seed_file_name": "sim_telarray_instrument_seeds.txt",  # name only; no directory
        }
        self.array_models = self._initialize_array_models()
        self._simulation_runner = self._initialize_simulation_runner()

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

    def _initialize_array_models(self):
        """
        Initialize array simulation models (one per model version).

        Returns
        -------
        list
            List of ArrayModel objects.
        """
        versions = gen.ensure_iterable(self.args_dict.get("model_version", []))

        return [
            ArrayModel(
                label=self.label,
                site=self.args_dict.get("site"),
                layout_name=self.args_dict.get("array_layout_name"),
                db_config=self.db_config,
                model_version=version,
                sim_telarray_seeds={
                    "seed": self._get_seed_for_random_instrument_instances(
                        self.sim_telarray_seeds["seed"], version
                    ),
                    "random_instrument_instances": self.sim_telarray_seeds[
                        "random_instrument_instances"
                    ],
                    "seed_file_name": self.sim_telarray_seeds["seed_file_name"],
                },
                simtel_path=self.args_dict.get("simtel_path", None),
                calibration_device_types=self._get_calibration_device_types(
                    self.args_dict.get("run_mode")
                ),
                overwrite_model_parameters=self.args_dict.get("overwrite_model_parameters", None),
            )
            for version in versions
        ]

    def _get_seed_for_random_instrument_instances(self, seed, model_version):
        """
        Generate seed for random instances of the instrument.

        Parameters
        ----------
        seed : str
            Seed string given through configuration.
        model_version: str
            Model version.

        Returns
        -------
        int
            Seed for random instances of the instrument.
        """
        if seed:
            return int(seed.split(",")[0].strip())

        seed = semver_to_int(model_version) * 10000000
        seed = seed + 1000000 if self.args_dict.get("site") != "North" else seed + 2000000
        seed = seed + (int)(self.args_dict.get("zenith_angle", 0.0 * u.deg).value) * 1000
        return seed + (int)(self.args_dict.get("azimuth_angle", 0.0 * u.deg).value)

    def _corsika_configuration(self):
        """
        Define CORSIKA configurations based on the simulation model.

        For 'corsika_sim_telarray', this is a list since multiple configurations
        might be defined to run in a single job using multipipe.

        Returns
        -------
        CorsikaConfig or list of CorsikaConfig
            CORSIKA configuration(s) based on the simulation model.
        """
        corsika_configurations = []
        for array_model in self.array_models:
            corsika_configurations.append(
                CorsikaConfig(
                    array_model=array_model,
                    label=self.label,
                    args_dict=self.args_dict,
                    db_config=self.db_config,
                    dummy_simulations=self._is_calibration_run(self.run_mode),
                )
            )
        return (
            corsika_configurations
            if self.simulation_software == "corsika_sim_telarray"
            else corsika_configurations[0]
        )

    def _initialize_simulation_runner(self):
        """
        Initialize corsika configuration and simulation runners.

        Returns
        -------
        CorsikaRunner or SimulatorArray or CorsikaSimtelRunner
            Simulation runner object.
        """
        corsika_configurations = self._corsika_configuration()

        runner_class = {
            "corsika": CorsikaRunner,
            "sim_telarray": SimulatorArray,
            "corsika_sim_telarray": CorsikaSimtelRunner,
        }.get(self.simulation_software)

        runner_args = {
            "label": self.label,
            "corsika_config": corsika_configurations,
            "simtel_path": self.args_dict.get("simtel_path"),
            "use_multipipe": runner_class is CorsikaSimtelRunner,
        }

        if runner_class is not SimulatorArray:
            runner_args["keep_seeds"] = self.args_dict.get("corsika_test_seeds", False)
            runner_args["curved_atmosphere_min_zenith_angle"] = self.args_dict.get(
                "curved_atmosphere_min_zenith_angle", 65 * u.deg
            )
        if runner_class is not CorsikaRunner:
            runner_args["sim_telarray_seeds"] = self.sim_telarray_seeds
        if runner_class is CorsikaSimtelRunner:
            runner_args["sequential"] = self.args_dict.get("sequential", False)
            runner_args["calibration_config"] = (
                self.args_dict if self._is_calibration_run(self.run_mode) else None
            )

        return runner_class(**runner_args)

    def simulate(self):
        """
        Prepare and submit a run script as a job.

        Writes submission scripts using the simulation runners and submits the
        run script to the job manager. Collects generated files.
        """
        run_script = self._simulation_runner.prepare_run_script(
            run_number=self.run_number, input_file=None, extra_commands=self._extra_commands
        )

        job_manager = JobManager(test=self._test)
        job_manager.submit(
            run_script=run_script,
            run_out_file=self._simulation_runner.get_file_name(
                file_type="sub_log", run_number=self.run_number
            ),
            log_file=self._simulation_runner.get_file_name(
                file_type=("log"), run_number=self.run_number
            ),
        )

        self._fill_list_of_generated_files()

    def _guess_run_from_file(self, file):
        """
        Extract the run number from the given file name.

        Input file names can follow any pattern with the
        string 'run' followed by the run number.

        Parameters
        ----------
        file: Path
            Simulation file name

        Returns
        -------
        int
            The extracted run number. If extraction fails, returns 1 and logs a warning.
        """
        file_name = str(Path(file).name)

        try:
            run_str = re.search(r"run\d*", file_name).group()
            return int(run_str[3:])
        except (ValueError, AttributeError):
            self.logger.warning(f"Run number could not be guessed from {file_name} using run = 1")
            return 1

    def _fill_list_of_generated_files(self):
        """Fill a dictionary with lists of generated files."""
        keys = [
            "simtel_output",
            "sub_out",
            "log",
            "input",
            "histogram",
            "corsika_log",
            "event_data",
        ]
        results = {key: [] for key in keys}

        def get_file_name(name, **kwargs):
            return str(self._simulation_runner.get_file_name(file_type=name, **kwargs))

        results["sub_out"].append(get_file_name("sub_log", mode="out", run_number=self.run_number))

        for i in range(len(self.array_models)):
            results["simtel_output"].append(
                get_file_name("simtel_output", run_number=self.run_number, model_version_index=i)
            )

            if "sim_telarray" in self.simulation_software:
                for file_type in ("log", "histogram", "event_data"):
                    results[file_type].append(
                        get_file_name(
                            file_type,
                            simulation_software="sim_telarray",
                            run_number=self.run_number,
                            model_version_index=i,
                        )
                    )

            if "corsika" in self.simulation_software:
                results["corsika_log"].append(
                    get_file_name(
                        "corsika_log",
                        simulation_software="corsika",
                        run_number=self.run_number,
                        model_version_index=i,
                    )
                )

        for key in keys:
            self._results[key].extend(results[key])

        print("AAAA", self._results)

    def get_file_list(self, file_type="simtel_output"):
        """
        Get list of files generated by simulations.

        Not all file types are available for all simulation types.
        Returns an empty list for an unknown file type.

        Parameters
        ----------
        file_type : str
            File type to be listed.

        Returns
        -------
        list
            List with the full path of all output files.

        """
        return self._results[file_type]

    def _make_resources_report(self, input_file_list):
        """
        Prepare a simple report on computing wall clock time used in the simulations.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        str
           string reporting on computing resources

        """
        if len(self._results["sub_out"]) == 0 and input_file_list is None:
            return "Mean wall time/run [sec]: np.nan"

        runtime = []

        _resources = {}
        _resources = self._simulation_runner.get_resources(run_number=self.run_number)
        if _resources.get("runtime"):
            runtime.append(_resources["runtime"])

        mean_runtime = np.mean(runtime)

        resource_summary = f"Mean wall time/run [sec]: {mean_runtime}"
        if "n_events" in _resources and _resources["n_events"] > 0:
            resource_summary += f", #events/run: {_resources['n_events']}"

        return resource_summary

    def report(self, input_file_list=None):
        """
        Report on simulations and computing resources used.

        Includes run time per run only at this point.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        """
        self.logger.info(
            f"Production run complete for primary {self.args_dict['primary']} showers "
            f"from {self.args_dict['azimuth_angle']} azimuth and "
            f"{self.args_dict['zenith_angle']} zenith "
            f"at {self.args_dict['site']} site, using {self.args_dict['model_version']} model."
        )
        self.logger.info(
            f"Computing for {self.simulation_software} Simulations: "
            f"{self._make_resources_report(input_file_list)}"
        )

    def save_file_lists(self):
        """Save files lists for output and log files."""
        for file_type in ["simtel_output", "log", "corsika_log", "histogram"]:
            file_name = self.io_handler.get_output_directory().joinpath(f"{file_type}_files.txt")
            file_list = self.get_file_list(file_type=file_type)
            if all(element is not None for element in file_list) and len(file_list) > 0:
                self.logger.info(f"Saving list of {file_type} files to {file_name}")
                with open(file_name, "w", encoding="utf-8") as f:
                    for line in self.get_file_list(file_type=file_type):
                        f.write(f"{line}\n")
            else:
                self.logger.debug(f"No files to save for {file_type} files.")

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

        input_files = self.get_file_list(file_type="simtel_output")
        output_files = self.get_file_list(file_type="event_data")
        for input_file, output_file in zip(input_files, output_files):
            generator = SimtelIOEventDataWriter([input_file])
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
            f"Packing the output files for registering on the grid ({directory_for_grid_upload})"
        )
        output_files = self.get_file_list(file_type="simtel_output")
        log_files = self.get_file_list(file_type="log")
        corsika_log_files = self.get_file_list(file_type="corsika_log")
        histogram_files = self.get_file_list(file_type="histogram")
        reduced_event_files = (
            self.get_file_list(file_type="event_data")
            if self.args_dict.get("save_reduced_event_lists")
            else []
        )

        directory_for_grid_upload = (
            Path(directory_for_grid_upload)
            if directory_for_grid_upload
            else self.io_handler.get_output_directory().joinpath("directory_for_grid_upload")
        )
        directory_for_grid_upload.mkdir(parents=True, exist_ok=True)

        # If there are more than one model version,
        # duplicate the corsika log file to have one for each model version with the "right name".
        if len(self.array_models) > 1 and corsika_log_files:
            self._copy_corsika_log_file_for_all_versions(corsika_log_files)

        # Group files by model version
        for model in self.array_models:
            model_version = model.model_version
            model_files = gen.ensure_iterable(model.pack_model_files())

            # Filter files for this model version
            model_logs = [f for f in log_files if model_version in f]
            model_hists = [f for f in histogram_files if model_version in f]
            model_corsika_logs = [f for f in corsika_log_files if model_version in f]

            if model_logs:
                tar_file_name = Path(model_logs[0]).name.replace("log.gz", "log_hist.tar.gz")
                tar_file_path = directory_for_grid_upload.joinpath(tar_file_name)
                # Add all relevant model, log, histogram, and CORSIKA log files to the tarball
                files_to_tar = model_logs + model_hists + model_corsika_logs + model_files
                gen.pack_tar_file(tar_file_path, files_to_tar)

        for file_to_move in output_files + reduced_event_files:
            source_file = Path(file_to_move)
            destination_file = directory_for_grid_upload / source_file.name
            if destination_file.exists():
                self.logger.warning(f"Overwriting existing file: {destination_file}")
            shutil.move(source_file, destination_file)

        self.logger.info(f"Output files for the grid placed in {directory_for_grid_upload!s}")

    def validate_metadata(self):
        """Validate metadata in the sim_telarray output files."""
        if "sim_telarray" not in self.simulation_software:
            self.logger.info("No sim_telarray files to validate.")
            return

        for model in self.array_models:
            files = self.get_file_list(file_type="simtel_output")
            output_file = next((f for f in files if model.model_version in f), None)
            if output_file:
                self.logger.info(f"Validating metadata for {output_file}")
                assert_sim_telarray_metadata(output_file, model)
                self.logger.info(f"Metadata for sim_telarray file {output_file} is valid.")
            else:
                self.logger.warning(
                    f"No sim_telarray file found for model version {model.model_version}: {files}"
                )

    def _copy_corsika_log_file_for_all_versions(self, corsika_log_files):
        """
        Create copies of the CORSIKA log file for each model version.

        Adds a header comment to each copy explaining its relationship to the original.

        Parameters
        ----------
        corsika_log_files: list
            List containing the original CORSIKA log file path.
        """
        original_log = Path(corsika_log_files[0])
        # Find which model version the original log belongs to
        original_version = next(
            model.model_version
            for model in self.array_models
            if re.search(
                rf"(?<![0-9A-Za-z]){re.escape(model.model_version)}(?![0-9A-Za-z])",
                original_log.name,
            )
        )

        for model in self.array_models:
            if model.model_version == original_version:
                continue

            new_log = original_log.parent / original_log.name.replace(
                original_version, model.model_version
            )

            with gzip.open(new_log, "wt", encoding="utf-8") as new_file:
                # Write the header to the new file
                header = (
                    f"###############################################################\n"
                    f"Copy of CORSIKA log file from model version {original_version}.\n"
                    f"Applicable also for {model.model_version} (same CORSIKA configuration,\n"
                    f"different sim_telarray model versions in the same run).\n"
                    f"###############################################################\n\n"
                )
                new_file.write(header)

                # Copy the content of the original log file, ignoring invalid characters
                with gzip.open(original_log, "rt", encoding="utf-8", errors="ignore") as orig_file:
                    for line in orig_file:
                        new_file.write(line)

            corsika_log_files.append(str(new_log))

    @staticmethod
    def _is_calibration_run(run_mode):
        """
        Check if this simulation is a calibration run.

        Parameters
        ----------
        run_mode: str
            Run mode of the simulation.

        Returns
        -------
        bool
            True if it is a calibration run, False otherwise.
        """
        return run_mode in [
            "pedestals",
            "dark_pedestals",
            "nsb_only_pedestals",
            "direct_injection",
        ]

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

    def verify_simulations(self):
        """
        Verify simulations.

        This includes checking the number of simulated events.

        """
        self.logger.info("Verifying simulations.")

        expected_shower_events = self.args_dict.get("nshow", 0)
        # core scatter is a list: first element is the usage factor
        expected_mc_events = expected_shower_events * self.args_dict.get("core_scatter", [0])[0]

        self.logger.info(
            f"Expected number of shower events: {expected_shower_events}, "
            f"expected number of MC events: {expected_mc_events}"
        )

        if self.simulation_software in ["corsika_sim_telarray", "sim_telarray"]:
            self._verify_simulated_events_in_sim_telarray(
                expected_shower_events, expected_mc_events
            )
        if self.args_dict.get("save_reduced_event_lists"):
            self._verify_simulated_events_in_reduced_event_lists(expected_mc_events)

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
        for file in self.get_file_list(file_type="simtel_output"):
            shower_events, mc_events = get_simulated_events(file)

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
        for file in self.get_file_list(file_type="event_data"):
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
