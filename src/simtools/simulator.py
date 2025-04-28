"""Simulator class for managing simulations of showers and array of telescopes."""

import gzip
import logging
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import simtools.utils.general as gen
from simtools.corsika.corsika_config import CorsikaConfig
from simtools.io_operations import io_handler
from simtools.job_execution.job_manager import JobManager
from simtools.model.array_model import ArrayModel
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.simtel.simulator_array import SimulatorArray
from simtools.testing.sim_telarray_metadata import assert_sim_telarray_metadata

__all__ = [
    "InvalidRunsToSimulateError",
    "Simulator",
]


class InvalidRunsToSimulateError(Exception):
    """Exception for invalid runs to simulate."""


class Simulator:
    """
    Simulator is managing the simulation of showers and of the array of telescopes.

    It interfaces with simulation software packages (e.g., CORSIKA or sim_telarray).

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
    mongo_db_config: dict
        MongoDB configuration.
    """

    def __init__(
        self,
        args_dict,
        label=None,
        extra_commands=None,
        mongo_db_config=None,
    ):
        """Initialize Simulator class."""
        self._logger = logging.getLogger(__name__)
        self.args_dict = args_dict

        self.simulation_software = self.args_dict["simulation_software"]
        self._logger.debug(f"Init Simulator {self.simulation_software}")
        self.label = label

        self.io_handler = io_handler.IOHandler()

        self.runs = self._initialize_run_list()
        self._results = defaultdict(list)
        self._test = self.args_dict.get("test", False)
        self.submit_engine = self.args_dict.get("submit_engine", "local")
        self._submit_options = self.args_dict.get("submit_options", None)
        self._extra_commands = extra_commands

        self.sim_telarray_seeds = {
            "seed": self.args_dict.get("sim_telarray_instrument_seeds"),
            "random_instrument_instances": self.args_dict.get(
                "sim_telarray_random_instrument_instances"
            ),
            "seed_file_name": "sim_telarray_instrument_seeds.txt",  # name only; no directory
        }
        self.array_models = self._initialize_array_models(mongo_db_config)
        self._simulation_runner = self._initialize_simulation_runner(mongo_db_config)

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
        gen.InvalidConfigDataError

        """
        if simulation_software not in ["sim_telarray", "corsika", "corsika_sim_telarray"]:
            self._logger.error(f"Invalid simulation software: {simulation_software}")
            raise gen.InvalidConfigDataError
        self._simulation_software = simulation_software.lower()

    def _initialize_array_models(self, mongo_db_config):
        """
        Initialize array simulation models.

        Parameters
        ----------
        mongo_db_config: dict
            Database configuration.

        Returns
        -------
        list
            List of ArrayModel objects.
        """
        model_version = self.args_dict.get("model_version", [])
        versions = [model_version] if not isinstance(model_version, list) else model_version

        return [
            ArrayModel(
                label=self.label,
                site=self.args_dict.get("site"),
                layout_name=self.args_dict.get("array_layout_name"),
                mongo_db_config=mongo_db_config,
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

        def semver_to_int(version: str):
            major, minor, patch = map(int, version.split("."))
            return major * 10000 + minor * 100 + patch

        seed = semver_to_int(model_version) * 10000000
        seed = seed + 1000000 if self.args_dict.get("site") != "North" else seed + 2000000
        seed = seed + (int)(self.args_dict["zenith_angle"].value) * 1000
        return seed + (int)(self.args_dict["azimuth_angle"].value)

    def _initialize_run_list(self):
        """
        Initialize run list using the configuration values 'run_number_start' and 'number_of_runs'.

        Returns
        -------
        list
            List of run numbers.

        Raises
        ------
        KeyError
            If 'run_number_start' or 'number_of_runs' are not found in the configuration.
        """
        try:
            return self._validate_run_list_and_range(
                run_list=None,
                run_range=[
                    self.args_dict["run_number_start"],
                    self.args_dict["run_number_start"] + self.args_dict["number_of_runs"],
                ],
            )
        except KeyError as exc:
            self._logger.error(
                "Error in initializing run list (missing 'run_number_start' or 'number_of_runs')"
            )
            raise exc

    def _validate_run_list_and_range(self, run_list, run_range):
        """
        Prepare list of run numbers from a list or from a range.

        If both arguments are given, they will be merged into a single list.

        Attributes
        ----------
        run_list: list
            list of runs (integers)
        run_range:list
            min and max of range of runs to be simulated (two list entries)

        Returns
        -------
        list
            list of unique run numbers (integers)

        """
        if run_list is None and run_range is None:
            self._logger.debug("Nothing to validate - run_list and run_range not given.")
            return None

        validated_runs = []
        if run_list is not None:
            if not isinstance(run_list, list):
                run_list = [run_list]
            if not all(isinstance(r, int) for r in run_list):
                msg = "run_list must contain only integers."
                self._logger.error(msg)
                raise InvalidRunsToSimulateError
            validated_runs = list(run_list)

        if run_range is not None:
            if not all(isinstance(r, int) for r in run_range) or len(run_range) != 2:
                msg = "run_range must contain two integers only."
                self._logger.error(msg)
                raise InvalidRunsToSimulateError

            run_range = np.arange(run_range[0], run_range[1])
            self._logger.debug(f"run_range: {run_range}")
            validated_runs.extend(list(run_range))

        validated_runs_unique = sorted(set(validated_runs))
        self._logger.info(f"run_list: {validated_runs_unique}")
        return list(validated_runs_unique)

    def _initialize_simulation_runner(self, db_config):
        """
        Initialize corsika configuration and simulation runners.

        Parameters
        ----------
        db_config: dict
            Database configuration.

        Returns
        -------
        CorsikaRunner or SimulatorArray or CorsikaSimtelRunner
            Simulation runner object.
        """
        corsika_configurations = []
        for array_model in self.array_models:
            corsika_configurations.append(
                CorsikaConfig(
                    array_model=array_model,
                    label=self.label,
                    args_dict=self.args_dict,
                    db_config=db_config,
                )
            )

        runner_class = {
            "corsika": CorsikaRunner,
            "sim_telarray": SimulatorArray,
            "corsika_sim_telarray": CorsikaSimtelRunner,
        }.get(self.simulation_software)

        # In case of CorsikaSimtelRunner we should pass all corsika_configurations
        # to the runner, since we define multiple configurations to run in a single job
        # using multipipe. In other cases we pass only the first one (there's only one anyway).
        runner_args = {
            "label": self.label,
            "corsika_config": (
                corsika_configurations
                if runner_class is CorsikaSimtelRunner
                else corsika_configurations[0]
            ),
            "simtel_path": self.args_dict.get("simtel_path"),
            "use_multipipe": runner_class is CorsikaSimtelRunner,
        }

        if runner_class is not SimulatorArray:
            runner_args["keep_seeds"] = self.args_dict.get("corsika_test_seeds", False)
        if runner_class is not CorsikaRunner:
            runner_args["sim_telarray_seeds"] = self.sim_telarray_seeds

        return runner_class(**runner_args)

    def _fill_results_without_run(self, input_file_list):
        """
        Fill results dict without calling submit (e.g., for testing).

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.
        """
        input_file_list = self._enforce_list_type(input_file_list)

        for file in input_file_list:
            run = self._guess_run_from_file(file)
            self._fill_results(file, run)
            if run not in self.runs:
                self.runs.append(run)

    def simulate(self, input_file_list=None):
        """
        Submit a run script as a job.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.
        """
        self._logger.info(f"Submission command: {self.submit_engine}")

        runs_and_files_to_submit = self._get_runs_and_files_to_submit(
            input_file_list=input_file_list
        )
        self._logger.info(
            f"Starting submission for {len(runs_and_files_to_submit)} "
            f"run{'s' if len(runs_and_files_to_submit) > 1 else ''}"
        )

        for run_number, input_file in runs_and_files_to_submit.items():
            run_script = self._simulation_runner.prepare_run_script(
                run_number=run_number, input_file=input_file, extra_commands=self._extra_commands
            )

            job_manager = JobManager(
                submit_engine=self.submit_engine,
                submit_options=self._submit_options,
                test=self._test,
            )
            job_manager.submit(
                run_script=run_script,
                run_out_file=self._simulation_runner.get_file_name(
                    file_type="sub_log", run_number=run_number
                ),
                log_file=self._simulation_runner.get_file_name(
                    file_type=("log"), run_number=run_number
                ),
            )

            self._fill_results(input_file, run_number)

    def _get_runs_and_files_to_submit(self, input_file_list=None):
        """
        Return a dictionary with run numbers and simulation files.

        The latter are expected to be given for the simtel simulator.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        runs_and_files: dict
            dictionary with run number as key and (if available) simulation
            file name as value

        Raises
        ------
        ValueError
            If no runs are to be submitted.

        """
        _runs_and_files = {}
        self._logger.debug(f"Getting runs and files to submit ({input_file_list})")

        if self.simulation_software == "sim_telarray":
            input_file_list = self._enforce_list_type(input_file_list)
            _runs_and_files = {self._guess_run_from_file(file): file for file in input_file_list}
        elif self.simulation_software in ["corsika", "corsika_sim_telarray"]:
            _runs_and_files = dict.fromkeys(self._get_runs_to_simulate())
        if len(_runs_and_files) == 0:
            raise ValueError("No runs to submit.")
        return _runs_and_files

    @staticmethod
    def _enforce_list_type(input_file_list):
        """
        Enforce the input list to be a list.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        list
            List of input files.
        """
        if not input_file_list:
            return []
        return input_file_list if isinstance(input_file_list, list) else [input_file_list]

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
            self._logger.warning(f"Run number could not be guessed from {file_name} using run = 1")
            return 1

    def _fill_results(self, file, run_number):
        """
        Fill the results dict with input, output, hist, and log files.

        Parameters
        ----------
        file: str
            input file name
        run_number: int
            run number

        """
        keys = ["output", "sub_out", "log", "input", "hist", "corsika_log"]
        results = {key: [] for key in keys}

        if "sim_telarray" in self.simulation_software:
            results["input"].append(str(file))

        results["sub_out"].append(
            str(
                self._simulation_runner.get_file_name(
                    file_type="sub_log",
                    mode="out",
                    run_number=run_number,
                )
            )
        )

        for model_version_index, _ in enumerate(self.array_models):
            results["output"].append(
                str(
                    self._simulation_runner.get_file_name(
                        file_type="output",
                        run_number=run_number,
                        model_version_index=model_version_index,
                    )
                )
            )
            if "sim_telarray" in self.simulation_software:
                results["log"].append(
                    str(
                        self._simulation_runner.get_file_name(
                            file_type="log",
                            simulation_software="sim_telarray",
                            run_number=run_number,
                            model_version_index=model_version_index,
                        )
                    )
                )
                results["hist"].append(
                    str(
                        self._simulation_runner.get_file_name(
                            file_type="histogram",
                            simulation_software="sim_telarray",
                            run_number=run_number,
                            model_version_index=model_version_index,
                        )
                    )
                )
            if "corsika" in self.simulation_software:
                results["corsika_log"].append(
                    str(
                        self._simulation_runner.get_file_name(
                            file_type="corsika_log",
                            simulation_software="corsika",
                            run_number=run_number,
                            model_version_index=model_version_index,
                        )
                    )
                )

        for key in keys:
            self._results[key].extend(results[key])

    def get_file_list(self, file_type="output"):
        """
        Get list of files generated by simulations.

        Options are "input", "output", "hist", "log", "corsika_log".
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
        self._logger.info(f"Getting list of {file_type} files")
        return self._results[file_type]

    def print_list_of_files(self, file_type="output"):
        """
        Print list of output files generated by simulations.

        Options are "input", "output", "hist", "log".

        Parameters
        ----------
        file_type : str
            File type to be listed.

        """
        self._logger.info(f"Printing list of {file_type} files")
        for file in self._results[file_type]:
            print(file)

    def _make_resources_report(self, input_file_list):
        """
        Prepare a simple report on computing wall clock time used in the simulations.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        dict
           Dictionary with reports on computing resources

        """
        if len(self._results["sub_out"]) == 0:
            if input_file_list is None:
                return {"Wall time/run [sec]": np.nan}
            self._fill_results_without_run(input_file_list)

        runtime = []

        _resources = {}
        for run in self.runs:
            _resources = self._simulation_runner.get_resources(run_number=run)
            if _resources.get("runtime"):
                runtime.append(_resources["runtime"])

        mean_runtime = np.mean(runtime)

        resource_summary = {}
        resource_summary["Wall time/run [sec]"] = mean_runtime
        if "n_events" in _resources and _resources["n_events"] > 0:
            resource_summary["#events/run"] = _resources["n_events"]
            resource_summary["Wall time/1000 events [sec]"] = (
                mean_runtime * 1000 / _resources["n_events"]
            )

        return resource_summary

    def resources(self, input_file_list=None):
        """
        Print a simple report on computing resources used.

        Includes run time per run only at this point.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        """
        resources = self._make_resources_report(input_file_list)
        print("-----------------------------")
        print(f"Computing Resources Report - {self.simulation_software} Simulations")
        for key, value in resources.items():
            print(f"{key} = {value:.2f}")
        print("-----------------------------")

    def _get_runs_to_simulate(self, run_list=None, run_range=None):
        """
        Process run_list and run_range and return the validated list of runs.

        Attributes
        ----------
        run_list: list
            list of runs (integers)
        run_range:list
            min and max of range of runs to be simulated (two list entries)

        Returns
        -------
        list
            list of unique run numbers (integers)

        """
        if run_list is None and run_range is None:
            return [] if self.runs is None else self.runs
        return self._validate_run_list_and_range(run_list, run_range)

    def save_file_lists(self):
        """Save files lists for output and log files."""
        for file_type in ["output", "log", "corsika_log", "hist"]:
            file_name = self.io_handler.get_output_directory(label=self.label).joinpath(
                f"{file_type}_files.txt"
            )
            file_list = self.get_file_list(file_type=file_type)
            if all(element is not None for element in file_list) and len(file_list) > 0:
                self._logger.info(f"Saving list of {file_type} files to {file_name}")
                with open(file_name, "w", encoding="utf-8") as f:
                    for line in self.get_file_list(file_type=file_type):
                        f.write(f"{line}\n")
            else:
                self._logger.debug(f"No files to save for {file_type} files.")

    def pack_for_register(self, directory_for_grid_upload=None):
        """
        Pack simulation output files for registering on the grid.

        Creates separate tarballs for each model version's log files.

        Parameters
        ----------
        directory_for_grid_upload: str
            Directory for the tarball with output files.

        """
        self._logger.info(
            f"Packing the output files for registering on the grid ({directory_for_grid_upload})"
        )
        output_files = self.get_file_list(file_type="output")
        log_files = self.get_file_list(file_type="log")
        corsika_log_files = self.get_file_list(file_type="corsika_log")
        histogram_files = self.get_file_list(file_type="hist")

        directory_for_grid_upload = (
            Path(directory_for_grid_upload)
            if directory_for_grid_upload
            else self.io_handler.get_output_directory(label=self.label).joinpath(
                "directory_for_grid_upload"
            )
        )
        directory_for_grid_upload.mkdir(parents=True, exist_ok=True)

        # If there are more than one model version,
        # duplicate the corsika log file to have one for each model version with the "right name".
        if len(self.array_models) > 1 and corsika_log_files:
            self._copy_corsika_log_file_for_all_versions(corsika_log_files)

        # Group files by model version
        for model in self.array_models:
            model_version = model.model_version

            # Filter files for this model version
            model_logs = [f for f in log_files if model_version in f]
            model_hists = [f for f in histogram_files if model_version in f]
            model_corsika_logs = [f for f in corsika_log_files if model_version in f]

            if model_logs:
                tar_file_name = Path(model_logs[0]).name.replace("log.gz", "log_hist.tar.gz")
                tar_file_path = directory_for_grid_upload.joinpath(tar_file_name)

                with tarfile.open(tar_file_path, "w:gz") as tar:
                    files_to_tar = model_logs + model_hists + model_corsika_logs
                    for file_to_tar in files_to_tar:
                        tar.add(file_to_tar, arcname=Path(file_to_tar).name)

        for file_to_move in output_files:
            source_file = Path(file_to_move)
            destination_file = directory_for_grid_upload / source_file.name
            if destination_file.exists():
                self._logger.warning(f"Overwriting existing file: {destination_file}")
            shutil.move(source_file, destination_file)

        self._logger.info(f"Output files for the grid placed in {directory_for_grid_upload!s}")

    def validate_metadata(self):
        """Validate metadata in the sim_telarray output files."""
        if "sim_telarray" not in self.simulation_software:
            self._logger.info("No sim_telarray files to validate.")
            return

        for model in self.array_models:
            files = self.get_file_list(file_type="output")
            output_file = next((f for f in files if model.model_version in f), None)
            if output_file:
                self._logger.info(f"Validating metadata for {output_file}")
                assert_sim_telarray_metadata(output_file, model)
                self._logger.info(f"Metadata for sim_telarray file {output_file} is valid.")
            else:
                self._logger.warning(
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

            with gzip.open(new_log, "wt") as new_file:
                new_file.write(
                    f"###############################################################\n"
                    f"Copy of CORSIKA log file from model version {original_version}.\n"
                    f"Applicable also for {model.model_version} (same CORSIKA configuration,\n"
                    f"different sim_telarray model versions in the same run).\n"
                    f"###############################################################\n\n"
                )
                with gzip.open(original_log, "rt") as orig_file:
                    shutil.copyfileobj(orig_file, new_file)

            corsika_log_files.append(str(new_log))
