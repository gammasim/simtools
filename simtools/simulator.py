"""Simulator class for managing simulations of showers and array of telescopes."""

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import simtools.utils.general as gen
from simtools.corsika.corsika_config import CorsikaConfig
from simtools.job_execution.job_manager import JobManager
from simtools.model.array_model import ArrayModel
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.simtel.simulator_array import SimulatorArray

__all__ = [
    "Simulator",
    "InvalidRunsToSimulateError",
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
        Configuration dictionary \
        (includes simulation_software, simulation_model, simulation_parameters groups)
    label: str
        Instance label.
    submit_command: str
        Job submission command.
    extra_commands: str or list of str
        Extra commands to be added to the run script before the run command,
    mongo_db_config: dict
        MongoDB configuration.
    test: bool
        If True, no jobs are submitted; only run scripts are prepared
    """

    def __init__(
        self,
        args_dict,
        label=None,
        submit_command=None,
        extra_commands=None,
        mongo_db_config=None,
        test=False,
    ):
        """Initialize Simulator class."""
        self._logger = logging.getLogger(__name__)
        self.args_dict = args_dict

        self.simulation_software = self.args_dict["simulation_software"]
        self._logger.debug(f"Init Simulator {self.simulation_software}")
        self.label = label

        self.runs = self._initialize_run_list()
        self._results = defaultdict(list)
        self._test = test
        self._submit_command = submit_command
        self._extra_commands = extra_commands

        self.array_model = self._initialize_array_model(mongo_db_config)
        self._simulation_runner = self._initialize_simulation_runner()
        self.runner_services = self._simulation_runner.runner_service

    @property
    def simulation_software(self):
        """The attribute simulation_software"""
        return self._simulation_software

    @simulation_software.setter
    def simulation_software(self, simulation_software):
        """
        Set and test simulation_software type

        Parameters
        ----------
        simulation_software: choices: [simtel, corsika, corsika_simtel]
            implemented are sim_telarray and CORSIKA or corsika_simtel
            (running CORSIKA and piping it directly to sim_telarray)

        Raises
        ------
        gen.InvalidConfigDataError

        """
        if simulation_software not in ["simtel", "corsika", "corsika_simtel"]:
            raise gen.InvalidConfigDataError
        self._simulation_software = simulation_software.lower()

    def _initialize_array_model(self, mongo_db_config):
        """
        Initialize array simulation model.

        Parameters
        ----------
        mongo_db_config: dict
            Database configuration.

        Returns
        -------
        ArrayModel
            ArrayModel object.
        """
        return ArrayModel(
            label=self.label,
            site=self.args_dict.get("site"),
            layout_name=self.args_dict.get("array_layout_name"),
            mongo_db_config=mongo_db_config,
            model_version=self.args_dict.get("model_version", None),
        )

    def _initialize_run_list(self):
        """
        Initialize run list.

        Returns
        -------
        list
            List of run numbers.
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
            self._logger.error(f"Error in initializing run list: {exc}")
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

            run_range = np.arange(run_range[0], run_range[1] + 1)
            self._logger.debug(f"run_range: {run_range}")
            validated_runs.extend(list(run_range))

        validated_runs_unique = sorted(set(validated_runs))
        self._logger.info(f"run_list: {validated_runs_unique}")
        return list(validated_runs_unique)

    def _initialize_simulation_runner(self):
        """
        Initialize simulation runners.

        Returns
        -------
        CorsikaRunner or SimulatorArray or CorsikaSimtelRunner
            Simulation runner object.
        """
        if self.simulation_software == "corsika":
            return CorsikaRunner(
                label=self.label,
                corsika_config=CorsikaConfig(
                    array_model=self.array_model,
                    label=self.label,
                    args_dict=self.args_dict,
                ),
                simtel_path=self.args_dict.get("simtel_path"),
                keep_seeds=False,
                use_multipipe=False,
            )
        if self.simulation_software == "simtel":
            return SimulatorArray(self.args_dict)
        if self.simulation_software == "corsika_simtel":
            return CorsikaSimtelRunner(
                label=self.label,
                corsika_config=CorsikaConfig(
                    array_model=self.array_model,
                    label=self.label,
                    args_dict=self.args_dict,
                ),
                simtel_path=self.args_dict.get("simtel_path"),
                keep_seeds=False,
                use_multipipe=False,
            )
        return None

    def _fill_results_without_run(self, input_file_list):
        """
        Fill in the results dict without calling submit.

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
        self._logger.info(f"Submission command: {self._submit_command}")

        runs_and_files_to_submit = self._get_runs_and_files_to_submit(
            input_file_list=input_file_list
        )
        self._logger.info(
            f"Starting submission for {len(runs_and_files_to_submit)} "
            f"run{'s' if len(runs_and_files_to_submit) > 1 else ''}"
        )

        for run, file in runs_and_files_to_submit.items():
            run_script = self._simulation_runner.prepare_run_script(
                run_number=run, input_file=file, extra_commands=self._extra_commands
            )

            job_manager = JobManager(submit_command=self._submit_command, test=self._test)
            job_manager.submit(
                run_script=run_script,
                run_out_file=self.runner_services.get_file_name(
                    file_type="sub_log", **self.runner_services.get_info_for_file_name(run)
                ),
                log_file=self.runner_services.get_file_name(
                    file_type=("log"),
                    **self.runner_services.get_info_for_file_name(run),
                ),
            )

            self._fill_results(file, run)

    def file_list(self, input_file_list=None):
        """
        List output files obtained with simulation run.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        """
        runs_and_files_to_submit = self._get_runs_and_files_to_submit(
            input_file_list=input_file_list
        )

        for run, _ in runs_and_files_to_submit.items():
            output_file_name = self._simulation_runner.get_file_name(
                file_type="output", **self.runner_services.get_info_for_file_name(run)
            )
            print(f"{str(output_file_name)} (file exists: {Path.exists(output_file_name)})")

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

        """
        _runs_and_files = {}
        self._logger.debug("Getting runs and files to submit ({input_file_list})")

        if self.simulation_software == "simtel":
            _file_list = self._enforce_list_type(input_file_list)
            for file in _file_list:
                _runs_and_files[self._guess_run_from_file(file)] = file
        if self.simulation_software in ["corsika", "corsika_simtel"]:
            _run_list = self._get_runs_to_simulate()
            for run in _run_list:
                _runs_and_files[run] = None
        return _runs_and_files

    @staticmethod
    def _enforce_list_type(input_file_list):
        """Enforce the input list to be a list."""
        if not input_file_list:
            return []
        if not isinstance(input_file_list, list):
            return [input_file_list]

        return input_file_list

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
            msg = f"Run number could not be guessed from {file_name} using run = 1"
            self._logger.warning(msg)
            return 1

    def _fill_results(self, file, run):
        """
        Fill the results dict with input, output and log files.

        Parameters
        ----------
        file: str
            input file name
        run: int
            run number

        """
        info_for_file_name = self.runner_services.get_info_for_file_name(run)
        self._results["output"].append(
            str(self.runner_services.get_file_name(file_type="output", **info_for_file_name))
        )
        self._results["sub_out"].append(
            str(
                self.runner_services.get_file_name(
                    file_type="sub_log", **info_for_file_name, mode="out"
                )
            )
        )
        if self.simulation_software in ["simtel", "corsika_simtel"]:
            self._results["log"].append(
                str(self.runner_services.get_file_name(file_type="log", **info_for_file_name))
            )
            self._results["input"].append(str(file))
            self._results["hist"].append(
                str(self.runner_services.get_file_name(file_type="histogram", **info_for_file_name))
            )
        else:
            self._results["corsika_autoinputs_log"].append(
                str(self.runner_services.get_file_name(file_type="log", **info_for_file_name))
            )
            self._results["input"].append(None)
            self._results["hist"].append(None)
            self._results["log"].append(None)

    def get_list_of_output_files(self, run_list=None, run_range=None):
        """
        Get list of output files.

        Parameters
        ----------
        run_list: list
            List of run numbers.
        run_range: list
            List of len 2 with the limits of the range of the run numbers.

        Returns
        -------
        list
            List with the full path of all the output files.

        """
        self._logger.info("Getting list of output files")

        if run_list or run_range or len(self._results["output"]) == 0:
            runs_to_list = self._get_runs_to_simulate(run_list=run_list, run_range=run_range)

            for run in runs_to_list:
                output_file_name = self._simulation_runner.get_file_name(
                    file_type="output", **self.runner_services.get_info_for_file_name(run)
                )
                self._results["output"].append(str(output_file_name))
        return self._results["output"]

    def get_list_of_histogram_files(self):
        """
        Get list of histogram files.

        (not applicable to all simulation types)

        Returns
        -------
        list
            List with the full path of all the histogram files.
        """
        self._logger.info("Getting list of histogram files")
        return self._results["hist"]

    def get_list_of_input_files(self):
        """
        Get list of input files.

        Returns
        -------
        list
            List with the full path of all the input files.
        """
        self._logger.info("Getting list of input files")
        return self._results["input"]

    def get_list_of_log_files(self):
        """
        Get list of log files.

        Returns
        -------
        list
            List with the full path of all the log files.
        """
        self._logger.info("Getting list of log files")
        if self.simulation_software in ["simtel", "corsika_simtel"]:
            return self._results["log"]
        return self._results["log"]

    def print_list_of_output_files(self):
        """Print list of output files."""
        self._logger.info("Printing list of output files")
        self._print_list_of_files(which="output")

    def print_list_of_histogram_files(self):
        """Print list of histogram files."""
        self._logger.info("Printing list of histogram files")
        self._print_list_of_files(which="hist")

    def print_list_of_input_files(self):
        """Print list of output files."""
        self._logger.info("Printing list of input files")
        self._print_list_of_files(which="input")

    def print_list_of_log_files(self):
        """Print list of log files."""
        self._logger.info("Printing list of log files")
        if self.simulation_software in ["simtel", "corsika_simtel"]:
            self._print_list_of_files(which="log")
        else:
            self._print_list_of_files(which="corsika_autoinputs_log")

    def _make_resources_report(self, input_file_list):
        """
        Prepare a simple report on computing resources used.

        Includes wall clock time per run only at this point.

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
                return {"Walltime/run [sec]": np.nan}
            self._fill_results_without_run(input_file_list)

        runtime = []

        _resources = {}
        for run in self.runs:
            _resources = self._simulation_runner.get_resources(run_number=run)
            if "runtime" in _resources and _resources["runtime"]:
                runtime.append(_resources["runtime"])

        mean_runtime = np.mean(runtime)

        resource_summary = {}
        resource_summary["Walltime/run [sec]"] = mean_runtime
        if "n_events" in _resources and _resources["n_events"] > 0:
            resource_summary["#events/run"] = _resources["n_events"]
            resource_summary["Walltime/1000 events [sec]"] = (
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
            if self.runs is None:
                msg = "Runs to simulate were not given - aborting"
                self._logger.error(msg)
                return []
            return self.runs
        return self._validate_run_list_and_range(run_list, run_range)

    def _print_list_of_files(self, which):
        """
        Print list of files of a certain type.

        Parameters
        ----------
        which str
            file type (e.g., log)

        """
        if which not in self._results:
            self._logger.error(f"Invalid file type {which}")
            raise KeyError
        for file in self._results[which]:
            print(file)
