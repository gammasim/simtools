import logging
import re
from collections import defaultdict
from copy import copy
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.corsika_simtel.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.io_operations import io_handler
from simtools.job_execution.job_manager import JobManager
from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.utils import names

__all__ = [
    "Simulator",
    "InvalidRunsToSimulate",
]


class InvalidRunsToSimulate(Exception):
    """Exception for invalid runs to simulate."""


class Simulator:
    """
    Simulator is responsible for managing simulation of showers and array of telescopes. \
    It interfaces with simulation software-specific packages, like CORSIKA or sim_telarray.

    The configuration is set as a dict config_data or a yaml \
    file config_file.

    Example of config_data for shower simulations:

    .. code-block:: python

        config_data = {
            'data_directory': '.',
            'site': 'South',
            'layout_name': 'Prod5',
            'run_range': [1, 100],
            'nshow': 10,
            'primary': 'gamma',
            'erange': [100 * u.GeV, 1 * u.TeV],
            'eslope': -2,
            'zenith': 20 * u.deg,
            'azimuth': 0 * u.deg,
            'viewcone': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }

    Example of config_data for array simulations:

    .. code-block:: python

        config_data = {
            'data_directory': '(..)/data',
            'primary': 'gamma',
            'zenith': 20 * u.deg,
            'azimuth': 0 * u.deg,
            'viewcone': 0 * u.deg,
            # ArrayModel
            'site': 'North',
            'layout_name': '1LST',
            'model_version': 'Prod5',
            'default': {
                'LST': '1'
            },
            'MST-01': 'FlashCam-D'
        }

    Parameters
    ----------
    simulator: choices: [simtel, corsika]
        implemented are sim_telarray and CORSIKA
    simulator_source_path: str or Path
        Location of executables for simulation software \
            (e.g. path with CORSIKA or sim_telarray)
    label: str
        Instance label.
    config_data: dict
        Dict with shower or array model configuration data.
    config_file: str or Path
        Path to yaml file containing configurable data.
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
        simulator,
        simulator_source_path,
        label=None,
        config_data=None,
        config_file=None,
        submit_command=None,
        extra_commands=None,
        mongo_db_config=None,
        test=False,
    ):
        """
        Initialize Simulator class.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Init Simulator {simulator}")

        self.label = label
        self.simulator = simulator
        self.runs = []
        self._results = defaultdict(list)
        self.test = test

        self._corsika_config_data = None
        self.site = None
        self.layout_name = None
        self._corsika_parameters_file = None
        self.config = None
        self.array_model = None
        self._simulation_runner = None

        self.io_handler = io_handler.IOHandler()
        self._output_directory = self.io_handler.get_output_directory(self.label, self.simulator)
        self._simulator_source_path = Path(simulator_source_path)
        self._submit_command = submit_command
        self._extra_commands = extra_commands
        self._mongo_db_config = mongo_db_config

        self._load_configuration_and_simulation_model(config_data, config_file)

        self._set_simulation_runner()

    @property
    def simulator(self):
        """The attribute simulator"""
        return self._simulator

    @simulator.setter
    def simulator(self, simulator):
        """
        Set and test simulator type

        Parameters
        ----------
        simulator: choices: [simtel, corsika, corsika_simtel]
            implemented are sim_telarray and CORSIKA or corsika_simtel
            (running CORSIKA and piping it directly to sim_telarray)

        Raises
        ------
        gen.InvalidConfigData

        """

        if simulator not in ["simtel", "corsika", "corsika_simtel"]:
            raise gen.InvalidConfigData
        self._simulator = simulator.lower()

    def _load_configuration_and_simulation_model(self, config_data=None, config_file=None):
        """
        Load configuration data and initialize simulation models.

        Parameters
        ----------
        config_data: dict
            Dict with simulator configuration data.
        config_file: str or Path
            Path to yaml file containing configurable data.

        """
        simulator_config_data = gen.collect_data_from_file_or_dict(config_file, config_data)
        if self.simulator == "corsika":
            self._load_corsika_config_and_model(simulator_config_data)
        if self.simulator == "simtel":
            self._load_sim_tel_config_and_model(simulator_config_data)
        if self.simulator == "corsika_simtel":
            config_showers, config_arrays = self._separate_corsika_and_simtel_config_data(
                simulator_config_data
            )
            self._load_corsika_config_and_model(config_showers)
            self._load_sim_tel_config_and_model(config_arrays)

    def _load_corsika_config_and_model(self, config_data):
        """
        Validate configuration data for CORSIKA shower simulation and
        remove entries needed for CorsikaRunner.

        Parameters
        ----------
        config_data: dict
            Dict with simulator configuration data.

        """

        self._corsika_config_data = copy(config_data)

        try:
            self.site = names.validate_site_name(self._corsika_config_data.pop("site"))
            self.layout_name = names.validate_array_layout_name(
                self._corsika_config_data.pop("layout_name")
            )
        except KeyError:
            self._logger.error("Missing parameter in simulation configuration data")
            raise

        self.runs = self._validate_run_list_and_range(
            self._corsika_config_data.pop("run_list", None),
            self._corsika_config_data.pop("run_range", None),
        )

        self._corsika_parameters_file = self._corsika_config_data.pop(
            "corsika_parameters_file", None
        )

    def _load_sim_tel_config_and_model(self, config_data):
        """
        Load array model and configuration parameters for array simulations

        Parameters
        ----------
        config_data: dict
            Dict with simulator configuration data.

        """
        _array_model_config, _rest_config = self._collect_array_model_parameters(config_data)

        _parameter_file = self.io_handler.get_input_data_file(
            "parameters", "array-simulator_parameters.yml"
        )
        _parameters = gen.collect_data_from_file_or_dict(_parameter_file, None)
        self.config = gen.validate_config_data(_rest_config, _parameters, ignore_unidentified=True)

        self.array_model = ArrayModel(
            label=self.label,
            array_config_data=_array_model_config,
            mongo_db_config=self._mongo_db_config,
        )

    def _separate_corsika_and_simtel_config_data(self, config_data):
        """
        Separate the CORSIKA and sim_telarray simulation configuration to two dictionaries.

        Parameters
        ----------
        config_data: dict
            Dictionary with both the CORSIKA and sim_telarray simulation configuration data.

        Returns
        -------
        dict
            Configuration of shower simulations.
        dict
            Configuration of array simulations.

        """

        common = copy(config_data.pop("common", {}))
        config_showers = copy(config_data.pop("showers", {})) | common
        config_arrays = copy(config_data.pop("array", {})) | common

        return config_showers, config_arrays

    def _validate_run_list_and_range(self, run_list, run_range):
        """
        Prepares list of run numbers from a list or from a range.
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
                raise InvalidRunsToSimulate

            self._logger.debug(f"run_list: {run_list}")
            validated_runs = list(run_list)

        if run_range is not None:
            if not all(isinstance(r, int) for r in run_range) or len(run_range) != 2:
                msg = "run_range must contain two integers only."
                self._logger.error(msg)
                raise InvalidRunsToSimulate

            run_range = np.arange(run_range[0], run_range[1] + 1)
            self._logger.debug(f"run_range: {run_range}")
            validated_runs.extend(list(run_range))

        validated_runs_unique = sorted(set(validated_runs))
        return list(validated_runs_unique)

    def _collect_array_model_parameters(self, config_data):
        """
        Separate configuration and model parameters from configuration data.

        Parameters
        ----------
        config_data: dict
            Dict with configuration data.

        """
        _array_model_data = {}
        _rest_data = copy(config_data)

        try:
            _array_model_data["site"] = names.validate_site_name(_rest_data.pop("site"))
            _array_model_data["layout_name"] = names.validate_array_layout_name(
                _rest_data.pop("layout_name")
            )
            _array_model_data["model_version"] = _rest_data.pop("model_version")
            _array_model_data["default"] = _rest_data.pop("default")
        except KeyError:
            self._logger.error("Missing parameter in simulation configuration data")
            raise

        # Reading telescope keys
        tel_keys = [k for k in _rest_data.keys() if k[1:4] in ["ST-", "CT-"]]
        for key in tel_keys:
            _array_model_data[key] = _rest_data.pop(key)

        return _array_model_data, _rest_data

    def _set_simulation_runner(self):
        """
        Set simulation runners

        """
        common_args = {
            "label": self.label,
            "simtel_source_path": self._simulator_source_path,
        }
        corsika_args = {
            "mongo_db_config": self._mongo_db_config,
            "site": self.site,
            "layout_name": self.layout_name,
            "corsika_parameters_file": self._corsika_parameters_file,
            "corsika_config_data": self._corsika_config_data,
        }
        if self.simulator in ["simtel", "corsika_simtel"]:
            simtel_args = {
                "array_model": self.array_model,
                "config_data": {
                    "simtel_data_directory": self.config.data_directory,
                    "primary": self.config.primary,
                    "zenith_angle": self.config.zenith_angle * u.deg,
                    "azimuth_angle": self.config.azimuth_angle * u.deg,
                },
            }
        if self.simulator == "corsika":
            self._set_corsika_runner(common_args | corsika_args)
        if self.simulator == "simtel":
            self._set_simtel_runner(common_args | simtel_args)
        if self.simulator == "corsika_simtel":
            self._set_corsika_simtel_runner(common_args, corsika_args, simtel_args)

    def _set_corsika_runner(self, simulator_args):
        """
        Creating CorsikaRunner.

        """
        self._simulation_runner = CorsikaRunner(**simulator_args)

    def _set_simtel_runner(self, simulator_args):
        """
        Creating a SimtelRunnerArray.

        """
        self._simulation_runner = SimtelRunnerArray(**simulator_args)

    def _set_corsika_simtel_runner(self, common_args, corsika_args, simtel_args):
        """
        Creating CorsikaRunner.

        """
        self._simulation_runner = CorsikaSimtelRunner(common_args, corsika_args, simtel_args)

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

            job_manager = JobManager(submit_command=self._submit_command, test=self.test)
            job_manager.submit(
                run_script=run_script,
                run_out_file=self._simulation_runner.get_file_name(
                    file_type="sub_log", **self._simulation_runner.get_info_for_file_name(run)
                ),
                log_file=self._simulation_runner.get_file_name(
                    file_type="corsika_autoinputs_log" if self.simulator == "corsika" else "log",
                    **self._simulation_runner.get_info_for_file_name(run),
                ),
            )

            self._fill_results(file, run)

    def file_list(self, input_file_list=None):
        """
        List output files obtained with simulation run

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
                file_type="output", **self._simulation_runner.get_info_for_file_name(run)
            )
            print(f"{str(output_file_name)} (file exists: {Path.exists(output_file_name)})")

    def _get_runs_and_files_to_submit(self, input_file_list=None):
        """
        Return a dictionary with run numbers and (if applicable) simulation
        files. The latter are expected to be given for the simtel simulator.

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        runs_and_files: dict
            dictionary with runnumber as key and (if availble) simulation
            file name as value

        """

        _runs_and_files = {}

        if self.simulator == "simtel":
            _file_list = self._enforce_list_type(input_file_list)
            for file in _file_list:
                _runs_and_files[self._guess_run_from_file(file)] = file
        if self.simulator in ["corsika", "corsika_simtel"]:
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
        Finds the run number for a given input file name.
        Input file names can follow any pattern with the
        string 'run' followed by the run number.
        If not found, returns 1.

        Parameters
        ----------
        file: Path
            Simulation file name

        """
        file_name = str(Path(file).name)

        try:
            run_str = re.search("run[0-9]*", file_name).group()
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

        info_for_file_name = self._simulation_runner.get_info_for_file_name(run)
        self._results["output"].append(
            str(self._simulation_runner.get_file_name(file_type="output", **info_for_file_name))
        )
        self._results["sub_out"].append(
            str(
                self._simulation_runner.get_file_name(
                    file_type="sub_log", **info_for_file_name, mode="out"
                )
            )
        )
        if self.simulator in ["simtel", "corsika_simtel"]:
            self._results["log"].append(
                str(self._simulation_runner.get_file_name(file_type="log", **info_for_file_name))
            )
            self._results["input"].append(str(file))
            self._results["hist"].append(
                str(
                    self._simulation_runner.get_file_name(
                        file_type="histogram", **info_for_file_name
                    )
                )
            )
        else:
            self._results["corsika_autoinputs_log"].append(
                str(
                    self._simulation_runner.get_file_name(
                        file_type="corsika_autoinputs_log", **info_for_file_name
                    )
                )
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
                    file_type="output", **self._simulation_runner.get_info_for_file_name(run)
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
        if self.simulator in ["simtel", "corsika_simtel"]:
            return self._results["log"]
        return self._results["corsika_autoinputs_log"]

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
        if self.simulator in ["simtel", "corsika_simtel"]:
            self._print_list_of_files(which="log")
        else:
            self._print_list_of_files(which="corsika_autoinputs_log")

    def _make_resources_report(self, input_file_list):
        """
        Prepare a simple report on computing resources used
        (includes wall clock time per run only at this point)

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
        Print a simple report on computing resources used
        (includes run time per run only at this point)

        Parameters
        ----------
        input_file_list: str or list of str
            Single file or list of files of shower simulations.

        """
        resources = self._make_resources_report(input_file_list)
        print("-----------------------------")
        print(f"Computing Resources Report - {self.simulator} Simulations")
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
                msg = "Runs to simulate were not given as arguments nor in config_data - aborting"
                self._logger.error(msg)
                return []

            return self.runs

        return self._validate_run_list_and_range(run_list, run_range)

    def _print_list_of_files(self, which):
        """
        Print list of files of a certain type

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
