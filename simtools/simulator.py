import logging
import re
from collections import defaultdict
from copy import copy
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.util.general as gen
from simtools import io_handler
from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.job_submission.job_manager import JobManager
from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_histograms import SimtelHistograms
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.util import names

__all__ = [
    "Simulator",
    "InvalidRunsToSimulate",
]


class InvalidRunsToSimulate(Exception):
    pass


class Simulator:
    """
    Simulator is responsible for managing simulation of showers and array of telescopes. \
    It interfaces with simulation software-specific packages, like CORSIKA or sim_telarray.

    The configuration is set as a dict configData or a yaml \
    file configFile.

    Example of configData for shower simulations:

    .. code-block:: python

        configData = {
            'dataDirectory': '.',
            'site': 'South',
            'layoutName': 'Prod5',
            'runRange': [1, 100],
            'nshow': 10,
            'primary': 'gamma',
            'erange': [100 * u.GeV, 1 * u.TeV],
            'eslope': -2,
            'zenith': 20 * u.deg,
            'azimuth': 0 * u.deg,
            'viewcone': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }

    Example of configData for array simulations:

    .. code-block:: python

        configData = {
            'dataDirectory': '(..)/data',
            'primary': 'gamma',
            'zenith': 20 * u.deg,
            'azimuth': 0 * u.deg,
            'viewcone': 0 * u.deg,
            # ArrayModel
            'site': 'North',
            'layoutName': '1LST',
            'modelVersion': 'Prod5',
            'default': {
                'LST': '1'
            },
            'MST-01': 'FlashCam-D'
        }

    Attributes
    ----------
    label : str
        Instance label.
    config : NamedTuple
        Configurable parameters.
    arrayModel : ArrayModel
        Instance of ArrayModel.

    Methods
    -------
    run(inputFileList):
        Run simulation.
    simulate(inputFileList, submitCommand=None, extraCommands=None, test=False):
        Submit a run script as a job.
    print_histograms():
        Print histograms and save a pdf file.
    printOutputFiles():
        Print list of output files of simulation run.
    get_list_of_output_files():
        Get list of output files.
    get_list_of_input_files():
        Get list of input files.
    get_list_of_log_files():
        Get list of log files.
    print_list_of_output_files():
        Print list of output files.
    print_list_of_input_files():
        Print list of output files.
    print_list_of_log_files():
        Print list of log files.
    """

    def __init__(
        self,
        simulator,
        simulatorSourcePath,
        label=None,
        configData=None,
        configFile=None,
        submitCommand=None,
        extraCommands=None,
        mongoDBConfig=None,
        test=False,
    ):
        """
        Simulator init.

        Parameters
        ----------
        label: str
            Instance label.
        simulator: choices: [simtel, corsika]
            implemented are sim_telarray and CORSIKA
        simulatorSourcePath: str or Path
            Location of exectutables for simulation software \
                (e.g. path with CORSIKA or sim_telarray)
        configData: dict
            Dict with shower or array model configuration data.
        configFile: str or Path
            Path to yaml file containing configurable data.
        submitCommand: str
            Job submission command.
        extraCommands: str or list of str
            Extra commands to be added to the run script before the run command,
        mongoDBConfig: dict
            MongoDB configuration.
        test: bool
            If True, no jobs are submitted; only run scripts are prepared
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Simulator {}".format(simulator))

        self.label = label
        self._set_simulator(simulator)
        self.runs = list()
        self._results = defaultdict(list)
        self.test = test

        self.io_handler = io_handler.IOHandler()
        self._outputDirectory = self.io_handler.get_output_directory(self.label, self.simulator)
        self._simulatorSourcePath = Path(simulatorSourcePath)
        self._submitCommand = submitCommand
        self._extraCommands = extraCommands
        self._mongoDBConfig = mongoDBConfig

        self._load_configuration_and_simulation_model(configData, configFile)

        self._set_simulation_runner()

    def _set_simulator(self, simulator):
        """
        Set and test simulator type

        Parameters
        ----------
        simulator: choices: [simtel, corsika]
            implemented are sim_telarray and CORSIKA

        Raises
        ------
        gen.InvalidConfigData

        """

        if simulator not in ["simtel", "corsika"]:
            raise gen.InvalidConfigData
        self.simulator = simulator.lower()

    def _load_configuration_and_simulation_model(self, configData=None, configFile=None):
        """
        Load configuration data and initialize simulation models.

        Parameters
        ----------
        configData: dict
            Dict with simulator configuration data.
        configFile: str or Path
            Path to yaml file containing configurable data.

        """
        configData = gen.collect_data_from_yaml_or_dict(configFile, configData)
        if self.simulator == "simtel":
            self._load_sim_tel_config_and_model(configData)
        elif self.simulator == "corsika":
            self._load_corsika_config_and_model(configData)

    def _load_corsika_config_and_model(self, configData):
        """
        Validate configuration data for CORSIKA shower simulation and
        remove entries need needed for CorsikaRunner.

        Parameters
        ----------
        configData: dict
            Dict with simulator configuration data.

        """

        self._corsikaConfigData = copy(configData)

        try:
            self.site = names.validate_site_name(self._corsikaConfigData.pop("site"))
            self.layoutName = names.validate_layout_array_name(
                self._corsikaConfigData.pop("layoutName")
            )
        except KeyError:
            self._logger.error("Missing parameter in simulation configuration data")
            raise

        self.runs = self._validate_run_list_and_range(
            self._corsikaConfigData.pop("runList", None),
            self._corsikaConfigData.pop("runRange", None),
        )

        self._corsikaParametersFile = self._corsikaConfigData.pop("corsikaParametersFile", None)

    def _load_sim_tel_config_and_model(self, configData):
        """
        Load array model and configuration parameters for array simulations

        Parameters
        ----------
        configData: dict
            Dict with simulator configuration data.

        """
        _arrayModelConfig, _restConfig = self._collect_array_model_parameters(configData)

        _parameterFile = self.io_handler.get_input_data_file(
            "parameters", "array-simulator_parameters.yml"
        )
        _parameters = gen.collect_data_from_yaml_or_dict(_parameterFile, None)
        self.config = gen.validate_config_data(_restConfig, _parameters)

        self.arrayModel = ArrayModel(
            label=self.label,
            arrayConfigData=_arrayModelConfig,
            mongoDBConfig=self._mongoDBConfig,
        )

    def _validate_run_list_and_range(self, runList, runRange):
        """
        Prepares list of run numbers from a list or from a range.
        If both arguments are given, they will be merged into a single list.

        Attributes
        ----------
        runList: list
            list of runs (integers)
        runRange:list
            min and max of range of runs to be simulated (two list entries)

        Returns
        -------
        list
            list of unique run numbers (integers)

        """
        if runList is None and runRange is None:
            self._logger.debug("Nothing to validate - runList and runRange not given.")
            return None

        validatedRuns = list()
        if runList is not None:
            if not all(isinstance(r, int) for r in runList):
                msg = "runList must contain only integers."
                self._logger.error(msg)
                raise InvalidRunsToSimulate
            else:
                self._logger.debug("runList: {}".format(runList))
                validatedRuns = list(runList)

        if runRange is not None:
            if not all(isinstance(r, int) for r in runRange) or len(runRange) != 2:
                msg = "runRange must contain two integers only."
                self._logger.error(msg)
                raise InvalidRunsToSimulate
            else:
                runRange = np.arange(runRange[0], runRange[1] + 1)
                self._logger.debug("runRange: {}".format(runRange))
                validatedRuns.extend(list(runRange))

        validatedRunsUnique = sorted(set(validatedRuns))
        return list(validatedRunsUnique)

    def _collect_array_model_parameters(self, configData):
        """
        Separate configuration and model parameters from configuration data.

        Parameters
        ----------
        configData: dict
            Dict with configuration data.

        """
        _arrayModelData = dict()
        _restData = copy(configData)

        try:
            _arrayModelData["site"] = names.validate_site_name(_restData.pop("site"))
            _arrayModelData["layoutName"] = names.validate_layout_array_name(
                _restData.pop("layoutName")
            )
            _arrayModelData["modelVersion"] = _restData.pop("modelVersion")
            _arrayModelData["default"] = _restData.pop("default")
        except KeyError:
            self._logger.error("Missing parameter in simulation configuration data")
            raise

        # Reading telescope keys
        telKeys = [k for k in _restData.keys() if k[1:4] in ["ST-", "CT-"]]
        for key in telKeys:
            _arrayModelData[key] = _restData.pop(key)

        return _arrayModelData, _restData

    def _set_simulation_runner(self):
        """
        Set simulation runners

        """
        if self.simulator == "simtel":
            self._set_simtel_runner()
        elif self.simulator == "corsika":
            self._set_corsika_runner()

    def _set_corsika_runner(self):
        """
        Creating CorsikaRunner.

        """
        self._simulationRunner = CorsikaRunner(
            label=self.label,
            site=self.site,
            layoutName=self.layoutName,
            simtelSourcePath=self._simulatorSourcePath,
            corsikaParametersFile=self._corsikaParametersFile,
            corsikaConfigData=self._corsikaConfigData,
        )

    def _set_simtel_runner(self):
        """
        Creating a SimtelRunnerArray.

        """
        self._simulationRunner = SimtelRunnerArray(
            label=self.label,
            arrayModel=self.arrayModel,
            simtelSourcePath=self._simulatorSourcePath,
            configData={
                "simtelDataDirectory": self.config.dataDirectory,
                "primary": self.config.primary,
                "zenithAngle": self.config.zenithAngle * u.deg,
                "azimuthAngle": self.config.azimuthAngle * u.deg,
            },
        )

    def _fill_results_without_run(self, inputFileList):
        """
        Fill in the results dict without calling submit.

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        """
        inputFileList = self._enforce_list_type(inputFileList)

        for file in inputFileList:
            run = self._guess_run_from_file(file)
            self._fill_results(file, run)
            self.runs.append(run)

    def simulate(self, inputFileList=None):
        """
        Submit a run script as a job.

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        """

        self._logger.info("Submission command: {}".format(self._submitCommand))

        runs_and_files_to_submit = self._get_runs_and_files_to_submit(inputFileList=inputFileList)
        self._logger.info("Starting submission for {} runs".format(len(runs_and_files_to_submit)))

        for run, file in runs_and_files_to_submit.items():

            runScript = self._simulationRunner.get_run_script(
                runNumber=run, inputFile=file, extraCommands=self._extraCommands
            )

            job_manager = JobManager(submitCommand=self._submitCommand, test=self.test)
            job_manager.submit(
                run_script=runScript,
                run_out_file=self._simulationRunner.get_sub_log_file(runNumber=run, mode=""),
            )

            self._fill_results(file, run)

    def file_list(self, inputFileList=None):
        """
        List output files obtained with simulation run

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        """

        runs_and_files_to_submit = self._get_runs_and_files_to_submit(inputFileList=inputFileList)

        for run, _ in runs_and_files_to_submit.items():
            print(
                "{} (file exists: {})".format(
                    str(self._simulationRunner.get_output_file(run)),
                    Path.exists(self._simulationRunner.get_output_file(run)),
                )
            )

    def _get_runs_and_files_to_submit(self, inputFileList=None):
        """
        Return a dictionary with run numbers and (if applicable) simulation
        files. The latter are expected to be given for the simtel simulator.

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        runs_and_files: dict
            dictionary with runnumber as key and (if availble) simulation
            file name as value

        """

        _runs_and_files = {}

        if self.simulator == "simtel":
            _file_list = self._enforce_list_type(inputFileList)
            for file in _file_list:
                _runs_and_files[self._guess_run_from_file(file)] = file
        elif self.simulator == "corsika":
            _run_list = self._get_runs_to_simulate()
            for run in _run_list:
                _runs_and_files[run] = None

        return _runs_and_files

    @staticmethod
    def _enforce_list_type(inputFileList):
        """Enforce the input list to be a list."""
        if not inputFileList:
            return list()
        elif not isinstance(inputFileList, list):
            return [inputFileList]

        return inputFileList

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
        fileName = str(Path(file).name)

        try:
            runStr = re.search("run[0-9]*", fileName).group()
            return int(runStr[3:])
        except (ValueError, AttributeError):
            msg = "Run number could not be guessed from {} using run = 1".format(fileName)
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

        self._results["output"].append(str(self._simulationRunner.get_output_file(run)))
        self._results["sub_out"].append(
            str(self._simulationRunner.get_sub_log_file(run, mode="out"))
        )
        self._results["log"].append(str(self._simulationRunner.get_log_file(run)))
        if self.simulator == "simtel":
            self._results["input"].append(str(file))
            self._results["hist"].append(str(self._simulationRunner.get_histogram_file(run)))
        else:
            self._results["input"].append(None)
            self._results["hist"].append(None)

    def print_histograms(self, inputFileList=None):
        """
        Print histograms and save a pdf file.

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        path
            Path of the pdf file.
        """

        figName = None

        if self.simulator == "simtel":
            if len(self._results["hist"]) == 0 and inputFileList is not None:
                self._fill_results_without_run(inputFileList)

            figName = self._outputDirectory.joinpath("histograms.pdf")
            histFileList = self.get_list_of_histogram_files()
            simtelHistograms = SimtelHistograms(histFileList)
            simtelHistograms.plot_and_save_figures(figName)

        return figName

    def get_list_of_output_files(self, runList=None, runRange=None):
        """
        Get list of output files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Returns
        -------
        list
            List with the full path of all the output files.

        """
        self._logger.info("Getting list of output files")

        if runList or runRange or len(self._results["output"]) == 0:
            runsToList = self._get_runs_to_simulate(runList=runList, runRange=runRange)

            for run in runsToList:
                self._results["output"].append(
                    str(self._simulationRunner.get_output_file(runNumber=run))
                )

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
        self._print_list_of_files(which="log")

    def _make_resources_report(self, inputFileList):
        """
        Prepare a simple report on computing resources used
        (includes wall clock time per run only at this point)

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        Returns
        -------
        dict
           Dictionary with reports on computing resources

        """

        if len(self._results["sub_out"]) == 0 and inputFileList is not None:
            self._fill_results_without_run(inputFileList)

        runtime = list()

        _resources = {}
        for run in self.runs:
            _resources = self._simulationRunner.get_resources(runNumber=run)
            if "runtime" in _resources and _resources["runtime"]:
                runtime.append(_resources["runtime"])

        meanRuntime = np.mean(runtime)

        resource_summary = dict()
        resource_summary["Walltime/run [sec]"] = meanRuntime
        if "nEvents" in _resources and _resources["nEvents"] > 0:
            resource_summary["#events/run"] = _resources["nEvents"]
            resource_summary["Walltime/1000 events [sec]"] = (
                meanRuntime * 1000 / _resources["nEvents"]
            )

        return resource_summary

    def resources(self, inputFileList=None):
        """
        Print a simple report on computing resources used
        (includes run time per run only at this point)

        Parameters
        ----------
        inputFileList: str or list of str
            Single file or list of files of shower simulations.

        """
        resources = self._make_resources_report(inputFileList)
        print("-----------------------------")
        print("Computing Resources Report - {} Simulations".format(self.simulator))
        for key, value in resources.items():
            print("{} = {:.2f}".format(key, value))
        print("-----------------------------")

    def _get_runs_to_simulate(self, runList=None, runRange=None):
        """
        Process runList and runRange and return the validated list of runs.

        Attributes
        ----------
        runList: list
            list of runs (integers)
        runRange:list
            min and max of range of runs to be simulated (two list entries)

        Returns
        -------
        list
            list of unique run numbers (integers)

        """
        if runList is None and runRange is None:
            if self.runs is None:
                msg = (
                    "Runs to simulate were not given as arguments nor " + "in configData - aborting"
                )
                self._logger.error(msg)
                return list()
            else:
                return self.runs
        else:
            return self._validate_run_list_and_range(runList, runRange)

    def _print_list_of_files(self, which):
        """
        Print list of files of a certain type

        Parameters
        ----------
        which str
            file type (e.g., log)

        """

        if which not in self._results:
            self._logger.error("Invalid file type {}".format(which))
            raise KeyError
        for f in self._results[which]:
            print(f)
