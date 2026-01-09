"""Base class for running sim_telarray simulations."""

import logging

import simtools.utils.general as gen
from simtools.job_execution import job_manager
from simtools.runners.runner_services import RunnerServices


class SimtelExecutionError(Exception):
    """Exception for sim_telarray execution error."""


class InvalidOutputFileError(Exception):
    """Exception for invalid output file."""


class SimtelRunner:
    """
    Base class for running simulations based on the sim_telarray software stack.

    The sim_telarray software stack includes sim_telarray itself and e.g., testeff,
    LightEmission, and other software packages.

    Parameters
    ----------
    label: str
        Instance label. Important for output file naming.
    corsika_config: CorsikaConfig
        CORSIKA configuration.
    is_calibration_run: bool
        Flag to indicate if this is a calibration run.
    """

    def __init__(
        self,
        label=None,
        corsika_config=None,
        is_calibration_run=False,
    ):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)

        self.label = label
        self._base_directory = None
        self.is_calibration_run = is_calibration_run

        self.runs_per_set = 1

        self.runner_service = RunnerServices(corsika_config, "sim_telarray", label)
        self.file_list = None

    def run(self, test=False, input_file=None, run_number=None):
        """
        Make run command and run sim_telarray.

        Parameters
        ----------
        test: bool
            If True, make simulations faster.
        input_file: str or Path
            Full path of the input CORSIKA file.
        run_number: int
            Run number.
        """
        self._logger.debug("Running sim_telarray")

        command, stdout_file, stderr_file = self._make_run_command(
            run_number=run_number, input_file=input_file
        )
        if test:
            self._logger.info(f"Running (test) with command: {command}")
            job_manager.submit(
                command,
                out_file=stdout_file,
                err_file=stderr_file,
                env={"SIM_TELARRAY_CONFIG_PATH": ""},
            )
        else:
            self._logger.debug(f"Running ({self.runs_per_set}x) with command: {command}")
            for _ in range(self.runs_per_set):
                job_manager.submit(
                    command,
                    out_file=stdout_file,
                    err_file=stderr_file,
                    env={"SIM_TELARRAY_CONFIG_PATH": ""},
                )

    def _raise_simtel_error(self):
        """
        Raise sim_telarray execution error.

        Final 30 lines from the log file are collected and printed.

        Raises
        ------
        SimtelExecutionError
        """
        if hasattr(self, "_log_file"):
            msg = gen.get_log_excerpt(self._log_file)
        else:
            msg = "Simtel log file does not exist."
        raise SimtelExecutionError(msg)

    def _make_run_command(self, run_number=None, input_file=None):
        """
        Make the sim_telarray run command.

        Returns a list of command arguments.
        """
        self._logger.debug(
            "make_run_command is being called from the base class - "
            "it should be implemented in the sub class"
        )
        input_file = input_file if input_file else "nofile"
        run_number = run_number if run_number else 1
        return [f"{input_file}-{run_number}"], None, None

    @staticmethod
    def get_config_option(par, value=None, weak_option=False):
        """Build sim_telarray command and return as string."""
        option_syntax = "-W" if weak_option else "-C"
        c = f" {option_syntax} {par}"
        c += f"={value}" if value is not None else ""
        return c

    def get_resources(self, run_number=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(run_number)
