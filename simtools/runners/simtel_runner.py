"""Base class for running sim_telarray simulations."""

import logging
import os
from pathlib import Path

import simtools.utils.general as gen

__all__ = ["InvalidOutputFileError", "SimtelExecutionError", "SimtelRunner"]

# pylint: disable=no-member
# The line above is needed because there are methods which are used in this class
# but are implemented in the classes inheriting from it.


class SimtelExecutionError(Exception):
    """Exception for simtel_array execution error."""


class InvalidOutputFileError(Exception):
    """Exception for invalid output file."""


class SimtelRunner:
    """
    Base class for running sim_telarray simulations.

    Parameters
    ----------
    simtel_path: str or Path
        Location of sim_telarray installation.
    label: str
        Instance label. Important for output file naming.
    """

    def __init__(self, simtel_path, label=None):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)

        self._simtel_path = Path(simtel_path)
        self.label = label
        self._script_dir = None
        self._base_directory = None

        self.runs_per_set = 1

    def __repr__(self):
        """Return a string representation of the SimtelRunner object."""
        return f"SimtelRunner(label={self.label})\n"

    def prepare_run_script(self, test=False, input_file=None, run_number=None, extra_commands=None):
        """
        Build and return the full path of the bash run script containing the sim_telarray command.

        Parameters
        ----------
        test: bool
            Test flag for faster execution.
        input_file: str or Path
            Full path of the input CORSIKA file.
        run_number: int
            Run number.
        extra_commands: str
            Additional commands for running simulations given in config.yml.

        Returns
        -------
        Path
            Full path of the run script.
        """
        self._logger.debug("Creating run bash script")
        self._logger.debug(f"Run number: {run_number} {self.corsika_config.run_number}")

        script_file_path = self.get_file_name(file_type="sub_script", run_number=run_number)

        self._logger.debug(f"Run bash script - {script_file_path}")

        self._logger.debug(f"Extra commands to be added to the run script {extra_commands}")

        command = self._make_run_command(run_number=run_number, input_file=input_file)
        with script_file_path.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")
            file.write("set -e\n")
            file.write("set -o pipefail\n")
            file.write("\nSECONDS=0\n")

            if extra_commands is not None:
                file.write("# Writing extras\n")
                for line in extra_commands:
                    file.write(f"{line}\n")
                file.write("# End of extras\n\n")

            n = 1 if test else self.runs_per_set
            for _ in range(n):
                file.write(f"{command}\n\n")

            file.write('\necho "RUNTIME: $SECONDS"\n')

        os.system(f"chmod ug+x {script_file_path}")
        return script_file_path

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

        command = self._make_run_command(run_number=run_number, input_file=input_file)

        if test:
            self._logger.info(f"Running (test) with command: {command}")
            self._run_simtel_and_check_output(command)
        else:
            self._logger.debug(f"Running ({self.runs_per_set}x) with command: {command}")
            for _ in range(self.runs_per_set):
                self._run_simtel_and_check_output(command)

        self._check_run_result(run_number=run_number)

    @staticmethod
    def _simtel_failed(sys_output):
        """Test if simtel process ended successfully.

        Returns
        -------
        bool
            1 if sys_output is different than 0, and 1 otherwise.
        """
        return sys_output != 0

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

        self._logger.error(msg)
        raise SimtelExecutionError(msg)

    def _run_simtel_and_check_output(self, command):
        """
        Run the sim_telarray command and check the exit code.

        Raises
        ------
        SimtelExecutionError
            if run was not successful.
        """
        sys_output = os.system(command)
        if self._simtel_failed(sys_output):
            self._raise_simtel_error()

    def _make_run_command(self, run_number=None, input_file=None):
        self._logger.debug(
            "make_run_command is being called from the base class - "
            "it should be implemented in the sub class"
        )
        input_file = input_file if input_file else "nofile"
        run_number = run_number if run_number else 1
        return f"{input_file}-{run_number}"

    @staticmethod
    def get_config_option(par, value=None, weak_option=False):
        """
        Build sim_telarray command.

        Parameters
        ----------
        par: str
            Parameter name.
        value: str
            Parameter value.
        weak_option: bool
            If True, use -W option instead of -C.

        Returns
        -------
        str
            Command for sim_telarray.
        """
        option_syntax = "-W" if weak_option else "-C"
        c = f" {option_syntax} {par}"
        c += f"={value}" if value is not None else ""
        return c
