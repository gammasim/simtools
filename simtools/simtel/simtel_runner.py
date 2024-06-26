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
    SimtelRunner is the base class of the sim_telarray interfaces.

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
        self._script_file = None

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

        self._script_dir = self._base_directory.joinpath("scripts")
        self._script_dir.mkdir(parents=True, exist_ok=True)
        self._script_file = self._script_dir.joinpath(
            f"run{run_number if run_number is not None else ''}-simtel"
        )
        self._logger.debug(f"Run bash script - {self._script_file}")

        self._logger.debug(f"Extra commands to be added to the run script {extra_commands}")

        command = self._make_run_command(input_file=input_file, run_number=run_number)
        with self._script_file.open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n\n")

            # Make sure to exit on failed commands and report their error code
            file.write("set -e\n")
            file.write("set -o pipefail\n")

            # Setting SECONDS variable to measure runtime
            file.write("\nSECONDS=0\n")

            if extra_commands is not None:
                file.write("# Writing extras\n")
                for line in extra_commands:
                    file.write(f"{line}\n")
                file.write("# End of extras\n\n")

            n = 1 if test else self.runs_per_set
            for _ in range(n):
                file.write(f"{command}\n\n")

            # Printing out runtime
            file.write('\necho "RUNTIME: $SECONDS"\n')

        os.system(f"chmod ug+x {self._script_file}")
        return self._script_file

    def run(self, test=False, force=False, input_file=None, run_number=None):
        """
        Make run command and run sim_telarray.

        Parameters
        ----------
        test: bool
            If True, make simulations faster.
        force: bool
            If True, remove possible existing output files and run again.
        input_file: str or Path
            Full path of the input CORSIKA file.
        run_number: int
            Run number.
        """
        self._logger.debug("Running sim_telarray")

        if not hasattr(self, "_make_run_command"):
            msg = "run method cannot be executed without the _make_run_command method"
            self._logger.error(msg)
            raise RuntimeError(msg)

        if not self._shall_run() and not force:
            self._logger.info("Skipping because output exists and force = False")
            return

        command = self._make_run_command(input_file=input_file, run_number=run_number)

        if test:
            self._logger.info(f"Running (test) with command: {command}")
            self._run_simtel_and_check_output(command)
        else:
            self._logger.debug(f"Running ({self.runs_per_set}x) with command: {command}")
            self._run_simtel_and_check_output(command)

            for _ in range(self.runs_per_set - 1):
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

    def _shall_run(self):
        self._logger.debug(
            "shall_run is being called from the base class - returning False -"
            "it should be implemented in the sub class"
        )
        return False

    @staticmethod
    def _config_option(par, value=None, weak_option=False):
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
