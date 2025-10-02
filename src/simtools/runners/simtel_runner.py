"""Base class for running sim_telarray simulations."""

import logging
import stat
import subprocess
from pathlib import Path

import simtools.utils.general as gen
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
    simtel_path: str or Path
        Location of sim_telarray installation.
    label: str
        Instance label. Important for output file naming.
    corsika_config: CorsikaConfig
        CORSIKA configuration.
    use_multipipe: bool
        Use multipipe to run CORSIKA and sim_telarray.
    """

    def __init__(self, simtel_path, label=None, corsika_config=None, use_multipipe=False):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)

        self._simtel_path = Path(simtel_path)
        self.label = label
        self._base_directory = None

        self.runs_per_set = 1

        self.runner_service = RunnerServices(corsika_config, label)
        self._directory = self.runner_service.load_data_directories(
            "corsika_sim_telarray" if use_multipipe else "sim_telarray"
        )

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

        script_file_path.chmod(script_file_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
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

        command, stdout_file, stderr_file = self._make_run_command(
            run_number=run_number, input_file=input_file
        )

        if test:
            self._logger.info(f"Running (test) with command: {command}")
            self._run_simtel_and_check_output(command, stdout_file, stderr_file)
        else:
            self._logger.debug(f"Running ({self.runs_per_set}x) with command: {command}")
            for _ in range(self.runs_per_set):
                self._run_simtel_and_check_output(command, stdout_file, stderr_file)

        self._check_run_result(run_number=run_number)

    def _check_run_result(self, run_number=None):  # pylint: disable=all
        """Check if simtel output file exists."""
        pass

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

    def _run_simtel_and_check_output(self, command, stdout_file, stderr_file):
        """
        Run the sim_telarray command and check the exit code.

        Raises
        ------
        SimtelExecutionError
            if run was not successful.
        """
        stdout_file = stdout_file if stdout_file else "/dev/null"
        stderr_file = stderr_file if stderr_file else "/dev/null"
        with (
            open(f"{stdout_file}", "w", encoding="utf-8") as stdout,
            open(f"{stderr_file}", "w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                stdout=stdout,
                stderr=stderr,
            )

        if result.returncode != 0:
            self._logger.error(result.stderr)
            self._raise_simtel_error()
        return result.returncode

    def _make_run_command(self, run_number=None, input_file=None):
        self._logger.debug(
            "make_run_command is being called from the base class - "
            "it should be implemented in the sub class"
        )
        input_file = input_file if input_file else "nofile"
        run_number = run_number if run_number else 1
        return f"{input_file}-{run_number}", None, None

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

    def get_resources(self, run_number=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(run_number)

    def get_file_name(
        self,
        simulation_software="sim_telarray",
        file_type=None,
        run_number=None,
        mode="",
        model_version_index=0,
    ):
        """
        Get the full path of a file for a given run number.

        Parameters
        ----------
        simulation_software: str
            Simulation software.
        file_type: str
            File type.
        run_number: int
            Run number.
        model_version_index: int
            Index of the model version.
            This is used to select the correct simulator_array instance in case
            multiple array models are simulated.

        Returns
        -------
        str
            File name with full path.
        """
        if simulation_software.lower() != "sim_telarray":
            raise ValueError(
                f"simulation_software ({simulation_software}) is not supported in SimulatorArray"
            )
        return self.runner_service.get_file_name(
            file_type=file_type,
            run_number=run_number,
            mode=mode,
            _model_version_index=model_version_index,
        )
