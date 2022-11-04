import logging
import os

import simtools.util.general as gen
from simtools.model.array_model import ArrayModel
from simtools.model.telescope_model import TelescopeModel

__all__ = ["SimtelRunner"]


class SimtelExecutionError(Exception):
    pass


class InvalidOutputFile(Exception):
    pass


class SimtelRunner:
    """
    SimtelRunner is the base class of the sim_telarray interfaces.

    Attributes
    ----------
    label: str, optional
        Instance label.

    Methods
    -------
    get_run_script(self, test=False, input_file=None, run_number=None)
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.
    run(test=False, force=False, input=None)
        Run sim_telarray. test=True will make it faster and force=True will remove existing files
        and run again.
    """

    def __init__(
        self,
        simtel_source_path,
        label=None,
    ):
        """
        SimtelRunner.

        Parameters
        ----------
        label: str, optional
            Instance label. Important for output file naming.
        simtel_source_path: str (or Path), optional
            Location of sim_telarray installation.
        """
        self._logger = logging.getLogger(__name__)

        self._simtel_source_path = simtel_source_path
        self.label = label

        self.RUNS_PER_SET = 1

    def __repr__(self):
        return "SimtelRunner(label={})\n".format(self.label)

    def _validate_telescope_model(self, tel):
        """Validate TelescopeModel"""
        if isinstance(tel, TelescopeModel):
            self._logger.debug("TelescopeModel is valid")
            return tel
        else:
            msg = "Invalid TelescopeModel"
            self._logger.error(msg)
            raise ValueError(msg)

    def _validate_array_model(self, array):
        """Validate ArrayModel"""
        if isinstance(array, ArrayModel):
            self._logger.debug("ArrayModel is valid")
            return array
        else:
            msg = "Invalid ArrayModel"
            self._logger.error(msg)
            raise ValueError(msg)

    def get_run_script(self, test=False, input_file=None, run_number=None, extra_commands=None):
        """
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.

        Parameters
        ----------
        test: bool
            Test flag for faster execution.
        input_file: str or Path
            Full path of the input CORSIKA file.
        run_number: int
            Run number.
        extra_commands: str
            Additional commands for running simulations given in config.yml

        Returns
        -------
        Path
            Full path of the run script.
        """
        self._logger.debug("Creating run bash script")

        self._script_dir = self._base_directory.joinpath("scripts")
        self._script_dir.mkdir(parents=True, exist_ok=True)
        self._script_file = self._script_dir.joinpath(
            "run{}-simtel".format(run_number if run_number is not None else "")
        )
        self._logger.debug("Run bash script - {}".format(self._script_file))

        self._logger.debug("Extra commands to be added to the run script {}".format(extra_commands))

        command = self._make_run_command(input_file=input_file, run_number=run_number)
        with self._script_file.open("w") as file:
            file.write("#!/usr/bin/bash\n\n")

            # Setting SECONDS variable to measure runtime
            file.write("\nSECONDS=0\n")

            if extra_commands is not None:
                file.write("# Writing extras\n")
                for line in extra_commands:
                    file.write("{}\n".format(line))
                file.write("# End of extras\n\n")

            N = 1 if test else self.RUNS_PER_SET
            for _ in range(N):
                file.write("{}\n\n".format(command))

            # Printing out runtime
            file.write('\necho "RUNTIME: $SECONDS"\n')

        os.system("chmod ug+x {}".format(self._script_file))
        return self._script_file

    def run(self, test=False, force=False, input_file=None, run_number=None):
        """
        Basic sim_telarray run method.

        Parameters
        ----------
        test: bool
            If True, make simulations faster.
        force: bool
            If True, remove possible existing output files and run again.
        """
        self._logger.debug("Running sim_telarray")

        if not hasattr(self, "_make_run_command"):
            msg = "run method cannot be executed without the _make_run_command method"
            self._logger.error(msg)
            raise RuntimeError(msg)

        if not self._shall_run(run_number) and not force:
            self._logger.info("Skipping because output exists and force = False")
            return

        command = self._make_run_command(input_file=input_file, run_number=run_number)

        if test:
            self._logger.info("Running (test) with command:{}".format(command))
            self._run_simtel_and_check_output(command)
        else:
            self._logger.info("Running ({}x) with command:{}".format(self.RUNS_PER_SET, command))
            self._run_simtel_and_check_output(command)

            for _ in range(self.RUNS_PER_SET - 1):
                self._run_simtel_and_check_output(command)

        self._check_run_result(run_number=run_number)

    @staticmethod
    def _simtel_failed(sys_output):
        return sys_output != 0

    def _raise_simtel_error(self):
        """
        Raise sim_telarray execution error. Final 30 lines from the log file
        are collected and printed.
        """
        if hasattr(self, "_log_file"):
            log_lines = gen.collect_final_lines(self._log_file, 30)
            msg = (
                "Simtel Error - See below the relevant part of the simtel log file.\n"
                + "===== from simtel log file ======\n"
                + log_lines
                + "================================="
            )
        else:
            msg = "Simtel log file does not exist."

        self._logger.error(msg)
        raise SimtelExecutionError(msg)

    def _run_simtel_and_check_output(self, command):
        """
        Run the sim_telarray command and check the exit code.
        """
        sys_output = os.system(command)
        if self._simtel_failed(sys_output):
            self._raise_simtel_error()

    def _shall_run(self, run_number=None):
        self._logger.debug(
            "shall_run is being called from the base class - returning False -"
            + "it should be implemented in the sub class"
        )
        return False

    @staticmethod
    def _config_option(par, value=None):
        """Util function for building sim_telarray command."""
        c = " -C {}".format(par)
        c += "={}".format(value) if value is not None else ""
        return c
