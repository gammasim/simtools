"""Interface to workload managers like gridengine or HTCondor."""

import logging
import os
from pathlib import Path

import simtools.utils.general as gen

__all__ = ["JobExecutionError", "JobManager"]


class JobExecutionError(Exception):
    """Job execution error."""


class JobManager:
    """
    Interface to workload managers like gridengine or HTCondor.

    Expects that jobs are described by shell scripts.

    Parameters
    ----------
    submit_engine : str
        Job submission system. Default is local.
    test : bool
        Testing mode without sub submission.
    """

    engines = {
        "gridengine": "qsub",
        "htcondor": "condor_submit",
        "local": "",
        "test_wms": "test_wms",  # used for testing only
    }

    def __init__(self, submit_engine=None, submit_options=None, test=False):
        """Initialize JobManager."""
        self._logger = logging.getLogger(__name__)
        self.submit_engine = submit_engine
        self.submit_options = submit_options
        self.test = test
        self.run_script = None
        self.run_out_file = None

        self.check_submission_system()

    @property
    def submit_engine(self):
        """Get the submit command."""
        return self._submit_engine

    @submit_engine.setter
    def submit_engine(self, value):
        """
        Set the submit command.

        Parameters
        ----------
        value : str
            Name of submit engine.

        Raises
        ------
        ValueError
            if invalid submit engine.
        """
        if value is None:
            value = "local"
        if value not in self.engines:
            raise ValueError(f"Invalid submit command: {value}")
        self._submit_engine = value

    def check_submission_system(self):
        """
        Check that the requested workload manager exist on the system.

        Raises
        ------
        MissingWorkloadManagerError
            if workflow manager is not found.
        """
        if self.submit_engine is None or self.submit_engine == "local":
            return

        if gen.program_is_executable(self.engines[self.submit_engine]):
            return

    def submit(self, run_script=None, run_out_file=None, log_file=None):
        """
        Submit a job described by a shell script.

        Parameters
        ----------
        run_script: str
            Shell script describing the job to be submitted.
        run_out_file: str or Path
            Redirect output/error/job stream to this file (out,err,job suffix).
        log_file: str or Path
            The log file of the actual simulator (CORSIKA or sim_telarray).
            Provided in order to print the log excerpt in case of run time error.
        """
        self.run_script = str(run_script)
        run_out_file = Path(run_out_file)
        self.run_out_file = str(run_out_file.parent.joinpath(run_out_file.stem))

        self._logger.info(f"Submitting script {self.run_script}")
        self._logger.info(f"Job output stream {self.run_out_file + '.out'}")
        self._logger.info(f"Job error stream {self.run_out_file + '.err'}")
        self._logger.info(f"Job log stream {self.run_out_file + '.job'}")

        if self.submit_engine == "gridengine":
            self._submit_gridengine()
        elif self.submit_engine == "htcondor":
            self._submit_htcondor()
        elif self.submit_engine == "local":
            self._submit_local(log_file)

    def _submit_local(self, log_file):
        """
        Run a job script on the command line (no submission to a workload manager).

        Parameters
        ----------
        log_file: str or Path
            The log file of the actual simulator (CORSIKA or sim_telarray).
            Provided in order to print the log excerpt in case of run time error.
        """
        self._logger.info("Running script locally")

        shell_command = f"{self.run_script} > {self.run_out_file}.out 2> {self.run_out_file}.err"

        if not self.test:
            sys_output = os.system(shell_command)
            if sys_output != 0:
                msg = gen.get_log_excerpt(f"{self.run_out_file}.err")
                self._logger.error(msg)
                if log_file.exists() and gen.get_file_age(log_file) < 5:
                    msg = gen.get_log_excerpt(log_file)
                    self._logger.error(msg)
                raise JobExecutionError("See excerpt from log file above\n")
        else:
            self._logger.info("Testing (local)")

    def _submit_htcondor(self):
        """Submit a job described by a shell script to HTcondor."""
        _condor_file = self.run_script + ".condor"
        try:
            with open(_condor_file, "w", encoding="utf-8") as file:
                file.write(f"Executable = {self.run_script}\n")
                file.write(f"Output = {self.run_out_file + '.out'}\n")
                file.write(f"Error = {self.run_out_file + '.err'}\n")
                file.write(f"Log = {self.run_out_file + '.job'}\n")
                if self.submit_options:
                    submit_option_list = self.submit_options.split(",")
                    for option in submit_option_list:
                        file.write(option.lstrip() + "\n")
                file.write("queue 1\n")
        except FileNotFoundError as exc:
            self._logger.error(f"Failed creating condor submission file {_condor_file}")
            raise JobExecutionError from exc

        self._execute(self.submit_engine, self.engines[self.submit_engine] + " " + _condor_file)

    def _submit_gridengine(self):
        """Submit a job described by a shell script to gridengine."""
        this_sub_cmd = self.engines[self.submit_engine]
        this_sub_cmd = this_sub_cmd + " -o " + self.run_out_file + ".out"
        this_sub_cmd = this_sub_cmd + " -e " + self.run_out_file + ".err"

        self._execute(self.submit_engine, this_sub_cmd + " " + self.run_script)

    def _execute(self, engine, shell_command):
        """
        Execute a shell command using a specific engine.

        Parameters
        ----------
        engine : str
            Engine to use.
        shell_command : str
            Shell command to execute.
        """
        self._logger.info(f"Submitting script to {engine}")
        self._logger.debug(shell_command)
        if not self.test:
            os.system(shell_command)
        else:
            self._logger.info(f"Testing ({engine})")
            self._logger.info(shell_command)
