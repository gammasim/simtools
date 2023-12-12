import logging
import os
from copy import copy
from pathlib import Path

import simtools.utils.general as gen

__all__ = ["JobManager", "MissingWorkloadManager", "JobExecutionError"]


class MissingWorkloadManager(Exception):
    """Exception for missing work load manager."""


class JobExecutionError(Exception):
    """Exception for job execution error (usually CORSIKA or sim_telarray)."""


class JobManager:
    """
    JobManager provides an interface to workload managers like gridengine or HTCondor.

    Parameters
    ----------
    submit_command: str
        Job submission command.
    test: bool
        Testing mode without sub submission.

    Raises
    ------
    MissingWorkloadManager
        if requested workflow manager not found.
    """

    def __init__(self, submit_command=None, test=False):
        """
        Initialize JobManager
        """
        self._logger = logging.getLogger(__name__)
        self.submit_command = submit_command
        self.test = test
        self.run_script = None
        self.run_out_file = None

        try:
            self.test_submission_system()
        except MissingWorkloadManager:
            self._logger.error(f"Requested workflow manager not found: {self.submit_command}")
            raise

    def test_submission_system(self):
        """
        Check that the requested workload manager exist on the system this script is executed.

        Raises
        ------
        MissingWorkloadManager
            if workflow manager is not found.
        """

        if self.submit_command is None:
            return
        if self.submit_command.find("qsub") >= 0:
            if gen.program_is_executable("qsub"):
                return
            raise MissingWorkloadManager
        if self.submit_command.find("condor_submit") >= 0:
            if gen.program_is_executable("condor_submit"):
                return
            raise MissingWorkloadManager
        if self.submit_command.find("local") >= 0:
            return

        raise MissingWorkloadManager

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

        if self.submit_command.find("qsub") >= 0:
            self._submit_gridengine()
        elif self.submit_command.find("condor_submit") >= 0:
            self._submit_htcondor()
        elif self.submit_command.find("local") >= 0:
            self._submit_local(log_file)

    def _submit_local(self, log_file):
        """
        Run a job script on the command line
        (no submission to a workload manager)

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
        """
        Submit a job described by a shell script to HTcondor

        """

        _condor_file = self.run_script + ".condor"
        self._logger.info(f"Submitting script to HTCondor ({_condor_file})")
        try:
            with open(_condor_file, "w", encoding="utf-8") as file:
                file.write(f"Executable = {self.run_script}\n")
                file.write(f"Output = {self.run_out_file + '.out'}\n")
                file.write(f"Error = {self.run_out_file + '.err'}\n")
                file.write(f"Log = {self.run_out_file + '.job'}\n")
                file.write("queue 1\n")
        except FileNotFoundError:
            self._logger.error(f"Failed creating condor submission file {_condor_file}")

        shell_command = self.submit_command + " " + _condor_file
        if not self.test:
            os.system(shell_command)
        else:
            self._logger.info("Testing (HTcondor)")

    def _submit_gridengine(self):
        """
        Submit a job described by a shell script to gridengine

        """

        this_sub_cmd = copy(self.submit_command)
        this_sub_cmd = this_sub_cmd + " -o " + self.run_out_file + ".out"
        this_sub_cmd = this_sub_cmd + " -e " + self.run_out_file + ".err"

        self._logger.info("Submitting script to gridengine")

        shell_command = this_sub_cmd + " " + self.run_script
        self._logger.debug(shell_command)
        if not self.test:
            os.system(shell_command)
        else:
            self._logger.info("Testing (gridengine)")
            self._logger.info(shell_command)
