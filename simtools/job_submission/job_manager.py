import logging
import os
from copy import copy

import simtools.util.general as gen

__all__ = ["JobManager"]


class MissingWorkloadManager(Exception):
    pass


class JobManager:
    """
    JobManager provides an interface to workload managers
    like gridengine or HTCondor.

    Attributes
    ----------
    submitCommand: str
       job submission command (allowed is qsub, condor_submit, local).
    test: bool
       testing mode without sub submission

    Methods
    -------
    test_submission_system
       Check that the requested workload manager exist
    submit(run_script, run_out_file)
       Submit a job described by a shell script

    """

    def __init__(self, submitCommand=None, test=False):
        """
        JobManager init

        Parameters
        ----------
        submitCommand: str
            job submission command
        test: bool
        testing mode without sub submission

        """
        self._logger = logging.getLogger(__name__)
        self.submitCommand = submitCommand
        self.test = test

        try:
            self.test_submission_system()
        except MissingWorkloadManager:
            self._logger.error(
                "Requested workflow manager not found: {}".format(self.submitCommand)
            )
            raise

    def test_submission_system(self):
        """
        Check that the requested workload manager exist on the
        system this script is executed

        Raises
        ------
        MissingWorkloadManager
            if workflow manager is not found

        """

        if self.submitCommand is None:
            return
        elif self.submitCommand.find("qsub") >= 0:
            if gen.program_is_executable("qsub"):
                return
            raise MissingWorkloadManager
        elif self.submitCommand.find("condor_submit") >= 0:
            if gen.program_is_executable("condor_submit"):
                return
            raise MissingWorkloadManager
        elif self.submitCommand.find("local") >= 0:
            return

        raise MissingWorkloadManager

    def submit(
        self,
        run_script=None,
        run_out_file=None,
    ):
        """
        Submit a job described by a shell script

        Parameters
        ----------
        run_script: string
            Shell script descring the job to be submitted
        run_out_file: string
            Redirect output/error/job stream to this file (out,err,job suffix)

        """
        self.run_script = str(run_script)
        self.run_out_file = str(run_out_file).replace(".log", "")

        self._logger.info("Submitting script {}".format(self.run_script))
        self._logger.info("Job output stream {}".format(self.run_out_file + ".out"))
        self._logger.info("Job error stream {}".format(self.run_out_file + ".err"))
        self._logger.info("Job log stream {}".format(self.run_out_file + ".job"))

        if self.submitCommand.find("qsub") >= 0:
            self._submit_gridengine()
        elif self.submitCommand.find("condor_submit") >= 0:
            self._submit_HTcondor()
        elif self.submitCommand.find("local") >= 0:
            self._submit_local()

    def _submit_local(self):
        """
        Run a job script on the command line
        (no submission to a workload manager)

        """

        self._logger.info("Running script locally")

        shellCommand = (
            self.run_script + " > " + self.run_out_file + ".out" " 2> " + self.run_out_file + ".err"
        )

        if not self.test:
            os.system(shellCommand)
        else:
            self._logger.info("Testing (local)")

    def _submit_HTcondor(self):
        """
        Submit a job described by a shell script to HTcondor

        """

        _condor_file = self.run_script + ".condor"
        self._logger.info("Submitting script to HTCondor ({})".format(_condor_file))
        try:
            with open(_condor_file, "w") as file:
                file.write("Executable = {}\n".format(self.run_script))
                file.write("Output = {}\n".format(self.run_out_file + ".out"))
                file.write("Error = {}\n".format(self.run_out_file + ".err"))
                file.write("Log = {}\n".format(self.run_out_file + ".job"))
                file.write("queue 1\n")
        except FileNotFoundError:
            self._logger.error("Failed creating condor submission file {}".format(_condor_file))

        shellCommand = self.submitCommand + " " + _condor_file
        if not self.test:
            os.system(shellCommand)
        else:
            self._logger.info("Testing (HTcondor")

    def _submit_gridengine(self):
        """
        Submit a job described by a shell script to gridengine

        """

        thisSubCmd = copy(self.submitCommand)
        thisSubCmd = thisSubCmd + " -o " + self.run_out_file + ".out"
        thisSubCmd = thisSubCmd + " -e " + self.run_out_file + ".err"

        self._logger.info("Submitting script to gridengine")

        shellCommand = thisSubCmd + " " + self.run_script
        self._logger.debug(shellCommand)
        if not self.test:
            os.system(shellCommand)
        else:
            self._logger.info("Testing (gridengine)")
