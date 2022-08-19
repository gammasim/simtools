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

    Methods
    -------

    """

    def __init__(
        self,
        label=None,
        submitCommand=None,
        test=False
    ):
        self._logger = logging.getLogger(__name__)
        self.label = label
        self.submitCommand = submitCommand
        self.test = test

        try:
            self.test_submission_system()
        except MissingWorkloadManager:
            self._logger.error(
                "Requested worklow manager not found: {}".format(
                    self.submitCommand))
            raise

    def test_submission_system(self):
        """
        Check the requested workload manager exist on the
        system this script is executed

        Raises
        ------
        MissingWorkloadManager
            if workflow manager is not found

        """

        if (self.submitCommand.find("qsub") >= 0
                and not gen.program_is_executable('qsub')):
            raise MissingWorkloadManager

    def submit(
        self,
        run_script=None,
        run_out_file=None,
        run_error_file=None,
    ):
        """
        Submit a job described by a shell script

        Parameters
        ----------
        run_script: string
            Shell script descring the job to be submitted
        run_out_file: string
            Redirect output stream to this file
        run_error_file: string
            Redirect error stream to this file

        """
        self.run_script = run_script
        self.run_out_file = run_out_file
        self.run_error_file = run_error_file

        if self.submitCommand.find("qsub") >= 0:
            self._submit_gridengine()

    def _submit_gridengine(self):
        """
        Submit a job described by a shell script to gridengine

        Parameters
        ----------
        run_script: string
            Shell script descring the job to be submitted
        run_out_file: string
            Redirect output stream to this file
        run_error_file: string
            Redirect error stream to this file

        """

        thisSubCmd = copy(self.submitCommand)
        if 'log_out' in thisSubCmd:
            thisSubCmd = thisSubCmd.replace(
                'log_out', str(self.run_out_file))
        if 'log_err' in thisSubCmd:
            thisSubCmd = thisSubCmd.replace(
                'log_err', str(self.run_error_file))

        self._logger.info(
            'Submitting script {} to gridengine'.format(self.run_script))

        shellCommand = thisSubCmd + ' ' + str(self.run_script)
        self._logger.debug(shellCommand)
        if not self.test:
            os.system(shellCommand)
        else:
            self._logger.info('Testing')
