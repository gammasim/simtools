"""Interface to workload managers to run jobs on a compute node."""

import logging
import subprocess
from pathlib import Path

import simtools.utils.general as gen

__all__ = ["JobExecutionError", "JobManager"]


class JobExecutionError(Exception):
    """Job execution error."""


class JobManager:
    """
    Job manager for submitting jobs to a compute node.

    Expects that jobs can be described by shell scripts.

    Parameters
    ----------
    test : bool
        Testing mode without sub submission.
    """

    def __init__(self, test=False):
        """Initialize JobManager."""
        self._logger = logging.getLogger(__name__)
        self.test = test
        self.run_script = None
        self.run_out_file = None

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

        submit_result = self.submit_local(log_file)
        if submit_result != 0:
            raise JobExecutionError(f"Job submission failed with return code {submit_result}")

    def submit_local(self, log_file):
        """
        Run a job script on the command line.

        Parameters
        ----------
        log_file: str or Path
            The log file of the actual simulator (CORSIKA or sim_telarray).
            Provided in order to print the log excerpt in case of run time error.

        Returns
        -------
        int
            Return code of the executed script
        """
        self._logger.info("Running script locally")

        if self.test:
            self._logger.info("Testing (local)")
            return 0

        result = None
        try:
            with (
                open(f"{self.run_out_file}.out", "w", encoding="utf-8") as stdout,
                open(f"{self.run_out_file}.err", "w", encoding="utf-8") as stderr,
            ):
                result = subprocess.run(
                    f"{self.run_script}",
                    shell=True,
                    check=True,
                    text=True,
                    stdout=stdout,
                    stderr=stderr,
                )
        except subprocess.CalledProcessError as exc:
            self._logger.error(gen.get_log_excerpt(f"{self.run_out_file}.err"))
            if log_file.exists() and gen.get_file_age(log_file) < 5:
                self._logger.error(gen.get_log_excerpt(log_file))
            raise JobExecutionError("See excerpt from log file above\n") from exc

        return result.returncode if result else 0
