"""Interface to workload managers like gridengine or HTCondor."""

import logging
import subprocess
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
        self._submit_engine = value or "local"
        if self._submit_engine not in self.engines:
            raise ValueError(f"Invalid submit command: {self._submit_engine}")

    def check_submission_system(self):
        """
        Check that the requested workload manager exist on the system.

        Raises
        ------
        JobExecutionError
            if workflow manager is not found.
        """
        try:
            if self.submit_engine in (None, "local") or gen.program_is_executable(
                self.engines[self.submit_engine]
            ):
                return
        except KeyError:
            pass
        raise JobExecutionError(f"Submit engine {self.submit_engine} not found")

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

        submit_result = 0
        if self.submit_engine == "local":
            submit_result = self._submit_local(log_file)
        else:
            submit_result = getattr(self, f"_submit_{self.submit_engine}")()

        if submit_result != 0:
            raise JobExecutionError(f"Job submission failed with return code {submit_result}")

    def _submit_local(self, log_file):
        """
        Run a job script on the command line (no submission to a workload manager).

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

    def _submit_htcondor(self):
        """Submit a job described by a shell script to HTcondor."""
        _condor_file = self.run_script + ".condor"
        lines = [
            f"Executable = {self.run_script}",
            f"Output = {self.run_out_file}.out",
            f"Error = {self.run_out_file}.err",
            f"Log = {self.run_out_file}.job",
        ]
        if self.submit_options:
            lines.extend(option.lstrip() for option in self.submit_options.split(","))
        lines.append("queue 1")
        try:
            with open(_condor_file, "w", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")
        except FileNotFoundError as exc:
            self._logger.error(f"Failed creating condor submission file {_condor_file}")
            raise JobExecutionError from exc

        return self._execute(self.submit_engine, [self.engines[self.submit_engine], _condor_file])

    def _submit_gridengine(self):
        """Submit a job described by a shell script to gridengine."""
        this_sub_cmd = [
            self.engines[self.submit_engine],
            "-o",
            self.run_out_file + ".out",
            "-e",
            self.run_out_file + ".err",
            self.run_script,
        ]
        return self._execute(self.submit_engine, this_sub_cmd)

    def _execute(self, engine, shell_command):
        """
        Execute a shell command using a specific engine.

        Parameters
        ----------
        engine : str
            Engine to use.
        shell_command : list
            List of shell command plus arguments.
        """
        self._logger.info(f"Submitting script to {engine}")
        self._logger.debug(shell_command)
        result = None
        if not self.test:
            result = subprocess.run(shell_command, shell=True, check=True)
        else:
            self._logger.info(f"Testing ({engine}: {shell_command})")

        return result.returncode if result else 0
