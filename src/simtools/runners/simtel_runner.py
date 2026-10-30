"""Base class for running sim_telarray simulations."""

import logging

from simtools.context import RunContext
from simtools.job_execution import execution
from simtools.runners.runner_services import RunnerServices

SIM_TELARRAY_ENV = {
    "SIM_TELARRAY_CONFIG_PATH": "",
}


def sim_telarray_env_as_string():
    """Return the sim_telarray environment variables as a string."""
    return " ".join(f'{key}="{value}" ' for key, value in SIM_TELARRAY_ENV.items())


class SimtelRunner:
    """
    Base class for running simulations based on the sim_telarray software stack.

    The sim_telarray software stack includes sim_telarray itself and auxiliary tools,
    LightEmission, and other software packages.

    Parameters
    ----------
    label: str
        Instance label. Important for output file naming.
    config: CorsikaConfig or dict
        Configuration parameters.
    """

    def __init__(self, label=None, config=None, array_model=None, context=None):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)

        self.label = label
        self.context = context or getattr(config, "context", None) or RunContext.from_environment()
        self._base_directory = None

        self.runs_per_set = 1
        # (Instrument seed, simulation seed) of each run of a set; without them (or None)
        # sim_telarray draws its seeds.
        self.run_seeds = []

        self.runner_service = RunnerServices(
            config,
            run_type="sim_telarray",
            label=label,
            array_model=array_model,
            context=self.context,
        )
        self.file_list = None

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

        command, stdout_file, stderr_file = self.make_run_command(
            run_number=run_number, input_file=input_file
        )
        runs = 1 if test else self.runs_per_set
        label = "test" if test else f"{self.runs_per_set}x"
        self._logger.info(f"Running ({label}) with command: {command}")
        for run_index in range(runs):
            execution.run_command(
                self._command_for_run(command, run_index),
                stdout=stdout_file,
                stderr=stderr_file,
                env=SIM_TELARRAY_ENV,
            )

    def _command_for_run(self, command, run_index):
        """Return the command of one run of a set, with the seeds of that run."""
        seeds = self.run_seeds[run_index] if self.run_seeds else None
        if seeds is None:
            return command
        return [*command[:-1], "-C", f"random_seed={seeds[0]},{seeds[1]}", command[-1]]

    def make_run_command(self, run_number=None, input_file=None):
        """Make the sim_telarray run command (to implemented in subclasses)."""
        raise NotImplementedError("Must be implemented in concrete subclass")

    def get_resources(self, runtime=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(runtime=runtime)
