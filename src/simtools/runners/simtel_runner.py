"""Base class for running sim_telarray simulations."""

import logging

from simtools.job_execution import job_manager
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

    The sim_telarray software stack includes sim_telarray itself and e.g., testeff,
    LightEmission, and other software packages.

    Parameters
    ----------
    label: str
        Instance label. Important for output file naming.
    config: CorsikaConfig or dict
        Configuration parameters.
    is_calibration_run: bool
        Flag to indicate if this is a calibration run.
    """

    def __init__(self, label=None, config=None, is_calibration_run=False):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)

        self.label = label
        self._base_directory = None
        self.is_calibration_run = is_calibration_run

        self.runs_per_set = 1

        self.runner_service = RunnerServices(config, run_type="sim_telarray", label=label)
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
        for _ in range(runs):
            job_manager.submit(
                command,
                out_file=stdout_file,
                err_file=stderr_file,
                env=SIM_TELARRAY_ENV,
            )

    def make_run_command(self, run_number=None, input_file=None):
        """Make the sim_telarray run command (to implemented in subclasses)."""
        raise NotImplementedError("Must be implemented in concrete subclass")

    def get_resources(self, run_number=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(run_number)
