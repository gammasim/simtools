"""Simulation runner for array simulations."""

import logging

from simtools.io_operations import io_handler
from simtools.runners.runner_services import RunnerServices
from simtools.runners.simtel_runner import InvalidOutputFileError, SimtelRunner

__all__ = ["SimulatorArray"]


class SimulatorArray(SimtelRunner):
    """
    SimulatorArray is the interface with sim_telarray to perform array simulations.

    Parameters
    ----------
    array_model: str
        Instance of ArrayModel class.
    label: str
        Instance label. Important for output file naming.
    simtel_path: str or Path
        Location of sim_telarray installation.
    """

    def __init__(
        self,
        corsika_config,
        simtel_path,
        label=None,
    ):
        """Initialize SimulatorArray."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimulatorArray")
        super().__init__(label=label, simtel_path=simtel_path)

        self.corsika_config = corsika_config
        self.io_handler = io_handler.IOHandler()
        self._log_file = None

        self.runner_service = RunnerServices(corsika_config, label)
        self._directory = self.runner_service.load_data_directories("simtel")

    def _shall_run(self, **kwargs):
        """Tells if simulations should be run again based on the existence of output files."""
        output_file = self.runner_service.get_file_name(
            file_type="output", **self.runner_service.get_info_for_file_name(kwargs["run_number"])
        )
        return not output_file.exists()

    def _make_run_command(self, run_number=None, input_file=None):
        """
        Build and return the command to run simtel_array.

        Parameters
        ----------
        kwargs: dict
            The dictionary must include the following parameters (unless listed as optional):
                input_file: str
                    Full path of the input CORSIKA file
                run_number: int (optional)
                    run number
        """
        info_for_file_name = self.runner_service.get_info_for_file_name(run_number)
        self._log_file = self.runner_service.get_file_name(file_type="log", **info_for_file_name)
        histogram_file = self.runner_service.get_file_name(
            file_type="histogram", **info_for_file_name
        )
        output_file = self.runner_service.get_file_name(file_type="output", **info_for_file_name)

        # Array
        command = str(self._simtel_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {self.corsika_config.array_model.get_config_file()}"
        command += f" -I{self.corsika_config.array_model.get_config_directory()}"
        command += super()._config_option("telescope_theta", self.corsika_config.zenith_angle)
        command += super()._config_option("telescope_phi", self.corsika_config.azimuth_angle)
        command += super()._config_option("power_law", "2.5")
        command += super()._config_option("histogram_file", histogram_file)
        command += super()._config_option("output_file", output_file)
        command += super()._config_option("random_state", "none")
        command += super()._config_option("show", "all")
        command += f" {input_file}"
        command += f" > {self._log_file} 2>&1 || exit"

        return command

    def _check_run_result(self, **kwargs):
        """Check run results."""
        output_file = self.runner_service.get_file_name(
            file_type="output", **self.runner_service.get_info_for_file_name(kwargs["run_number"])
        )
        if not output_file.exists():
            msg = f"sim_telarray output file {output_file} does not exist."
            self._logger.error(msg)
            raise InvalidOutputFileError(msg)
        self._logger.debug(f"simtel_array output file {output_file} exists.")
