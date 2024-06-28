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
        use_multipipe=False,
    ):
        """Initialize SimulatorArray."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimulatorArray")
        super().__init__(label=label, simtel_path=simtel_path)

        self.corsika_config = corsika_config
        self.io_handler = io_handler.IOHandler()
        self._log_file = None

        self.runner_service = RunnerServices(corsika_config, label)
        self._directory = self.runner_service.load_data_directories(
            "corsika_simtel" if use_multipipe else "simtel"
        )

    def _shall_run(self, **kwargs):
        """Tells if simulations should be run again based on the existence of output files."""
        output_file = self.get_file_name(file_type="output", run_number=kwargs["run_number"])
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
        self._log_file = self.get_file_name(file_type="log", run_number=run_number)
        histogram_file = self.get_file_name(file_type="histogram", run_number=run_number)
        output_file = self.get_file_name(file_type="output", run_number=run_number)

        # Array
        command = str(self._simtel_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {self.corsika_config.array_model.get_config_file()}"
        command += f" -I{self.corsika_config.array_model.get_config_directory()}"
        command += super().get_config_option("telescope_theta", self.corsika_config.zenith_angle)
        command += super().get_config_option("telescope_phi", self.corsika_config.azimuth_angle)
        command += super().get_config_option("power_law", "2.5")
        command += super().get_config_option("histogram_file", histogram_file)
        command += super().get_config_option("output_file", output_file)
        command += super().get_config_option("random_state", "none")
        command += super().get_config_option("show", "all")
        command += f" {input_file}"
        command += f" > {self._log_file} 2>&1 || exit"

        return command

    def _check_run_result(self, **kwargs):
        """Check run results."""
        output_file = self.get_file_name(file_type="output", run_number=kwargs["run_number"])
        if not output_file.exists():
            msg = f"sim_telarray output file {output_file} does not exist."
            self._logger.error(msg)
            raise InvalidOutputFileError(msg)
        self._logger.debug(f"simtel_array output file {output_file} exists.")

    def get_resources(self, run_number=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(run_number)

    def get_file_name(self, simulation_software="simtel", file_type=None, run_number=None, mode=""):
        """
        Get the full path of a file for a given run number.

        Parameters
        ----------
        simulation_software: str
            Simulation software.
        file_type: str
            File type.
        run_number: int
            Run number.

        Returns
        -------
        str
            File name with full path.
        """
        if simulation_software.lower() != "simtel":
            raise ValueError(
                f"simulation_software ({simulation_software}) is not supported in SimulatorArray"
            )
        return self.runner_service.get_file_name(
            file_type=file_type, run_number=run_number, mode=mode
        )
