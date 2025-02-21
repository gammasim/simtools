"""Simulation runner for array simulations."""

import logging

from simtools.io_operations import io_handler
from simtools.runners.simtel_runner import InvalidOutputFileError, SimtelRunner
from simtools.utils.general import clear_default_sim_telarray_cfg_directories

__all__ = ["SimulatorArray"]


class SimulatorArray(SimtelRunner):
    """
    SimulatorArray is the interface with sim_telarray to perform array simulations.

    Parameters
    ----------
    corsika_config_data: CorsikaConfig
        CORSIKA configuration.
    simtel_path: str or Path
        Location of source of the sim_telarray/CORSIKA package.
    label: str
        Instance label.
    use_multipipe: bool
        Use multipipe to run CORSIKA and sim_telarray.
    """

    def __init__(
        self,
        corsika_config,
        simtel_path,
        label=None,
        use_multipipe=False,
        sim_telarray_seeds=None,
    ):
        """Initialize SimulatorArray."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimulatorArray")
        super().__init__(
            label=label,
            simtel_path=simtel_path,
            corsika_config=corsika_config,
            use_multipipe=use_multipipe,
        )

        self.sim_telarray_seeds = sim_telarray_seeds
        self.corsika_config = corsika_config
        self.io_handler = io_handler.IOHandler()
        self._log_file = None

    def _make_run_command(self, run_number=None, input_file=None):
        """
        Build and return the command to run simtel_array.

        Parameters
        ----------
        input_file: str
            Full path of the input CORSIKA file
        run_number: int (optional)
            run number

        Returns
        -------
        str
            Command to run sim_telarray.
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
        command += super().get_config_option(
            "power_law",
            SimulatorArray.get_power_law_for_sim_telarray_histograms(
                self.corsika_config.primary_particle
            ),
        )
        command += super().get_config_option("histogram_file", histogram_file)
        command += super().get_config_option("output_file", output_file)
        command += super().get_config_option("random_state", "none")
        if self.sim_telarray_seeds:
            command += super().get_config_option("random_seed", self.sim_telarray_seeds)
        command += super().get_config_option("show", "all")
        command += f" {input_file}"
        command += f" > {self._log_file} 2>&1 || exit"

        return clear_default_sim_telarray_cfg_directories(command)

    def _check_run_result(self, run_number=None):
        """
        Check if simtel output file exists.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        bool
            True if simtel output file exists.

        Raises
        ------
        InvalidOutputFileError
            If simtel output file does not exist.
        """
        output_file = self.get_file_name(file_type="output", run_number=run_number)
        if not output_file.exists():
            msg = f"sim_telarray output file {output_file} does not exist."
            self._logger.error(msg)
            raise InvalidOutputFileError(msg)
        self._logger.debug(f"simtel_array output file {output_file} exists.")
        return True

    @staticmethod
    def get_power_law_for_sim_telarray_histograms(primary):
        """
        Get the power law index for sim_telarray.

        Events will be histogrammed in sim_telarray with a weight according to
        the difference between this exponent and the one used for the shower simulations.

        Parameters
        ----------
        primary: str
            Primary particle.

        Returns
        -------
        float
            Power law index.
        """
        power_laws = {
            "gamma": 2.5,
            "electron": 3.3,
        }
        if primary.name in power_laws:
            return power_laws[primary.name]

        return 2.68
