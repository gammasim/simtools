"""Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality."""

import stat
from pathlib import Path

from simtools.runners.corsika_runner import CorsikaRunner
from simtools.simtel.simulator_array import SimulatorArray

__all__ = ["CorsikaSimtelRunner"]


class CorsikaSimtelRunner(CorsikaRunner, SimulatorArray):
    """
    Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality.

    Uses CorsikaConfig to manage the CORSIKA configuration and SimulatorArray
    for the sim_telarray configuration.

    Parameters
    ----------
    common_args: dict
        Arguments common to both CORSIKA and sim_telarray runners
    corsika_args: dict
        Arguments for the CORSIKA runner (see full list in CorsikaRunner documentation).
    simtel_args: dict
        Arguments for the sim_telarray runner (see full list in SimulatorArray documentation).
    """

    def __init__(
        self,
        corsika_config,
        simtel_path,
        label=None,
        keep_seeds=False,
        use_multipipe=False,
    ):
        self.corsika_config = corsika_config
        self.corsika_config.set_output_file_and_directory(use_multipipe)
        CorsikaRunner.__init__(
            self,
            corsika_config=corsika_config,
            simtel_path=simtel_path,
            label=label,
            keep_seeds=keep_seeds,
            use_multipipe=use_multipipe,
        )
        SimulatorArray.__init__(
            self,
            corsika_config=corsika_config,
            simtel_path=simtel_path,
            label=label,
            use_multipipe=use_multipipe,
        )

    def prepare_run_script(self, use_pfp=False, **kwargs):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        use_pfp: bool
            Whether to use the preprocessor in preparing the CORSIKA input file
        kwargs: dict
            The following optional parameters can be provided:
                run_number: int
                    Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        self.export_multipipe_script(**kwargs)
        return CorsikaRunner.prepare_run_script(self, use_pfp=use_pfp, **kwargs)

    def export_multipipe_script(self, **kwargs):
        """
        Write the multipipe script used in piping CORSIKA to sim_telarray.

        Parameters
        ----------
        kwargs: dict
            The following optional parameters can be provided:
                run_number: int
                    Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        kwargs = {
            "run_number": None,
            **kwargs,
        }
        run_number = kwargs["run_number"]

        run_command = self._make_run_command(
            run_number=run_number,
            input_file="-",  # instruct sim_telarray to take input from standard output
        )
        multipipe_file = Path(self.corsika_config.config_file_path.parent).joinpath(
            self.corsika_config.get_corsika_config_file_name("multipipe")
        )
        with open(multipipe_file, "w", encoding="utf-8") as file:
            file.write(f"{run_command}")
        self._logger.info(f"Multipipe script - {multipipe_file}")
        self._export_multipipe_executable(multipipe_file)

    def _export_multipipe_executable(self, multipipe_file):
        """
        Write multipipe executable used to call the multipipe_corsika command.

        Parameters
        ----------
        multipipe_file: str or Path
            The name of the multipipe file which contains all of the multipipe commands.
        """
        multipipe_executable = Path(self.corsika_config.config_file_path.parent).joinpath(
            "run_cta_multipipe"
        )
        with open(multipipe_executable, "w", encoding="utf-8") as file:
            multipipe_command = Path(self._simtel_path).joinpath(
                "sim_telarray/bin/multipipe_corsika "
                f"-c {multipipe_file}"
                " || echo 'Fan-out failed'"
            )
            file.write(f"{multipipe_command}")

        multipipe_executable.chmod(multipipe_executable.stat().st_mode | stat.S_IEXEC)

    def _make_run_command(self, run_number=None, input_file=None):
        """
        Build and return the command to run simtel_array.

        Parameters
        ----------
        kwargs: dict
            The dictionary must include the following parameters (unless listed as optional):
                input_file: str
                    Full path of the input CORSIKA file.
                    Use '-' to tell sim_telarray to read from standard output
                run_number: int
                    run number

        """
        info_for_file_name = self.runner_service.get_info_for_file_name(run_number)
        try:
            weak_pointing = any(pointing in self.label for pointing in ["divergent", "convergent"])
        except TypeError:  # allow for sel.label to be None
            weak_pointing = False

        command = str(self._simtel_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {self.corsika_config.array_model.get_config_file()}"
        command += f" -I{self.corsika_config.array_model.get_config_directory()}"
        command += super()._config_option(
            "telescope_theta", self.corsika_config.zenith_angle, weak_option=weak_pointing
        )
        command += super()._config_option(
            "telescope_phi", self.corsika_config.azimuth_angle, weak_option=weak_pointing
        )
        command += super()._config_option(
            "power_law", abs(self.corsika_config.get_config_parameter("ESLOPE"))
        )
        command += super()._config_option(
            "histogram_file", self.runner_service.get_file_name("histogram", **info_for_file_name)
        )
        command += super()._config_option(
            "output_file", self.runner_service.get_file_name("output", **info_for_file_name)
        )
        command += super()._config_option("random_state", "none")
        command += super()._config_option("show", "all")
        command += f" {input_file}"
        _log_file = self.runner_service.get_file_name("log", **info_for_file_name)
        command += f" | gzip > {_log_file} 2>&1 || exit"

        return command
