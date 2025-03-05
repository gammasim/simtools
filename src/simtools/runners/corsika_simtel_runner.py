"""Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality."""

import logging
import stat
from pathlib import Path

from simtools.runners.corsika_runner import CorsikaRunner
from simtools.simtel.simulator_array import SimulatorArray
from simtools.utils.general import clear_default_sim_telarray_cfg_directories

__all__ = ["CorsikaSimtelRunner"]


class CorsikaSimtelRunner:
    """
    Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality.

    Uses CorsikaConfig to manage the CORSIKA configuration and SimulatorArray
    for the sim_telarray configuration.

    Parameters
    ----------
    corsika_config : CorsikaConfig
        Configuration parameters for CORSIKA.
    simtel_path : str or Path
        Location of the sim_telarray package.
    label : str
        Label.
    keep_seeds : bool
        Use seeds based on run number and primary particle. If False, use sim_telarray seeds.
    use_multipipe : bool
        Use multipipe to run CORSIKA and sim_telarray.
    """

    def __init__(
        self,
        corsika_config,
        simtel_path,
        label=None,
        keep_seeds=False,
        use_multipipe=False,
        sim_telarray_seeds=None,
    ):
        self._logger = logging.getLogger(__name__)
        self.corsika_config = corsika_config
        self._simtel_path = simtel_path
        self.sim_telarray_seeds = sim_telarray_seeds
        self.label = label

        self.corsika_config.set_output_file_and_directory(use_multipipe)
        self.corsika_runner = CorsikaRunner(
            corsika_config=corsika_config,
            simtel_path=simtel_path,
            label=label,
            keep_seeds=keep_seeds,
            use_multipipe=use_multipipe,
        )
        self.simulator_array = SimulatorArray(
            corsika_config=corsika_config,
            simtel_path=simtel_path,
            label=label,
            use_multipipe=use_multipipe,
            sim_telarray_seeds=sim_telarray_seeds,
        )

    def prepare_run_script(
        self, run_number=None, input_file=None, extra_commands=None, use_pfp=False
    ):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        run_number: int
            Run number.
        use_pfp: bool
            Whether to use the preprocessor in preparing the CORSIKA input file

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        self._export_multipipe_script(run_number)
        return self.corsika_runner.prepare_run_script(
            run_number=run_number,
            input_file=input_file,
            extra_commands=extra_commands,
            use_pfp=use_pfp,
        )

    def _export_multipipe_script(self, run_number):
        """
        Write the multipipe script used in piping CORSIKA to sim_telarray.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        run_command = self._make_run_command(
            run_number=run_number,
            input_file="-",  # instruct sim_telarray to take input from standard output
        )
        multipipe_file = Path(self.corsika_config.config_file_path.parent).joinpath(
            self.corsika_config.get_corsika_config_file_name("multipipe")
        )
        with open(multipipe_file, "w", encoding="utf-8") as file:
            file.write(f"{run_command}")
        self._logger.info(f"Multipipe script: {multipipe_file}")
        self._write_multipipe_script(multipipe_file)

    def _write_multipipe_script(self, multipipe_file):
        """
        Write script used to call the multipipe_corsika command.

        Parameters
        ----------
        multipipe_file: str or Path
            The name of the multipipe file which contains all of the multipipe commands.
        """
        multipipe_script = Path(self.corsika_config.config_file_path.parent).joinpath(
            "run_cta_multipipe"
        )
        with open(multipipe_script, "w", encoding="utf-8") as file:
            multipipe_command = Path(self._simtel_path).joinpath(
                f"sim_telarray/bin/multipipe_corsika -c {multipipe_file} || echo 'Fan-out failed'"
            )
            file.write(f"{multipipe_command}")

        multipipe_script.chmod(multipipe_script.stat().st_mode | stat.S_IEXEC)

    def _make_run_command(self, run_number=None, input_file=None):
        """
        Build and return the command to run simtel_array.

        Parameters
        ----------
        run_number: int
            Run number.
        input_file: str
            Full path of the input CORSIKA file.
            Use '-' to tell sim_telarray to read from standard output

        Returns
        -------
        str:
            Command to run sim_telarray.
        """
        try:
            weak_pointing = any(pointing in self.label for pointing in ["divergent", "convergent"])
        except TypeError:  # allow for self.label to be None
            weak_pointing = False

        command = str(self._simtel_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {self.corsika_config.array_model.get_config_file()}"
        command += f" -I{self.corsika_config.array_model.get_config_directory()}"
        command += self.simulator_array.get_config_option(
            "telescope_theta", self.corsika_config.zenith_angle, weak_option=weak_pointing
        )
        command += self.simulator_array.get_config_option(
            "telescope_phi", self.corsika_config.azimuth_angle, weak_option=weak_pointing
        )
        command += self.simulator_array.get_config_option(
            "power_law",
            SimulatorArray.get_power_law_for_sim_telarray_histograms(
                self.corsika_config.primary_particle
            ),
        )
        command += self.simulator_array.get_config_option(
            "histogram_file",
            self.get_file_name(
                simulation_software="simtel", file_type="histogram", run_number=run_number
            ),
        )
        command += self.simulator_array.get_config_option("random_state", "none")
        if self.sim_telarray_seeds:
            command += self.simulator_array.get_config_option(
                "random_seed", self.sim_telarray_seeds
            )
        command += self.simulator_array.get_config_option("show", "all")
        command += self.simulator_array.get_config_option(
            "output_file",
            self.simulator_array.get_file_name(
                simulation_software="simtel", file_type="output", run_number=run_number
            ),
        )
        command += f" {input_file}"
        _log_file = self.simulator_array.get_file_name(
            simulation_software="simtel", file_type="log", run_number=run_number
        )
        command += f" | gzip > {_log_file} 2>&1 || exit"

        # Remove the default sim_telarray configuration directories
        return clear_default_sim_telarray_cfg_directories(command)

    def get_file_name(self, simulation_software=None, file_type=None, run_number=None, mode=None):
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
        if simulation_software is None:
            # preference to simtel output (multipipe)
            simulation_software = "simtel" if self.simulator_array else "corsika"

        runner = self.corsika_runner if simulation_software == "corsika" else self.simulator_array
        return runner.get_file_name(file_type=file_type, run_number=run_number, mode=mode)
