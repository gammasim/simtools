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
    corsika_config : CorsikaConfig or list of CorsikaConfig
        A list of "CorsikaConfig" instances which
        contain the CORSIKA configuration parameters.
    simtel_path : str or Path
        Location of the sim_telarray package.
    label : str
        Label.
    keep_seeds : bool
        Use seeds based on run number and primary particle. If False, use sim_telarray seeds.
    use_multipipe : bool
        Use multipipe to run CORSIKA and sim_telarray.
    sim_telarray_seeds : dict
        Dictionary with configuration for sim_telarray random instrument setup.
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
        if not isinstance(corsika_config, list):
            corsika_config = [corsika_config]
        self.corsika_config = corsika_config
        # the base corsika config is the one used to define the CORSIKA specific parameters.
        # The others are used for the array configurations.
        self.base_corsika_config = corsika_config[0]
        self._simtel_path = simtel_path
        self.sim_telarray_seeds = sim_telarray_seeds
        self.label = label

        self.base_corsika_config.set_output_file_and_directory(use_multipipe)
        self.corsika_runner = CorsikaRunner(
            corsika_config=self.base_corsika_config,
            simtel_path=simtel_path,
            label=label,
            keep_seeds=keep_seeds,
            use_multipipe=use_multipipe,
        )
        # The simulator array should be defined for every CORSIKA configuration
        # because it allows to define multiple sim_telarray instances
        self.simulator_array = []
        for _corsika_config in self.corsika_config:
            self.simulator_array.append(
                SimulatorArray(
                    corsika_config=_corsika_config,
                    simtel_path=simtel_path,
                    label=label,
                    use_multipipe=use_multipipe,
                    sim_telarray_seeds=sim_telarray_seeds,
                )
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
        multipipe_file = Path(self.base_corsika_config.config_file_path.parent).joinpath(
            self.base_corsika_config.get_corsika_config_file_name("multipipe")
        )
        with open(multipipe_file, "w", encoding="utf-8") as file:
            for corsika_config, simulator_array in zip(self.corsika_config, self.simulator_array):
                run_command = self._make_run_command(
                    run_number=run_number,
                    input_file="-",  # instruct sim_telarray to take input from standard output
                    corsika_config=corsika_config,
                    simulator_array=simulator_array,
                )
                file.write(f"{run_command}")
                file.write("\n")
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
        multipipe_script = Path(self.base_corsika_config.config_file_path.parent).joinpath(
            "run_cta_multipipe"
        )
        with open(multipipe_script, "w", encoding="utf-8") as file:
            multipipe_command = Path(self._simtel_path).joinpath(
                f"sim_telarray/bin/multipipe_corsika -c {multipipe_file} || echo 'Fan-out failed'"
            )
            file.write(f"{multipipe_command}")

        multipipe_script.chmod(multipipe_script.stat().st_mode | stat.S_IEXEC)

    def _make_run_command(self, run_number, input_file, corsika_config, simulator_array):
        """
        Build and return the command to run sim_telarray.

        Parameters
        ----------
        run_number: int
            Run number.
        input_file: str
            Full path of the input CORSIKA file.
            Use '-' to tell sim_telarray to read from standard output
        corsika_config: CorsikaConfig
            CORSIKA configuration.
        simulator_array: SimulatorArray
            SimulatorArray instance.

        Returns
        -------
        str:
            Command to run sim_telarray.
        """
        try:
            weak_pointing = any(pointing in self.label for pointing in ["divergent", "convergent"])
        except TypeError:  # allow for self.label to be None
            weak_pointing = False

        corsika_config.array_model.export_all_simtel_config_files()

        command = str(self._simtel_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {corsika_config.array_model.config_file_path}"
        command += f" -I{corsika_config.array_model.get_config_directory()}"
        command += simulator_array.get_config_option(
            "telescope_theta", corsika_config.zenith_angle, weak_option=weak_pointing
        )
        command += simulator_array.get_config_option(
            "telescope_phi", corsika_config.azimuth_angle, weak_option=weak_pointing
        )
        command += simulator_array.get_config_option(
            "power_law",
            SimulatorArray.get_power_law_for_sim_telarray_histograms(
                corsika_config.primary_particle
            ),
        )
        command += simulator_array.get_config_option(
            "histogram_file",
            simulator_array.get_file_name(
                simulation_software="simtel", file_type="histogram", run_number=run_number
            ),
        )
        command += simulator_array.get_config_option("random_state", "none")

        if self.sim_telarray_seeds and self.sim_telarray_seeds.get("random_instances"):
            config_dir = Path(corsika_config.array_model.config_file_path).parent
            command += simulator_array.get_config_option(
                "random_seed",
                f"file-by-run:{config_dir}/{self.sim_telarray_seeds['seed_file_name']},auto",
            )
        elif self.sim_telarray_seeds and self.sim_telarray_seeds.get("seed"):
            command += simulator_array.get_config_option(
                "random_seed", self.sim_telarray_seeds["seed"]
            )
        command += simulator_array.get_config_option("show", "all")
        command += simulator_array.get_config_option(
            "output_file",
            simulator_array.get_file_name(
                simulation_software="simtel", file_type="output", run_number=run_number
            ),
        )
        command += f" {input_file}"
        _log_file = simulator_array.get_file_name(
            simulation_software="simtel", file_type="log", run_number=run_number
        )
        command += f" | gzip > {_log_file} 2>&1 || exit"

        logging.debug(f"sim_telarray command to be included in the multipipe cfg: {command}")

        # Remove the default sim_telarray configuration directories
        return clear_default_sim_telarray_cfg_directories(command)

    def get_file_name(
        self,
        simulation_software=None,
        file_type=None,
        run_number=None,
        mode=None,
        model_version_index=0,
    ):
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
        mode: str
            Mode to use for the file name.
        model_version_index: int
            Index of the model version.
            This is used to select the correct simulator_array instance
            in case multiple array models are simulated.

        Returns
        -------
        str
            File name with full path.
        """
        if simulation_software is None:
            # preference to simtel output (multipipe)
            simulation_software = "simtel" if self.simulator_array else "corsika"

        runner = (
            self.corsika_runner
            if simulation_software == "corsika"
            else self.simulator_array[model_version_index]
        )
        return runner.get_file_name(file_type=file_type, run_number=run_number, mode=mode)
