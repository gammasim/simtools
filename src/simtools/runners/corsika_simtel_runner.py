"""Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality."""

import logging
import stat
from pathlib import Path

import simtools.utils.general as gen
from simtools import settings
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.runner_services import RunnerServices
from simtools.simtel.simulator_array import SimulatorArray


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
    label : str
        Label.
    keep_seeds : bool
        Use seeds based on run number and primary particle. If False, use sim_telarray seeds.
    use_multipipe : bool
        Use multipipe to run CORSIKA and sim_telarray.
    sim_telarray_seeds : dict
        Dictionary with configuration for sim_telarray random instrument setup.
    calibration_config : dict
        Configuration for the calibration of the sim_telarray data.
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        keep_seeds=False,
        use_multipipe=False,
        sim_telarray_seeds=None,
        sequential=False,
        calibration_config=None,
        curved_atmosphere_min_zenith_angle=None,
    ):
        self._logger = logging.getLogger(__name__)
        self.corsika_config = gen.ensure_iterable(corsika_config)
        # the base corsika config is the one used to define the CORSIKA specific parameters.
        # The others are used for the array configurations.
        self.base_corsika_config = self.corsika_config[0]
        self.sim_telarray_seeds = sim_telarray_seeds
        self.label = label
        self.sequential = "--sequential" if sequential else ""

        self.runner_service = RunnerServices(self.base_corsika_config, label)
        self._directory = self.runner_service.load_data_directories(
            "corsika_sim_telarray" if use_multipipe else "sim_telarray"
        )

        self.corsika_runner = CorsikaRunner(
            corsika_config=self.base_corsika_config,
            label=label,
            keep_seeds=keep_seeds,
            use_multipipe=use_multipipe,
            curved_atmosphere_min_zenith_angle=curved_atmosphere_min_zenith_angle,
        )
        # The simulator array should be defined for every CORSIKA configuration
        # because it allows to define multiple sim_telarray instances
        self.simulator_array = []
        for _corsika_config in self.corsika_config:
            self.simulator_array.append(
                SimulatorArray(
                    corsika_config=_corsika_config,
                    label=label,
                    use_multipipe=use_multipipe,
                    sim_telarray_seeds=sim_telarray_seeds,
                    calibration_config=calibration_config,
                )
            )

    def prepare_run(self, run_number=None, input_file=None, extra_commands=None):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        self._export_multipipe_script(run_number)
        return self.corsika_runner.prepare_run(input_file=input_file, extra_commands=extra_commands)

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
        multipipe_file = self.runner_service.get_file_name("multipipe_config")

        with open(multipipe_file, "w", encoding="utf-8") as file:
            for simulator_array in self.simulator_array:
                run_command = simulator_array.make_run_command(
                    run_number=run_number,
                    input_file="-",  # instruct sim_telarray to take input from standard output
                    weak_pointing=self._determine_pointing_option(self.label),
                )
                file.write(f"{run_command}")
                file.write("\n")
        self._logger.info(f"Multipipe script: {multipipe_file}")
        self._write_multipipe_script(multipipe_file)

    @staticmethod
    def _determine_pointing_option(label):
        """
        Determine the pointing option for sim_telarray.

        Parameters
        ----------
        label: str
            Label of the simulation.

        Returns
        -------
        str:
            Pointing option.
        """
        try:
            return any(pointing in label for pointing in ["divergent", "convergent"])
        except TypeError:  # allow for pointing_option to be None
            pass
        return False

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
            multipipe_command = settings.config.sim_telarray_path.joinpath(
                f"bin/multipipe_corsika -c {multipipe_file} {self.sequential} "
                "|| echo 'Fan-out failed'"
            )
            file.write(f"{multipipe_command}")

        multipipe_script.chmod(multipipe_script.stat().st_mode | stat.S_IEXEC)

    def get_file_name(
        self,
        simulation_software=None,
        file_type=None,
        run_number=None,
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
            # preference to sim_telarray output (multipipe)
            simulation_software = "sim_telarray" if self.simulator_array else "corsika"

        runner = (
            self.corsika_runner
            if simulation_software == "corsika"
            else self.simulator_array[model_version_index]
        )
        return runner.get_file_name(file_type=file_type, run_number=run_number)

    def get_resources(self, run_number=None):
        """Return computing resources used."""
        return self.corsika_runner.get_resources(run_number)
