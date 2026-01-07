"""Run simulations with CORSIKA and pipe it to sim_telarray using the multipipe functionality."""

import logging
import stat

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
    is_calibration_run : bool
        Flag to indicate if this is a calibration run.
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        keep_seeds=False,
        sim_telarray_seeds=None,
        sequential=False,
        curved_atmosphere_min_zenith_angle=None,
        is_calibration_run=False,
    ):
        self._logger = logging.getLogger(__name__)
        self.corsika_config = gen.ensure_iterable(corsika_config)
        # the base corsika config is the one used to define the CORSIKA specific parameters.
        # The others are used for the array configurations.
        self.base_corsika_config = self.corsika_config[0]
        self.sim_telarray_seeds = sim_telarray_seeds
        self.label = label
        self.sequential = "--sequential" if sequential else ""

        self.runner_service = RunnerServices(self.base_corsika_config, "multi_pipe", label)
        self.file_list = None

        self.corsika_runner = CorsikaRunner(
            corsika_config=self.base_corsika_config,
            label=label,
            keep_seeds=keep_seeds,
            use_multipipe=True,
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
                    sim_telarray_seeds=sim_telarray_seeds,
                    is_calibration_run=is_calibration_run,
                )
            )

    def prepare_run(self, run_number=None, sub_script=None, corsika_file=None, extra_commands=None):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        run_number: int
            Run number.
        corsika_file: str or Path
            Path to the CORSIKA input file.
        extra_commands: str
            Additional commands for running simulations.
        """
        self.file_list = self.runner_service.load_files(run_number=run_number)
        self._export_multipipe_script(run_number)
        self.corsika_runner.prepare_run(
            run_number=run_number,
            sub_script=sub_script,
            corsika_file=self.runner_service.get_file_name(
                file_type="multi_pipe_script", run_number=run_number
            )
            if not corsika_file
            else corsika_file,
            extra_commands=extra_commands,
        )
        self.update_file_list_from_runners()

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
        multipipe_file = self.runner_service.get_file_name(
            "multi_pipe_config", run_number=run_number
        )

        with open(multipipe_file, "w", encoding="utf-8") as file:
            for simulator_array in self.simulator_array:
                run_command = simulator_array.make_run_command(
                    run_number=run_number,
                    corsika_input_file="-",  # instruct sim_telarray to take input from stdout
                    weak_pointing=self._determine_pointing_option(self.label),
                )
                file.write(f"{run_command}")
                file.write("\n")
        self._logger.info(f"Multipipe script: {multipipe_file}")
        self._write_multipipe_script(multipipe_file, run_number)

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

    def _write_multipipe_script(self, multipipe_file, run_number):
        """
        Write script used to call the multipipe_corsika command.

        Parameters
        ----------
        multipipe_file: str or Path
            The name of the multipipe file which contains all of the multipipe commands.
        run_number: int
            Run number.
        """
        multipipe_script = self.runner_service.get_file_name(
            "multi_pipe_script", run_number=run_number
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

    def update_file_list_from_runners(self):
        """
        Get list of generated files (independent of type).

        Parameters
        ----------
        file_type : str
            File type to be listed.

        Returns
        -------
        list
            List with the full path of all output files.

        """
        if self.file_list is None:
            self.file_list = self.corsika_runner.file_list
        else:
            self.file_list.update(self.corsika_runner.file_list)

        for simulator_array in self.simulator_array:
            _tmp_list = simulator_array.file_list
            for key, data in _tmp_list.items():
                if key in self.file_list:
                    # in case of multiple sim_telarray instances, make list of files
                    if not isinstance(self.file_list[key], list):
                        self.file_list[key] = [self.file_list[key]]
                    self.file_list[key].append(data)
                else:
                    self.file_list[key] = [data]

        print("UUUUU", self.file_list)

        return self.file_list
