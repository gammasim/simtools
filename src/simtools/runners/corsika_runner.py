"""Generate run scripts and directories for CORSIKA simulations."""

import logging
import stat
from pathlib import Path

from simtools import settings
from simtools.corsika.run_directory import link_run_directory
from simtools.io import io_handler
from simtools.runners.runner_services import RunnerServices


class CorsikaRunner:
    """
    Prepare and run CORSIKA simulations.

    Generate run scripts and directories for CORSIKA simulations. Run simulations if requested.

    CorsikaRunner is configured through a CorsikaConfig instance.

    Parameters
    ----------
    corsika_config_data: CorsikaConfig
        CORSIKA configuration.
    label: str
        Instance label.
    keep_seeds: bool
        Use seeds based on run number and primary particle. If False, use sim_telarray seeds.
    use_multipipe: bool
        Use multipipe to run CORSIKA and sim_telarray.
    curved_atmosphere_min_zenith_angle: Quantity
        Minimum zenith angle for which to use the curved-atmosphere CORSIKA binary.
    """

    def __init__(
        self,
        corsika_config,
        label=None,
        keep_seeds=False,
        use_multipipe=False,
        curved_atmosphere_min_zenith_angle=None,
    ):
        """Initialize CorsikaRunner."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")
        self.label = label

        self.corsika_config = corsika_config
        self._keep_seeds = keep_seeds
        self._use_multipipe = use_multipipe
        self.curved_atmosphere_min_zenith_angle = curved_atmosphere_min_zenith_angle

        self.io_handler = io_handler.IOHandler()

        self.runner_service = RunnerServices(corsika_config, label)
        simulation_software = "corsika" if not self._use_multipipe else "corsika_sim_telarray"
        self.runner_service.load_data_directories(simulation_software)

    def prepare_run(self, run_number=None, extra_commands=None, input_file=None):  # pylint: disable=unused-argument
        """
        Prepare CORSIKA run script and run directory.

        The CORSIKA run directory includes all input files needed for the simulation.

        Parameters
        ----------
        extra_commands: str
            Additional commands for running simulations.

        Returns
        -------
        List:
            List of files defined for the run.
        """
        run_files = self.runner_service.load_files(
            "corsika" if not self._use_multipipe else "corsika_sim_telarray", run_number=run_number
        )

        self.corsika_config.generate_corsika_input_file(
            self._use_multipipe,
            self._keep_seeds,
            run_files["corsika_input"],
            run_files["corsika_output"]
            if not self._use_multipipe
            else run_files["multi_pipe_script"],
        )

        self._logger.debug(f"Extra commands to be added to the run script: {extra_commands}")
        corsika_run_dir = run_files["corsika_output"].parent
        link_run_directory(corsika_run_dir, self._corsika_executable())

        self._export_run_script(run_files, corsika_run_dir, extra_commands)
        return run_files

    def _corsika_executable(self):
        """Get the CORSIKA executable path."""
        if self.corsika_config.use_curved_atmosphere:
            self._logger.debug("Using curved-atmosphere CORSIKA binary.")
            return Path(settings.config.corsika_exe_curved)
        self._logger.debug("Using flat-atmosphere CORSIKA binary.")
        return Path(settings.config.corsika_exe)

    def _export_run_script(self, run_files, corsika_run_dir, extra_commands):
        """Export CORSIKA run script."""
        corsika_log_file = run_files["corsika_log"].with_suffix("")
        with open(run_files["sub_script"], "w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n")
            file.write("set -e\n")
            file.write("set -o pipefail\n")

            # Setting SECONDS variable to measure runtime
            file.write("\nSECONDS=0\n")

            if extra_commands is not None:
                file.write("\n# Writing extras\n")
                file.write(f"{extra_commands}\n")
                file.write("# End of extras\n\n")

            file.write(f"export CORSIKA_DATA={corsika_run_dir}\n")
            file.write('mkdir -p "$CORSIKA_DATA"\n')
            file.write('cd "$CORSIKA_DATA" || exit 2\n')

            file.write("\n# Running corsika\n")
            file.write(
                f"{self._corsika_executable()} < {run_files['corsika_input']} "
                f"> {corsika_log_file} 2>&1\n"
            )
            file.write("\n# Cleanup\n")
            file.write(f"gzip {corsika_log_file}\n")

            file.write('\necho "RUNTIME: $SECONDS"\n')

        run_files["sub_script"].chmod(
            run_files["sub_script"].stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
        )
        return run_files["sub_script"]

    def get_resources(self, run_number=None):
        """Return computing resources used."""
        return self.runner_service.get_resources(run_number)

    def get_file_name(
        self,
        simulation_software="corsika",
        file_type=None,
        run_number=None,
        model_version_index=0,
    ):  # pylint: disable=unused-argument
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
            This is used to select the correct simulator_array instance in case
            multiple array models are simulated. (Not used here.)

        Returns
        -------
        str
            File name with full path.
        """
        if simulation_software.lower() != "corsika":
            raise ValueError(
                f"simulation_software ({simulation_software}) is not supported in CorsikaRunner"
            )
        try:
            return self.runner_service.get_file_name(file_type=file_type, run_number=run_number)
        except KeyError:
            # TODO is this a good solution?
            return self.runner_service.get_file_name(
                file_type=f"{simulation_software}_{file_type}", run_number=run_number
            )
