"""Generate run scripts and directories for CORSIKA simulations."""

import logging
import stat

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
        self._directory = self.runner_service.load_data_directories("corsika")

    def prepare_run(self, run_number=None, extra_commands=None, input_file=None):
        """
        Prepare CORSIKA run script and run directory.

        The CORSIKA run directory includes all input files needed for the simulation.

        Parameters
        ----------
        extra_commands: str
            Additional commands for running simulations.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        if run_number is not None and run_number != self.corsika_config.run_number:
            raise ValueError(
                f"run_number mismatch (given {run_number}, "
                f"expected {self.corsika_config.run_number})"
            )
        if input_file is not None:
            self._logger.warning("input_file parameter is not used in CorsikaRunner")

        run_files = {}
        for file_type in ["sub_script", "corsika_log", "corsika_input", "corsika_output"]:
            run_files[file_type] = self.get_file_name(
                file_type=file_type, run_number=self.corsika_config.run_number
            )
        self.corsika_config.generate_corsika_input_file(
            self._use_multipipe,
            self._keep_seeds,
            run_files["corsika_input"],
            run_files["corsika_output"],
        )

        self._logger.debug(f"Extra commands to be added to the run script: {extra_commands}")
        self._logger.debug(f"CORSIKA data will be set to {self._directory['corsika']}")

        for key, file_path in run_files.items():
            self._logger.debug(f"{key}: {file_path}")

        return self._export_run_script(run_files, extra_commands)

    def _export_run_script(self, run_files, extra_commands):
        """Export CORSIKA run script."""
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

            file.write(f"export CORSIKA_DATA={self._directory['corsika']}\n")
            file.write('mkdir -p "$CORSIKA_DATA"\n')
            file.write('cd "$CORSIKA_DATA" || exit 2\n')

            file.write("\n# Running corsika\n")
            # TODO file.write(autoinputs_command)
            file.write("\n# Cleanup\n")
            file.write(f"gzip {run_files['corsika_log']}\n")

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
